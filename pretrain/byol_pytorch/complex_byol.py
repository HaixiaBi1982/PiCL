import copy
import random
from functools import wraps

import torch
from complexPyTorch.complexLayers import ComplexLinear, ComplexBatchNorm1d, ComplexReLU
from torch import nn
import torch.nn.functional as F

from dataset_preparing.dataset_aug import add_complex_noise, rotate90, rotate180, rotate270, \
    HFlip_rotate90, HFlip_rotate180, HFlip_rotate270, HorizontalFlip, \
    batch_augmentation_of_wishartPSG


def flatten(t):
    return t.reshape(t.shape[0], -1)


def singleton(cache_key):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance

            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance

        return wrapper

    return inner_fn


def get_module_device(module):
    return next(module.parameters()).device


def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val


# loss fn
def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)

    loss1 = 2 - 2 * (x * y).sum(dim=-1)
    loss2 = torch.norm((x - y), p=2, dim=-1) ** 2

    return loss2


# augmentation utils
class Complex_RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p

    def forward(self, x):
        if random.random() > self.p:
            return x
        if x.dtype == torch.complex64:
            o_real = self.fn(x.real)
            o_imag = self.fn(x.imag)
            return torch.complex(o_real, o_imag)
        else:
            return self.fn(x)


# exponential moving average
class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)


# 复数 MLP
def MLP(dim, projection_size, hidden_size):
    return nn.Sequential(
        ComplexLinear(dim, hidden_size),
        ComplexBatchNorm1d(hidden_size),
        ComplexReLU(),
        ComplexLinear(hidden_size, projection_size)
    )


def SimSiamMLP(dim, projection_size, hidden_size):
    return nn.Sequential(
        ComplexLinear(dim, hidden_size),
        ComplexBatchNorm1d(hidden_size),
        ComplexReLU(),
        ComplexLinear(hidden_size, hidden_size),
        ComplexBatchNorm1d(hidden_size),
        ComplexReLU(),
        ComplexLinear(hidden_size, projection_size),
        ComplexBatchNorm1d(projection_size, affine=False)
    )


# a wrapper class for the base neural network
# will manage the interception of the hidden layer output
# and pipe it into the projecter and predictor nets
class NetWrapper(nn.Module):
    def __init__(self, net, projection_size, projection_hidden_size, layer=-2, use_simsiam_mlp=False):
        super().__init__()
        self.net = net
        self.layer = layer

        self.projector = None
        self.projection_size = projection_size
        self.projection_hidden_size = projection_hidden_size

        self.use_simsiam_mlp = use_simsiam_mlp

        self.hidden = {}
        self.hook_registered = False

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _find_layer(self):
        if type(self.layer) == str:
            modules = dict([*self.net.named_modules()])
            return modules.get(self.layer, None)
        elif type(self.layer) == int:
            children = [*self.net.children()]
            return children[self.layer]
        return None

    def _hook(self, _, input, output):
        device = input[0].device
        self.hidden[device] = flatten(output)

    def _register_hook(self):
        layer = self._find_layer()
        assert layer is not None, f'hidden layer ({self.layer}) not found'
        handle = layer.register_forward_hook(self._hook)
        self.hook_registered = True

    @singleton('projector')
    def _get_projector(self, hidden):
        _, dim = hidden.shape
        create_mlp_fn = MLP if not self.use_simsiam_mlp else SimSiamMLP
        projector = create_mlp_fn(dim, self.projection_size, self.projection_hidden_size)
        return projector.to(self.device)

    def get_representation(self, x):
        if self.layer == -1:
            return self.net(x)

        if not self.hook_registered:
            self._register_hook()

        self.hidden.clear()
        _ = self.net(x)
        hidden = self.hidden[x.device]
        self.hidden.clear()

        assert hidden is not None, f'hidden layer {self.layer} never emitted an output'
        return hidden

    def forward(self, x, return_projection=True):
        representation = self.get_representation(x)
        if not return_projection:
            return representation

        projector = self._get_projector(representation)
        projection = projector(representation.data)
        return projection, representation


class ComplexBYOL(nn.Module):
    def __init__(
            self,
            net,
            image_size,
            mean,
            std,
            hidden_layer=-1,
            projection_size=256,
            projection_hidden_size=4096,
            moving_average_decay=0.99,
            use_momentum=True,
            dataset=None,
    ):
        super().__init__()
        self.net = net
        self.mean = mean
        self.std = std


        self.WeakPSG = [rotate90, rotate180, rotate270, HFlip_rotate90, HFlip_rotate180, HFlip_rotate270, HorizontalFlip]
        self.NoisePSG = torch.nn.Sequential(add_complex_noise())
        self.WishartPSG = batch_augmentation_of_wishartPSG(dataset=dataset)
        self.PSG_categories = ['WeakPSG', 'NoisePSG', 'WishartPSG']

        self.online_encoder = NetWrapper(net, projection_size, projection_hidden_size, layer=hidden_layer,
                                         use_simsiam_mlp=not use_momentum)
        self.original_online_encoder = copy.deepcopy(self.online_encoder)

        self.use_momentum = use_momentum
        self.target_encoder = None
        self.target_ema_updater = EMA(moving_average_decay)

        self.online_predictor = MLP(projection_size, projection_size, projection_hidden_size)

        # get device of network and make wrapper same device
        device = get_module_device(net)
        self.to(device)

        # # send a mock image tensor to instantiate singleton parameters
        # test = torch.randn(2, 6, image_size, image_size, device=device, dtype=float32)
        # test = torch.complex(test, test)
        # self.forward(test)

    @singleton('target_encoder')
    def _get_target_encoder(self):
        target_encoder = copy.deepcopy(self.original_online_encoder)
        set_requires_grad(target_encoder, False)
        return target_encoder

    def set_requires_grad(model, val):
        for p in model.parameters():
            p.requires_grad = val

    def reset_moving_average(self):
        del self.target_encoder
        self.target_encoder = None

    def update_moving_average(self):
        assert self.use_momentum, 'you do not need to update the moving average, since you have turned off momentum for the target encoder'
        assert self.target_encoder is not None, 'target encoder has not been created yet'
        update_moving_average(self.target_ema_updater, self.target_encoder, self.online_encoder)

    def operation_for_category(self, category):
        if category == 'WeakPSG':
            return random.sample(self.WeakPSG, 1)[0]
        elif category == 'NoisePSG':
            return self.NoisePSG
        elif category == 'WishartPSG':
            return self.WishartPSG
        else:
            assert 0, "Invalid category"

    def forward(
            self,
            x,
            position,
            return_embedding=False,
            return_projection=True
    ):

        assert not (self.training and x.shape[
            0] == 1), 'you must have greater than 1 sample when training, due to the batchnorm in the projection layer'

        if return_embedding:
            return self.online_encoder(x, return_projection=return_projection)

        random_select = random.sample(self.PSG_categories, 2)
        augment1, augment2 = self.operation_for_category(random_select[0]), self.operation_for_category(random_select[1])

        image_one = augment1(x, position) if random_select[0] == "WishartPSG" else augment1(x)
        image_two = augment2(x, position) if random_select[1] == "WishartPSG" else augment2(x)

        online_proj_one, _ = self.online_encoder(image_one)
        online_proj_two, _ = self.online_encoder(image_two)

        online_pred_one = self.online_predictor(online_proj_one)
        online_pred_two = self.online_predictor(online_proj_two)

        with torch.no_grad():
            target_encoder = self._get_target_encoder() if self.use_momentum else self.online_encoder

            target_proj_one, _ = target_encoder(image_one)
            target_proj_two, _ = target_encoder(image_two)

            target_proj_one.detach_()
            target_proj_two.detach_()

        loss_one = loss_fn(online_pred_one, target_proj_two)
        loss_two = loss_fn(online_pred_two, target_proj_one)

        loss = (loss_one + loss_two) / 2
        return loss.mean()


def get_requires_grd(model):
    for p in model.parameters():
        if not p.requires_grad:
            return p.requires_grad
    return True
