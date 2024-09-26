import copy
import math
import random
import numpy as np
import torch
from torch import nn
from torchvision import transforms as T


class RandomApply_for_n_times(nn.Module):
    def __init__(self, fn, n):
        super().__init__()
        self.fn = fn
        self.n = n

    def forward(self, x):
        p = 1 / self.n
        r = random.random()
        if r == 0:
            return x
        for i in range(self.n):
            x = self.fn(x)
            if i * p < r <= (i + 1) * p:
                return x


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


# Noise PSG
class add_complex_noise(torch.nn.Module):
    def __init__(self, scale=0.15):
        super().__init__()
        self.scale = scale

    def forward(self, data):
        data = data.cpu().numpy()
        W, H = data.shape[1], data.shape[2]
        for w in range(W):
            for h in range(H):
                for i in range(data.shape[0]):
                    n_a = np.random.normal(1.0, self.scale, 1)
                    assert n_a > 0
                    if (data[i, w, h].imag != 0).any():  # complex elements
                        n_p = np.random.uniform(low=0, high=2 * math.pi, size=None)
                        noise = n_a * np.exp(1j * n_p)
                        data[i, w, h] = data[i, w, h] * noise
                    else:  # real elements
                        noise = n_a ** 2
                        data[i, w, h] = data[i, w, h] * noise
        return torch.tensor(data).cuda()


# WishartPSG
def dataset_augmentation_of_wishartPSG(dataset, augment_list=None):
    original_dataset = copy.deepcopy(dataset)
    original_length = copy.deepcopy(len(original_dataset))
    new_dataset = copy.deepcopy(original_dataset)
    new_dataset.data = []
    new_dataset.labels = []
    new_dataset.position = []
    if augment_list is None:
        print("Augment_list is None!!!")
        return new_dataset

    for i in range(len(augment_list)):
        if augment_list[i] is not None:
            L = copy.deepcopy(len(new_dataset))
            count = 0
            for j in range(original_length):
                original_dataset.data[j] = torch.as_tensor(original_dataset.data[j])

                data_list, position_list = augment_list[i](original_dataset.data[j], original_dataset.position[j])
                assert len(data_list)==len(position_list)!=0
                for k in range(len(data_list)):
                    new_dataset.data.append(torch.tensor(data_list[k]))
                    new_dataset.labels.append(original_dataset.labels[j])
                    new_dataset.position.append(position_list[k])
                    count += 1
            print("{} --augment_wishartPSG_{}({} pixel selected)--> {} Aug:{}".format(L, i + 1, augment_list[i].num, len(new_dataset), count))
    return new_dataset


class batch_augmentation_of_wishartPSG(torch.nn.Module):
    def __init__(self, dataset, random_select=False, number_of_selected_pixels=1):
        super().__init__()
        self.sampler = PSG_OF_Nearset_Wishart(dataset, random_select, number_of_selected_pixels)

    def forward(self, batch, position_list):
        aug_data_list = []
        for i in range(batch.shape[0]):
            data = batch[i, ...].cpu().numpy()
            position = (position_list[0][i].cpu().item(), position_list[1][i].cpu().item())
            min_patch_list, _ = self.sampler(data, position)
            aug_data_list.append(min_patch_list[0])
        aug_data = torch.tensor(np.array(aug_data_list)).to(batch.device)
        return aug_data


class PSG_OF_Nearset_Wishart(torch.nn.Module):
    def __init__(self, dataset, random_select=False, number_of_selected_pixels=1):
        super().__init__()
        self.data = dataset.total_data
        self.window_center = dataset.window_center
        self.window_size = dataset.window_size
        self.random_select = random_select
        if self.random_select:
            print("!!! random_select for wishartPSG !!!")
        self.num = number_of_selected_pixels
        assert not dataset.std

        self.label = dataset.total_labels
        self.class_num = dataset.class_num

    def forward(self, data, position):
        #  find the nearest patch in self.data
        surroundings = get_surrounding_coordinates(position, self.window_center, self.window_size)
        candidate_pixel_data = find_indexes(surroundings, self.data, position)
        if not self.random_select:
            # compute wishart distance
            distance_list = np.zeros([len(candidate_pixel_data)])
            position_list = []

            patch_show=False
            if patch_show:
                W_d_m = np.zeros([self.window_size, self.window_size])

            for i, di in enumerate(candidate_pixel_data):
                distance_list[i] = Wishart_distance(data[:, self.window_center[0], self.window_center[1]], di[0])
                position_list.append(di[1])
                if patch_show:
                    W_d_m[di[1][0]-position[0]-self.window_center[0], di[1][1]-position[1]-self.window_center[1]] = distance_list[i]

            sorted_indices = np.argsort(distance_list)
            smallest_indices = sorted_indices[:self.num]

            # The n elements closest to wishart are selected as positive samples
            min_patch_list = []
            min_position_list = []
            for min_indices in smallest_indices:
                min_data = candidate_pixel_data[min_indices]

                assert position_list[min_indices] == min_data[1]
                assert self.window_center[0] == self.window_center[1] == self.window_size // 2
                min_patch = extract_subarray(self.data, min_data[1], self.window_size)
                min_patch_list.append(min_patch)
                min_position_list.append(min_data[1])
            return min_patch_list, min_position_list
        else:
            random.seed(42)
            rand_patch_list = []
            rand_position_list = []
            for i in range(self.num):
                rand_data, rand_position = random.choice(candidate_pixel_data)
                rand_patch = extract_subarray(self.data, rand_position, self.window_size)
                rand_patch_list.append(rand_patch)
                rand_position_list.append(rand_position)
            return rand_patch_list, rand_position_list



def extract_subarray(arr, position, window_size):
    start_x = position[0] - window_size // 2 -1
    start_y = position[1] - window_size // 2 -1
    end_x = start_x + window_size
    end_y = start_y + window_size
    if start_x < 0 or start_y < 0 or end_x > arr.shape[1] or end_y > arr.shape[2]:
        pad_x = max(0 - start_x, 0)
        pad_y = max(0 - start_y, 0)
        pad_end_x = max(end_x - arr.shape[1], 0)
        pad_end_y = max(end_y - arr.shape[2], 0)
        arr = np.pad(arr, ((0, 0), (pad_x, pad_end_x), (pad_y, pad_end_y)), mode='constant')
        start_x += pad_x
        start_y += pad_y
        end_x += pad_x
        end_y += pad_y
    subarray = arr[:, start_x:end_x, start_y:end_y]
    assert subarray.shape == (6, window_size, window_size)
    return subarray

def get_surrounding_coordinates(coordinates, window_center, window_size):
    x, y = coordinates
    result = []
    for i in range(x - window_center[0], x + window_size - window_center[0]):
        for j in range(y - window_center[1], y + window_size - window_center[1]):
            result.append((i, j))
    return result

def find_indexes(surroundings, total_data, position):
    indexes = []
    assert position[0]<=total_data.shape[1] and position[1]<=total_data.shape[2]
    for item in surroundings:
        if position != (item[0], item[1]):
            try:
                indexes.append((total_data[:, item[0]-1, item[1]-1], (item[0], item[1])))
            except:
                pass
    return indexes

def data6_to_T_matrix(data6):
    T_matrix = np.array(
    [
        [data6[0], data6[1], data6[2]],
        [np.conj(data6[1]), data6[3], data6[4]],
        [np.conj(data6[2]), np.conj(data6[4]), data6[5]]
    ])
    return T_matrix

def Wishart_distance(T1, T2):
    T1 = data6_to_T_matrix(copy.deepcopy(T1))
    T2 = data6_to_T_matrix(copy.deepcopy(T2))
    inv_T1 = np.linalg.inv(T1)
    inv_T2 = np.linalg.inv(T2)
    D12 = (np.log(np.linalg.det(T1)) + np.log(np.linalg.det(T2)) + np.trace(inv_T1@T2 + inv_T2@T1))/2
    return D12.real


def dataset_augmentation(dataset, classes=None, augment_list=None):
    original_dataset = copy.deepcopy(dataset)
    original_length = copy.deepcopy(len(original_dataset))
    new_dataset = copy.deepcopy(original_dataset)
    new_dataset.data = []
    new_dataset.labels = []
    new_dataset.position = []
    sample_center = original_dataset.__get_window_center__()

    if classes is not None:
        classes += 1
    # 0-14 mapping to 1-15

    if augment_list is None:
        print("Augment_list is None!!!")
        return new_dataset

    for i in range(len(augment_list)):
        if augment_list[i] is not None:
            L = copy.deepcopy(len(new_dataset))
            count = 0
            for j in range(original_length):
                original_dataset.data[j] = torch.as_tensor(original_dataset.data[j])
                if classes is None:
                    d = augment_list[i](original_dataset.data[j])
                    if d.shape==(6, original_dataset.window_size, original_dataset.window_size):
                        assert d.shape!=[1]
                        new_dataset.data.append(augment_list[i](original_dataset.data[j]).cpu())
                        new_dataset.labels.append(original_dataset.labels[j])
                        new_dataset.position.append(original_dataset.position[j])
                        count += 1
                    else:
                        pass
                else:
                    if original_dataset.labels[j][sample_center] == classes:
                        if augment_list[i](original_dataset.data[j]) is not None:
                            new_dataset.data.append(augment_list[i](original_dataset.data[j]).cpu())
                            new_dataset.labels.append(original_dataset.labels[j])
                            new_dataset.position.append(original_dataset.position[j])
                            count += 1

            print("{} --augment_fn_{}--> {} Aug:{}".format(L, i + 1, len(new_dataset), count))
    return new_dataset


def dataset_sampling(original_dataset, sampling_rate, random_seed=1):  # 数据采样
    random.seed(random_seed)
    class_num = original_dataset.__get_class_num__()
    sample_center = original_dataset.__get_window_center__()

    label_list = []
    for i in range(class_num):
        label_list.append([])

    data_list = copy.deepcopy(label_list)
    position_list = copy.deepcopy(label_list)

    for i in range(len(original_dataset)):
        label = original_dataset.labels[i][sample_center] - 1
        assert 0 <= label <= class_num - 1
        label_list[label].append(original_dataset.labels[i])
        data_list[label].append(original_dataset.data[i])
        position_list[label].append(original_dataset.position[i])

    num = np.zeros(class_num)
    for i in range(len(original_dataset)):
        label = original_dataset.labels[i][sample_center] - 1
        assert 0 <= label <= class_num - 1
        num[label] += 1

    sampled_dataset = copy.deepcopy(original_dataset)
    sampled_dataset.data = []
    sampled_dataset.labels = []
    sampled_dataset.position = []

    for i in range(class_num):
        used_data_num = math.ceil(sampling_rate * len(label_list[i]))
        indexes = random.sample(range(len(label_list[i])), used_data_num)

        for j in indexes:
            sampled_dataset.data.append(torch.as_tensor(data_list[i][j]))
            sampled_dataset.labels.append(torch.as_tensor(label_list[i][j]))
            sampled_dataset.position.append(position_list[i][j])
    print("-" * 30)
    print("sampling_rate={:.2%} used_data_num:{}\n".format(sampling_rate, len(sampled_dataset)), end="")
    print("-" * 30)

    return sampled_dataset


def count_dataset_classes_num(dataset):
    global label_list
    class_num = dataset.__get_class_num__()
    sample_center = dataset.__get_window_center__()

    n = [0] * class_num
    for i in range(len(dataset)):
        label = dataset.labels[i][sample_center] - 1
        assert 0 <= label <= class_num - 1
        n[int(label)] += 1

    if class_num == 15:
        label_list = ['Water', 'Barley', 'Peas', 'Stembeans', 'Beet', 'Forest',
                      'Bare soil', 'Grass',
                      'Rapeseed', 'Lucerne', 'Wheat 2', 'Wheat', 'Buildings', 'Potatoes', 'Wheat 3']
    else:
        assert 0
    classes_num = dict(zip(label_list, list(n)))
    total = 0
    for key, value in classes_num.items():
        print('{}:{}({:.2%})'.format(key, value, value / len(dataset)))
        total += value / len(dataset)
    print("total:{}({:.2%})\n".format(len(dataset), total))

    return n


# Define aug methods
HorizontalFlip = torch.nn.Sequential(T.RandomHorizontalFlip(p=1))

rotate90 = torch.nn.Sequential(Complex_RandomApply(T.RandomRotation((90, 90)), p=1))
rotate180 = torch.nn.Sequential(Complex_RandomApply(T.RandomRotation((180, 180)), p=1))
rotate270 = torch.nn.Sequential(Complex_RandomApply(T.RandomRotation((270, 270)), p=1))

HFlip_rotate90 = torch.nn.Sequential(T.RandomHorizontalFlip(p=1),
                                     Complex_RandomApply(T.RandomRotation((90, 90)), p=1))
HFlip_rotate180 = torch.nn.Sequential(T.RandomHorizontalFlip(p=1),
                                      Complex_RandomApply(T.RandomRotation((180, 180)), p=1))
HFlip_rotate270 = torch.nn.Sequential(T.RandomHorizontalFlip(p=1),
                                      Complex_RandomApply(T.RandomRotation((270, 270)), p=1))



def dataset_addition(dataset1, dataset2):
    new_dataset = copy.deepcopy(dataset1)
    new_dataset.data = []
    new_dataset.labels = []
    new_dataset.position = []

    new_dataset.data = dataset1.data + dataset2.data
    new_dataset.labels = dataset1.labels + dataset2.labels
    new_dataset.position = dataset1.position + dataset2.position

    return new_dataset
