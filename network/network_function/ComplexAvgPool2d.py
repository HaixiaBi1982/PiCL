import torch
from torch.nn import Module
from torch.nn.functional import avg_pool2d


class ComplexAvgPool2d(Module):

    def __init__(self, kernel_size, stride=None, padding=0,
                 return_indices=False, ceil_mode=False):
        super(ComplexAvgPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.ceil_mode = ceil_mode

    def forward(self, input):
        return complex_avg_pool2d(input, kernel_size=self.kernel_size,
                                  stride=self.stride, padding=self.padding,
                                  ceil_mode=self.ceil_mode
                                  )


def complex_avg_pool2d(input, *args, **kwargs):
    '''
    Perform complex average pooling.
    '''
    absolute_value_real = avg_pool2d(input.real, *args, **kwargs)
    absolute_value_imag = avg_pool2d(input.imag, *args, **kwargs)

    return absolute_value_real.type(torch.complex64) + 1j * absolute_value_imag.type(torch.complex64)
