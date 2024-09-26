import torch
from torch.nn import Module

class ComplexAdaptiveMaxPool2d(Module):

    def __init__(self, size):
        super(ComplexAdaptiveMaxPool2d, self).__init__()
        self.size = size

    def forward(self, input):
        max_pool = torch.nn.AdaptiveMaxPool2d(self.size)
        real = input.real
        imag = input.imag
        real = max_pool(real)
        imag = max_pool(imag)
        output = torch.complex(real, imag)

        return output


class ComplexAdaptiveMaxPool1d(Module):

    def __init__(self, size):
        super(ComplexAdaptiveMaxPool1d, self).__init__()
        self.size = size

    def forward(self, input):
        max_pool = torch.nn.AdaptiveMaxPool1d(self.size)
        real = input.real
        real = max_pool(real)
        imag = input.imag
        imag = max_pool(imag)
        output = torch.complex(real, imag)
        return output


if __name__ == "__main__":
    input = torch.randn(size=[1, 2, 4, 4], dtype=torch.complex64)
    max_pool = ComplexAdaptiveMaxPool2d(6)
    output = max_pool(input)
    print(output.shape)
