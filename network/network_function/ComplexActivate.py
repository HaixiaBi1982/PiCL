import torch
from torch import sigmoid
from torch.nn import Module
import torch.nn.functional as F


def complex_sigmoid(input):
    return sigmoid(input.real).type(torch.complex64) + 1j * sigmoid(input.imag).type(torch.complex64)


class ComplexSigmoid(Module):

    def forward(self, input):
        return complex_sigmoid(input)


# ---------------------------------------------------------------------------------------------------------------------


def complex_softmax(input):
    return F.softmax(input.real, dim=1).type(torch.complex64) + 1j * F.softmax(input.imag, dim=1).type(torch.complex64)


class ComplexSoftmax(Module):

    def forward(self, input):
        return complex_softmax(input)
