import torch
import torch.nn as nn
from complexPyTorch.complexLayers import ComplexConv2d, ComplexLinear, ComplexReLU
from torch.nn import Sequential, Flatten

from network.network_function import ComplexSigmoid, ComplexSoftmax
from network.network_function.ComplexAdaptiveMaxPool import ComplexAdaptiveMaxPool2d
from network.network_function.ComplexAvgPool2d import ComplexAvgPool2d


class ChannelAttention(torch.nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)  # B*C*W*H---(squeeze)--->B*C*1*1
        self.max_pool = ComplexAdaptiveMaxPool2d(1)  # B*C*W*H---(squeeze)--->B*C*1*1

        self.fc = torch.nn.Sequential(
            ComplexLinear(channel, channel // reduction),
            ComplexReLU(),
            ComplexLinear(channel // reduction, channel),
            ComplexSigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        U_avg = self.avg_pool(x).view(b, c)
        U_max = self.max_pool(x).view(b, c)
        y = self.fc(U_avg).view(b, c, 1, 1) + self.fc(U_max).view(b, c, 1, 1)
        return x * y

class SpatialAttention(torch.nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = ComplexConv2d(2, 1, 3, 1, 1,
                                  bias=False)
        'kernel size must be 3 or 7'
        self.sigmoid = ComplexSigmoid()

    def forward(self, x):
        U_avg = torch.mean(x, dim=1, keepdim=True)

        U_max_real, _ = torch.max(x.real, dim=1, keepdim=True)
        U_max_imag, _ = torch.max(x.imag, dim=1, keepdim=True)
        U_max = torch.complex(U_max_real, U_max_imag)

        y = torch.cat([U_avg, U_max], dim=1)

        y = self.conv(y)
        y = self.sigmoid(y)
        return x * y

class ComplexAttentionNet(nn.Module):

    def __init__(self, class_num=15):
        super(ComplexAttentionNet, self).__init__()

        self.conv1 = ComplexConv2d(6, 6, 3, 1)
        self.avg_pool1 = ComplexAvgPool2d((2, 2), 2)
        self.conv2 = ComplexConv2d(6, 12, 3, 1)
        self.linear = ComplexLinear(108, 15)
        self.softmax = ComplexSoftmax()

        self.channel_attention1 = ChannelAttention(6, reduction=6)
        self.channel_attention2 = ChannelAttention(12, reduction=6)

        self.classifier = Sequential(
            Flatten(),
            ComplexLinear(108, class_num),
            ComplexSoftmax()
        )

    def forward(self, x, input_representation=False, output_representation=False):
        if input_representation:
            x = self.classifier(x)
            return x
        else:
            x = self.conv1(x)
            x = self.channel_attention1(x)
            x = self.avg_pool1(x)
            x = self.conv2(x)
            x = self.channel_attention2(x)
            if output_representation:
                return x
            else:
                x = self.classifier(x)
                return x

if __name__ == "__main__":
    input = torch.rand(1, 12, 3, 3)
    input = torch.complex(input, input)
    input = input.cuda()
    net = ComplexAttentionNet()
    net = net.cuda()
    output = net(input, input_representation=True)
    print(output.shape)
