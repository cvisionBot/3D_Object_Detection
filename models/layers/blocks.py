import math
import torch
from torch import nn

from ..layers.convolution import Conv2dBn, Conv2dBnAct, DepthwiseConvBn, DepthwiseConvBnAct
from ..layers.attention import SE_Block
from ..layers.activation import Swish


class MBConv1(nn.Module):
    def __init__(self, in_channels, kernel_size, out_channels, stride):
        super(MBConv1, self).__init__()
        self.conv_act = Conv2dBnAct(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1, dilation=1,
                                        groups=1, padding_mode='zeros', act=Swish())
        self.dconv = DepthwiseConvBnAct(in_channels=in_channels, kernel_size=kernel_size, stride=1, dilation=1,
                                            padding_mode='zeros', act=Swish())
        self.se = SE_Block(in_channels=in_channels)
        self.conv_bn = Conv2dBn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride)
        self.identity = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride)
    
    def forward(self, input):
        output = self.conv_act(input)
        output = self.dconv(output)
        output = self.se(output)
        output = self.conv_bn(output)
        if input.size() != output.size():
            input = self.identity(input)
        output = input + output
        return output


class MBConv6(nn.Module):
    def __init__(self, in_channels, kernel_size, out_channels, stride, act=None, SE=False):
        super(MBConv6, self).__init__()
        self.se = SE
        self.act = nn.ReLU() if act is None else act
        self.conv_act = Conv2dBnAct(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1,
                                        groups=1, padding_mode='zeros', act=self.act)
        self.dconv = DepthwiseConvBnAct(in_channels=in_channels, kernel_size=kernel_size, stride=stride, dilation=1, padding_mode='zeros', act=self.act)
        self.conv_bn = Conv2dBn(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.se = SE_Block(in_channels=in_channels)
        self.identity = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride)

    def forward(self, input):
        output = self.conv_act(input)
        output = self.dconv(output)
        if self.se:
            output = self.se(output)
        output = self.conv_bn(output)
        if input.size() != output.size():
            input = self.identity(input)
        output = output + input
        return output

