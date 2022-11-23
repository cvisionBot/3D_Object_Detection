import torch
from torch import nn

from ..layers.utils import weight_initialize
from ..layers.convolution import Conv2dBnAct, DepthwiseSepConvBnAct


class BiFPNBlock(nn.Module):
    def __init__(self, feature_size, epsilon=0.0001):
        super(BiFPNBlock, self).__init__()
        self.epsilon = epsilon

        self.p3_rout = DepthwiseSepConvBnAct(feature_size, feature_size, 1, 1)
        self.p4_rout = DepthwiseSepConvBnAct(feature_size, feature_size, 1, 1)
        self.p5_rout = DepthwiseSepConvBnAct(feature_size, feature_size, 1, 1)
        self.p6_rout = DepthwiseSepConvBnAct(feature_size, feature_size, 1, 1)

        self.p4_out = DepthwiseSepConvBnAct(feature_size, feature_size, 1, 1)
        self.p5_out = DepthwiseSepConvBnAct(feature_size, feature_size, 1, 1)
        self.p6_out = DepthwiseSepConvBnAct(feature_size, feature_size, 1, 1)
        self.p7_out = DepthwiseSepConvBnAct(feature_size, feature_size, 1, 1)

        self.upsample = nn.Upsample(scale_factor=2)
        self.downsample = nn.Upsample(scale_factor=0.5)

    def forward(self, input):
        p3_x, p4_x, p5_x, p6_x, p7_x = input

        # # Top-Down Pathway
        p7_rout = p7_x
        p7_up = self.upsample(p7_x)
        p6_rout = self.p6_rout(p6_x + p7_up)
        p6_up = self.upsample(p6_rout)
        p5_rout = self.p5_rout(p5_x + p6_up)
        p5_up = self.upsample(p5_rout)
        p4_rout = self.p4_rout(p4_x + p5_up)
        p4_up = self.upsample(p4_rout)
        p3_rout = self.p3_rout(p3_x + p4_up)

        # Bottom-Up Pathway
        p3_down = self.downsample(p3_x)
        p4_out = self.p4_out(p4_rout + p3_down)
        p4_down = self.downsample(p4_out)
        p5_out = self.p5_out(p5_rout + p4_down)
        p5_down = self.downsample(p5_out)
        p6_out = self.p6_out(p6_rout + p5_down)
        p6_down = self.downsample(p6_out)
        p7_out = self.p7_out(p7_rout + p6_down)

        return [p3_rout, p4_out, p5_out, p6_out, p7_out]



class BiFeaturePyramidNetwork(nn.Module):
    def __init__(self, fpn_sizes, feature_size=256, num_iter=2, epsilon=0.0001):
        super(BiFeaturePyramidNetwork, self).__init__()

        self.feature_size = feature_size
        s3_size, s4_size, s5_size = fpn_sizes

        self.p3 = nn.Conv2d(s3_size, self.feature_size, kernel_size=1, stride=1, padding=0)
        self.p4 = nn.Conv2d(s4_size, self.feature_size, kernel_size=1, stride=1, padding=0)
        self.p5 = nn.Conv2d(s5_size, self.feature_size, kernel_size=1, stride=1, padding=0)

        self.p6 = nn.Conv2d(s5_size, self.feature_size, kernel_size=3, stride=2, padding=1)
        self.p7 = Conv2dBnAct(in_channels=self.feature_size, out_channels=self.feature_size, kernel_size=3, stride=2)

        bifpns = []
        for _ in range(num_iter):
            bifpns.append(BiFPNBlock(feature_size))
        self.bifpn = nn.Sequential(*bifpns)


    def forward(self, input):
        stage3, stage4, stage5 = input
        p3 = self.p3(stage3)
        p4 = self.p4(stage4)
        p5 = self.p5(stage5)
        p6 = self.p6(stage5)
        p7 = self.p7(p6)

        features = [p3, p4, p5, p6, p7]
        output = self.bifpn(features)
        return output