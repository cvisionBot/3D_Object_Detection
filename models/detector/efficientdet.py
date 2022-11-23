import math
import torch
from torch import nn

from ..layers.utils import weight_initialize
from ..detector.bifpn import BiFeaturePyramidNetwork

class EfficientDet(nn.Module):
    def __init__(self, Backbone, BiFPN, num_classes, in_channels=3):
        super(EfficientDet, self).__init__()

        self.backbone = Backbone(in_channels)
        fpn_sizes = self.backbone.stage_channels
        self.bifpn = BiFeaturePyramidNetwork(fpn_sizes)

    def forward(self, input):
        stem = self.backbone.stem(input)
        s1 = self.backbone.layer1(stem)
        s2 = self.backbone.layer2(s1)
        s3 = self.backbone.layer3(s2)
        s4 = self.backbone.layer4(s3)
        s5 = self.backbone.layer5(s4)

        features = self.bifpn([s2, s3, s5])


if __name__ == '__main__':
    from models.backbone.efficientnet import EfficientNet
    model = EfficientDet(
        Backbone=EfficientNet, BiFPN=BiFeaturePyramidNetwork, num_classes=1000
    )
    print(model(torch.rand(1, 3, 256, 256)))