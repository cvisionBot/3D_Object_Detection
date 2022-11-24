import math
import torch
from torch import nn

from ..layers.convolution import Conv2dAct
from ..layers.utils import weight_initialize
from ..detector.bifpn import BiFeaturePyramidNetwork


class ClassificationTaskHead(nn.Module):
    def __init__(self, num_classes, in_features, num_anchors=9, feature_size=256, prior=0.01):
        super(ClassificationTaskHead, self).__init__()

        self.num_classes = num_classes

        self.conv1 = Conv2dAct(in_channels=in_features, out_channels=feature_size, kernel_size=3, stride=1)
        self.conv2 = Conv2dAct(in_channels=feature_size, out_channels=feature_size, kernel_size=3, stride=1)
        self.conv3 = Conv2dAct(in_channels=feature_size, out_channels=feature_size, kernel_size=3, stride=1)
        self.conv4 = Conv2dAct(in_channels=feature_size, out_channels=feature_size, kernel_size=3, stride=1)

        self.output = nn.Conv2d(feature_size, num_anchors * num_classes, kernel_size=3, stride=1)
        self.output_act = nn.Sigmoid()

        weight_initialize(self)
        self.output.weight.data.fill_(0)
        self.output.bias.data.fill_(-math.log((1.0 - prior) / prior))

    def forward(self, input):
        output = self.conv1(input)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        output = self.output(output)
        output = self.output_act(output)

        # b, c, h, w = out.shape
        b = torch.tensor(output.shape)
        b = b[0] # if use onnx size()함수 사용
        return output.contiguous().view(b, self.numclasses, -1)




class EfficientDet(nn.Module):
    def __init__(self, Backbone, BiFPN, ClassificationSubNet, num_classes, in_channels=3):
        super(EfficientDet, self).__init__()

        self.backbone = Backbone(in_channels)
        fpn_sizes = self.backbone.stage_channels
        self.bifpn = BiFeaturePyramidNetwork(fpn_sizes)

        feature_size = self.bifpn.feature_size
        self.classification = ClassificationSubNet(num_classes, in_features=feature_size)

    def forward(self, input):
        stem = self.backbone.stem(input)
        s1 = self.backbone.layer1(stem)
        s2 = self.backbone.layer2(s1)
        s3 = self.backbone.layer3(s2)
        s4 = self.backbone.layer4(s3)
        s5 = self.backbone.layer5(s4)

        features = self.bifpn([s2, s3, s5])

        # prediction
        classifications = torch.cat([self.classification(f) for f in features], dim=2)
        #regression = torch.cat([self.regression(f) for f in features], dim=2)
        return classifications


if __name__ == '__main__':
    from models.backbone.efficientnet import EfficientNet
    model = EfficientDet(
        Backbone=EfficientNet, BiFPN=BiFeaturePyramidNetwork, num_classes=1000
    )
    print(model(torch.rand(1, 3, 256, 256)))