import torch
from torch import nn

from ..layers.convolution import Conv2dBnAct
from ..layers.blocks import MBConv1, MBConv6
from ..layers.activation import Swish
from ..layers.utils import weight_initialize


class _EfficientStem(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(_EfficientStem, self).__init__()
        self.EfficientStem = Conv2dBnAct(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, dilation=1,
                                groups=1, padding_mode='zeros', act=Swish())
    def forward(self, input):
        return self.EfficientStem(input)

class _EfficientNetB0(nn.Module):
    def __init__(self, in_channels, classes):
        super(_EfficientNetB0, self).__init__()
        self.stem = _EfficientStem(in_channels=in_channels, out_channels=32)
        self.stage_channels = [40, 80, 320]
        # layer configs about EfficientDet
        # Block_idx, in_channels, kernel_size, out_channels, stride
        layer1 = [[1, 32, 3, 16, 1], [6, 16, 3, 24, 1], [6, 24, 3, 24, 2]]
        layer2 = [[6, 24, 5, 40, 1], [6, 40, 5, 40, 2]]
        layer3 = [[6, 40, 3, 80, 1], [6, 80, 3, 80, 1], [6, 80, 3, 80, 2]]
        layer4 = [[6, 80, 5, 112, 1], [6, 112, 5, 112, 1], [6, 112, 5, 112, 1],
                  [6, 112, 5, 192, 1], [6, 192, 5, 192, 1], [6, 192, 5, 192, 1], [6, 192, 5, 192, 2]]
        layer5 = [[6, 192, 3, 320, 1]]

        self.layer1 = Make_Layers(layer1)
        self.layer2 = Make_Layers(layer2)
        self.layer3 = Make_Layers(layer3)
        self.layer4 = Make_Layers(layer4)
        self.layer5 = Make_Layers(layer5)

        self.classification = nn.Sequential(
            Conv2dBnAct(in_channels=320, out_channels=1280, kernel_size=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(1280, classes, 1)
        )

    def forward(self, input):
        stem = self.stem(input)
        print('stem shape : ', stem.shape)
        s1 = self.layer1(stem)
        print('s1 shape : ', s1.shape)
        s2 = self.layer2(s1)
        print('s2 shape : ', s2.shape)
        s3 = self.layer3(s2)
        print('s3 shape : ', s3.shape)
        s4 = self.layer4(s3)
        print('s4 shape : ', s4.shape)
        s5 = self.layer5(s4)
        print('s5 shape : ', s5.shape)
        pred = self.classification(s5)
        print('pred shape : ', pred.shape)
        b, c, _, _ = pred.size()
        pred = pred.view(b, c)
        return {'pred' : pred}


class Make_Layers(nn.Module):
    def __init__(self, layers_configs):
        super(Make_Layers, self).__init__()
        self.layers_configs = layers_configs
        self.layer = self.MBConv(self.layers_configs)

    def MBConv(self, layers_configs):
        layers = []
        for b, i, k, o, s in layers_configs:
            if b == 1:
                layers.append(MBConv1(in_channels=i, kernel_size=k, out_channels=o, stride=s))
            else:
                layers.append(MBConv6(in_channels=i, kernel_size=k, out_channels=o, stride=s, act=Swish()))
        return nn.Sequential(*layers)
    
    def forward(self, input):
        return self.layer(input)



def EfficientNet(in_channels, classes=1000, varient='B0'):
    if varient == 'B0':
        model = _EfficientNetB0(in_channels=in_channels, classes=classes)
    weight_initialize(model)
    return model


if __name__ == "__main__":
    model = EfficientNet(in_channels=3, classes=1000, varient='B0')
    model(torch.rand(1, 3, 224, 224))