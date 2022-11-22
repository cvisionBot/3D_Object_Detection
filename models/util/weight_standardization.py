from torch import nn


def weight_standardization(model):
    for m in model.modules():
        if isinstance(m, WSConv2d):
            weight = m.weight
            weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)

            weight = weight - weight_mean
            std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
            weight = weight / std.expand_as(weight)
            m.weight = nn.Parameter(weight)


class WSConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super(WSConv2d, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                        dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)
        
    def forward(self, input):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return nn.functional.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)