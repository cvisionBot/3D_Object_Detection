import torch
from torch import nn

from ..layers.utils import weight_initialize


class BiFeaturePyramidNetwork(nn.Module):
    def __init__(self, fpn_sizes, feature_size=256):
        super(BiFeaturePyramidNetwork, self).__init__()

    def forward(self, input):
        return