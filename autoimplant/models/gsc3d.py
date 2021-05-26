import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _triple


class GatedSpatialConv3d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=0, dilation=1, groups=1, bias=False):
        """

        :param in_channels:
        :param out_channels:
        :param kernel_size:
        :param stride:
        :param padding:
        :param dilation:
        :param groups:
        :param bias:
        """

        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)
        super(GatedSpatialConv3d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _triple(0), groups, bias, 'zeros')
        self._gate_conv = nn.Sequential(
            nn.BatchNorm3d(in_channels+1),
            nn.Conv3d(in_channels+1, in_channels+1, 1),
            nn.ReLU(), 
            nn.Conv3d(in_channels+1, 1, 1),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )

    def forward(self, input_features, gating_features):
        """

        :param input_features:  [NxCxHxWxD]  featuers comming from the shape branch (canny branch).
        :param gating_features: [Nx1xHxWxD] features comming from the texture branch (resnet). Only one channel feature map.
        :return:
        """
        alphas = self._gate_conv(torch.cat([input_features, gating_features], dim=1))
        input_features = (input_features * (alphas + 1)) 
        return F.conv3d(input_features, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
  
    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
