import torch
import torch.nn as nn
from .pw_group_conv import PWGroupConv
from .channel_shuffle import ChannelShuffle
from .dwconv_layer import DWConvLayer

class ShuffleNetUnit(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, groups=3):
        super().__init__()
        self.stride = stride
        mid_channels = out_channels // 4  # bottleneck

        # 1x1 pointwise group conv + BN + ReLU
        self.pw_group1 = nn.Sequential(
            PWGroupConv(in_channels, mid_channels, groups=groups),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )

        # channel shuffle
        self.shuffle = ChannelShuffle(groups)

        # 3x3 depthwise conv
        self.dw_conv = DWConvLayer(mid_channels, mid_channels, stride=stride)

        # 1x1 pointwise group conv + BN
        self.pw_group2 = nn.Sequential(
            PWGroupConv(mid_channels, out_channels, groups=groups),
            nn.BatchNorm2d(out_channels)
        )

        # shortcut
        if stride == 1 and in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)

    def forward(self, x):
        out = self.pw_group1(x)
        out = self.shuffle(out)
        out = self.dw_conv(out)
        out = self.pw_group2(out)

        if self.stride == 1 and x.shape[1] == out.shape[1]:
            out = out + x  
        else:
            out = torch.cat([out, self.shortcut(x)], dim=1)  

        return out
