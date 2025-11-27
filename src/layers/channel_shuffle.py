import torch
import torch.nn as nn

class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        b, c, h, w = x.size()
        g = self.groups
        assert c % g == 0, "Channels must be divisible by groups"

        channels_per_group = c // g

        # (B, g, c/g, H, W)
        x = x.view(b, g, channels_per_group, h, w)

        # transpose: (B, c/g, g, H, W)
        x = x.transpose(1, 2).contiguous()

        # flatten back
        x = x.view(b, c, h, w)
        return x
