import torch.nn as nn

class DWConvLayer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=channels,   # depthwise
                bias=False
            ),
            nn.BatchNorm2d(channels),
            # ShuffleNet: NO ReLU here (Xception'daki gibi)
        )

    def forward(self, x):
        return self.conv(x)
