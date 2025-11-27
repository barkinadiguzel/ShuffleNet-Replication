import torch.nn as nn

class Conv1Layer(nn.Module):
    def __init__(self, out_channels=24):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)
