import torch
import torch.nn as nn
from src.layers.conv1_layer import Conv1Layer
from src.layers.pool_layers.maxpool_layer import MaxPoolLayer
from src.layers.flatten_layer import FlattenLayer
from src.layers.fc_layer import FCLayer
from src.blocks.shufflenet_unit import ShuffleNetUnit
from src.layers.channel_shuffle import ChannelShuffle

class ShuffleNet(nn.Module):
    def __init__(self, num_classes=1000, groups=3, width_multiplier=1.0):
        super().__init__()
        self.groups = groups
        
        # Stage output channels table (g=3 seçtik)
        out_channels_stage2 = int(240 * width_multiplier)
        out_channels_stage3 = int(480 * width_multiplier)
        out_channels_stage4 = int(960 * width_multiplier)
        
        # Stem
        self.conv1 = Conv1Layer(3, 24)
        self.maxpool = MaxPoolLayer(kernel_size=3, stride=2, padding=1)
        
        # Stage2
        self.stage2 = nn.Sequential(
            ShuffleNetUnit(24, out_channels_stage2, stride=2, groups=groups),
            ShuffleNetUnit(out_channels_stage2, out_channels_stage2, stride=1, groups=groups)
        )
        
        # Stage3
        self.stage3 = nn.Sequential(
            ShuffleNetUnit(out_channels_stage2, out_channels_stage3, stride=2, groups=groups),
            ShuffleNetUnit(out_channels_stage3, out_channels_stage3, stride=1, groups=groups)
        )
        
        # Stage4
        self.stage4 = nn.Sequential(
            ShuffleNetUnit(out_channels_stage3, out_channels_stage4, stride=2, groups=groups),
            ShuffleNetUnit(out_channels_stage4, out_channels_stage4, stride=1, groups=groups)
        )
        
        # Global Pooling + FC
        self.globalpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = FlattenLayer()
        self.fc = FCLayer(out_channels_stage4, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.globalpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
