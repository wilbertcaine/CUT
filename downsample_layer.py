import torch
import torch.nn as nn

class Downsample(nn.Module):
    def __init__(self, features):
        super().__init__()
        layers = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(features, features, kernel_size=3, stride=2)
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, input):
        return self.model(input)