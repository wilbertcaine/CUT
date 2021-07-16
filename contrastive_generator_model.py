import torch
import torch.nn as nn

class ResnetBlock(nn.Module):
    def __init__(self, features):
        super().__init__()
        layers = []
        for i in range(2):
            layers += [
                nn.ReflectionPad2d(1),
                nn.Conv2d(features, features, kernel_size=3),
                nn.InstanceNorm2d(features),
            ]
            if i==0:
                layers += [
                    nn.ReLU(True)
                ]
        self.model = nn.Sequential(*layers)

    def forward(self, input):
        return input + self.model(input)

# class Downsample(nn.Module):
#     def __init__(self, features):
#         super().__init__()
#         layers = [
#             nn.ReflectionPad2d(1),
#             nn.ConvTranspose2d(258, 256, kernel_size)
#         ]
#         self.model = nn.Sequential(*layers)
#
#     def forward(self, input):
#         return self.model(input)

class Generator(nn.Module):
    def __init__(self, in_channels=3, features=64, residuals=9):
        super().__init__()
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, features, kernel_size=7),
            nn.InstanceNorm2d(features),
            nn.ReLU(True)
        ]
        features_prev = features
        for i in range(2):
            features *= 2
            layers += [
                nn.Conv2d(features_prev, features, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm2d(features),
                nn.ReLU(True),
                # nn.ReflectionPad2d(1)
            ]
            features_prev = features
        for i in range(residuals):
            layers += [ResnetBlock(features_prev)]
        for i in range(2):
            features //= 2
            layers += [
                # nn.ReplicationPad2d(1),
                nn.Conv2d(features_prev, features, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm2d(features),
                nn.ReLU(True)
            ]
            features_prev = features
        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(features_prev, in_channels, kernel_size=7),
            nn.Tanh(),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, input):
        return(self.model(input))

def test():
    x = torch.randn((5, 3, 256, 256))
    print(x.shape)
    model = Generator(in_channels=3)
    preds = model(x)
    print(preds.shape)

if __name__ == "__main__":
    test()
