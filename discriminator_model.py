import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=64):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, features, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)
        ]
        features_prev = features
        for i in range(3):
            features *= 2
            layers += [
                nn.Conv2d(features_prev, features, kernel_size=4, stride=2 if i<2 else 1, padding=1),
                nn.InstanceNorm2d(features),
                nn.LeakyReLU(0.2, True)
            ]
            features_prev = features
        features = 1
        layers += [
            nn.Conv2d(features_prev, features, kernel_size=4, stride=1, padding=1)
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, input):
        return(self.model(input))

    def set_requires_grad(self, requires_grad=False):
        for param in self.parameters():
            param.requires_grad = requires_grad

def test():
    x = torch.randn((5, 3, 256, 256))
    print(x.shape)
    model = Discriminator(in_channels=3)
    preds = model(x)
    print(preds.shape)

if __name__ == "__main__":
    test()
