import torch
import torch.nn as nn


class Fire(nn.Module):

    def __init__(self, in_planes, s1x1_planes,
                 e1x1_planes, e3x3_planes):
        super().__init__()

        self.s1x1 = nn.Conv2d(in_planes, s1x1_planes, kernel_size=1)
        self.e1x1 = nn.Conv2d(s1x1_planes, e1x1_planes, kernel_size=1)
        self.e3x3 = nn.Conv2d(s1x1_planes, e3x3_planes,
                              kernel_size=3, padding=1)

        self.s1x1_activation = nn.ReLU(inplace=True)
        self.e1x1_activation = nn.ReLU(inplace=True)
        self.e3x3_activation = nn.ReLU(inplace=True)

    def forward(self, X):
        X = self.s1x1_activation(self.s1x1(X))

        return torch.cat([
            self.e1x1_activation(self.e1x1(X)),
            self.e3x3_activation(self.e3x3(X)),
        ], dim=1)
