from models import modules
import torch.nn as nn


class MeetMatch(nn.Module):

    def __init__(self, in_planes=3, n_features=10):
        super().__init__()
        self.in_planes = in_planes
        self.n_features = n_features

        self.extractor = nn.Sequential(
            nn.Conv2d(in_planes, 32, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d((300, 300)),
            modules.Fire(32, 16, 32, 32),
            modules.Fire(64, 16, 32, 32),
            nn.MaxPool2d(kernel_size=3, stride=2),
            modules.Fire(64, 32, 64, 64),
            modules.Fire(128, 32, 64, 64),
            modules.Fire(128, 32, 64, 64),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(128, self.n_features, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.extractor(x)
        x = self.classifier(x)
        return x.view(x.size(0), self.n_features)
