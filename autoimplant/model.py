import torch.nn as nn


class ModelX8(nn.Module):
    def __init__(self, ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.MaxPool3d(kernel_size=8),
            nn.Dropout(p=0.5),
            nn.Conv3d(1, 64, (10, 10, 10), stride=3),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 256, (7, 7, 7), stride=3),
            nn.ReLU(inplace=True)
        )

        self.linear = nn.Sequential(
            nn.Linear(256 * 5 ** 3, 256 * 5 ** 3),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(256, 64, (7, 7, 7), stride=3),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(64, 1, (10, 10, 10), stride=3)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, 256 * 5 ** 3)

        x = self.linear(x)
        x = x.view((-1, 256, 5, 5, 5))

        x = self.decoder(x)

        x = self.sigmoid(x)

        return x
