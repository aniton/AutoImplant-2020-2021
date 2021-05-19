import torch.nn as nn


class Autoencoder(nn.Module):

    def __init__(self, scale_factor=8):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.MaxPool2d(kernel_size=scale_factor),
            nn.Dropout(p=0.5),
            nn.Conv3d(1, 64, (9, 9, 9), stride=3),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 192, (6, 6, 6), stride=3),
            nn.ReLU(inplace=True)
        )

        self.linear = nn.Sequential(
            nn.Linear(24000, 24000),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(192, 64, (6, 6, 6), stride=3),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(64, 1, (9, 9, 9), stride=3),
            nn.Upsample(scale_factor=scale_factor)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(24000)

        x = self.linear(x)
        x = x.view((192, 5, 5, 5))

        x = x.unsqueeze(0)
        x = self.decoder(x)

        x = x.view(216000)
        x = self.sigmoid(x)

        return x


def loss_fn(outputs, targets):
    loss = nn.BCELoss()

    return loss(outputs, targets[0])
