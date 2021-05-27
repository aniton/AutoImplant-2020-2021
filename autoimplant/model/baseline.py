import torch.nn as nn
import torch.nn.functional as F


class ModelX8(nn.Module):
    def __init__(self, ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv3d(1, 64, (10, 10, 10), stride=3),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 192, (7, 7, 7), stride=3),
            nn.ReLU(inplace=True)
        )

        self.linear = nn.Sequential(
            nn.Linear(192 * 5 ** 3, 192 * 5 ** 3),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(192, 64, (7, 7, 7), stride=3),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(64, 1, (10, 10, 10), stride=3)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.max_pool3d(x, kernel_size=8)

        x = self.encoder(x)
        x = x.view(-1, 192 * 5 ** 3)

        x = self.linear(x)
        x = x.view((-1, 192, 5, 5, 5))

        x = self.decoder(x)

        x = self.sigmoid(x)

        if not self.training:
            x = F.upsample(x, scale_factor=8, mode='trilinear', align_corners=False)

        return x


class SecondStageLi(nn.Module):
    def __init__(self, ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )

        self.bottleneck = nn.Sequential(
            nn.Conv3d(16, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(32, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(32, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(32, 16, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(16, 1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(1),
            nn.ReLU(inplace=True),
            nn.Conv3d(1, 1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(1),
            nn.ReLU(inplace=True),
            nn.Conv3d(1, 1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(1),

        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.max_pool3d(x, kernel_size=2)

        x = self.encoder(x)

        x = self.bottleneck(x)

        x = self.decoder(x)

        x = self.sigmoid(x)

        if not self.training:
            x = F.upsample(x, scale_factor=2, mode='trilinear', align_corners=False)

        return x


class SecondStageUNet(nn.Module):
    def __init__(self, ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv3d(1, 3, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm3d(3),
            nn.ReLU(inplace=True),
            nn.Conv3d(3, 8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2),

            nn.Conv3d(8, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2),

            nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2)
        )

        self.bottleneck = nn.Sequential(
            nn.Conv3d(32, 64, (1, 1, 1), stride=1, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 32, (1, 1, 1), stride=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear'),
            nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode='trilinear'),
            nn.Conv3d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode='trilinear'),
            nn.Conv3d(8, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(3),
            nn.ReLU(inplace=True),
            nn.Conv3d(3, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(3),
            nn.ReLU(inplace=True),
            nn.Conv3d(3, 1, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm3d(1),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.max_pool3d(x, kernel_size=2)

        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)

        x = self.sigmoid(x)

        if not self.training:
            x = F.upsample(x, scale_factor=2, mode='trilinear', align_corners=False)

        return x
