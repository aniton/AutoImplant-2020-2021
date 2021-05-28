from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F


class PreActResBlock3d(nn.Module):
    def __init__(self, channels, kernels=3):
        super(PreActResBlock3d, self).__init__()
        if type(kernels) is int:
            kernels = [kernels] * (len(channels) - 1)
        assert len(kernels) == (len(channels) - 1)
        self.block = nn.Sequential()
        for i, c in enumerate(channels):
            if i + 1 == len(channels):
                break
            self.block.add_module(f'bn{i}', nn.BatchNorm3d(channels[i]))
            self.block.add_module(f'act{i}', nn.ReLU(inplace=False))
            self.block.add_module(f'conv{i}',
                                  nn.Conv3d(c, channels[i + 1],
                                            kernels[i], stride=1,
                                            padding=kernels[i] // 2,
                                            bias=False))

        self.skip_conv = nn.Conv3d(channels[0], channels[-1], kernel_size=1)

    def forward(self, x):
        skip = self.skip_conv(x)
        x = self.block(x)
        return x + skip


def upsample(ch1, ch2):
    return nn.Sequential(OrderedDict([
        ('conv', nn.Conv3d(ch1, ch2, kernel_size=1)),
        ('ups', nn.Upsample(scale_factor=2, mode='trilinear'))]))


class UNet3d_thin(nn.Module):
    '''
    First place UNet with less layers
    '''

    def __init__(self, ):
        super(UNet3d_thin, self).__init__()

        self.conv1 = PreActResBlock3d(channels=[1, 16, 32], kernels=[3, 3])
        self.conv2 = PreActResBlock3d(channels=[32, 48, 64], kernels=[3, 3])
        self.conv3 = PreActResBlock3d(channels=[64, 96, 128], kernels=[3, 3])

        self.down1 = nn.Conv3d(32, 32, kernel_size=1, stride=2)
        self.down2 = nn.Conv3d(64, 64, kernel_size=1, stride=2)
        self.down3 = nn.Conv3d(128, 128, kernel_size=1, stride=2)

        self.bottleneck = PreActResBlock3d(channels=[128, 256], kernels=3)

        self.deconv1 = PreActResBlock3d(channels=[64, 32], kernels=3)
        self.deconv2 = PreActResBlock3d(channels=[128, 64], kernels=3)
        self.deconv3 = PreActResBlock3d(channels=[256, 128], kernels=3)

        self.up3 = upsample(256, 128)
        self.up2 = upsample(128, 64)
        self.up1 = upsample(64, 32)

        self.last_conv = nn.Conv3d(32, 1, kernel_size=1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.max_pool3d(x, kernel_size=4)

        # Encoder
        conv1 = self.conv1(x)
        conv2 = self.conv2(self.down1(conv1))
        conv3 = self.conv3(self.down2(conv2))

        # Bottleneck
        # import ipdb; ipdb.set_trace()
        dec3 = self.up3(self.bottleneck(self.down3(conv3)))

        # Decoder
        x = self.up2(self.deconv3(torch.cat([dec3, conv3], dim=1)))  # dec2
        x = self.up1(self.deconv2(torch.cat([x, conv2], dim=1)))  # dec1
        x = self.sigmoid(self.last_conv(self.deconv1(torch.cat([x, conv1], dim=1))))

        if not self.training:
            x = F.upsample(x, scale_factor=4, mode='trilinear', align_corners=False)

        return x

class UNet3d_thick(nn.Module):
    '''
    First place UNet with less layers
    '''
    def __init__(self, ):
        super(UNet3d_thick, self).__init__()

        self.conv1 = PreActResBlock3d(channels=[1, 16, 32], kernels=[3, 3])
        self.conv2 = PreActResBlock3d(channels=[32, 48, 64], kernels=[3, 3])
        self.conv3 = PreActResBlock3d(channels=[64, 96, 128], kernels=[3, 3])
        self.conv4 = PreActResBlock3d(channels=[128, 192, 256], kernels=[3, 3])
        
        self.down1 = nn.Conv3d(32, 32, kernel_size=1, stride=2)
        self.down2 = nn.Conv3d(64, 64, kernel_size=1, stride=2)
        self.down3 = nn.Conv3d(128, 128, kernel_size=1, stride=2)
        self.down4 = nn.Conv3d(256, 256, kernel_size=1, stride=2)
        
        self.bottleneck = PreActResBlock3d(channels=[256, 512], kernels=3)
        
        self.deconv1 = PreActResBlock3d(channels=[64, 32], kernels=3)
        self.deconv2 = PreActResBlock3d(channels=[128, 64], kernels=3)
        self.deconv3 = PreActResBlock3d(channels=[256, 128], kernels=3)
        self.deconv4 = PreActResBlock3d(channels=[512, 256], kernels=3)
        
        self.up4 = upsample(512, 256)
        self.up3 = upsample(256, 128)
        self.up2 = upsample(128, 64)
        self.up1 = upsample(64, 32)
        
        self.last_conv = nn.Conv3d(32, 1, kernel_size=1)
        
        self.sigmoid = nn.Sigmoid()
        

    def forward(self, x):
        x = F.max_pool3d(x, kernel_size=4)
        
        # Encoder
        conv1 = self.conv1(x) 
        conv2 = self.conv2(self.down1(conv1))
        conv3 = self.conv3(self.down2(conv2))
        conv4 = self.conv4(self.down3(conv3))
        
        # Bottleneck
        # import ipdb; ipdb.set_trace()
        dec4 = self.up4(self.bottleneck(self.down4(conv4)))
        
        # Decoder
        x = self.up3(self.deconv4(torch.cat([dec4, conv4], dim=1))) # dec3
        x = self.up2(self.deconv3(torch.cat([x, conv3], dim=1))) # dec2
        x = self.up1(self.deconv2(torch.cat([x, conv2], dim=1))) # dec1
        x = self.sigmoid(self.last_conv(self.deconv1(torch.cat([x, conv1], dim=1))))

        if not self.training:
            x = F.upsample(x, scale_factor=4, mode='trilinear', align_corners=False)
        
        return x
