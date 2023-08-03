import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, middle_channels, kernel_size=2, stride=2
        )
        self.conv = ConvBlock(middle_channels, out_channels)

    def forward(self, x):
        return self.conv(self.up(x))


class Decoder(nn.Module):
    def __init__(self, num_classes, num_filters):
        super(Decoder, self).__init__()
        self.decoder4 = DecoderBlock(num_filters[3], num_filters[2] * 2, num_filters[2])
        self.decoder3 = DecoderBlock(num_filters[2], num_filters[1] * 2, num_filters[1])
        self.decoder2 = DecoderBlock(num_filters[1], num_filters[0] * 2, num_filters[0])
        self.decoder1 = ConvBlock(num_filters[0], num_filters[0])
        self.final = nn.Conv2d(num_filters[0], num_classes, kernel_size=1)

    def forward(self, x4, x3, x2, x1):
        x4 = self.decoder4(x4)
        x3 = self.decoder3(torch.cat([x4, x3], 1))
        x2 = self.decoder2(torch.cat([x3, x2], 1))
        x1 = self.decoder1(torch.cat([x2, x1], 1))
        return self.final(x1)
