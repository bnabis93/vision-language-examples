from torch import nn


class Decoder(nn.Module):
    def __init__(self, num_classes):
        super(Decoder, self).__init__()

        self.transconv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.transconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.transconv3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.transconv4 = nn.ConvTranspose2d(32, num_classes, kernel_size=2, stride=2)

    def forward(self, x):
        x = self.transconv1(x)
        x = self.transconv2(x)
        x = self.transconv3(x)
        x = self.transconv4(x)
        return x
