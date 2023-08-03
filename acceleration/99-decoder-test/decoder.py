import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)

    def forward(self, x, skip_connection):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.up(x)
        x = torch.cat([x, skip_connection], dim=1)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels_list, out_channels_list):
        super(Decoder, self).__init__()
        assert len(in_channels_list) == len(out_channels_list)
        num_stages = len(in_channels_list)
        self.num_stages = num_stages
        self.stages = nn.ModuleList()

        for i in range(num_stages):
            stage = DecoderBlock(in_channels_list[i], out_channels_list[i])
            self.stages.append(stage)

    def forward(self, x, skip_connections):
        assert len(skip_connections) == self.num_stages
        for i in range(self.num_stages):
            x = self.stages[i](x, skip_connections[i])
        return x
