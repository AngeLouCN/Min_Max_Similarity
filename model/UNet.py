import torch
import torch.nn as nn


class CONV_Block(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out


class Unet(nn.Module):
    def __init__(self, num_classes=1, input_channels=3, **kwargs):
        super().__init__()

        nb_filter = [64, 128, 256, 512, 1024]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = CONV_Block(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = CONV_Block(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = CONV_Block(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = CONV_Block(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = CONV_Block(nb_filter[3], nb_filter[4], nb_filter[4])
        self.conv3_1 = CONV_Block(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_2 = CONV_Block(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_3 = CONV_Block(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_4 = CONV_Block(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def _make_pred_layer(self, block, dilation_series, padding_series, NoLabels):
        return block(dilation_series, padding_series, NoLabels)

    def forward(self, input):
        x0_0 = self.conv0_0(input)   #64 256 256
        x1_0 = self.conv1_0(self.pool(x0_0)) # 128 128 128 
        x2_0 = self.conv2_0(self.pool(x1_0)) # 256 64
        x3_0 = self.conv3_0(self.pool(x2_0)) # 512 32
        x4_0 = self.conv4_0(self.pool(x3_0)) # 1024 16

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1)) #32
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)
        return output
