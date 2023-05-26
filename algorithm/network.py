"""
This is the xception network slightly adapted from
https://github.com/tstandley/Xception-PyTorch/blob/master/xception.py
and
https://github.com/hoya012/pytorch-Xception/blob/master/Xception_pytorch.ipynb

It is based on
Francois Chollet
Xception: Deep Learning with Depthwise Separable Convolutions
https://arxiv.org/pdf/1610.02357.pdf
"""

import torch.nn as nn


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1):
        super(SeparableConv2d, self).__init__()

        self.depthwise = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, dilation=dilation, groups=in_channels, bias=False)

        self.pointwise = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1,
                                   padding=0, dilation=1, groups=1, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class PointwiseBlock(nn.Module):

    def __init__(self, in_channels, filters):
        super(PointwiseBlock, self).__init__()

        self.in_channels = in_channels
        self.filters = filters

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=filters[0], kernel_size=1, stride=1, bias=True)
        self.bn1 = nn.BatchNorm2d(filters[0])

        self.conv2 = nn.Conv2d(in_channels=filters[0], out_channels=filters[1], kernel_size=1, stride=1, bias=True)
        self.bn2 = nn.BatchNorm2d(filters[1])

        self.conv3 = nn.Conv2d(in_channels=filters[1], out_channels=filters[2], kernel_size=1, stride=1, bias=True)
        self.bn3 = nn.BatchNorm2d(filters[2])

        self.relu = nn.ReLU(inplace=True)
        self.conv_shortcut = nn.Conv2d(in_channels=in_channels, out_channels=filters[2], kernel_size=1, stride=1, bias=True)
        self.bn_shortcut = nn.BatchNorm2d(filters[2])

    def forward(self, x):
        if self.in_channels == self.filters[-1]:
            # identity shortcut
            shortcut = x
        else:
            shortcut = self.conv_shortcut(x)
            shortcut = self.bn_shortcut(shortcut)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = out + shortcut
        out = self.relu(out)

        return out


class SepConvBlock(nn.Module):

    def __init__(self, in_channels, filters):
        super(SepConvBlock, self).__init__()

        self.in_channels = in_channels
        self.filters = filters

        self.sepconv1 = SeparableConv2d(in_channels=in_channels, out_channels=filters[0], kernel_size=3)
        self.bn1 = nn.BatchNorm2d(filters[0])

        self.sepconv2 = SeparableConv2d(in_channels=filters[0], out_channels=filters[1], kernel_size=3)
        self.bn2 = nn.BatchNorm2d(filters[1])

        self.relu = nn.ReLU(inplace=False)
        self.conv_shortcut = nn.Conv2d(in_channels=in_channels, out_channels=filters[1], kernel_size=1, stride=1, bias=True)
        self.bn_shortcut = nn.BatchNorm2d(filters[1])

    def forward(self, x):
        if self.in_channels == self.filters[-1]:
            # identity shortcut
            shortcut = x
        else:
            shortcut = self.conv_shortcut(x)
            shortcut = self.bn_shortcut(shortcut)

        out = self.relu(x)
        out = self.sepconv1(out)
        out = self.bn1(out)

        out = self.relu(out)
        out = self.sepconv2(out)
        out = self.bn2(out)

        out = out + shortcut

        return out


class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()

        self.num_blocks = 8

        self.last_layer = nn.Conv2d(in_channels=728, out_channels=2, kernel_size=1, stride=1, bias=True)
        self.first_block = PointwiseBlock(in_channels=9, filters=[128, 256, 728])
        self.sepconv_blocks = self._make_sepconv_blocks()
        self.var_activation = nn.ReLU(inplace=False)

        # initialize parameters
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_sepconv_blocks(self):
        blocks = []
        for i in range(self.num_blocks):
            blocks.append(SepConvBlock(in_channels=728, filters=[728, 728]))
        return nn.ModuleList(blocks)

    def forward(self, x):
        height = x[:, -1, :, :][:, None, :, :]
        height = height.expand(-1, 728, -1, -1)
        x = x[:, :-1, :, :]
        x = self.first_block(x)
        for i, layer in enumerate(self.sepconv_blocks):
            if i == 6:
                x = x + height
            x = layer(x)
        return self.last_layer(x)