from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import build_encoder


# from segmentation models
class Conv3x3GNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=False):
        super().__init__()
        self.upsample = upsample
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3, 3),
                              stride=1, padding=1, bias=False),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.block(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        return x


class FPNBlock(nn.Module):
    def __init__(self, pyramid_channels, skip_channels):
        super().__init__()
        self.skip_conv = nn.Conv2d(skip_channels, pyramid_channels, kernel_size=1)

    def forward(self, x):
        x, skip = x

        x = F.interpolate(x, scale_factor=2, mode='nearest')
        skip = self.skip_conv(skip)

        x = x + skip
        return x


class SegmentationBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_upsamples=0):
        super().__init__()

        blocks = [
            Conv3x3GNReLU(in_channels, out_channels, upsample=bool(n_upsamples))
        ]

        if n_upsamples > 1:
            for _ in range(1, n_upsamples):
                blocks.append(Conv3x3GNReLU(out_channels, out_channels, upsample=True))

        self.block = nn.Sequential(*blocks)

    def forward(self, x):
        return self.block(x)


class Decoder(nn.Module):
    def __init__(
            self,
            num_classes,
            encoder_channels,
            pyramid_channels=256,
            segmentation_channels=128,
            final_upsampling=2,
            dropout=0.2,
            mish=False,
            merge='cat'
    ):
        super().__init__()
        in_channels = encoder_channels
        self.merge = merge
        self.in_channels = in_channels
        print('[Decoder::__init__] merge:', self.merge)
        print('[Decoder::__init__] final_upsampling:', final_upsampling)
        print('[Decoder::__init__] encoder_channels:', encoder_channels)

        self.final_upsampling = final_upsampling
        self.conv1 = nn.Conv2d(in_channels[0], pyramid_channels, kernel_size=(1, 1))

        self.p4 = FPNBlock(pyramid_channels, in_channels[1])
        self.p3 = FPNBlock(pyramid_channels, in_channels[2])
        self.p2 = FPNBlock(pyramid_channels, in_channels[3])

        self.s5 = SegmentationBlock(pyramid_channels, segmentation_channels, n_upsamples=4)
        self.s4 = SegmentationBlock(pyramid_channels, segmentation_channels, n_upsamples=3)
        self.s3 = SegmentationBlock(pyramid_channels, segmentation_channels, n_upsamples=2)
        self.s2 = SegmentationBlock(pyramid_channels, segmentation_channels, n_upsamples=1)

        self.dropout = nn.Dropout2d(p=dropout)

        if self.merge == 'cat':
            self.merge_op = nn.Sequential(
                nn.Conv2d(segmentation_channels*4, segmentation_channels, kernel_size=3,
                          stride=1, padding=1, bias=False),
                nn.BatchNorm2d(segmentation_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
        elif self.merge == 'add':
            self.merge_op = nn.Sequential(
                nn.Conv2d(segmentation_channels, segmentation_channels, kernel_size=3,
                          stride=1, padding=1, bias=False),
                nn.BatchNorm2d(segmentation_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
        else:
            self.merge_op = None

        self.final_conv = nn.Conv2d(segmentation_channels, num_classes, kernel_size=1, padding=0)
        self.initialize()

    def forward(self, x, encodes):
        c5, c4, c3, c2, _ = encodes

        p5 = self.conv1(c5)
        p4 = self.p4([p5, c4])
        p3 = self.p3([p4, c3])
        p2 = self.p2([p3, c2])

        s5 = self.s5(p5)
        s4 = self.s4(p4)
        s3 = self.s3(p3)
        s2 = self.s2(p2)

        if self.merge == 'add':
            x = s5 + s4 + s3 + s2
            x = self.merge_op(x)
        elif self.merge == 'cat':
            x = torch.cat([s5, s4, s3, s2], dim=1)
            x = self.merge_op(x)
        else:
            assert False

        x = self.dropout(x)
        x = self.final_conv(x)

        if self.final_upsampling is not None and self.final_upsampling > 1:
            x = F.interpolate(x, scale_factor=self.final_upsampling, mode='bilinear', align_corners=True)

        output_dict = {'logits': x}
        return output_dict

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class FPN(nn.Module):
    def __init__(self, encoder, num_classes, **kwargs):
        super().__init__()
        self.encoder = build_encoder(encoder)
        self.decoder = Decoder(num_classes=num_classes, encoder_channels=self.encoder.channels, **kwargs)

    def forward(self, x):
        encodes = self.encoder(x)
        return self.decoder(x, encodes)


