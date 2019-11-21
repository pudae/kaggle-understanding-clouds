from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

import kvt
from kvt.utils import build_from_config
from kvt.models.segmentations.unet import DecoderBlock
from kvt.registry import MODELS


@kvt.HOOKS.register
class UNetModelBuilderHook(object):
    def __call__(self, config):
        model = build_from_config(config, MODELS)
        model.decoder = Decoder(encoder_channels=model.encoder.channels, **config.params)

        return model


class Decoder(nn.Module):
    def __init__(self,
                 num_classes,
                 encoder_channels,
                 dropout=0.2,
                 out_channels=[256, 128, 64, 32, 16],
                 use_cls_head=False,
                 **_):
        super().__init__()
        in_channels = encoder_channels
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.use_cls_head = use_cls_head

        self.center = DecoderBlock(in_channels=in_channels[0], out_channels=in_channels[0],
                                   up_sample=False)

        self.layer1 = DecoderBlock(in_channels[0]+in_channels[1], out_channels[0])
        self.layer2 = DecoderBlock(in_channels[2]+out_channels[0], out_channels[1])
        self.layer3 = DecoderBlock(in_channels[3]+out_channels[1], out_channels[2])
        self.layer4 = DecoderBlock(in_channels[4]+out_channels[2], out_channels[3])
        self.layer5 = DecoderBlock(out_channels[3], out_channels[4])

        self.final = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv2d(out_channels[-1], num_classes, kernel_size=1)
        )

        if self.use_cls_head:
            feature_size = in_channels[0] // 4
            self.cls_head = nn.Sequential(
                    nn.Linear(in_channels[0], feature_size),
                    nn.BatchNorm1d(feature_size),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                    nn.Linear(feature_size, num_classes))

    def forward(self, x, encodes):
        skips = encodes[1:]

        center = self.center(encodes[0])

        decodes = self.layer1([center, skips[0]])
        decodes = self.layer2([decodes, skips[1]])
        decodes = self.layer3([decodes, skips[2]])
        decodes = self.layer4([decodes, skips[3]])
        decodes = self.layer5([decodes, None])

        decodes4 = F.interpolate(decodes, x.size()[2:], mode='bilinear')
        output_dict = {'logits': self.final(decodes4)}

        if self.use_cls_head:
            p = torch.sigmoid(output_dict['logits'])
            p = torch.max(p, dim=1, keepdims=True)[0]
            p = F.interpolate(p, center.size()[2:], mode='bilinear')
            p = p.detach()
            c5_pool = center * p
            c5_pool = F.adaptive_avg_pool2d(c5_pool, 1)
            c5_pool = c5_pool.squeeze(-1).squeeze(-1)

            assert len(c5_pool.size()) == 2
            cls_logits = self.cls_head(c5_pool)
            output_dict.update({'cls_logits': cls_logits})

        return output_dict
