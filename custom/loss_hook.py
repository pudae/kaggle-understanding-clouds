from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tqdm
import numpy as np
import torch
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F

import kvt
import kvt.losses


@kvt.HOOKS.register
class CloudLossHook:
    def __init__(self, ignore_negative=False, cls_weight=1.0):
        self.loss_cls = nn.BCEWithLogitsLoss()

        print('[CloudLossHook] cls_weight:', cls_weight)
        print('[CloudLossHook] ignore_negative:', ignore_negative)
        self.cls_weight = cls_weight
        self.ignore_negative = ignore_negative

    def __call__(self, loss_fn, outputs, labels, data, is_train):
        B,C,H,W = labels.size()

        cls_tp = 1
        if 'cls_logits' in outputs:
            assert self.loss_cls is not None
            cls_logits = outputs['cls_logits']
            cls_labels = labels.view(B, C, H*W)
            cls_labels = torch.sum(cls_labels, dim=2)
            cls_labels = (cls_labels > 0).float()
            loss_cls = self.loss_cls(input=outputs['cls_logits'], target=cls_labels)

            if self.ignore_negative:
                cls_prob = torch.sigmoid(cls_logits)
                cls_pred = (cls_prob > 0.5).int()
                cls_tp = (cls_pred == cls_labels).int() * cls_labels
                cls_tp = cls_tp.detach().unsqueeze(-1).unsqueeze(-1)
        else:
            loss_cls = torch.Tensor([0.0]).float().cuda()
            if self.ignore_negative:
                cls_labels = labels.view(B, C, H*W)
                cls_labels = torch.sum(cls_labels, dim=2)
                cls_labels = (cls_labels > 0).float()
                cls_tp = cls_labels
                cls_tp = cls_tp.detach().unsqueeze(-1).unsqueeze(-1)

        logits = outputs['logits']
        logits = logits * cls_tp
        logits = logits.view(-1,H,W)
        labels = labels.view(-1,H,W)
        loss_seg = loss_fn(input=logits, target=labels)

        loss = loss_seg + loss_cls * self.cls_weight
        
        loss_dict = {
                'loss': loss,
                'loss_seg': loss_seg,
                'loss_cls': loss_cls
                }
        return loss_dict


@kvt.HOOKS.register
class CloudLossWeightedHook:
    def __init__(self, seg_loss_names=['BCEWithLogitsLoss', 'DiceLoss'], seg_weights=[0.75,0.25],
                 ignore_negative=False, cls_weight=1.0):
        self.loss_cls = nn.BCEWithLogitsLoss()
        self.seg_losses = [self._to_loss_fn(ln) for ln in seg_loss_names]
        self.seg_weights = seg_weights
        print('[CloudLossWeightedHook] cls_weight:', cls_weight)
        print('[CloudLossWeightedHook] seg_loss_names:', seg_loss_names)
        print('[CloudLossWeightedHook] seg_weights:', seg_weights)
        print('[CloudLossWeightedHook] ignore_negative:', ignore_negative)

        self.cls_weight = cls_weight
        self.ignore_negative = ignore_negative

    def _to_loss_fn(self, loss_name):
        if loss_name == 'BCEWithLogitsLoss':
            return nn.BCEWithLogitsLoss()
        elif loss_name == 'BinaryFocalLoss':
            return kvt.losses.BinaryFocalLoss()
        elif loss_name == 'DiceLoss':
            return kvt.losses.DiceLoss()
        else:
            return None

    def __call__(self, loss_fn, outputs, labels, data, is_train):
        B,C,H,W = labels.size()

        cls_tp = None
        if 'cls_logits' in outputs:
            assert self.loss_cls is not None
            cls_logits = outputs['cls_logits']
            cls_labels = labels.view(B, C, H*W)
            cls_labels = torch.sum(cls_labels, dim=2)
            cls_labels = (cls_labels > 0).float()
            loss_cls = self.loss_cls(input=outputs['cls_logits'], target=cls_labels)
        else:
            loss_cls = torch.Tensor([0.0]).float().cuda()

        if self.ignore_negative:
            cls_labels = labels.view(B, C, H*W)
            cls_labels = torch.sum(cls_labels, dim=2)
            cls_labels = (cls_labels > 0).float()
            cls_labels = cls_labels.detach().unsqueeze(-1).unsqueeze(-1)

        logits = outputs['logits'].view(B*C,H,W)
        cls_labels = cls_labels.view(B*C,1,1)
        seg_losses = [seg_loss_fn(input=logits*cls_labels, target=labels.view(B*C,H,W)) for seg_loss_fn in self.seg_losses]
        seg_losses = [w*l for w,l in zip(self.seg_weights, seg_losses)]

        loss_seg = sum(seg_losses)
        loss = loss_seg + loss_cls * self.cls_weight

        return {'loss': loss,
                'loss_seg': loss_seg,
                'loss_cls': loss_cls
                }
