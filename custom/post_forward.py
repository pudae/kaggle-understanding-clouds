from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import torch
import torch.nn.functional as F

import kvt


@kvt.HOOKS.register
class CloudPostForwardHook:
    def __call__(self, outputs, images=None, labels=None, data=None, is_train=False):
        if isinstance(outputs, dict):
            logits = outputs['logits']
            if 'cls_logits' in outputs:
                cls_logits = outputs['cls_logits']
            else:
                cls_logits = None

            if logits.dim() == 4:
                probabilities = torch.sigmoid(logits)
                B,C,H,W = probabilities.size()
                if 350 != H and 525 != W:
                    probabilities = F.interpolate(probabilities, size=(350,525),mode='bilinear')
                    if labels is not None:
                        labels_resized = F.interpolate(labels.float(), size=(350,525),mode='nearest')
                    else:
                        labels_resized = None

                outputs_dict = {
                        'logits': logits,
                        'probabilities': probabilities}
                if labels_resized is not None:
                    outputs_dict['labels'] = labels_resized

                if cls_logits is not None:
                    cls_probabilities = torch.sigmoid(cls_logits)
                    outputs_dict.update({
                        'cls_logits': cls_logits,
                        'cls_probabilities': cls_probabilities})
            else:   ## TTA
                probabilities = torch.sigmoid(logits)
                B,T,C,H,W = probabilities.size()
                p_list = []
                p_list.append(probabilities[:,0])
                p_list.append(torch.flip(probabilities[:,1], dims=[3]))
                if T > 2:
                    p_list.append(torch.flip(probabilities[:,2], dims=[2]))
                if T > 3:
                    p_list.append(torch.flip(probabilities[:,3], dims=[2,3]))

                probabilities = sum(p_list) / len(p_list)

                B,C,H,W = probabilities.size()

                if 350 != H and 525 != W:
                    probabilities = F.interpolate(probabilities, size=(350,525),mode='bilinear')
                    if labels is not None:
                        labels_resized = F.interpolate(labels.float(), size=(350,525),mode='nearest')
                    else:
                        labels_resized = None

                outputs_dict = {
                        'logits': logits,
                        'probabilities': probabilities}

                if labels_resized is not None:
                    outputs_dict['labels'] = labels_resized

                if cls_logits is not None:
                    cls_probabilities = torch.sigmoid(cls_logits)
                    cls_probabilities = torch.mean(cls_probabilities, dim=1)
                    outputs_dict.update({
                        'cls_logits': cls_logits,
                        'cls_probabilities': cls_probabilities})
        else:
            assert isinstance(outputs, torch.Tensor)
            outputs_dict = {'logits': outputs}
            outputs_dict['probabilities'] = torch.sigmoid(outputs)
        outputs_dict['image_id'] = data['image_id']
        return outputs_dict
