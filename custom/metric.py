from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import pandas as pd
import torch

import kvt


def compute_metrics(predicts, labels):
    N,  H, W = predicts.shape

    predicts = predicts.reshape((-1, H*W))
    labels = labels.reshape((-1, H*W))

    sum_p = np.sum(predicts, axis=1)
    sum_l = np.sum(labels, axis=1)
    intersection = np.sum(np.logical_and(predicts, labels), axis=1)

    numer = 2*intersection
    denom = sum_p + sum_l
    dice = numer / (denom + 1e-6)

    empty_indices = np.where(sum_l <= 0)[0]
    non_empty_indices = np.where(sum_l > 0)[0]
    if len(non_empty_indices) == 0:
        non_empty_mean_dice = 0.0
    else:
        non_empty_dice = dice[non_empty_indices]
        non_empty_mean_dice = float(np.mean(non_empty_dice))

    all_non_empty_index = np.where(numer > 0)[0]
    all_empty_index = np.where(denom == 0)[0]
    dice[all_empty_index] = 1
    mean_dice = float(np.mean(dice))

    cls_accuracy = (len(all_non_empty_index) + len(all_empty_index)) / N

    correct_indices = np.where((sum_p > 0) == (sum_l > 0))[0]
    incorrect_indices = np.where((sum_p > 0) != (sum_l > 0))[0]

    tp = len(np.where(sum_l[correct_indices] > 0)[0])
    tn = len(np.where(sum_l[correct_indices] == 0)[0])

    fp = len(np.where(sum_l[incorrect_indices] == 0)[0])
    fn = len(np.where(sum_l[incorrect_indices] > 0)[0])

    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)

    tnr = tn / (tn + fp + 1e-10)
    fpr = fp / (fp + tn + 1e-10)

    return {'score': mean_dice,
            'mean_dice': mean_dice,
            'mean_dice_non_empty': non_empty_mean_dice,
            'cls_acc': cls_accuracy,
            'precision': precision,
            'recall': recall,
            'tnr': tnr,
            'fpr': fpr}


@kvt.HOOKS.register
class CloudMetricHook:
    def __call__(self, outputs, labels, data, is_train, split):
        probabilities = outputs['probabilities']
        labels = outputs['labels']

        is_train = False
        if isinstance(probabilities, torch.Tensor):
            assert isinstance(probabilities, torch.Tensor)
            assert isinstance(labels, torch.Tensor)
            labels = labels.detach().cpu().numpy()
            probabilities = probabilities.detach().cpu().numpy()

        if 'cls_probabilities' in outputs:
            cls_probabilities = outputs['cls_probabilities']
            if isinstance(cls_probabilities, torch.Tensor):
                cls_probabilities = cls_probabilities.detach().cpu().numpy()
        else:
            cls_probabilities = None

        assert probabilities.shape == labels.shape
        cls_thres = np.array([0.7,0.7,0.7,0.7])
        thres = np.array([0.4,0.4,0.4,0.4])

        predictions = (probabilities > thres[None,:,None,None]).astype(int)
        if cls_probabilities is not None:
            cls_predictions = (cls_probabilities > cls_thres).astype(int)
            predictions = predictions * cls_predictions[:,:,None,None]

        B,C,H,W = predictions.shape
        return compute_metrics(predictions.reshape(-1,H,W), labels.reshape(-1,H,W))
