from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import abc

import numpy as np
import torch


class MetricHookBase(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __call__(self, outputs, labels, is_train, split):
        pass


class DefaultMetricHook(MetricHookBase):
    def __call__(self, outputs, labels, is_train, split):
        if isinstance(outputs, dict):
            logits = outputs['logits']
        else:
            logits = outputs

        if isinstance(logits, np.ndarray):
            assert isinstance(labels, np.ndarray)
            assert len(logits.shape) == 2
            predicts = np.argmax(logits, axis=1)
            correct = np.sum((predicts == labels).astype(int))
            total = predicts.shape[0]
            accuracy = correct / total
        else:
            assert isinstance(logits, torch.Tensor)
            assert isinstance(labels, torch.Tensor)
            assert logits.ndim == 2
            predicts = torch.argmax(logits, dim=1)
            correct = torch.sum(predicts == labels)
            total = predicts.size(0)
            accuracy = correct / total

        return {'score': accuracy, 'accuracy': accuracy}
