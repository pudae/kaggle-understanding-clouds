from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import abc


class LoggerHookBase(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __call__(self, writer, split, outputs, labels, log_dict,
                 epoch, step=None, num_steps_in_epoch=None):
        pass


class DefaultLoggerHook(LoggerHookBase):
    def __call__(self, writer, split, outputs, labels, log_dict,
                 epoch, step=None, num_steps_in_epoch=None):
        if step is not None:
            assert num_steps_in_epoch is not None
            log_step = epoch * 10000 + (step / num_steps_in_epoch) * 10000
            log_step = int(log_step)
        else:
            log_step = epoch

        for key, value in log_dict.items():
            writer.add_scalar(f'{split}/{key}', log_dict[key], log_step)
