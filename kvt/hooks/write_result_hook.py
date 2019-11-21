from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import abc


class WriteResultHookBase(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __call__(self, split, output_path, outputs, labels=None, data=None,
                 is_train=False):
        pass


class DefaultWriteResultHook(WriteResultHookBase):
    def __call__(self, split, output_path, outputs, labels=None, data=None,
                 is_train=False):
        pass

