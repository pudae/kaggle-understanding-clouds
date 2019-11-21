from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import abc

import kvt.registry
from kvt.registry import BACKBONES, MODELS
from kvt.utils import build_from_config


class ModelBuilderHookBase(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __call__(self, config):
        pass


class DefaultModelBuilderHook(ModelBuilderHookBase):
    def __call__(self, config):
        #######################################################################
        # classification models
        #######################################################################
        if BACKBONES.get(config.name) is not None:
            return self.build_classifcation_model(config)

        #######################################################################
        # segmentation models
        #######################################################################
        return build_from_config(config, MODELS)

    def build_classification_model(config):
        model = build_from_config(config, BACKBONES)

        # mobilenet
        if config.name.startswith('mobilenet'):
            model.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(backbone.last_channel, config.params.num_classes),
            )

        # squeezenet
        if config.name.startswith('squeezenet'):
            model.classifier = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Conv2d(512, config.params.num_classes, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1))
            )

        # mnasnet
        if config.name.startswith('mnasnet'):
            model.classifier = nn.Sequential(
                    nn.Dropout(p=0.2, inplace=True),
                    nn.Linear(1280, config.params.num_classes)
            )

        # wsl
        if 'wsl' in config.name:
            model.fc = nn.Linear(model.fc.in_channels, config.params.num_classes)

        # TODO: other models

        return model
