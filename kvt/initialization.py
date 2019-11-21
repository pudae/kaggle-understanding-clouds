from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pkgutil

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models

import pretrainedmodels

import kvt.hooks
import kvt.losses
import kvt.models.backbones
import kvt.models.segmentations


from kvt.registry import (
        BACKBONES,
        MODELS,
        LOSSES,
        OPTIMIZERS,
        SCHEDULERS,
        DATASETS,
        TRANSFORMS,
        HOOKS
)


def register_torch_modules():
    # register backbones
    for name, cls in kvt.models.backbones.__dict__.items():
        if not callable(cls):
            continue
        BACKBONES.register(cls)

    # pretrained models
    for name, cls in pretrainedmodels.__dict__.items():
        if not callable(cls):
            continue
        BACKBONES.register(cls)

    # torchvision models
    for name, cls in torchvision.models.__dict__.items():
        if not callable(cls):
            continue
        BACKBONES.register(cls)

    # segmentation models
    for name, cls in kvt.models.segmentations.__dict__.items():
        if not callable(cls):
            continue
        MODELS.register(cls)

    # register losses
    losses = [
        nn.L1Loss,
        nn.MSELoss,
        nn.CrossEntropyLoss,
        nn.CTCLoss,
        nn.NLLLoss,
        nn.PoissonNLLLoss,
        nn.KLDivLoss,
        nn.BCELoss,
        nn.BCEWithLogitsLoss,
        nn.MarginRankingLoss,
        nn.HingeEmbeddingLoss,
        nn.MultiLabelMarginLoss,
        nn.SmoothL1Loss,
        nn.SoftMarginLoss,
        nn.MultiLabelSoftMarginLoss,
        nn.CosineEmbeddingLoss,
        nn.MultiMarginLoss,
        nn.TripletMarginLoss,

        kvt.losses.DiceLoss,
        kvt.losses.FocalLoss,
        kvt.losses.BinaryFocalLoss,
        kvt.losses.LovaszSoftmaxLoss,
        kvt.losses.LovaszHingeLoss,
    ]

    for loss in losses:
        LOSSES.register(loss)

    # register optimizers
    optimizers = [
        optim.Adadelta,
        optim.Adagrad,
        optim.Adam,
        optim.AdamW,
        optim.SparseAdam,
        optim.Adamax,
        optim.ASGD,
        optim.LBFGS,
        optim.RMSprop,
        optim.Rprop,
        optim.SGD,
    ]
    for optimizer in optimizers:
        OPTIMIZERS.register(optimizer)

    # register schedulers
    schedulers = [
            optim.lr_scheduler.StepLR,
            optim.lr_scheduler.MultiStepLR,
            optim.lr_scheduler.ExponentialLR,
            optim.lr_scheduler.CosineAnnealingLR,
            optim.lr_scheduler.ReduceLROnPlateau,
            optim.lr_scheduler.CyclicLR,
            optim.lr_scheduler.OneCycleLR,
    ]
    for scheduler in schedulers:
        SCHEDULERS.register(scheduler)


def register_default_hooks():
    HOOKS.register(kvt.hooks.DefaultLossHook)
    HOOKS.register(kvt.hooks.DefaultForwardHook)
    HOOKS.register(kvt.hooks.DefaultPostForwardHook)
    HOOKS.register(kvt.hooks.DefaultMetricHook)
    HOOKS.register(kvt.hooks.DefaultModelBuilderHook)
    HOOKS.register(kvt.hooks.DefaultLoggerHook)
    HOOKS.register(kvt.hooks.DefaultWriteResultHook)


def initialize():
    register_torch_modules()
    register_default_hooks()
