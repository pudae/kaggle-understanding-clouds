import os
import math
from collections import defaultdict

import tqdm

import numpy as np
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from kvt.builder import (
        build_hooks,
        build_model,
        build_dataloaders
)
import kvt.utils


def adjust_learning_rate(optimizer, lr):
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr
  return lr


def moving_average(net1, net2, alpha=1):
  for param1, param2 in zip(net1.parameters(), net2.parameters()):
    param1.data *= (1.0 - alpha)
    param1.data += param2.data * alpha


def _check_bn(module, flag):
  if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
    flag[0] = True


def check_bn(model):
  flag = [False]
  model.apply(lambda module: _check_bn(module, flag))
  return flag[0]


def reset_bn(module):
  if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
    module.running_mean = torch.zeros_like(module.running_mean)
    module.running_var = torch.zeros_like(module.running_var)


def _get_momenta(module, momenta):
  if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
    momenta[module] = module.momentum


def _set_momenta(module, momenta):
  if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
    module.momentum = momenta[module]


def bn_update(loader, model):
    """
        BatchNorm buffers update (if any).
        Performs 1 epochs to estimate buffers average using train dataset.
        :param loader: train dataset loader for buffers average estimation.
        :param model: model being update
        :return: None
    """
    if not check_bn(model):
        return
    model.train()
    momenta = {}
    model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    n = 0

    split = loader ['split']
    dataset_mode = loader['mode']
    dataloader = loader['dataloader']
    for input_dict in tqdm.tqdm(dataloader):
        input_var = input_dict['image'].cuda()
        b = input_var.size(0)

        momentum = b / (n + b)
        for module in momenta.keys():
            module.momentum = momentum

        model(input_var)
        n += b

    model.apply(lambda module: _set_momenta(module, momenta))


def schedule(config, epoch):
  t = epoch / config.swa.start
  lr_ratio = config.swa.lr / config.optimizer.params.lr
  if t <= 0.5:
    factor = 1.0
  elif t <= 0.9:
    factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
  else:
    factor = lr_ratio
  return config.optimizer.params.lr * factor


def detach_params(model):
  for param in model.parameters():
    param.detach_()

  return model


def get_checkpoints(config):
    assert 'swa' in config
    num_checkpoint = config.swa.num_checkpoint

    if 'epoch_end' in config.swa:
        epoch_end = config.swa.epoch_end
    else:
        epoch_end = None

    checkpoint_dir = os.path.join(config.train.dir, 'checkpoint')

    if epoch_end is not None:
        checkpoints = [os.path.join(checkpoint_dir, 'epoch_{:04d}.pth'.format(e))
                       for e in range(epoch_end+1)]
    else:
        checkpoints = [os.path.join(checkpoint_dir, 'best.score.{:04d}.pth'.format(e))
                       for e in range(100)
                       if os.path.exists(os.path.join(checkpoint_dir, 'best.score.{:04d}.pth'.format(e)))]

    checkpoints = [p for p in checkpoints if os.path.exists(p)]
    checkpoints = checkpoints[-num_checkpoint:]
    print('-------------------------------------------------------------------')
    print('checkpoint_dir:', checkpoint_dir)
    print('\n'.join(checkpoints))
    return checkpoints


def run(config):
    # build hooks
    hooks = build_hooks(config)

    # build model
    model = build_model(config, hooks).cuda()

    # load checkpoint
    checkpoints = get_checkpoints(config)

    # build datasets
    config.dataset.splits = [v for v in config.dataset.splits if v.split == 'train' or v.split == 'train_all']
    print(config.dataset.splits)

    dataloaders = build_dataloaders(config)
    dataloaders = [dataloader for dataloader in dataloaders if dataloader['split'] == 'train' or dataloader['split'] == 'train_all']
    assert len(dataloaders) == 1, f'len(dataloaders)({len(dataloaders)}) not 1'

    dataloader = dataloaders[0]

    kvt.utils.load_checkpoint(model, None, checkpoints[0])
    for i, checkpoint in enumerate(checkpoints[1:]):
        model2 = build_model(config, hooks).cuda()
        last_epoch, _ = kvt.utils.load_checkpoint(model2, None, checkpoint)
        if 'ema' in config.swa:
            moving_average(model, model2, config.swa.ema)
        else:
            moving_average(model, model2, 1. / (i + 2))

    with torch.no_grad():
        bn_update(dataloader, model)

    if 'ema' in config.swa:
        output_name = 'ema'
    else:
        output_name = 'swa'

    print('save {}'.format(output_name))
    kvt.utils.save_checkpoint(config, model, None, 0, 0,
                              name=output_name,
                              weights_dict={'state_dict': model.state_dict()})
