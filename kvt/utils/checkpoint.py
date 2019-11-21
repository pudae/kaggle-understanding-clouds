from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil

import torch


def get_last_checkpoint(checkpoint_dir):
    checkpoints = [checkpoint
                   for checkpoint in os.listdir(checkpoint_dir)
                   if checkpoint.startswith('epoch_') and checkpoint.endswith('.pth')]
    if checkpoints:
        return os.path.join(checkpoint_dir, list(sorted(checkpoints))[-1])
    return None


def get_initial_checkpoint(config):
    checkpoint_dir = os.path.join(config.train.dir, 'checkpoint')
    return get_last_checkpoint(checkpoint_dir)


def get_checkpoint(config, name):
    checkpoint_dir = os.path.join(config.train.dir, 'checkpoint')
    return os.path.join(checkpoint_dir, name)


def remove_old_checkpoint(checkpoint_dir, keep):
    checkpoints = [checkpoint
                   for checkpoint in os.listdir(checkpoint_dir)
                   if checkpoint.startswith('epoch_') and checkpoint.endswith('.pth')]
    checkpoints = sorted(checkpoints)
    for checkpoint in checkpoints[:-keep]:
        os.remove(os.path.join(checkpoint_dir, checkpoint))


def copy_last_n_checkpoints(config, n, name):
    checkpoint_dir = os.path.join(config.train.dir, 'checkpoint')
    checkpoints = [checkpoint
                   for checkpoint in os.listdir(checkpoint_dir)
                   if checkpoint.startswith('epoch_') and checkpoint.endswith('.pth')]
    checkpoints = sorted(checkpoints)
    for i, checkpoint in enumerate(checkpoints[-n:]):
        shutil.copyfile(os.path.join(checkpoint_dir, checkpoint),
                        os.path.join(checkpoint_dir, name.format(i)))


def load_checkpoint(model, optimizer, checkpoint):
    print('load checkpoint from', checkpoint)
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_dict'])

    step = checkpoint['step'] if 'step' in checkpoint else -1
    last_epoch = checkpoint['epoch'] if 'epoch' in checkpoint else -1

    return last_epoch, step


def save_checkpoint(config, model, optimizer, epoch, step=0, keep=None, weights_dict=None, name=None):
    checkpoint_dir = os.path.join(config.train.dir, 'checkpoint')

    if name:
        checkpoint_path = os.path.join(checkpoint_dir, '{}.pth'.format(name))
    else:
        checkpoint_path = os.path.join(checkpoint_dir, 'epoch_{:04d}.pth'.format(epoch))

    state_dict = {}
    for key, value in model.state_dict().items():
        if key.startswith('module.'):
            key = key[len('module.'):]
        state_dict[key] = value

    if weights_dict is None:
        weights_dict = {
          'state_dict': state_dict,
          'optimizer_dict' : optimizer.state_dict(),
          'epoch' : epoch,
          'step' : step,
        }
    torch.save(weights_dict, checkpoint_path)

    if keep is not None and keep > 0:
        remove_old_checkpoint(checkpoint_dir, keep)
