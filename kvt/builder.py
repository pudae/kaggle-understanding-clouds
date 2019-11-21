from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from easydict import EasyDict as edict

import torch
from torch.utils.data import DataLoader

from kvt.registry import (
        LOSSES,
        OPTIMIZERS,
        SCHEDULERS,
        DATASETS,
        TRANSFORMS,
        HOOKS
)


from kvt.utils import build_from_config


def build_dataloaders(config):
    dataloaders = []
    for split_config in config.dataset.splits:
        dataset_config = edict({'name': config.dataset.name,
                                'params': config.dataset.params})
        dataset_config.params.update(split_config)
        print(dataset_config)

        transform = build_from_config(
                config.transform, TRANSFORMS,
                default_args={'split': dataset_config.params.split})
        dataset = build_from_config(
                dataset_config, DATASETS,
                default_args={'transform': transform})

        is_train = dataset_config.params.mode == 'train'
        if is_train:
            batch_size = config.train.batch_size
        else:
            batch_size = config.evaluation.batch_size
        dataloader = DataLoader(dataset,
                                shuffle=is_train,
                                batch_size=batch_size,
                                drop_last=is_train,
                                num_workers=config.transform.num_preprocessor,
                                pin_memory=True)

        dataloaders.append({'split': dataset_config.params.split,
                            'mode': dataset_config.params.mode,
                            'dataloader': dataloader})
    return dataloaders


def build_hooks(config):
    # build default hooks
    build_model_hook_config = {'name': 'DefaultModelBuilderHook'}
    forward_hook_config = {'name': 'DefaultForwardHook'}
    post_forward_hook_config = {'name': 'DefaultPostForwardHook'}
    loss_hook_config = {'name': 'DefaultLossHook'}
    metric_hook_config = {'name': 'DefaultMetricHook'}
    logger_hook_config = {'name': 'DefaultLoggerHook'}
    write_result_hook_config = {'name': 'DefaultWriteResultHook'}

    if 'hooks' in config:
        hooks = config.hooks
        if 'build_model' in hooks:
            build_model_hook_config.update(hooks.build_model)
        if 'forward' in hooks:
            forward_hook_config.update(hooks.forward)
        if 'post_forward' in hooks:
            post_forward_hook_config.update(hooks.post_forward)
        if 'loss' in hooks:
            loss_hook_config.update(hooks.loss)
        if 'metric' in hooks:
            metric_hook_config.update(hooks.metric)
        if 'logger' in hooks:
            logger_hook_config.update(hooks.logger)
        if 'write_result' in hooks:
            write_result_hook_config.update(hooks.write_result)

    hooks_dict = {}
    hooks_dict['build_model_fn'] = build_from_config(build_model_hook_config, HOOKS)
    hooks_dict['forward_fn'] = build_from_config(forward_hook_config, HOOKS)
    hooks_dict['post_forward_fn'] = build_from_config(post_forward_hook_config, HOOKS)
    hooks_dict['loss_fn'] = build_from_config(loss_hook_config, HOOKS)
    hooks_dict['metric_fn'] = build_from_config(metric_hook_config, HOOKS)
    hooks_dict['logger_fn'] = build_from_config(logger_hook_config, HOOKS)
    hooks_dict['write_result_fn'] = build_from_config(write_result_hook_config, HOOKS)
    hooks = edict(hooks_dict)

    return hooks


def build_model(config, hooks):
    return hooks.build_model_fn(config.model)


def build_optimizer(config, **kwargs):
    return build_from_config(config.optimizer,
                             OPTIMIZERS,
                             default_args=kwargs)


def build_scheduler(config, **kwargs):
    return build_from_config(config.scheduler,
                             SCHEDULERS,
                             default_args=kwargs)


def build_loss(config):
    return build_from_config(config.loss, LOSSES)

