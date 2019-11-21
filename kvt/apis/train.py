import os
import math
from collections import defaultdict

import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn import GroupNorm, Conv2d, Linear

from tensorboardX import SummaryWriter

from kvt.builder import (
        build_hooks,
        build_model,
        build_loss,
        build_optimizer,
        build_scheduler,
        build_dataloaders
)
import kvt.utils


def prepare_directories(config):
    os.makedirs(os.path.join(config.train.dir, 'checkpoint'), exist_ok=True)


def train_single_epoch(config, model, split, dataloader,
                       hooks, optimizer, scheduler, epoch):
    model.train()

    batch_size = config.train.batch_size
    total_size = len(dataloader.dataset)
    total_step = math.ceil(total_size / batch_size)

    tbar = tqdm.tqdm(enumerate(dataloader), total=total_step)
    for i, data in tbar:
        images = data['image'].cuda()
        labels = data['label'].cuda()

        outputs = hooks.forward_fn(model=model, images=images, labels=labels,
                                   data=data, is_train=True)
        outputs = hooks.post_forward_fn(outputs=outputs, images=images, labels=labels,
                                        data=data, is_train=True)
        loss = hooks.loss_fn(outputs=outputs, labels=labels.float(), data=data, is_train=True)
        metric_dict = hooks.metric_fn(outputs=outputs, labels=labels, data=data, is_train=True, split=split)

        if isinstance(loss, dict):
            loss_dict = loss
            loss = loss_dict['loss']
        else:
            loss_dict = {'loss': loss}

        loss.backward()

        if config.train.gradient_accumulation_step is None:
            optimizer.step()
            optimizer.zero_grad()
        elif (i+1) % config.train.gradient_accumulation_step == 0:
            optimizer.step()
            optimizer.zero_grad()

        if config.scheduler.name == 'OneCycleLR':
            scheduler.step()

        log_dict = {key:value.item() for key, value in loss_dict.items()}
        log_dict['lr'] = optimizer.param_groups[0]['lr']
        log_dict.update(metric_dict)

        f_epoch = epoch + i / total_step
        tbar.set_description(f'{split}, {f_epoch:.2f} epoch')
        tbar.set_postfix(lr=optimizer.param_groups[0]['lr'],
                         loss=loss.item())

        hooks.logger_fn(split=split, outputs=outputs, labels=labels, log_dict=log_dict,
                        epoch=epoch, step=i, num_steps_in_epoch=total_step)


def evaluate_single_epoch(config, model, split, dataloader, hooks, epoch):
    model.eval()

    batch_size = config.evaluation.batch_size
    total_size = len(dataloader.dataset)
    total_step = math.ceil(total_size / batch_size)

    with torch.no_grad():
        losses = []
        aggregated_loss_dict = defaultdict(list)
        aggregated_outputs_dict = defaultdict(list)
        aggregated_outputs = []
        aggregated_labels = []

        aggregated_metric_dict = defaultdict(list)

        tbar = tqdm.tqdm(enumerate(dataloader), total=total_step)
        for i, data in tbar:
            images = data['image'].cuda() #to(device)
            labels = data['label'].cuda() #to(device)

            outputs = hooks.forward_fn(model=model, images=images, labels=labels,
                                       data=data, is_train=False)
            outputs = hooks.post_forward_fn(outputs=outputs, images=images, labels=labels,
                                            data=data, is_train=True)
            loss = hooks.loss_fn(outputs=outputs, labels=labels.float(), data=data, is_train=False)
            if isinstance(loss, dict):
                loss_dict = loss
                loss = loss_dict['loss']
            else:
                loss_dict = {'loss': loss}
            losses.append(loss.item())

            f_epoch = epoch + i / total_step
            tbar.set_description(f'{split}, {f_epoch:.2f} epoch')

            metric_dict = hooks.metric_fn(outputs=outputs, labels=labels, data=data, is_train=False, split=split)
            for key, value in metric_dict.items():
                aggregated_metric_dict[key].append(value)

            for key, value in loss_dict.items():
                aggregated_loss_dict[key].append(value.item())

    metric_dict = {key:sum(value)/len(value)
                   for key, value in aggregated_metric_dict.items()}

    log_dict = {key: sum(value)/len(value) for key, value in aggregated_loss_dict.items()}
    log_dict.update(metric_dict)

    hooks.logger_fn(split=split,
                    outputs=aggregated_outputs,
                    labels=aggregated_labels,
                    log_dict=log_dict,
                    epoch=epoch)

    return metric_dict['score']


def train(config, model, hooks, optimizer, scheduler, dataloaders, last_epoch):
    best_ckpt_score = -100000
    for epoch in range(last_epoch, config.train.num_epochs):
        # train 
        for dataloader in dataloaders:
            split = dataloader['split']
            dataset_mode = dataloader['mode']

            if dataset_mode != 'train':
                continue

            dataloader = dataloader['dataloader']
            train_single_epoch(config, model, split, dataloader, hooks,
                               optimizer, scheduler, epoch)

        score_dict = {}
        ckpt_score = None
        # validation
        for dataloader in dataloaders:
            split = dataloader['split']
            dataset_mode = dataloader['mode']

            if dataset_mode != 'validation':
                continue

            dataloader = dataloader['dataloader']
            score = evaluate_single_epoch(config, model, split, dataloader, hooks,
                                          epoch)
            score_dict[split] = score
            # Use score of the first split
            if ckpt_score is None:
                ckpt_score = score

        # update learning rate
        if config.scheduler.name == 'ReduceLROnPlateau':
            scheduler.step(ckpt_score)
        elif config.scheduler.name == 'CosineAnnealingLR':
            param_epoch = (epoch + 1) % config.scheduler.params.T_max
            print('param_epoch:', param_epoch)
            scheduler.step(param_epoch+1)
        elif config.scheduler.name != 'OneCycleLR' and config.scheduler.name != 'ReduceLROnPlateau':
            scheduler.step()

        if config.scheduler.name == 'CosineAnnealingLR' and epoch % config.scheduler.params.T_max == config.scheduler.params.T_max - 1:
            snapshot_idx = epoch // config.scheduler.params.T_max
            print('save snapshot:', epoch, config.scheduler.params.T_max, snapshot_idx)
            kvt.utils.save_checkpoint(config, model, optimizer, epoch, keep=None,
                                      name=f'snapshot.{snapshot_idx}')
        if ckpt_score > best_ckpt_score:
            best_ckpt_score = ckpt_score
            kvt.utils.save_checkpoint(config, model, optimizer, epoch, keep=None,
                                      name='best.score')
            kvt.utils.copy_last_n_checkpoints(config, 5, 'best.score.{:04d}.pth')

        if epoch % config.train.save_checkpoint_epoch == 0:
            kvt.utils.save_checkpoint(config, model, optimizer,
                                      epoch, keep=config.train.num_keep_checkpoint)


def to_data_parallel(config, model, optimizer):
    if 'sync_bn' in config.train:
        print('sycn bn!!')
        from kvt.sync_batchnorm import SynchronizedBatchNorm1d, DataParallelWithCallback, convert_model
        model = convert_model(model)
        model = model.cuda()
        if torch.cuda.device_count() > 1:
            model = DataParallelWithCallback(model, list(range(torch.cuda.device_count())))
        return model, optimizer

    if torch.cuda.device_count() == 1:
        model = model.cuda()
        return model, optimizer

    model = model.cuda()
    return torch.nn.DataParallel(model), optimizer


def group_weight(module):
    group_decay = []
    group_no_decay = []
    for m in module.modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, Conv2d):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, _BatchNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, GroupNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
    assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)
    return group_decay, group_no_decay


def run(config):
    # prepare directories
    prepare_directories(config)

    # build hooks
    hooks = build_hooks(config)

    # build model
    model = build_model(config, hooks)

    # build loss
    loss = build_loss(config)
    loss_fn = hooks.loss_fn
    hooks.loss_fn = lambda **kwargs: loss_fn(loss_fn=loss, **kwargs)
    
    # build optimizer
    if 'no_bias_decay' in config.train and config.train.no_bias_decay:
        if 'encoder_lr_ratio' in config.train:
            encoder_lr_ratio = config.train.encoder_lr_ratio
            group_decay_encoder, group_no_decay_encoder = group_weight(model.encoder)
            group_decay_decoder, group_no_decay_decoder = group_weight(model.decoder)
            base_lr = config.optimizer.params.lr
            params = [{'params': group_decay_decoder},
                      {'params': group_no_decay_decoder, 'weight_decay': 0.0},
                      {'params': group_decay_encoder, 'lr': base_lr * encoder_lr_ratio},
                      {'params': group_no_decay_encoder, 'lr': base_lr * encoder_lr_ratio, 'weight_decay': 0.0}]
        else:
            group_decay, group_no_decay = group_weight(model)
            params = [{'params': group_decay},
                      {'params': group_no_decay, 'weight_decay': 0.0}]
    elif 'encoder_lr_ratio' in config.train:
        denom = config.train.encoder_lr_ratio
        base_lr = config.optimizer.params.lr
        params = [{'params': model.decoder.parameters()},
                  {'params': model.encoder.parameters(), 'lr': base_lr * encoder_lr_ratio}]
    else:
        params = model.parameters()
    optimizer = build_optimizer(config, params=params)

    model = model.cuda()
    # load checkpoint
    checkpoint = kvt.utils.get_initial_checkpoint(config)
    if checkpoint is not None:
        last_epoch, step = kvt.utils.load_checkpoint(model, optimizer, checkpoint)
        print('epoch, step:', last_epoch, step)
    else:
        last_epoch, step = -1, -1

    model, optimizer = to_data_parallel(config, model, optimizer)

    # build scheduler
    scheduler = build_scheduler(config, optimizer=optimizer, 
                                last_epoch=last_epoch)

    # build datasets
    dataloaders = build_dataloaders(config)

    # build summary writer
    writer = SummaryWriter(logdir=config.train.dir)
    logger_fn = hooks.logger_fn
    hooks.logger_fn = lambda **kwargs: logger_fn(writer=writer, **kwargs)

    # train loop
    train(config=config,
          model=model,
          optimizer=optimizer,
          scheduler=scheduler,
          dataloaders=dataloaders,
          hooks=hooks,
          last_epoch=last_epoch+1)
