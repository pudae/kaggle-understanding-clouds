import os
import math
from collections import defaultdict

import tqdm

import numpy as np
import torch
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


def evaluate_split(config, model, split, dataloader, hooks):
    model.eval()

    batch_size = config.evaluation.batch_size
    total_size = len(dataloader.dataset)
    total_step = math.ceil(total_size / batch_size)

    with torch.no_grad():
        losses = []
        aggregated_outputs_dict = defaultdict(list)
        aggregated_outputs = []
        aggregated_labels = []

        tbar = tqdm.tqdm(enumerate(dataloader), total=total_step)
        for i, data in tbar:
            images = data['image'].cuda()
            labels = data['label'].cuda()

            outputs = hooks.forward_fn(model=model, images=images, labels=labels,
                                       data=data, is_train=False)
            outputs = hooks.post_forward_fn(outputs=outputs, images=images, data=data, is_train=False)
            loss = hooks.loss_fn(outputs=outputs, labels=labels, data=data, is_train=False)
            losses.append(loss.item())

            # aggregate outputs
            if isinstance(outputs, dict):
                for key, value in outputs.items():
                    aggregated_outputs_dict[key].append(value.cpu().numpy())
            else:
                aggregated_outputs.append(outputs.cpu().numpy())

            aggregated_labels.append(labels.cpu().numpy())

    def concatenate(v):
        # not a list or empty
        if not isinstance(v, list) or not v:
            return v

        # ndarray
        if isinstance(v[0], np.ndarray):
            return np.concatenate(v, axis=0)
        
        return v

    aggregated_outputs_dict = {key:concatenate(value) for key, value in aggregated_outputs_dict.items()}
    aggregated_outputs = concatenate(aggregated_outputs)
    aggregated_labels = concatenate(aggregated_labels)

    # list & empty
    if isinstance(aggregated_outputs, list) and not aggregated_outputs:
        aggregated_outputs = aggregated_outputs_dict

    metric_dict = hooks.metric_fn(outputs=aggregated_outputs,
                                  labels=aggregated_labels,
                                  is_train=False)

    return metric_dict['score']


def evaluate(config, model, hooks, dataloaders):
    # validation
    for dataloader in dataloaders:
        split = dataloader['split']
        dataset_mode = dataloader['mode']

        if dataset_mode != 'validation':
            continue

        dataloader = dataloader['dataloader']
        score = evaluate_split(config, model, split, dataloader, hooks)
        print(f'[{split}] score: {score}')


def run(config):
    # build hooks
    hooks = build_hooks(config)

    # build model
    model = build_model(config, hooks)

    # build loss
    loss = build_loss(config)
    loss_fn = hooks.loss_fn
    hooks.loss_fn = lambda **kwargs: loss_fn(loss_fn=loss, **kwargs)

    # load checkpoint
    checkpoint = config.checkpoint
    last_epoch, step = kvt.utils.load_checkpoint(model, None, checkpoint)

    # build datasets
    dataloaders = build_dataloaders(config)

    model = model.cuda()
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # train loop
    evaluate(config=config,
             model=model,
             dataloaders=dataloaders,
             hooks=hooks)

