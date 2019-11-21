import os
import math
from collections import defaultdict

import tqdm

import pandas as pd
import numpy as np
import torch
from tensorboardX import SummaryWriter

from kvt.builder import (
        build_hooks,
        build_model,
        build_dataloaders
)
import kvt.utils


def evaluate(config, model, hooks, dataloader):
    split = dataloader['split']
    dataset_mode = dataloader['mode']
    dataloader = dataloader['dataloader']

    model.eval()

    batch_size = config.evaluation.batch_size
    total_size = len(dataloader.dataset)
    total_step = math.ceil(total_size / batch_size)

    with torch.no_grad():
        aggregated_outputs_dict = defaultdict(list)
        aggregated_outputs = []
        aggregated_labels = []

        tbar = tqdm.tqdm(enumerate(dataloader), total=total_step)
        for i, data in tbar:
            images = data['image'].cuda() #to(device)
            labels = data['label'].cuda() #to(device)

            outputs = hooks.forward_fn(model=model, images=images, data=data, is_train=False)
            outputs = hooks.post_forward_fn(outputs=outputs, images=images, labels=labels, data=data, is_train=False)

            metric_dict = hooks.metric_fn(outputs=outputs, labels=labels, is_train=False)
            # for key, value in metric_dict.items():
            #     aggregated_metric_dict[key].append(value)
            if isinstance(outputs, dict):
                for key, value in outputs.items():
                    if isinstance(value, torch.Tensor):
                        aggregated_outputs_dict[key].append(value.cpu().numpy())
                    else:
                        aggregated_outputs_dict[key].extend(list(value))
            else:
                aggregated_outputs.append(outputs.cpu().numpy())
            aggregated_labels.append(labels.cpu().numpy())

    # metric_dict = {key:sum(value)/len(value)
    #                for key, value in aggregated_metric_dict.items()}
    # return metric_dict['score']
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

    if isinstance(aggregated_outputs, list) and not aggregated_outputs:
        aggregated_outputs = aggregated_outputs_dict

    return aggregated_outputs, aggregated_labels


def run(config):
    # build hooks
    hooks = build_hooks(config)

    # build model
    model = build_model(config, hooks)

    # load checkpoint
    checkpoint = config.checkpoint
    last_epoch, step = kvt.utils.load_checkpoint(model, None, checkpoint)
    print(f'last_epoch:{last_epoch}')

    model = model.cuda()
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # build datasets
    config.dataset.splits = [v for v in config.dataset.splits if v.split == config.inference.split]

    dataloaders = build_dataloaders(config)
    dataloaders = [dataloader for dataloader in dataloaders if dataloader['split'] == config.inference.split]
    assert len(dataloaders) == 1, f'len(dataloaders)({len(dataloaders)}) not 1'

    dataloader = dataloaders[0]

    records = []
    # train loop
    aggregated_outputs_base, aggregated_labels_base = evaluate(config=config,
                                                               model=model,
                                                               dataloader=dataloader,
                                                               hooks=hooks)

    base_score = hooks.metric_fn(outputs=aggregated_outputs_base,
                                 labels=aggregated_labels_base,
                                 is_train=False)['score']

    print('base_score:', base_score)
    records.append(('base_score', base_score))

    config.transform.name = 'aug_search'

    REPEAT = 3
    # for limit in np.arange(0.1, 0.6, 0.1):
    # for limit in [90,80,70,60]:
    for limit in [5,10,15,20]:
        scores = []
        config.transform.params.limit = limit

        dataloaders = build_dataloaders(config)
        dataloaders = [dataloader for dataloader in dataloaders if dataloader['split'] == config.inference.split]
        assert len(dataloaders) == 1, f'len(dataloaders)({len(dataloaders)}) not 1'

        dataloader = dataloaders[0]
        for _ in range(REPEAT):
            agg_outputs_aug, _ = evaluate(config=config,
                                          model=model,
                                          dataloader=dataloader,
                                          hooks=hooks)
            agg_outputs_ensemble = {}
            agg_outputs_ensemble['labels'] = aggregated_outputs_base['labels']
            agg_outputs_ensemble['probabilities'] = (aggregated_outputs_base['probabilities'] + agg_outputs_aug['probabilities']) / 2.0
            agg_outputs_ensemble['cls_probabilities'] = (aggregated_outputs_base['cls_probabilities'] + agg_outputs_aug['cls_probabilities']) / 2.0
            scores.append(hooks.metric_fn(outputs=agg_outputs_ensemble, labels=aggregated_labels_base, is_train=False)['score'])
        records.append((f'{limit:.02f}', sum(scores)/len(scores)))
        print(records[-1])

    df = pd.DataFrame.from_records(records, columns=['setting', 'score'])
    df.to_csv('aug.csv', index=False)

