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
        build_dataloaders
)
import kvt.utils


def inference(config, model, hooks, dataloader):
    # TODO: refactoring
    split = dataloader['split']
    dataset_mode = dataloader['mode']
    dataloader = dataloader['dataloader']

    model.eval()

    batch_size = config.evaluation.batch_size
    total_size = len(dataloader.dataset)
    total_step = math.ceil(total_size / batch_size)

    with torch.no_grad():
        aggregated_outputs_dict = defaultdict(list)
        aggregated_outputs_dict_batch = defaultdict(list)

        tbar = tqdm.tqdm(enumerate(dataloader), total=total_step)
        for i, data in tbar:
            images = data['image'].cuda() #to(device)
            
            if images.dim() == 5:
                B,T,C,H,W = images.size()
                images = images.view(-1,C,H,W)
                outputs = hooks.forward_fn(model=model, images=images, data=data, is_train=False)
                
                if isinstance(outputs, torch.Tensor):
                    assert outputs.dim() == 2
                    outputs = outputs.view(B,T,-1)
                else:
                    outputs['logits'] = outputs['logits'].view(B,T,-1,H,W)
                    if 'cls_logits' in outputs:
                        assert outputs['cls_logits'].dim() == 2
                        outputs['cls_logits'] = outputs['cls_logits'].view(B,T,-1)
            else:
                outputs = hooks.forward_fn(model=model, images=images, data=data, is_train=False)
            outputs = hooks.post_forward_fn(outputs=outputs, images=images, data=data, is_train=False)

            # aggregate outputs
            if isinstance(outputs, dict):
                for key, value in outputs.items():
                    if key == 'logits' or key == 'cls_logits':
                        continue
                    if isinstance(value, torch.Tensor):
                        if key != 'probabilities':
                            aggregated_outputs_dict[key].append(value.cpu().numpy())
                        aggregated_outputs_dict_batch[key].append(value.cpu().numpy())
                    else:
                        aggregated_outputs_dict[key].extend(list(value))
                        aggregated_outputs_dict_batch[key].extend(list(value))

            def concatenate(v):
                # not a list or empty
                if not isinstance(v, list) or not v:
                    return v

                # ndarray
                if isinstance(v[0], np.ndarray):
                    return np.concatenate(v, axis=0)
                
                return v
                
            aggregated_outputs_dict_batch = {key:concatenate(value) for key, value in aggregated_outputs_dict_batch.items()}
            hooks.write_result_fn(split, config.inference.output_path, outputs=aggregated_outputs_dict_batch)
            aggregated_outputs_dict_batch = defaultdict(list)

    def concatenate(v):
        # not a list or empty
        if not isinstance(v, list) or not v:
            return v

        # ndarray
        if isinstance(v[0], np.ndarray):
            return np.concatenate(v, axis=0)
        
        return v

    aggregated_outputs_dict = {key:concatenate(value) for key, value in aggregated_outputs_dict.items()}
    hooks.write_result_fn(split, config.inference.output_path, outputs=aggregated_outputs_dict)


def run(config):
    # build hooks
    hooks = build_hooks(config)

    # build model
    model = build_model(config, hooks)

    # load checkpoint
    checkpoint = config.checkpoint
    last_epoch, step = kvt.utils.load_checkpoint(model, None, checkpoint)
    print(f'last_epoch:{last_epoch}')

    # build datasets
    config.dataset.splits = [v for v in config.dataset.splits if v.split == config.inference.split]

    dataloaders = build_dataloaders(config)
    dataloaders = [dataloader for dataloader in dataloaders if dataloader['split'] == config.inference.split]
    assert len(dataloaders) == 1, f'len(dataloaders)({len(dataloaders)}) not 1'

    dataloader = dataloaders[0]

    model = model.cuda()
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # train loop
    inference(config=config,
              model=model,
              dataloader=dataloader,
              hooks=hooks)
