from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sacred import Experiment

ex = Experiment('kvt')

@ex.config
def default_config():
    # configuration for model
    backbone = {
        'name': 'resnet18',
        'params': {
            'pretrained': True
        }
    }

    # configuration for loss 
    loss = {
        'name': 'CrossEntropyLoss',
        'params': {}
    }

    # configuration for optimizer
    optimizer = {
        'name': 'Adam',
        'params': {
            'lr': 0.001,
            'weight_decay': 0.0005
        }
    }

    # configuration for scheduler
    scheduler = {
        'name': 'none',
        'params': {}
    }

    # configuration for trainsform
    transform = {
        'name': 'default_transform',
        'num_preprocessor': 8,
        'params': {}
    }

    # configuration for training
    train = {
        'dir': 'train_logs/base',
        'batch_size': 32,
        'log_step': 2,
        'gradient_accumulation_step': None,
        'num_epochs': 100,
        'save_checkpoint_epoch': 1,
        'num_keep_checkpoint': 5,
    }

    # configuration for evaluation
    evaluation = {
        'batch_size': 64
    }
