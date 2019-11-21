from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import pprint
from easydict import EasyDict as edict
from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds

import torch
import custom   # import all custom modules for registering objects.

from kvt.initialization import initialize
from kvt.utils import ex
from kvt.apis.train import run as run_train
from kvt.apis.evaluate import run as run_evaluate
from kvt.apis.inference import run as run_inference
from kvt.apis.swa import run as run_swa


ex.captured_out_filter = apply_backspaces_and_linefeeds

@ex.main
def main(_run, _config):
    config = edict(_config)
    pprint.PrettyPrinter(indent=2).pprint(config)


@ex.command
def train(_run, _config):
    config = edict(_config)
    print('------------------------------------------------')
    print('train')
    pprint.PrettyPrinter(indent=2).pprint(config)
    run_train(config)


@ex.command
def evaluate(_run, _config):
    config = edict(_config)
    print('------------------------------------------------')
    print('evaluate')
    pprint.PrettyPrinter(indent=2).pprint(config)
    run_evaluate(config)


@ex.command
def inference(_run, _config):
    config = edict(_config)
    print('------------------------------------------------')
    print('inference')
    pprint.PrettyPrinter(indent=2).pprint(config)
    run_inference(config)


@ex.command
def swa(_run, _config):
    config = edict(_config)
    print('------------------------------------------------')
    print('swa')
    pprint.PrettyPrinter(indent=2).pprint(config)
    run_swa(config)


if __name__ == '__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.backends.cudnn.deterministic=True

    initialize()
    ex.run_commandline()
