import torch.hub
import torch.nn as nn

def _resnext101_32xxxd_wsl(name, num_classes=1000):
    model = torch.hub.load('facebookresearch/WSL-Images', name)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def resnext101_32x8d_wsl(num_classes=1000):
    return _resnext101_32xxxd_wsl('resnext101_32x8d_wsl', num_classes=num_classes)


def resnext101_32x16d_wsl(num_classes=1000):
    return _resnext101_32xxxd_wsl('resnext101_32x16d_wsl', num_classes=num_classes)


def resnext101_32x32d_wsl(num_classes=1000):
    return _resnext101_32xxxd_wsl('resnext101_32x32d_wsl', num_classes=num_classes)


def resnext101_32x48d_wsl(num_classes=1000):
    return _resnext101_32xxxd_wsl('resnext101_32x48d_wsl', num_classes=num_classes)
