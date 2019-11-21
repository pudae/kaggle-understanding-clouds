from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import kvt


@kvt.HOOKS.register
class UNetModelBuilderHook(object):
    def __init__(self, **_):
        pass

    def __call__(self, config, backbone):
        print('UNetModelBuilderHook')
        encoder = build_encoder(config, backbone)
        if 'params' in config:
            params = config.params
        else:
            params = {}
        return build_unet(config.name, encoder, config.num_classes, **params)


@kvt.HOOKS.register
class FPNModelBuilderHook(object):
    def __init__(self, **_):
        pass

    def __call__(self, config, backbone):
        print('FPNModelBuilderHook')
        encoder = build_encoder(config, backbone)
        if 'params' in config:
            params = config.params
        else:
            params = {}
        return build_fpn(config.name, encoder, config.num_classes, **params)


@kvt.HOOKS.register
class CLSModelBuilderHook(object):
    def __init__(self, **_):
        pass

    def __call__(self, config, backbone):
        print('CLSModelBuilderHook')
        return build_cls(config.name, backbone)
