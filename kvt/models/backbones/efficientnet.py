from efficientnet_pytorch import EfficientNet


def efficientnet_b0(num_classes=1000, **kwargs):
    return EfficientNet('efficientnet-b0', num_classes=num_classes)


def efficientnet_b1(num_classes=1000, **kwargs):
    return EfficientNet('efficientnet-b1', num_classes=num_classes)


def efficientnet_b2(num_classes=1000, **kwargs):
    return EfficientNet('efficientnet-b2', num_classes=num_classes)


def efficientnet_b3(num_classes=1000, **kwargs):
    return EfficientNet('efficientnet-b3', num_classes=num_classes)


def efficientnet_b4(num_classes=1000, **kwargs):
    return EfficientNet('efficientnet-b4', num_classes=num_classes)


def efficientnet_b5(num_classes=1000, **kwargs):
    return EfficientNet('efficientnet-b5', num_classes=num_classes)


def efficientnet_b6(num_classes=1000, **kwargs):
    return EfficientNet('efficientnet-b6', num_classes=num_classes)


def efficientnet_b7(num_classes=1000, **kwargs):
    return EfficientNet('efficientnet-b7', num_classes=num_classes)
