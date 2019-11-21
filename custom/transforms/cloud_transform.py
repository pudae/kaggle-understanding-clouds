from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
import albumentations as albu

import kvt


def get_training_augmentation(resize_to=(320,640), crop_size=(288,576)):
    print('[get_training_augmentation] crop_size:', crop_size, ', resize_to:', resize_to) 

    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.ShiftScaleRotate(scale_limit=0.20, rotate_limit=10, shift_limit=0.1, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=0),
        albu.GridDistortion(p=0.5),
        albu.Resize(*resize_to),
        albu.RandomCrop(*crop_size),
        albu.ChannelShuffle(),
        albu.InvertImg(),
        albu.ToGray(),
        albu.Normalize(),
    ]

    return albu.Compose(train_transform)


def get_test_augmentation(resize_to=(320,640)):
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.Resize(*resize_to),
        albu.Normalize(),
    ]
    return albu.Compose(test_transform)


@kvt.TRANSFORMS.register
def cloud_transform(split, resize_to=(320,640), crop_size=(288,576), tta=1, **_):
    if isinstance(resize_to, str):
        resize_to = eval(resize_to)
    if isinstance(crop_size, str):
        crop_size = eval(crop_size)
    print('[cloud_transform] resize_to:', resize_to)
    print('[cloud_transform] crop_size:', crop_size)
    print('[cloud_transform] tta:', tta)

    train_aug = get_training_augmentation(resize_to, crop_size)
    test_aug = get_test_augmentation(resize_to)

    def transform(image, mask):
        if split == 'train':
            augmented = train_aug(image=image, mask=mask)
        else:
            augmented = test_aug(image=image, mask=mask)

        if tta > 1:
            images = []
            images.append(augmented['image'])
            images.append(test_aug(image=np.fliplr(image))['image'])
            if tta > 2:
                images.append(test_aug(image=np.flipud(image))['image'])
            if tta > 3:
                images.append(test_aug(image=np.flipud(np.fliplr(image)))['image'])
            image = np.stack(images, axis=0)
            image = np.transpose(image, (0,3,1,2))
        else:
            image = augmented['image']
            image = np.transpose(image, (2,0,1))
        mask = augmented['mask']
        mask = np.transpose(mask, (2,0,1))

        return image, mask

    return transform
