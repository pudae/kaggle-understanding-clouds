from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tqdm
import pandas as pd
import numpy as np
import cv2


def mask2rle(img):
    pixels= img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)
                                     

def rle2mask(height, width, encoded):
    img = np.zeros(height*width, dtype=np.uint8)

    if isinstance(encoded, float):
        return img.reshape((width, height)).T

    s = encoded.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape((width, height)).T


def resize(image, size=(1050, 700)):
    image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
    return image


for input_dir in ['data/test_images', 'data/train_images']:
    output_dir = input_dir + '_700'

    os.makedirs(output_dir, exist_ok=True)

    filenames = list(os.listdir(input_dir))
    for filename in tqdm.tqdm(filenames):
        inpath = os.path.join(input_dir, filename)
        outpath = os.path.join(output_dir, filename)
        image = cv2.imread(inpath)
        image = resize(image)
        cv2.imwrite(outpath, image)
