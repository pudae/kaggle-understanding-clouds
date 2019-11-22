from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import gc
import argparse
import itertools

import tqdm
import pandas as pd
import numpy as np
import cv2


LABEL_MAP = {
        'Fish': 0,
        'Flower': 1,
        'Gravel': 2,
        'Sugar': 3
        }


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


def load_predictions(input_dirs, df_train):
    print(input_dirs)
    image_ids = os.path.join(input_dirs[0], 'image_ids.csv')
    image_ids = pd.read_csv(image_ids)['Image']

    dfs = []
    for input_dir in input_dirs:
        filepath = os.path.join(input_dir, 'cls.csv')
        if os.path.exists(filepath):
            dfs.append(pd.read_csv(filepath, index_col='image_id'))

    ret = []
    cls_records = []
    for i, image_id in tqdm.tqdm(enumerate(image_ids), total=len(image_ids)):
        predictions = []
        for input_dir in input_dirs:
            filepath = os.path.join(input_dir, f'{image_id}.npz')
            if not os.path.exists(filepath):
                continue

            with np.load(filepath) as data:
                arr = data['arr_0']
            predictions.append(arr.astype(np.float16) / 255.0)

        predictions = np.mean(np.stack(predictions, axis=0), axis=0)

        cls_probs = []
        for c in range(0,4):
            image_id_with_cls = f'{image_id}_{c}'
            rle_encoded = df_train.loc[image_id_with_cls]['EncodedPixels']
            label = rle2mask(1400, 2100, rle_encoded)
            label = cv2.resize(label, (525,350), interpolation=cv2.INTER_NEAREST)
            ret.append((image_id_with_cls, predictions[c,:,:], label))

            cls_prob = np.sort(predictions[c,:,:].flatten())
            cls_prob = np.mean(cls_prob[-17500:])
            cls_probs.append(cls_prob)
        cls_records.append(tuple([image_id] + cls_probs))
        del predictions

    df_seg = pd.DataFrame.from_records(cls_records, columns=['image_id', 'p0', 'p1', 'p2', 'p3'])
    df_seg = df_seg.set_index('image_id')
    dfs.append(df_seg)

    df = sum(dfs) / len(dfs)
    df.values[np.arange(df.values.shape[0]), np.argmax(df.values, axis=1)] = 1.0

    gc.collect()
    return ret, df


def evaluate(predictions, df_cls, cls_thresholds=[0.5,0.5,0.5,0.5], thresholds=[0.5,0.5,0.5,0.5]):
    image_ids = []
    masks = []
    labels = []

    for p in predictions:
        image_id, mask, label = p
        cls_id = int(image_id[-1:])
        if df_cls is not None:
            cls_score = df_cls.loc[image_id[:-2]][f'p{cls_id}']
        else:
            cls_score = np.array([1.0])

        thres = thresholds[cls_id]
        cls_thres = cls_thresholds[cls_id]

        cls_prediction = (cls_score > cls_thres)
        mask_prediction = (mask > thres)
        mask_prediction = np.logical_and(mask_prediction, cls_prediction)

        image_ids.append(image_id)
        masks.append(mask_prediction)
        labels.append(label)

    masks = np.array(masks)
    labels = np.array(labels)

    return compute_metrics(masks, labels)


def compute_metrics(predicts, labels):
    N, H, W = predicts.shape

    predicts = predicts.reshape((-1, H*W))
    labels = labels.reshape((-1, H*W))

    sum_p = np.sum(predicts, axis=1)
    sum_l = np.sum(labels, axis=1)
    intersection = np.sum(np.logical_and(predicts, labels), axis=1)

    numer = 2*intersection
    denom = sum_p + sum_l
    dice = numer / (denom + 1e-6)

    empty_indices = np.where(sum_l <= 0)[0]
    non_empty_indices = np.where(sum_l > 0)[0]
    if len(non_empty_indices) == 0:
        non_empty_mean_dice = 0.0
    else:
        non_empty_dice = dice[non_empty_indices]
        non_empty_mean_dice = float(np.mean(non_empty_dice))

    all_non_empty_index = np.where(numer > 0)[0]
    all_empty_index = np.where(denom == 0)[0]
    dice[all_empty_index] = 1
    mean_dice = float(np.mean(dice))

    cls_accuracy = (len(all_non_empty_index) + len(all_empty_index)) / N

    correct_indices = np.where((sum_p > 0) == (sum_l > 0))[0]
    incorrect_indices = np.where((sum_p > 0) != (sum_l > 0))[0]

    tp = len(np.where(sum_l[correct_indices] > 0)[0])
    tn = len(np.where(sum_l[correct_indices] == 0)[0])

    fp = len(np.where(sum_l[incorrect_indices] == 0)[0])
    fn = len(np.where(sum_l[incorrect_indices] > 0)[0])

    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)

    tnr = tn / (tn + fp + 1e-10)
    fpr = fp / (fp + tn + 1e-10)

    return {'mean_dice': mean_dice,
            'mean_dice_non_empty': non_empty_mean_dice,
            'cls_acc': cls_accuracy,
            'precision': precision,
            'recall': recall,
            'tnr': tnr,
            'fpr': fpr,
            'tp/tn/fp/fn': [tp,tn,fp,fn]}


def parse_args():
    parser = argparse.ArgumentParser(description='evaluate')
    parser.add_argument('--input_dir', dest='input_dir',
                        help='the directory where inferenced files are located',
                        type=str)
    return parser.parse_args()


def main():
    print('evaluate')
    args = parse_args()

    df_train = pd.read_csv('data/train.csv')
    def _to_image_id_class_id(v):
        image_id = v[:v.find('_')]
        label = v[v.find('_')+1:]
        return f'{image_id}_{LABEL_MAP[label]}'

    df_train['ImageId_ClassId'] = df_train.Image_Label.map(_to_image_id_class_id)
    df_train = df_train.set_index('ImageId_ClassId')

    input_dirs = args.input_dir.split(',')
    pred_dev, df_cls_dev = load_predictions(
            [os.path.join(input_dir, 'dev') for input_dir in input_dirs], df_train)
    pred_test_dev, df_cls_test_dev = load_predictions(
            [os.path.join(input_dir, 'test_dev') for input_dir in input_dirs], df_train)

    cls_thresholds = [0.7,0.7,0.7,0.7]
    thresholds = [0.425,0.425,0.425,0.425]
    print(thresholds, cls_thresholds)
    print('dev:', evaluate(pred_dev, df_cls_dev, thresholds=thresholds, cls_thresholds=cls_thresholds))
    print('test_dev:', evaluate(pred_test_dev, df_cls_test_dev, thresholds=thresholds, cls_thresholds=cls_thresholds))


if __name__ == '__main__':
    main()


