from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse

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
LABEL_LIST = ['Fish', 'Flower', 'Gravel', 'Sugar']


def mask2rle(img):
    pixels= img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)
                                     

def rle2mask(height, width, encoded):
    if isinstance(encoded, float):
        img = np.zeros((height,width), dtype=np.uint8)
        return img

    s = encoded.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(height*width, dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape((width, height)).T


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


def make_submission(input_dirs, cls_thresholds=[0.5,0.5,0.5,0.5], thresholds=[0.5,0.5,0.5,0.5],
                    use_argmax=False):
    print('cls_thresholds:', cls_thresholds)
    print('thresholds:', thresholds)
    print('use_argmax:', use_argmax)
    ids = []
    rles = []

    dfs = []
    for input_dir in input_dirs:
        filepath = os.path.join(input_dir, 'cls.csv')
        if not os.path.exists(filepath):
            print(filepath, 'is not exist')
            continue
        dfs.append(pd.read_csv(filepath, index_col='image_id'))
    df_cls = sum(dfs) / len(dfs)
    num_cls = len(dfs)

    for i, (image_id, row) in tqdm.tqdm(enumerate(df_cls.iterrows()), total=len(df_cls)):
        predictions = []
        for input_dir in input_dirs:
            filepath = os.path.join(input_dir, f'{image_id}.npz')
            if not os.path.exists(filepath):
                continue

            with np.load(filepath) as data:
                arr = data['arr_0']
            predictions.append(arr.astype(np.float16) / 255.0)

        predictions = np.mean(np.stack(predictions, axis=0), axis=0)

        if use_argmax:
            cls_prob = np.sort(predictions.reshape(4,-1), axis=1)
            cls_prob = np.mean(cls_prob[:,-18000:], axis=1)
            cls_scores = (row[['p0', 'p1', 'p2', 'p3']].values * num_cls + cls_prob) / (num_cls + 1)
            cls_scores[np.argmax(cls_scores)] = 1

        for c in range(0,4):
            image_id_with_cls = f'{image_id}_{c}'
            cls_score = row[f'p{c}']

            if use_argmax:
                cls_score = cls_scores[c]
            else:
                cls_prob = np.sort(predictions[c,:,:].flatten())
                cls_prob = np.mean(cls_prob[-18000:])
                cls_score = (cls_score * num_cls + cls_prob) / (num_cls + 1)
            
            cls_prediction = (cls_score > cls_thresholds[c])

            prediction = predictions[c,:,:]
            prediction = (prediction > thresholds[c])
            prediction = np.logical_and(prediction, cls_prediction).astype(np.uint8)

            H,W = prediction.shape
            assert H == 350 and W == 525
            rle_encoded = mask2rle(prediction)
            assert np.all(rle2mask(H, W, rle_encoded) == prediction)
            ids.append(f'{image_id}_{LABEL_LIST[c]}')
            rles.append(rle_encoded)

    return pd.DataFrame({'Image_Label': ids, 'EncodedPixels': rles})


def evaluate(pred_path, label_path):
    df_pred = pd.read_csv(pred_path)
    df_label = pd.read_csv(label_path)

    preds = []
    labels = []
    for (_, row_p), (_, row_l) in tqdm.tqdm(zip(df_pred.iterrows(), df_label.iterrows()), total=len(df_pred)):
        preds.append(rle2mask(350, 525, row_p.EncodedPixels))
        labels.append(rle2mask(350, 525, row_l.EncodedPixels))

    preds = np.stack(preds, axis=0)
    labels = np.stack(labels, axis=0)
    print(preds.shape, labels.shape)
    metric = compute_metrics(preds, labels)
    print(metric)


def parse_args():
    parser = argparse.ArgumentParser(description='make submission')
    parser.add_argument('--input_dir', dest='input_dir',
                        help='the directory where inferenced files are located',
                        type=str)
    parser.add_argument('--output', dest='output',
                        help='the name of submission file',
                        required=True,
                        default=None, type=str)
    parser.add_argument('--submission', dest='submission',
                        help='the pseudo label file',
                        default='submissions/v3.s27.resnet34.384_576_768.resnext101.384.15e.s32.resnext101.384_544_576.with_s11.resnet34.resnext101.s12.swa.tta4.base.csv',
                        type=str)
    return parser.parse_args()


def main():
    print('make submission')
    args = parse_args()
    assert args.output is not None
    
    output_dir = os.path.dirname(args.output)
    os.makedirs(output_dir, exist_ok=True)

    input_dirs = args.input_dir.split(',')

    cls_thresholds = [0.7,0.7,0.7,0.7]
    thresholds = [0.425,0.425,0.425,0.425]
    # cls_thresholds = [0.7075,0.7025 ,0.715,0.7025]
    # thresholds = [0.425,0.45,0.45,0.4125]
    df = make_submission([os.path.join(input_dir, 'test') for input_dir in input_dirs],
                         cls_thresholds=cls_thresholds,
                         thresholds=thresholds,
                         use_argmax=False) #True)
    df.to_csv(args.output, index=False)

    # evaluate(args.output, args.submission)


if __name__ == '__main__':
    main()
