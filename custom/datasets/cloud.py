from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random

import numpy as np
import pandas as pd
import cv2
from torch.utils.data.dataset import Dataset

import kvt

from custom.datasets.utils import rle2mask


LABEL_MAP = {
        'Fish': 0,
        'Flower': 1,
        'Gravel': 2,
        'Sugar': 3
        }


@kvt.DATASETS.register
class CloudDataset(Dataset):
    def __init__(self,
                 data_dir,
                 split,
                 transform=None,
                 idx_fold=0,
                 num_fold=10,
                 csv_filename='train.ver0.csv',
                 submission_filename='sample_submission.csv',
                 **_):
        self.split = split
        self.idx_fold = idx_fold
        self.num_fold = num_fold
        self.transform = transform
        self.data_dir = data_dir
        self.csv_filename = csv_filename
        self.submission_filename = submission_filename

        self.df_examples, self.image_ids = self._load_examples()

        if self.split == 'test' or self.split == 'test_pseudo':
            self.images_dir = 'test_images'
        else:
            self.images_dir = 'train_images'
        if '700' in csv_filename:
            self.images_dir = self.images_dir + '_700'

    def _load_examples(self):
        if self.split == 'test' or self.split == 'test_pseudo':
            if '700' in self.csv_filename:
                submission_name = os.path.splitext(self.submission_filename)[0]
                csv_path = os.path.join(self.data_dir, f'{submission_name}_700.csv')
            else:
                csv_path = os.path.join(self.data_dir, self.submission_filename)
        else:
            csv_path = os.path.join(self.data_dir, self.csv_filename)

        df_examples = pd.read_csv(csv_path)
        if 'Image' not in df_examples.columns:
            df_examples['Image'] = df_examples.Image_Label.map(lambda v: v[:v.find('_')])
            df_examples['Label'] = df_examples.Image_Label.map(lambda v: v[v.find('_')+1:])
            df_examples['LabelIndex'] = df_examples.Label.map(lambda v: LABEL_MAP[v])

        df_examples = df_examples.fillna('')

        dev_idx = self.idx_fold
        test_dev_idx = self.num_fold # -self.idx_fold

        if self.split == 'test_dev':
            df_examples = df_examples[df_examples.Fold == test_dev_idx]
        elif self.split == 'dev':
            df_examples = df_examples[df_examples.Fold == dev_idx]
        elif self.split == 'train':
            df_examples = df_examples[(df_examples.Fold != dev_idx) & (df_examples.Fold != test_dev_idx)]

        df_examples = df_examples.set_index('Image')
        image_ids = list(df_examples.index.unique())

        return df_examples, image_ids

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        image_path = os.path.join(self.data_dir, self.images_dir, image_id)

        image = cv2.imread(image_path)
        df = self.df_examples.loc[image_id]
        assert len(df) == 4
        mask = self.make_label(image, df)

        if self.transform:
            image, mask = self.transform(image, mask)

        mask = mask.astype(np.int32)

        return {'image_id': image_id, 'image': image, 'label': mask}

    def make_label(self, image, df):
        H,W,C = image.shape
        mask = np.zeros((H,W,4), dtype=np.uint8)
        for i in range(4):
            row = df.iloc[i]
            cls_idx = int(row.LabelIndex)
            encoded = row.EncodedPixels
            if len(encoded) == 0:
                continue
            mask[:,:,cls_idx] = rle2mask(H, W, encoded)
        return mask

    def __len__(self):
        return len(self.image_ids)
