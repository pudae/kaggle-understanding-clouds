from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import pandas as pd

import kvt


@kvt.HOOKS.register
class CloudWriteResultHook:
    def __call__(self, split, output_path, outputs, labels=None, data=None, is_train=False):
        assert isinstance(outputs, dict), f'type(outputs): {type(outputs)}'
        assert 'image_id' in outputs, f'image_id not in outputs'

        image_ids = outputs['image_id']

        output_dir = os.path.join(output_path, split)
        os.makedirs(output_dir, exist_ok=True)

        if 'probabilities' in outputs:
            probabilities = outputs['probabilities']
            assert len(probabilities) == len(image_ids), \
                    f'len(probabilities)({len(probabilities)}) is not same with len(image_ids)({len(image_ids)})'

            probabilities = probabilities * 255 
            assert np.min(probabilities) >= 0.0
            assert np.max(probabilities) <= 255

            for image_id, seg in zip(image_ids, probabilities):
                output_filename = os.path.join(output_dir, f'{image_id}.npz')
                np.savez_compressed(output_filename, seg.astype(np.uint8))
        else:
            if 'cls_probabilities' in outputs:
                cls_probabilities = outputs['cls_probabilities']
            else:
                cls_probabilities = None

            output_filename = os.path.join(output_dir, 'image_ids.csv')
            df = pd.DataFrame(index=image_ids, data=image_ids, columns=['Image'])
            df.to_csv(output_filename, index=False)

            if cls_probabilities is not None:
                df = pd.DataFrame(index=image_ids, data=cls_probabilities, columns=['p0', 'p1', 'p2', 'p3'])
                df.index.name = 'image_id'
                output_filename = os.path.join(output_dir, 'cls.csv')
                df.to_csv(output_filename)
