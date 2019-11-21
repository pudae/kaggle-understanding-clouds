import tqdm
import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


N_SPLITS = 11
TRAIN_CSV_PATH = 'data/train.csv'

LABEL_MAP = {
        'Fish': 0,
        'Flower': 1,
        'Gravel': 2,
        'Sugar': 3
        }

df_train = pd.read_csv(TRAIN_CSV_PATH)
df_train['Image'] = df_train.Image_Label.map(lambda v: v[:v.find('_')])
df_train['Label'] = df_train.Image_Label.map(lambda v: v[v.find('_')+1:])
df_train['LabelIndex'] = df_train.Label.map(lambda v: LABEL_MAP[v])


X = []
y = []
image_ids = []

df_group = df_train.groupby('Image')
for i, (key, df) in tqdm.tqdm(enumerate(df_group), total=len(df_group)):
    X.append([i])
    ml = np.array([0,0,0,0])
    df = df.dropna()
    ml[np.array(df.LabelIndex)-1] = 1
    y.append(ml)
    image_ids.append(key)


random_state = 1234
mskf = MultilabelStratifiedKFold(n_splits=N_SPLITS, random_state=random_state)

df_train['Fold'] = 0
df_train = df_train.set_index('Image')
for f, (train_index, test_index) in enumerate(mskf.split(X, y)):
    for i in tqdm.tqdm(test_index):
        df_train.loc[image_ids[i], 'Fold'] = f

df_train = df_train.reset_index()
df_train.to_csv(f'data/train.ver0.csv', index=False)
