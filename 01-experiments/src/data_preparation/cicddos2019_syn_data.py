import os

import numpy as np
import pandas as pd

from src.features import selected_features
from src.utils import split_data

rstate = 1337


output_path = '../../lib/datasets/experiments/cicddos2019_syn'
if not os.path.isdir(output_path):
    os.makedirs(output_path)


datasets = {
    'cicddos2019_syn_train': 'datasets/filtered/cicddos2019/cicddos2019_train_syn.csv',
    'cicddos2019_syn_test': 'datasets/filtered/cicddos2019/cicddos2019_test_syn.csv',
}

cicddos2019_syn_train = pd.read_csv(
    datasets['cicddos2019_syn_train'],
    usecols=selected_features.keys())

cicddos2019_syn_test = pd.read_csv(
    datasets['cicddos2019_syn_test'],
    usecols=selected_features.keys())

cicddos2019_syn = pd.concat([cicddos2019_syn_train, cicddos2019_syn_test], axis=0)
cicddos2019_syn['label'] = [1] * cicddos2019_syn.shape[0]
cicddos2019_syn = cicddos2019_syn.sample(frac=1, random_state=rstate)
print('Shape of data: {}'.format(cicddos2019_syn.shape))


cicddos2019_syn_X_train, cicddos2019_syn_y_train, cicddos2019_syn_X_test, cicddos2019_syn_y_test = split_data(
    cicddos2019_syn.iloc[:, :-1], cicddos2019_syn.iloc[:, -1], percent_test=25
)

np.savez_compressed('{}/{}.npz'.format(output_path, 'syn_train_test'),
                    X_train=cicddos2019_syn_X_train,
                    y_train=cicddos2019_syn_y_train,
                    X_test=cicddos2019_syn_X_test,
                    y_test=cicddos2019_syn_y_test)
