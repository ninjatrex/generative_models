import os
from pprint import pprint

import numpy as np
import pandas as pd
from sdv.constraints import GreaterThan
from sdv.tabular import GaussianCopula

from src.utils import pickle_data

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
from src.features import selected_features

model_path = '../lib/models/generative'
if not os.path.isdir(model_path):
    os.makedirs(model_path)

rstate = 1337

syn_data_path = '../lib/datasets/experiments/cicddos2019_syn/syn_train_test.npz'
syn_data = np.load(syn_data_path)

fwd_total_length_packets_gt_fwd_packet_length_max = GreaterThan(
    low='fwd_packet_length_max',
    high='fwd_total_length_packets',
    handling_strategy='transform'
)

fwd_packet_length_max_gt_fwd_packet_length_mean = GreaterThan(
    low='fwd_packet_length_mean',
    high='fwd_packet_length_max',
    handling_strategy='transform'
)

fwd_packet_length_mean_gt_fwd_packet_length_std = GreaterThan(
    low='fwd_packet_length_std',
    high='fwd_packet_length_mean',
    handling_strategy='transform'
)


bwd_packet_length_mean_gt_bwd_packet_length_std = GreaterThan(
    low='bwd_packet_length_std',
    high='bwd_packet_length_mean',
    handling_strategy='transform'
)

constraints = [
    fwd_total_length_packets_gt_fwd_packet_length_max,
    fwd_packet_length_max_gt_fwd_packet_length_mean,
    fwd_packet_length_mean_gt_fwd_packet_length_std,
    bwd_packet_length_mean_gt_bwd_packet_length_std,
]

print('Load data (cicddos2019_syn_ddos) ...')
X_train, y_train, X_test, y_test = syn_data['X_train'], syn_data['y_train'], syn_data['X_test'], syn_data['y_test']
print('X_train: {}, y_train: {}, X_test: {}, y_test: {}'.format(
    X_train.shape, y_train.shape, X_test.shape, y_test.shape))

X_train = pd.DataFrame(data=X_train, columns=selected_features)
X_test = pd.DataFrame(data=X_test, columns=selected_features)

for feature in selected_features.keys():
    X_train[feature] = X_train[feature].apply(lambda x: round(x, 5))
X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
X_train.dropna(inplace=True)

import matplotlib.pyplot as plt
fig, axis = plt.subplots(5,2,figsize=(20, 10))
X_train.hist(ax=axis, bins=100)
plt.show()


print('Adapt correct data types for features ...')
syn_data = (X_train, X_test)
for data in syn_data:
    for feature, data_type in selected_features.items():
        data[feature] = data[feature].apply(lambda x: data_type(x))
    print(data.sample(10, random_state=rstate))


field_distributions = {
    'fwd_num_packets': 'gamma',
    'bwd_num_packets': 'gamma',
    'fwd_total_length_packets': 'gamma',
    # 'fwd_packet_length_max': 'beta',
    'fwd_packet_length_mean': 'student_t',
    'fwd_packet_length_std': 'gamma',
    'bwd_packet_length_mean': 'gamma',
    'bwd_packet_length_std': 'gamma',
    'fwd_avg_num_packets': 'student_t',
    'bwd_avg_num_packets': 'gamma'
}

print('Train GaussianCopula Model on training data...')
gc = GaussianCopula(constraints=constraints, field_distributions=field_distributions)

gc.fit(X_train)
pickle_data(gc, '{}/cicdds2019_syn_gc.pkl.gz'.format(model_path))

print('Show distributions of features ...')
pprint(gc.get_distributions())
X_gen = gc.sample(X_test.shape[0])
print(X_gen)
def after_processing(data: pd.DataFrame):
    for feature, data_type in selected_features.items():
        if data_type == float:
            data[feature] = data[feature].apply(lambda x: round(x, 5))
            data[feature] = data[feature].apply(lambda x: x if x > 0 else 0)
        data[feature] = data[feature].apply(lambda x: float(x))
after_processing(X_gen)
fig, axis = plt.subplots(5,2,figsize=(20, 10))
X_gen.hist(ax=axis, bins=100)
plt.show()

