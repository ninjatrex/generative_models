import numpy as np
import pandas as pd
from src.utils import unpickle_data

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
from src.features import selected_features

model_path = '../lib/models/generative'
rstate = 1337

syn_data_path = '../lib/datasets/experiments/cicddos2019_syn/syn_train_test.npz'
syn_data = np.load(syn_data_path)

print('Load data (cicddos2019_syn_ddos) ...')
X_test, y_test = syn_data['X_test'], syn_data['y_test']
print('X_test: {}, y_test: {}'.format(X_test.shape, y_test.shape))

X_test = pd.DataFrame(data=X_test, columns=selected_features)



gc_ddos2019_syn = unpickle_data('{}/cicdds2019_syn_gc.pkl.gz'.format(model_path))
print(gc_ddos2019_syn.get_distributions())
print('Sample {} data points from model ...'.format(X_test.shape[0]))
X_gen = gc_ddos2019_syn.sample(X_test.shape[0])
print(X_gen)

def after_processing(data: pd.DataFrame):
    for feature, data_type in selected_features.items():
        if data_type == float:
            data[feature] = data[feature].apply(lambda x: round(x, 5))
            data[feature] = data[feature].apply(lambda x: x if x > 0 else 0)
        data[feature] = data[feature].apply(lambda x: float(x))
after_processing(X_gen)
print(X_gen)

print(X_gen.describe())
print(X_test.describe())

print('Evaluate model by comparing sampled data with test data ...')
