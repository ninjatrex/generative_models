import gzip
import os
import pickle

import numpy as np
import pandas as pd


def shuffle_data(data):
    permutation = np.random.RandomState(seed=42).permutation(data.shape[0])
    return data[permutation]


def split_data(X, y, percent_test=10):
    num_elements_train = int(X.shape[0] * ((100 - percent_test) / 100))

    X_train = X[:num_elements_train]
    y_train = y[:num_elements_train]

    X_test = X[num_elements_train:]
    y_test = y[num_elements_train:]

    return X_train, y_train, X_test, y_test


def unpickle_data(pickle_file):
    with gzip.open(pickle_file, 'rb') as file:
        return pickle.load(file)


def pickle_data(data, pickle_file):
    with gzip.GzipFile(pickle_file, 'ab' if os.path.isfile(pickle_file) else 'wb') as file:
        pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)


def get_num_unique_labels(data: pd.DataFrame):
    return {label: len(data[data['label'] == label]) for label in pd.unique(data['label'])}


def get_samples_with_label(data: pd.DataFrame, label: str):
    return data[data['label'] == label]

def transform_labels(data: pd.DataFrame):
    return data['label'].apply(lambda label: 0 if label == 'benign' else 1)
