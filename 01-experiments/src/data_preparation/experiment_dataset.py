import os

import pandas as pd
from sdv.tabular import GaussianCopula

from src.features import selected_features

rstate = 1337

datasets = {
    'ids2017_benign_train': 'datasets/filtered/cicids2017/ids2017_train_benign.csv',
    'ids2017_benign_test': 'datasets/filtered/cicids2017/ids2017_test_benign.csv',

    'ids2017_dos_hulk_train': 'datasets/filtered/cicids2017/ids2017_train_dos_hulk.csv',
    'ids2017_dos_hulk_test': 'datasets/filtered/cicids2017/ids2017_test_dos_hulk.csv',

    'cicddos2019_syn': 'datasets/filtered/cicddos2019/cicddos2019_train_syn.csv',
}

output_path = 'datasets/experiments/stream_exp_prep'
if not os.path.isdir(output_path):
    os.makedirs(output_path)

print('Load datasets ...')

ids2017_benign_train = pd.read_csv(
    datasets['ids2017_benign_train'],
    usecols=selected_features.keys()).sample(
    n=20000,
    random_state=rstate)
ids2017_benign_train['label'] = [0] * ids2017_benign_train.shape[0]

ids2017_dos_golden_eye_train = pd.read_csv(
    datasets['ids2017_dos_hulk_train'],
    usecols=selected_features.keys()).sample(
    n=20000,
    random_state=rstate)
ids2017_dos_golden_eye_train['label'] = [1] * ids2017_dos_golden_eye_train.shape[0]

ids2017_benign_test = pd.read_csv(
    datasets['ids2017_benign_test'],
    usecols=selected_features.keys()).sample(
    n=10000,
    random_state=rstate)
ids2017_benign_test['label'] = [0] * ids2017_benign_test.shape[0]

ids2017_dos_golden_eye_test = pd.read_csv(
    datasets['ids2017_dos_hulk_test'],
    usecols=selected_features.keys()).sample(
    n=10000,
    random_state=rstate)
ids2017_dos_golden_eye_test['label'] = [1] * ids2017_dos_golden_eye_test.shape[0]

cicddos2019_syn = pd.read_csv(
    datasets['cicddos2019_syn'],
    usecols=selected_features.keys())
cicddos2019_syn['label'] = [1] * cicddos2019_syn.shape[0]

cicddos2019_syn_train = cicddos2019_syn.iloc[:int(cicddos2019_syn.shape[0]/2), :]
cicddos2019_syn_test = cicddos2019_syn.iloc[int(cicddos2019_syn.shape[0]/2):, :].sample(n=10000, random_state=rstate)

print('Create a generative model of cicddos2019_syn_train data ...')
model = GaussianCopula()
model.fit(cicddos2019_syn_train.iloc[:, :-1])
cicddos2019_syn_train_generated = model.sample(20000)
cicddos2019_syn_train_generated['label'] = [1] * cicddos2019_syn_train_generated.shape[0]

print('Create application ddos train dataset ...')
app_ddos_train_dataset = pd.concat([ids2017_benign_train, ids2017_dos_golden_eye_train], axis=0)
app_ddos_train_dataset = app_ddos_train_dataset.sample(frac=1, random_state=rstate)
app_ddos_train_dataset.to_csv('{}/{}.csv'.format(output_path, 'app_ddos_train_dataset'), index=False)

print('Create application ddos test dataset ...')
app_ddos_test_dataset = pd.concat([ids2017_benign_test, ids2017_dos_golden_eye_test], axis=0)
app_ddos_test_dataset = app_ddos_test_dataset.sample(frac=1, random_state=rstate)
app_ddos_test_dataset.to_csv('{}/{}.csv'.format(output_path, 'app_ddos_test_dataset'), index=False)

print('Create udp ddos train dataset with synthetic data ...')
syn_ddos_train_dataset = pd.concat([ids2017_benign_train, cicddos2019_syn_train_generated], axis=0)
syn_ddos_train_dataset = syn_ddos_train_dataset.sample(frac=1, random_state=rstate)
syn_ddos_train_dataset.to_csv('{}/{}.csv'.format(output_path, 'udp_ddos_train_dataset'), index=False)

print('Create udp ddos test dataset with real data')
syn_ddos_test_dataset = pd.concat([ids2017_benign_test, cicddos2019_syn_test], axis=0)
syn_ddos_test_dataset = syn_ddos_test_dataset.sample(frac=1, random_state=rstate)
syn_ddos_test_dataset.to_csv('{}/{}.csv'.format(output_path, 'udp_ddos_test_dataset'), index=False)




