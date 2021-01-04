import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.mixture import GaussianMixture
import numpy as np
import pandas as pd
from src.utils import unpickle_data, split_data
from src.features import selected_features
from sklearn.ensemble import RandomForestClassifier
from cf_matrix import *
rstate = 1337

model_path = '../../src/models/cicddos2019'
if not os.path.isdir(model_path):
    os.makedirs(model_path)

output_path = '../../lib/images/plots'
if not os.path.isdir(output_path):
    os.makedirs(output_path)

models = unpickle_data('{}/{}.pkl.gz'.format(model_path, 'cicddos2019_gmm'))
print(models)
pca_data = unpickle_data('{}/{}.pkl.gz'.format(model_path, 'cicddos2019_pca_data'))
print(pca_data)

pca: PCA = models['pca']
gmm_ben: GaussianMixture = models['gmm_ben']
gmm_syn: GaussianMixture = models['gmm_syn']

pca_data_ben: pd.DataFrame = pca_data['ben'].sample(n=100000, random_state=rstate)
pca_data_syn: pd.DataFrame = pca_data['syn'].sample(n=100000, random_state=rstate)

print('Real Syn: {} Real Benign: {}'.format(pca_data_syn.shape, pca_data_ben.shape))

print('Sample from generators ...')
gen_data_ben = np.nan_to_num(gmm_ben.sample(pca_data_ben.shape[0])[0])
gen_data_ben = pd.DataFrame(data=gen_data_ben, columns=['pc_1'])
print(gen_data_ben)

gen_data_syn = np.nan_to_num(gmm_syn.sample(pca_data_syn.shape[0])[0])
gen_data_syn = pd.DataFrame(data=gen_data_syn, columns=['pca_1'])
print(gen_data_syn)

print('Set Labels ...')
pca_data_ben['label'] = [0] * pca_data_ben.shape[0]
print(pca_data_ben)

pca_data_syn['label'] = [1] * pca_data_syn.shape[0]
print(pca_data_syn)

gen_data_ben['label'] = [0] * gen_data_ben.shape[0]
print(gen_data_ben)

gen_data_syn['label'] = [1] * gen_data_syn.shape[0]
print(gen_data_syn)

print('Reset Indices ...')
pca_data_ben = pca_data_ben.reset_index(drop=True)
print(pca_data_ben)
pca_data_syn = pca_data_syn.reset_index(drop=True)
print(pca_data_ben)
gen_data_ben = gen_data_ben.reset_index(drop=True)
print(gen_data_ben)
gen_data_syn = gen_data_syn.reset_index(drop=True)
print(gen_data_syn)

print('Concat, Shuffle and Split Real Dataset')
real_data = pd.concat([pca_data_ben, pca_data_syn], axis=0).sample(frac=1, random_state=rstate)
real_X_train, real_y_train, real_X_test, real_y_test = split_data(
    real_data.iloc[:, :-1], real_data.iloc[:, -1], percent_test=25)

print('Concat, Shuffle and Split Gen Dataset')
gen_data_ben = pd.DataFrame(data=np.nan_to_num(gen_data_ben), columns=['pc_1', 'label'])
gen_data_syn = pd.DataFrame(data=np.nan_to_num(gen_data_syn), columns=['pc_1', 'label'])
gen_data = pd.concat([gen_data_ben, gen_data_syn], axis=0).sample(frac=1, random_state=rstate)
gen_X_train, gen_y_train, gen_X_test, gen_y_test = split_data(
    gen_data.iloc[:, :-1], gen_data.iloc[:, -1], percent_test=25
)

print('Train Model on Real Data and Test on Real Data ...')
rfc_real = RandomForestClassifier(random_state=rstate)
rfc_real.fit(real_X_train, real_y_train)

rfc_real_results = rfc_real.predict(real_X_test)
print('Test model (trained on real data and tested on real data) ...')
print('accuracy_score: ', accuracy_score(real_y_test, rfc_real_results))
print('precision_score: ', precision_score(real_y_test, rfc_real_results, average=None))
print('recall_score: ', recall_score(real_y_test, rfc_real_results, average=None))
print('f1_score: ', f1_score(real_y_test, rfc_real_results, average=None))
confusion_m = confusion_matrix(real_y_test, rfc_real_results)
print('confusion_matrix: ', confusion_m)
labels = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
categories = ['Benign', 'Syn DDoS']
make_confusion_matrix(confusion_m,
                      group_names=labels,
                      categories=categories,
                      title='Trained on Real Data',
                      sum_stats=False)

plt.savefig('{}/cf_matrix_synddos_real_pca_trained.svg'.format(output_path), dpi=300)

print('Train Model on Synthetic Data and Test on Real Data ...')
rfc_gen = RandomForestClassifier(random_state=rstate)
rfc_gen.fit(gen_X_train, gen_y_train)

rfc_gen_results = rfc_gen.predict(real_X_test)
print('Test model (trained on synthetic data and tested on real data) ...')
print('accuracy_score: ', accuracy_score(real_y_test, rfc_gen_results))
print('precision_score: ', precision_score(real_y_test, rfc_gen_results, average=None))
print('recall_score: ', recall_score(real_y_test, rfc_gen_results, average=None))
print('f1_score: ', f1_score(real_y_test, rfc_gen_results, average=None))
confusion_m = confusion_matrix(real_y_test, rfc_gen_results)
print('confusion_matrix: ', confusion_m)

make_confusion_matrix(confusion_m,
                      group_names=labels,
                      categories=categories,
                      title='Trained on Synthetic Data',
                      sum_stats=False)

plt.savefig('{}/cf_matrix_synddos_gen_pca_trained.svg'.format(output_path), dpi=300)

real_data_train = real_data.iloc[:50000, :]
real_data_test = real_data.iloc[50000:75000, :]

mixed_data = pd.concat([
    real_data_train,
    gen_data.sample(n=50000, random_state=rstate)
])
mixed_X_train, mixed_y_train, mixed_X_test, mixed_y_test = split_data(
    mixed_data.iloc[:, :-1], mixed_data.iloc[:, -1], percent_test=25
)

print('Train Model on Synthetic Data and Test on Real Data ...')
rfc_mixed = RandomForestClassifier(random_state=rstate)
rfc_mixed.fit(mixed_X_train, mixed_y_train)

rfc_mixed_results = rfc_mixed.predict(real_data_test.iloc[:, :-1])
print('Test model (trained on synthetic data and tested on real data) ...')
print('accuracy_score: ', accuracy_score(real_data_test.iloc[:, -1], rfc_mixed_results))
print('precision_score: ', precision_score(real_data_test.iloc[:, -1], rfc_mixed_results, average=None))
print('recall_score: ', recall_score(real_data_test.iloc[:, -1], rfc_mixed_results, average=None))
print('f1_score: ', f1_score(real_data_test.iloc[:, -1], rfc_mixed_results, average=None))
confusion_m = confusion_matrix(real_data_test.iloc[:, -1], rfc_mixed_results)
print('confusion_matrix: ', confusion_m)