import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.mixture import GaussianMixture
from sklearn.svm import LinearSVC
from src.features import selected_features

from src.experiments.cf_matrix import make_confusion_matrix
from src.utils import unpickle_data, split_data

rstate = 1337

model_path = '../../src/models/cicddos2019'
if not os.path.isdir(model_path):
    os.makedirs(model_path)

output_path = '../../lib/images/plots'
if not os.path.isdir(output_path):
    os.makedirs(output_path)

models = unpickle_data('{}/{}.pkl.gz'.format(model_path, 'cicddos2019_gmm'))
print(models)
real_data = unpickle_data('{}/{}.pkl.gz'.format(model_path, 'cicddos2019_pca_data'))
print(real_data)

pca: PCA = models['pca']
gmm_ben: GaussianMixture = models['gmm_ben']
gmm_syn: GaussianMixture = models['gmm_syn']

pca_data_ben: pd.DataFrame = real_data['ben'].sample(n=100000, random_state=rstate)
pca_data_syn: pd.DataFrame = real_data['syn'].sample(n=100000, random_state=rstate)

print('Real Syn: {} Real Benign: {}'.format(pca_data_syn.shape, pca_data_ben.shape))

print('Sample from generators ...')
gen_data_ben = np.nan_to_num(gmm_ben.sample(pca_data_ben.shape[0])[0])
gen_data_syn = np.nan_to_num(gmm_syn.sample(pca_data_syn.shape[0])[0])

print('Inverse PCA Transformation ...')
gen_data_ben = pca.inverse_transform(gen_data_ben)
gen_data_ben = pd.DataFrame(data=gen_data_ben, columns=selected_features)
print(gen_data_ben)

gen_data_syn = pca.inverse_transform(gen_data_syn)
gen_data_syn = pd.DataFrame(data=gen_data_syn, columns=selected_features)
print(gen_data_syn)

real_data_ben = pca.inverse_transform(pca_data_ben)
real_data_ben = pd.DataFrame(data=real_data_ben, columns=selected_features)
print(real_data_ben)

real_data_syn = pca.inverse_transform(pca_data_syn)
real_data_syn = pd.DataFrame(data=real_data_syn, columns=selected_features)
print(real_data_syn)

print('Reset Indices ...')
real_data_ben = real_data_ben.reset_index(drop=True)
print(real_data_ben)
real_data_syn = real_data_syn.reset_index(drop=True)
print(real_data_syn)
gen_data_ben = gen_data_ben.reset_index(drop=True)
print(gen_data_ben)
gen_data_syn = gen_data_syn.reset_index(drop=True)
print(gen_data_syn)

print('Concat Data ...')
real_data = pd.concat([real_data_ben, real_data_syn], axis=0)
print(real_data)
gen_data = pd.concat([gen_data_ben, gen_data_syn], axis=0)
print(gen_data)

print('Label Data ...')
real_data['label'] = [0] * real_data.shape[0]
gen_data['label'] = [1] * gen_data.shape[0]

# pca_data = pd.DataFrame(data=np.nan_to_num(pca_data), columns=['pc_1', 'label'])
# gen_data = pd.DataFrame(data=np.nan_to_num(gen_data), columns=['pc_1', 'label'])

mixed_data = pd.concat([
    real_data, gen_data
], axis=0).sample(frac=1, random_state=rstate)

print('split data ...')
mixed_X_train, mixed_y_train, mixed_X_test, mixed_y_test = split_data(
    mixed_data.iloc[:, :-1], mixed_data.iloc[:, -1], percent_test=25)

print('mixed_X_train: {}, mixed_y_train: {}, mixed_X_test: {}, mixed_y_test: {}'.format(
    mixed_X_train.shape, mixed_y_train.shape, mixed_X_test.shape, mixed_y_test.shape))

print(mixed_X_train)
print(mixed_y_train)
print(mixed_X_test)
print(mixed_y_test)

print('Train LSVC Model ...')
lsvc = LinearSVC(random_state=rstate)

lsvc.fit(mixed_X_train, mixed_y_train)

lsvc_results = lsvc.predict(mixed_X_test)
print(lsvc_results)
print('Test model on capability of seperating real and synthetic data ...')
print('accuracy_score: ', accuracy_score(mixed_y_test, lsvc_results))
print('precision_score: ', precision_score(mixed_y_test, lsvc_results, average=None))
print('recall_score: ', recall_score(mixed_y_test, lsvc_results, average=None))
print('f1_score: ', f1_score(mixed_y_test, lsvc_results, average=None))
confusion_m = confusion_matrix(mixed_y_test, lsvc_results)
print('confusion_matrix: ', confusion_m)

labels = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
categories = ['Real', 'Synthetic']
make_confusion_matrix(confusion_m,
                      group_names=labels,
                      categories=categories,
                      title='Linear Support Vector Machine: \n Discriminate Real and Synthetic Data',
                      sum_stats=False)

plt.savefig('{}/cf_sep_real_syn_data_lsvc.svg'.format(output_path), dpi=300)
plt.show()

print('Train RFC Model ...')
rfc = RandomForestClassifier(random_state=rstate)

rfc.fit(mixed_X_train, mixed_y_train)

rfc_results = rfc.predict(mixed_X_test)
print(rfc_results)
print('Test model on capability of seperating real and synthetic data ...')
print('accuracy_score: ', accuracy_score(mixed_y_test, rfc_results))
print('precision_score: ', precision_score(mixed_y_test, rfc_results, average=None))
print('recall_score: ', recall_score(mixed_y_test, rfc_results, average=None))
print('f1_score: ', f1_score(mixed_y_test, rfc_results, average=None))
confusion_m = confusion_matrix(mixed_y_test, rfc_results)
print('confusion_matrix: ', confusion_m)

labels = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
categories = ['Real', 'Synthetic']
make_confusion_matrix(confusion_m,
                      group_names=labels,
                      categories=categories,
                      title='Random Forest Classifier: \n Discriminate Real and Synthetic Data',
                      sum_stats=False)

plt.savefig('{}/cf_sep_real_syn_data_rfc.svg'.format(output_path), dpi=300)
plt.show()