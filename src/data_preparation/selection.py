from sdv.tabular import GaussianCopula
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from src.features import all_features, drop
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

from src.utils import split_data

rstate = 1337

selected_features = [i for i in all_features if i not in drop]
print(len(selected_features))

# cicddos2019_train_syn_selected = pd.read_csv(
#     '../../lib/datasets/filtered/cicddos2019/cicddos2019_train_syn.csv',
#     usecols=selected_features)
# # cicddos2019_train_syn_selected['label'] = [1] * cicddos2019_train_syn_selected.shape[0]
#
# cicddos2019_train_benign_selected = pd.read_csv(
#     '../../lib/datasets/filtered/cicddos2019/cicddos2019_train_benign.csv',
#     usecols=selected_features).sample(n=200000, random_state=rstate)
# cicddos2019_train_benign_selected['label'] = [0] * cicddos2019_train_benign_selected.shape[0]
#
# pca = PCA(0.99, random_state=rstate)
# syn_pca = pca.fit_transform(cicddos2019_train_syn_selected)
# syn_pca = pd.DataFrame(data=syn_pca, columns=['pc_1'])
# print(syn_pca)
#
#
# gc_syn = GaussianCopula(
# )
# gc_syn.fit(syn_pca)
#
# syn_gen_pca = gc_syn.sample(100000)
# syn_gen_pca = pd.DataFrame(data=syn_gen_pca, columns=['pc_1'])
#
# syn_gen = pd.DataFrame(data=pca.inverse_transform(syn_gen_pca), columns=selected_features)
#
# fig, axis = plt.subplots(7,4,figsize=(20, 10))
# syn_gen.hist(ax=axis, bins=100)
# plt.show()
#
# fig, axis = plt.subplots(7,4,figsize=(20, 10))
# cicddos2019_train_syn_selected.sample(n=100000, random_state=rstate).hist(ax=axis, bins=100)
# plt.show()

ids2017_benign = pd.read_csv(
    '../../lib/datasets/filtered/cicids2017/ids2017_train_benign.csv',
    usecols=selected_features).sample(
    n=100000, random_state=rstate)
ids2017_benign['label'] = [0] * ids2017_benign.shape[0]

ids2017_hulk = pd.concat([
    pd.read_csv(
    '../../lib/datasets/filtered/cicids2017/ids2017_train_dos_hulk.csv',
    usecols=selected_features),
    pd.read_csv(
        '../../lib/datasets/filtered/cicids2017/ids2017_test_dos_hulk.csv',
        usecols=selected_features)]).sample(
    n=100000, random_state=rstate)
ids2017_hulk['label'] = [1] * ids2017_hulk.shape[0]

ids2017 = pd.concat([ids2017_benign, ids2017_hulk], axis=0)
ids2017 = ids2017.sample(frac=1, random_state=rstate)

ids2017_X_train, ids2017_y_train, ids2017_X_test, ids2017_y_test = split_data(
    ids2017.iloc[:, :-1], ids2017.iloc[:, -1], percent_test=25)

rfc_ids2017_only = RandomForestClassifier(random_state=rstate)
rfc_ids2017_only.fit(ids2017_X_train, ids2017_y_train)


print('IDS 2017 Only')
rf_results = rfc_ids2017_only.predict(ids2017_X_test)
print('accuracy_score: ', accuracy_score(ids2017_y_test, rf_results))
print('precision_score: ', precision_score(ids2017_y_test, rf_results, average=None))
print('recall_score: ', recall_score(ids2017_y_test, rf_results, average=None))
print('f1_score: ', f1_score(ids2017_y_test, rf_results, average=None))
print('confusion_matrix: ', confusion_matrix(ids2017_y_test, rf_results))

print('Now change attack traffic to 50 pct. hulk and 50 pct. syn ...')
attack = pd.concat([
    ids2017_hulk.iloc[:, :-1].sample(n=12500, random_state=rstate),
    pd.read_csv('../../lib/datasets/filtered/cicddos2019/cicddos2019_train_syn.csv',
        usecols=selected_features).sample(n=12500, random_state=rstate)
])
attack['label'] = [1] * attack.shape[0]

mixed = pd.concat([ids2017_benign.sample(n=25000, random_state=rstate), attack], axis=0)
mixed = mixed.sample(frac=1, random_state=rstate)

mixed_X, mixed_y = mixed.iloc[:, :-1], mixed.iloc[:, -1]


print('Mixed Data test')
rf_results = rfc_ids2017_only.predict(mixed_X)
print('accuracy_score: ', accuracy_score(mixed_y, rf_results))
print('precision_score: ', precision_score(mixed_y, rf_results, average=None))
print('recall_score: ', recall_score(mixed_y, rf_results, average=None))
print('f1_score: ', f1_score(mixed_y, rf_results, average=None))
print('confusion_matrix: ', confusion_matrix(mixed_y, rf_results))

print('Now take synthetic syn data and take that into account when training ...')
syn_ddos = pd.read_csv('../../lib/datasets/filtered/cicddos2019/cicddos2019_train_syn.csv',
        usecols=selected_features)

gc = GaussianCopula()
gc.fit(syn_ddos)
syn_ddos_gen = gc.sample(50000)

new_attack = pd.concat([
    ids2017_hulk.iloc[:, :-1].sample(50000, random_state=rstate),
    syn_ddos_gen])
new_attack['label'] = [1] * new_attack.shape[0]

new_dataset = pd.concat([
    new_attack,
    ids2017_benign
])

new_X_train, new_y_train, new_X_test, new_y_test = split_data(
    new_dataset.iloc[:, :-1], new_dataset.iloc[:, -1], percent_test=25)

rfc_with_gen = RandomForestClassifier(random_state=rstate)
rfc_with_gen.fit(new_X_train, new_y_train)

print('Test rfc with gen data on real data')
rf_gen_results = rfc_with_gen.predict(mixed_X)
print('accuracy_score: ', accuracy_score(mixed_y, rf_gen_results))
print('precision_score: ', precision_score(mixed_y, rf_gen_results, average=None))
print('recall_score: ', recall_score(mixed_y, rf_gen_results, average=None))
print('f1_score: ', f1_score(ids2017_y_test, mixed_y, average=None))
print('confusion_matrix: ', confusion_matrix(mixed_y, rf_gen_results))


# cicddos2019_pca = pd.concat([X_pca, y], axis=1)
#
# print(cicddos2019_pca)
#
# cicddos2019_pca_benign = cicddos2019_pca[cicddos2019_pca.label == 0]
# cicddos2019_pca_syn = cicddos2019_pca[cicddos2019_pca.label == 1]
# print(cicddos2019_pca_benign)
# print(cicddos2019_pca_syn)
#
#
#
# gc_benign = GaussianCopula()
# gc_benign.fit(cicddos2019_pca_benign.iloc[:, :-1])
#
# gc_syn = GaussianCopula()
# gc_syn.fit(cicddos2019_pca_syn.iloc[:, :-1])
#
# cicddos2019_pca_benign_gen = gc_benign.sample(200000)
# cicddos2019_pca_benign_gen['label'] = [0] * 200000
# cicddos2019_pca_syn_gen = gc_syn.sample(200000)
# cicddos2019_pca_syn_gen['label'] = [1] * 200000
#
# cicddos2019_pca_gen = pd.concat([cicddos2019_pca_benign_gen, cicddos2019_pca_syn_gen])
# cicddos2019_pca_gen = cicddos2019_pca_gen.sample(frac=1, random_state=rstate)
#
# rfc = RandomForestClassifier()
# rfc.fit(cicddos2019_pca_gen.iloc[:, :-1], cicddos2019_pca_gen.iloc[:, -1])
#
#
# print('Test utility of data - the lower the accuracy the higher the utility...')
# rf_results = rfc.predict(cicddos2019_pca.iloc[:, :-1])
# print('accuracy_score: ', accuracy_score(cicddos2019_pca.iloc[:, -1], rf_results))
# print('precision_score: ', precision_score(cicddos2019_pca.iloc[:, -1], rf_results, average=None))
# print('recall_score: ', recall_score(cicddos2019_pca.iloc[:, -1], rf_results, average=None))
# print('f1_score: ', f1_score(cicddos2019_pca.iloc[:, -1], rf_results, average=None))
# print('confusion_matrix: ', confusion_matrix(cicddos2019_pca.iloc[:, -1], rf_results))


