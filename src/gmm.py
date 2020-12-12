import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
pd.set_option('display.max_columns', None)
from src.features import all_features, drop

from sklearn.preprocessing import MaxAbsScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.naive_bayes import GaussianNB
from src.utils import split_data
from sdv.evaluation import evaluate

rstate = 1337
num_samples = 20000

scaler = MaxAbsScaler()

model_path = 'models/cicids2017'
if not os.path.isdir(model_path):
    os.makedirs(model_path)

selected_features = [i for i in all_features if i not in drop]
print(len(selected_features))

print('Load cicddos syn data')
cicddos_syn = pd.read_csv(
    '../lib/datasets/filtered/cicddos2019/cicddos2019_train_syn.csv', usecols=selected_features)

pca = PCA(0.99)
cicddos_syn_pca = pca.fit_transform(cicddos_syn)
print(cicddos_syn_pca)


print('Employ exhaustive search for optimal number of components (range of [10, 200, 20])')
n_components = np.arange(100, 200, 10)
models = [GaussianMixture(n, max_iter=1000, covariance_type='full', random_state=rstate, reg_covar=1e-5 )
          for n in n_components]
aics = [model.fit(cicddos_syn_pca).aic(cicddos_syn_pca) for model in models]
bics = [model.fit(cicddos_syn_pca).bic(cicddos_syn_pca) for model in models]

plt.plot(n_components, bics, label='BIC')
plt.plot(n_components, aics, label='AIC')
plt.xlabel('n_components')
plt.legend()
plt.show()


# def gmm_sampler(real_data: pd.DataFrame, num_samples: int):
#     pca_cicddos_benign = PCA(0.99, random_state=rstate)
#     cicddos_benign_pca = pca_cicddos_benign.fit_transform(real_data)
#     gmm = GaussianMixture(200, covariance_type='full', random_state=rstate)
#     gmm.fit(cicddos_benign_pca)
#
#     new_data = pd.DataFrame(
#         data=pca_cicddos_benign.inverse_transform(gmm.sample(num_samples)[0]),
#         columns=real_data.columns)
#
#     for column in new_data:
#         if selected_features[column] == int:
#             new_data[column] = round(new_data[column])
#     return new_data
#
#
# new_data = gmm_sampler(ids2017_train_benign, num_samples)
#
# print(evaluate(new_data, ids2017_train_benign, aggregate=False, metrics=['logistic_detection']))
#
# ids2017_train_benign['label'] = [0] * num_samples
# new_data['label'] = [1] * num_samples
#
# mixed = pd.concat([ids2017_train_benign, new_data])
#
# mixed = mixed.sample(frac=1, random_state=rstate)
#
# X_train, y_train, X_test, y_test = split_data(mixed.iloc[:, :-1], mixed.iloc[:, -1], percent_test=25)
#
# rf_model = GaussianNB()
# rf_model.fit(X_train, y_train)
#
# print('Test utility of data - the lower the accuracy the higher the utility...')
# rf_results = rf_model.predict(X_test)
# print('accuracy_score: ', accuracy_score(y_test, rf_results))
# print('precision_score: ', precision_score(y_test, rf_results, average=None))
# print('recall_score: ', recall_score(y_test, rf_results, average=None))
# print('f1_score: ', f1_score(y_test, rf_results, average=None))
# print('confusion_matrix: ', confusion_matrix(y_test, rf_results))
