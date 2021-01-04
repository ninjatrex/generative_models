import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 2000)
from src.features import selected_features

from sklearn.preprocessing import MaxAbsScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.naive_bayes import GaussianNB
from src.utils import split_data, pickle_data
from sklearn.svm import LinearSVC

rstate = 1337
num_samples = 20000

scaler = MaxAbsScaler()

model_path = 'models/cicddos2019'
if not os.path.isdir(model_path):
    os.makedirs(model_path)

print('Load cicddos syn data')
cicddos_syn = pd.read_csv(
    '../lib/datasets/filtered/cicddos2019/cicddos2019_train_syn.csv', usecols=selected_features).sample(
    n=400000, random_state=rstate)

cicddos_ben = pd.read_csv(
    '../lib/datasets/filtered/cicddos2019/cicddos2019_train_benign.csv', usecols=selected_features).sample(
    n=400000, random_state=rstate
)

print('Syn: {} Benign: {}'.format(cicddos_syn.shape[0], cicddos_ben.shape[0]))

cicddos = pd.concat([cicddos_ben, cicddos_syn], axis=0)

pca = PCA(0.99)
cicddos_pca = pd.DataFrame(data=pca.fit_transform(cicddos), columns=['pc_1'])
print(cicddos_pca)

cicddos_pca_ben = pd.DataFrame(data=cicddos_pca.iloc[:cicddos_ben.shape[0], :], columns=['pc_1'])
print(cicddos_pca_ben)

cicddos_pca_syn = pd.DataFrame(data=cicddos_pca.iloc[cicddos_syn.shape[0]:, :], columns=['pc_1'])
print(cicddos_pca_syn)

# for name, data in {'benign': cicddos_pca_ben, 'syn': cicddos_pca_syn}.items():
#     print('Employ exhaustive search for optimal number of components (range of [1, 200, 10])')
#     n_components = np.arange(1, 200, 10)
#     models = [GaussianMixture(n, covariance_type='full', random_state=rstate, reg_covar=1e-5)
#               for n in n_components]
#     aics = []
#     bics = []
#     for model in models:
#         print('Fit {} model with n_components = {}'.format(name, model.n_components))
#         model.fit(data)
#         aics.append(model.aic(data))
#         bics.append(model.bic(data))
#     components_ic = (aics, bics)
#     pickle_data(components_ic, '{}/{}_{}.pkl.gz'.format(model_path, name, 'information_criterion_gmm'))
#
#     plt.plot(n_components, bics, label='BIC')
#     plt.plot(n_components, aics, label='AIC')
#     plt.xlabel('n_components')
#     plt.legend()
#     plt.show()


print('Fit GMM for ben data with 50 components')
gmm_ben = GaussianMixture(n_components=50, covariance_type='full', random_state=rstate, reg_covar=1e-5)
gmm_ben.fit(cicddos_pca_ben)

print('Fit GMM for syn data with 110 components')
gmm_syn = GaussianMixture(n_components=110, covariance_type='full', random_state=rstate, reg_covar=1e-5)
gmm_syn.fit(cicddos_pca_syn)

models = {
    'gmm_ben': gmm_ben,
    'gmm_syn': gmm_syn,
    'pca': pca
}
pickle_data(models, '{}/{}.pkl.gz'.format(model_path, 'cicddos2019_gmm'))

cicddos_data = {
    'ben': cicddos_pca_ben,
    'syn': cicddos_pca_syn
}

pickle_data(cicddos_data, '{}/{}.pkl.gz'.format(model_path, 'cicddos2019_pca_data'))
