import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import numpy as np
import pandas as pd
from src.utils import unpickle_data
from src.features import selected_features

rstate = 1337

model_path = 'models/cicddos2019'
if not os.path.isdir(model_path):
    os.makedirs(model_path)


models = unpickle_data('{}/{}.pkl.gz'.format(model_path, 'cicddos2019_gmm'))
print(models)
pca_data = unpickle_data('{}/{}.pkl.gz'.format(model_path, 'cicddos2019_pca_data'))
print(pca_data)

pca: PCA = models['pca']
gmm_ben: GaussianMixture = models['gmm_ben']
gmm_syn: GaussianMixture = models['gmm_syn']

pca_data_ben: pd.DataFrame = pca_data['ben'].sample(n=10000, random_state=rstate)
pca_data_syn: pd.DataFrame = pca_data['syn'].sample(n=10000, random_state=rstate)


print('Sample from syn model')
syn_gen_pca = np.nan_to_num(np.array(gmm_syn.sample(10000)[0]))
syn_gen_pca = pd.DataFrame(data=syn_gen_pca, columns=['pc_1'])

print('Compare real and synthetic syn data ...')
print(syn_gen_pca.describe())
print(pca_data_syn.describe())

print('Sample from ben model')
ben_gen_pca = np.nan_to_num(np.array(gmm_ben.sample(10000)[0]))
ben_gen_pca = pd.DataFrame(data=ben_gen_pca, columns=['pc_1'])

print('Compare real and synthetic ben data ...')
print(ben_gen_pca.describe())
print(pca_data_ben.describe())


ben_data = [ben_gen_pca['pc_1'], pca_data_ben['pc_1']] # benign plot
syn_data = [syn_gen_pca['pc_1'], pca_data_syn['pc_1']] # syn plot

print('Plot data')
output_path = '../lib/images/plots'
if not os.path.isdir(output_path):
    os.makedirs(output_path)

fig1, ax_syn = plt.subplots(nrows=2, ncols=1, figsize=(16, 9))
my_x_ticks = np.arange(-260000, -225000, 10000)
# Remove top and right border
ax_syn[1].spines['top'].set_visible(False)
ax_syn[1].spines['right'].set_visible(False)
ax_syn[1].spines['left'].set_visible(False)# Remove y-axis tick marks
ax_syn[1].yaxis.set_ticks_position('none')# Add major gridlines in the y-axis
ax_syn[1].grid(color='grey', axis='y', linestyle='-', linewidth=0.25, alpha=0.5)
ax_syn[1].boxplot(syn_data, labels=['synthetic', 'real'], vert=False)
ax_syn[1].set_xlim([-260000, -225000])
ax_syn[1].xticks = my_x_ticks


ax_syn[0].spines['top'].set_visible(False)
ax_syn[0].spines['right'].set_visible(False)
ax_syn[0].spines['left'].set_visible(False)# Remove y-axis tick marks
ax_syn[0].yaxis.set_ticks_position('none')# Add major gridlines in the y-axis
ax_syn[0].grid(color='grey', axis='y', linestyle='-', linewidth=0.25, alpha=0.5)
ax_syn[0].hist(syn_data, range=[-260000, -225000], label=['synthetic', 'real'])
ax_syn[0].set_xlim([-260000, -225000])
ax_syn[0].legend()
ax_syn[0].xticks = my_x_ticks

fig1.suptitle('Syn DDoS Data: n=10.000')
fig1.text(0.5, 0.04, 'principal component value', ha='center', va='center')

plt.savefig('{}/syn_ddos_data_comparison.svg'.format(output_path), dpi=300)


fig2, ax_ben = plt.subplots(nrows=2, ncols=1, figsize=(16, 9))
my_x_ticks = np.arange(-255000, -245000, 1000)
# Remove top and right border
ax_ben[1].spines['top'].set_visible(False)
ax_ben[1].spines['right'].set_visible(False)
ax_ben[1].spines['left'].set_visible(False)# Remove y-axis tick marks
ax_ben[1].yaxis.set_ticks_position('none')# Add major gridlines in the y-axis
ax_ben[1].grid(color='grey', axis='y', linestyle='-', linewidth=0.25, alpha=0.5)
ax_ben[1].boxplot(ben_data, labels=['synthetic', 'real'], vert=False, color='Blues')
ax_ben[1].set_xlim([-255000, -245000])
ax_ben[1].xticks = my_x_ticks


ax_ben[0].spines['top'].set_visible(False)
ax_ben[0].spines['right'].set_visible(False)
ax_ben[0].spines['left'].set_visible(False)# Remove y-axis tick marks
ax_ben[0].yaxis.set_ticks_position('none')# Add major gridlines in the y-axis
ax_ben[0].grid(color='grey', axis='y', linestyle='-', linewidth=0.25, alpha=0.5)
ax_ben[0].hist(ben_data, range=[-255000, -245000], label=['synthetic', 'real'], color='Blues')
ax_ben[0].set_xlim([-255000, -245000])
ax_ben[0].legend()
ax_ben[0].xticks = my_x_ticks

fig2.suptitle('Benign Data: n=10.000')
fig2.text(0.5, 0.04, 'principal component value', ha='center', va='center')

plt.savefig('{}/ben_ddos_data_comparison.svg'.format(output_path), dpi=300)

plt.show()
