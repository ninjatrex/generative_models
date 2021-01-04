import pandas as pd
from sdv.tabular import GaussianCopula
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from src.features import selected_features
from sklearn.naive_bayes import GaussianNB
from src.utils import split_data

rstate = 1337

datasets = {
    'benign_train': 'datasets/filtered/cicids2017/ids2017_train_benign.csv',
    'dos_golden_eye_train': 'datasets/filtered/cicids2017/ids2017_train_dos_golden_eye.csv',
    'dos_hulk_train': 'datasets/filtered/cicids2017/ids2017_train_dos_hulk.csv',
    'dos_slowhttptest': 'datasets/filtered/cicids2017/ids2017_train_dos_slowhttptest.csv'
}


benign_train = pd.read_csv(datasets['benign_train'], usecols=selected_features.keys())
print('num_samples benign_train: {}'.format(benign_train.shape[0]))
benign_train = benign_train.sample(n=60000, random_state=rstate)
benign_train['label'] = [0] * benign_train.shape[0]
print('num_samples benign_train: {}'.format(benign_train.shape[0]))

dos_golden_eye_train = pd.read_csv(datasets['dos_golden_eye_train'], usecols=selected_features.keys())
print('num_samples dos_golden_eye_train: {}'.format(dos_golden_eye_train.shape[0]))
dos_golden_eye_train = dos_golden_eye_train.sample(n=10000, random_state=rstate)
dos_golden_eye_train['label'] = [1] * dos_golden_eye_train.shape[0]
print('num_samples dos_golden_eye_train: {}'.format(dos_golden_eye_train.shape[0]))

dos_hulk_train = pd.read_csv(datasets['dos_hulk_train'], usecols=selected_features.keys())
print('num_samples dos_hulk_train: {}'.format(dos_golden_eye_train.shape[0]))
dos_hulk_train = dos_hulk_train.sample(n=10000, random_state=rstate)
dos_hulk_train['label'] = [1] * dos_golden_eye_train.shape[0]
print('num_samples dos_hulk_train: {}'.format(dos_golden_eye_train.shape[0]))

dos_slowhttptest = pd.read_csv(datasets['dos_slowhttptest'], usecols=selected_features.keys())
print('num samples dos_slowhttptest: {}'.format(dos_slowhttptest.shape[0]))
dos_slowhttptest = dos_slowhttptest.sample(n=1000, random_state=rstate)
dos_slowhttptest['label'] = [1] * dos_slowhttptest.shape[0]
print('num samples dos_slowhttptest: {}'.format(dos_slowhttptest.shape[0]))

print('Concat samples into biased dataset...')
biased_train = pd.concat([benign_train, dos_golden_eye_train, dos_hulk_train, dos_slowhttptest], axis=0)
biased_train = biased_train.sample(frac=1, random_state=rstate)
X_train_biased, y_train_biased, X_val_biased, y_val_biased = split_data(
    biased_train.iloc[:, :-1], biased_train.iloc[:, -1], percent_test=25)
print('num benign biased dataset: {}; num attack biased dataset: {}'.format(
        biased_train[biased_train['label'] == 0].shape[0], biased_train[biased_train['label'] == 1].shape[0]
))

print('Train SVC on biased dataset...')
biased_rf = GaussianNB()
biased_rf.fit(X_train_biased, y_train_biased)

print('Test biased RandomForestClassifier...')
rf_results_biased = biased_rf.predict(X_val_biased)
print('accuracy_score: ', accuracy_score(y_val_biased, rf_results_biased))
print('precision_score: ', precision_score(y_val_biased, rf_results_biased, average=None))
print('recall_score: ', recall_score(y_val_biased, rf_results_biased, average=None))
print('f1_score: ', f1_score(y_val_biased, rf_results_biased, average=None))
print('confusion_matrix: ', confusion_matrix(y_val_biased, rf_results_biased))

print('Upsample biased dataset ...')
gen_model = GaussianCopula()
gen_model.fit(dos_slowhttptest)
dos_slowhttptest = pd.concat([gen_model.sample(39000), dos_slowhttptest])
print('num samples dos_slowhttptest: {}'.format(dos_slowhttptest.shape[0]))

print('Concat samples into balanced dataset...')
balanced_train = pd.concat([benign_train, dos_golden_eye_train, dos_hulk_train, dos_slowhttptest], axis=0)
balanced_train = balanced_train.sample(frac=1, random_state=rstate)
X_train_balanced, y_train_balanced, X_val_balanced, y_val_balanced = split_data(
    balanced_train.iloc[:, :-1], balanced_train.iloc[:, -1], percent_test=25)
print('num benign balanced dataset: {}; num attack balanced dataset: {}'.format(
        balanced_train[balanced_train['label'] == 0].shape[0], balanced_train[balanced_train['label'] == 1].shape[0]
))

print('Train SVC on biased dataset...')
balanced_rf = GaussianNB()
balanced_rf.fit(X_train_balanced, y_train_balanced)

print('Test balanced RandomForestClassifier...')
rf_results_balanced = biased_rf.predict(X_val_biased)
print('accuracy_score: ', accuracy_score(y_val_biased, rf_results_balanced))
print('precision_score: ', precision_score(y_val_biased, rf_results_balanced, average=None))
print('recall_score: ', recall_score(y_val_biased, rf_results_balanced, average=None))
print('f1_score: ', f1_score(y_val_biased, rf_results_balanced, average=None))
print('confusion_matrix: ', confusion_matrix(y_val_biased, rf_results_balanced))


