from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from skmultiflow.data import DataStream
from skmultiflow.trees import HoeffdingTreeClassifier
from skmultiflow.evaluation import EvaluatePrequential
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

datasets = {
    'app_ddos_train': 'datasets/experiments/stream_exp_prep/app_ddos_train_dataset.csv',
    'app_ddos_test': 'datasets/experiments/stream_exp_prep/app_ddos_test_dataset.csv',
    'udp_ddos_train': 'datasets/experiments/stream_exp_prep/udp_ddos_train_dataset.csv',
    'udp_ddos_test': 'datasets/experiments/stream_exp_prep/udp_ddos_test_dataset.csv'
}

app_ddos_train = pd.read_csv(datasets['app_ddos_train'])
app_ddos_test = pd.read_csv(datasets['app_ddos_test'])
udp_ddos_train = pd.read_csv(datasets['udp_ddos_train'])
udp_ddos_test = pd.read_csv(datasets['udp_ddos_test'])

rf_app_ddos = RandomForestClassifier()
rf_app_ddos.fit(X=app_ddos_train.iloc[:, :-1], y=app_ddos_train.iloc[:, -1])

print('Test app ddos rf on app ddos test set...')
rf_app_on_app_ddos_results = rf_app_ddos.predict(app_ddos_test.iloc[:, :-1])
print('accuracy_score: ', accuracy_score(app_ddos_test.iloc[:, -1], rf_app_on_app_ddos_results))
print('precision_score: ', precision_score(app_ddos_test.iloc[:, -1], rf_app_on_app_ddos_results, average=None))
print('recall_score: ', recall_score(app_ddos_test.iloc[:, -1], rf_app_on_app_ddos_results, average=None))
print('f1_score: ', f1_score(app_ddos_test.iloc[:, -1], rf_app_on_app_ddos_results, average=None))
print('confusion_matrix: ', confusion_matrix(app_ddos_test.iloc[:, -1], rf_app_on_app_ddos_results))

print('Test app ddos rf on udp ddos test set...')
rf_udp_on_udp_ddos_results = rf_app_ddos.predict(udp_ddos_test.iloc[:, :-1])
print('accuracy_score: ', accuracy_score(udp_ddos_test.iloc[:, -1], rf_udp_on_udp_ddos_results))
print('precision_score: ', precision_score(udp_ddos_test.iloc[:, -1], rf_udp_on_udp_ddos_results, average=None))
print('recall_score: ', recall_score(udp_ddos_test.iloc[:, -1], rf_udp_on_udp_ddos_results, average=None))
print('f1_score: ', f1_score(udp_ddos_test.iloc[:, -1], rf_udp_on_udp_ddos_results, average=None))
print('confusion_matrix: ', confusion_matrix(udp_ddos_test.iloc[:, -1], rf_udp_on_udp_ddos_results))

rf_udp_ddos = RandomForestClassifier()
rf_udp_ddos.fit(X=udp_ddos_train.iloc[:, :-1], y=udp_ddos_train.iloc[:, -1])

print('Test udp ddos rf on udp ddos test set...')
rf_udp_on_udp_ddos_results = rf_udp_ddos.predict(udp_ddos_test.iloc[:, :-1])
print('accuracy_score: ', accuracy_score(udp_ddos_test.iloc[:, -1], rf_udp_on_udp_ddos_results))
print('precision_score: ', precision_score(udp_ddos_test.iloc[:, -1], rf_udp_on_udp_ddos_results, average=None))
print('recall_score: ', recall_score(udp_ddos_test.iloc[:, -1], rf_udp_on_udp_ddos_results, average=None))
print('f1_score: ', f1_score(udp_ddos_test.iloc[:, -1], rf_udp_on_udp_ddos_results, average=None))
print('confusion_matrix: ', confusion_matrix(udp_ddos_test.iloc[:, -1], rf_udp_on_udp_ddos_results))

