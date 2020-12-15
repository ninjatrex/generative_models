from src.utils import *
import pandas as pd

datasets = {
    'ids2017_train': '../../lib/datasets/download/ids2017_train_transformed.csv',
    'ids2017_test': '../../lib/datasets/download/ids2017_test_transformed.csv',
}


output_path = '../../lib/datasets/filtered/cicids2017'
if not os.path.isdir(output_path):
    os.makedirs(output_path)

for dataset, path in datasets.items():
    cicids2017 = pd.read_csv(path)
    labels_with_nums = get_num_unique_labels(cicids2017)
    print(labels_with_nums)

    for label in labels_with_nums.keys():
        get_samples_with_label(cicids2017, label).to_csv('{}/{}_{}.csv'.format(output_path, dataset, label))
