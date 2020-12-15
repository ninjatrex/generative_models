from src.utils import *
import pandas as pd

datasets = {
    'cicddos2019_train': '../../lib/datasets/download/cicddos2019_train_transformed.csv',
    'cicddos2019_test': '../../lib/datasets/download/cicddos2019_test_transformed.csv',
}


output_path = '../../lib/datasets/filtered/cicddos2019'
if not os.path.isdir(output_path):
    os.makedirs(output_path)

for dataset, path in datasets.items():
    cicddos2019 = pd.read_csv(path)
    labels_with_nums = get_num_unique_labels(cicddos2019)
    print(labels_with_nums)

    for label in labels_with_nums.keys():
        get_samples_with_label(cicddos2019, label).to_csv('{}/{}_{}.csv'.format(output_path, dataset, label))