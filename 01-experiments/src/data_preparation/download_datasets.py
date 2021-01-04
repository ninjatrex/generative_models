import os
import sys
import requests
from tqdm import tqdm
from src.config import dataset_info


def download(url, path, chunk_size=1024):
    file_size = int(requests.head(url).headers["Content-Length"])
    with requests.get(url, stream=True) as r, open(path, "wb") as f, tqdm(
            unit="B",  # unit string to be displayed.
            unit_scale=True,  # let tqdm to determine the scale in kilo, mega..etc.
            unit_divisor=1024,  # is used when unit_scale is true
            total=file_size,  # the total iteration.
            file=sys.stdout,  # default goes to stderr, this is the display on console.
            desc=path  # prefix to be displayed on progress bar.
    ) as progress:
        for chunk in r.iter_content(chunk_size=chunk_size):
            # download the file chunk by chunk
            data_size = f.write(chunk)
            # on each chunk update the progress bar.
            progress.update(data_size)


download_dir = '../../lib/datasets/download'

# check for dataset dir
if not os.path.isdir(download_dir):
    os.makedirs(download_dir)

# download
for dataset, info in dataset_info.items():
    for type, url in info.items():
        file_path = '{}/{}_{}.csv'.format(download_dir, dataset, type)
        # check for already existing files
        if not os.path.isfile(file_path):
            print('Requesting {}_{}.csv for download...'.format(dataset, type))
            download(url, file_path)
