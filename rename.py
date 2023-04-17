import argparse
import os

import PIL.Image
from PIL import Image

from torch.utils.data import DataLoader
from tqdm import tqdm

def downsample(dataset_path):

    path = f'{dataset_path}/original'

    name_changes = []

    count = 0
    for file in os.scandir(path):
        name_changes.append((file.path, f'{path}/{count}.png'))
        count += 1

    for old, new in tqdm(name_changes):
        os.rename(old, new)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--data', required=True)

    args = parser.parse_args()

    downsample(args.data)