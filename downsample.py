import argparse
import os

import PIL.Image
from PIL import Image

from torch.utils.data import DataLoader
from tqdm import tqdm


def downsample_file(file, target_path, scale):

    new_file = f"{target_path}/{file.name}"

    with Image.open(file.path) as original:
        width = original.width // scale
        height = original.height // scale
        downsampled = original.resize((width, height), Image.BICUBIC)
        downsampled.save(new_file)

def downsample(dataset_path):

    original_path = f'{dataset_path}/original'
    two_times = f'{dataset_path}/two'
    four_times = f'{dataset_path}/four'

    if not os.path.exists(two_times):
        os.makedirs(two_times)

    if not os.path.exists(four_times):
        os.makedirs(four_times)

    files = []

    for file in os.scandir(original_path):
        files.append(file)

    for file in tqdm(files):
        downsample_file(file, two_times, 2)
        downsample_file(file, four_times, 4)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--data', required=True)

    args = parser.parse_args()

    downsample(args.data)