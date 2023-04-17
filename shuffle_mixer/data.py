import os

import numpy as np
import torch
import torchvision
from PIL import Image
from PIL.Image import Transpose
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms.functional import convert_image_dtype


def data_loader(folder, scale, batch_size, lr_size, workers, steps):
    dataset = UpsampleDataset(folder, scale, batch_size, lr_size, steps)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=workers)

    return loader


class UpsampleDataset(Dataset):

    def __init__(self, folder, scale, batch_size, lr_size, steps):

        if scale == 2:
            self.data_folder = f"{folder}/two"
        elif scale == 4:
            self.data_folder = f"{folder}/four"

        self.target_folder = f"{folder}/original"

        self.scale = scale

        count = 0
        for _ in os.scandir(self.data_folder):
            count += 1

        self.file_count = count

        self.batch_size = batch_size

        self.lr_size = lr_size
        self.hr_size = scale * lr_size

        self.steps = steps

    def __len__(self):
        return self.batch_size * self.steps

    def __getitem__(self, _):

        file = np.random.randint(0, self.file_count)

        data_path = f"{self.data_folder}/{file}.png"
        target_path = f"{self.target_folder}/{file}.png"

        # 0 and 1 to map to 0 to width - size
        horizontal_start = np.random.rand()
        vertical_start = np.random.rand()

        flip_horizontal = np.random.randint(0, 2) == 1
        flip_vertical = np.random.randint(0, 2) == 1

        rotate = np.random.randint(0, 2) == 1

        with Image.open(data_path) as data_image, Image.open(target_path) as target_image:
            data_left = horizontal_start * (data_image.width - self.lr_size)
            data_top = vertical_start * (data_image.height - self.lr_size)
            data = data_image.crop((
                data_left,
                data_top,
                data_left + self.lr_size,
                data_top + self.lr_size,
            ))
            target_left = horizontal_start * (target_image.width - self.hr_size)
            target_right = vertical_start * (target_image.height - self.hr_size)
            target = target_image.crop((
                target_left,
                target_right,
                target_left + self.hr_size,
                target_right + self.hr_size,
            ))

            if flip_horizontal:
                data = data.transpose(Transpose.FLIP_LEFT_RIGHT)
                target = target.transpose(Transpose.FLIP_LEFT_RIGHT)

            if flip_vertical:
                data = data.transpose(Transpose.FLIP_TOP_BOTTOM)
                target = target.transpose(Transpose.FLIP_TOP_BOTTOM)

            if rotate:
                data = data.transpose(Transpose.ROTATE_90)
                target = target.transpose(Transpose.ROTATE_90)

            data = torchvision.transforms.functional.pil_to_tensor(data)
            target = torchvision.transforms.functional.pil_to_tensor(target)

            data = convert_image_dtype(data)
            target = convert_image_dtype(target)

        return data, target
