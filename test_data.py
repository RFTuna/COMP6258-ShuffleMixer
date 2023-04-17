import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from shuffle_mixer.data import UpsampleDataset, data_loader

plt.rcParams["savefig.bbox"] = 'tight'


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = torchvision.transforms.functional.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

loader = data_loader('data/train', 4, 64, 64, 4)

for data, target in iter(loader):

    print(data.shape)
    print(target.shape)

    show(make_grid(data))
    show(make_grid(target))

plt.show()