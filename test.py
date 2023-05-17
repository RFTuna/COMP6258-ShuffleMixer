import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import os
from torchvision.utils import save_image

from shuffle_mixer.model import create_shuffle_mixer
import torch

import matplotlib.pyplot as plt


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = create_shuffle_mixer(tiny=False, four_times_scale=True)


checkpoint = torch.load("350000.ckpt")
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()

transform = transforms.ToTensor()

for image_file in os.listdir("test_data"):
    image_path = os.path.join("test_data", image_file)
    downsampled_path = os.path.join("test_output", f"{image_file}")
    downsampled_nearest_path = os.path.join("test_output", f"nearest {image_file}")
    downsampled_bicubic_path = os.path.join("test_output", f"bicubic {image_file}")
    downsampled_model_path = os.path.join("test_output", f"model {image_file}")

    # Open and preprocess the image
    image = Image.open(image_path).convert('RGB')
    width = image.width // 4
    height = image.height // 4
    image = image.resize((width, height), Image.BICUBIC)
    image = transform(image)

    save_image(image, downsampled_path)

    nearest = torchvision.transforms.functional.resize(image, (height * 4, width * 4), torchvision.transforms.InterpolationMode.NEAREST)
    bicubic = torchvision.transforms.functional.resize(image, (height * 4, width * 4), torchvision.transforms.InterpolationMode.BICUBIC)
    model_out = model(image.unsqueeze(0))

    save_image(nearest, downsampled_nearest_path)
    save_image(bicubic, downsampled_bicubic_path)
    save_image(model_out, downsampled_model_path)

plt.show()