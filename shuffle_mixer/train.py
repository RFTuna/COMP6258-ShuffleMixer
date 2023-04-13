from tqdm import tqdm
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter


def preprocessing_transform(scale_factor):
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomCrop(64 * scale_factor),
        transforms.RandomApply(
            [transforms.RandomRotation((90, 90))],
            0.5
        ),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip()
    ])


def data_loader(path, batch_size, scale_factor):
    dataset = torchvision.datasets.ImageFolder(path, transform=preprocessing_transform(scale_factor))
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)


def loss_fn(upsamled, expected):
    weight = 0.1

    residuals = upsamled - expected
    loss_p = torch.sum(torch.abs(residuals))

    upsampled_freq = torch.fft.fft2(upsamled)
    expected_freq = torch.fft.fft2(expected)
    freq_residuals = upsampled_freq - expected_freq
    loss_f = torch.sum(torch.abs(freq_residuals))

    return loss_p + weight * loss_f


class Train:

    # 3450 pieces of data with a batch size of 64 means 6000 epochs needed to reach 300,000 iterations
    def __init__(self, model, device, train_path, valid_path, summary_path, checkpoint_path, train_batch=64,
                 valid_batch=16, epochs=6000, checkpoint_epochs=10, learning_rate=5e-4):
        self.model = model.to(device)

        self.device = device

        learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        self.train_batch = train_batch
        self.valid_batch = valid_batch

        scale_factor = 4 if model.four_times_scale else 2
        self.downsample = transforms.Resize(64, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)

        # self.epoch_size = 600
        self.epochs = epochs

        self.checkpoint_epochs = checkpoint_epochs
        self.checkpoint_path = checkpoint_path

        self.writer = SummaryWriter(summary_path)

        self.train = data_loader(train_path, self.train_batch, scale_factor)
        self.valid = data_loader(valid_path, self.valid_batch, scale_factor)

    def epoch(self):

        loss = 0

        np.random.seed()
        for data in tqdm(iter(self.train)):
            hr_truth, _ = data
            hr_truth = hr_truth.to(self.device)

            lr = self.downsample(hr_truth)

            self.optimizer.zero_grad()

            hr_estimate = self.model(lr)

            loss = loss_fn(hr_estimate, hr_truth) / self.train_batch
            loss.backward()

            self.optimizer.step()

        return loss

    def plot_image(self, step):

        np.random.seed()
        hr_truth, _ = next(iter(self.train))
        hr_truth = hr_truth.to(self.device)

        lr = self.downsample(hr_truth)

        hr_estimate = self.model(lr)

        self.writer.add_images("Example/HR Truth", hr_truth, step)
        self.writer.add_images("Example/LR", lr, step)
        self.writer.add_images("Example/HR Estimate", hr_estimate, step)

    def validate(self):

        np.random.seed()
        hr_truth, _ = next(iter(self.valid))
        hr_truth = hr_truth.to(self.device)

        lr = self.downsample(hr_truth)

        hr_estimate = self.model(lr)

        loss = loss_fn(hr_estimate, hr_truth) / self.valid_batch

        return loss

    def run(self):

        for epoch in range(self.epochs):

            print(f"EPOCH: {epoch}")

            self.model.train(True)

            train_loss = self.epoch()

            self.model.train(False)

            valid_loss = self.validate()

            print(f"LOSS train {train_loss} validate {valid_loss}")

            self.writer.add_scalars("Loss", {"train": train_loss, "validate": valid_loss}, epoch + 1)
            self.plot_image(epoch + 1)
            self.writer.flush()

            if (epoch + 1) % self.checkpoint_epochs == 0:
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_State_dict": self.optimizer.state_dict(),
                    "loss": train_loss
                }, f"{self.checkpoint_path}/{epoch}.ckpt")
