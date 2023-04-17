from datetime import datetime

from tqdm import tqdm
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from shuffle_mixer.data import data_loader


def loss_fn(upsamled, expected):
    weight = 0.1

    residuals = upsamled - expected
    loss_p = torch.sum(torch.abs(residuals))

    upsampled_freq = torch.fft.rfft2(upsamled)
    expected_freq = torch.fft.rfft2(expected)
    upsampled_freq = torch.stack([upsampled_freq.real, upsampled_freq.imag], dim=-1)
    expected_freq = torch.stack([expected_freq.real, expected_freq.imag], dim=-1)
    loss_f = torch.sum(torch.abs(upsampled_freq - expected_freq))

    return loss_p + weight * loss_f


class Trainer:

    def __init__(self, model, device, train_path, valid_path, summary_path, checkpoint_path, lr_size=64, batch_size=64,
                 iterations=300000, loss_iterations=10, val_iterations=100, plot_iterations=1000, checkpoint_epochs=10,
                 learning_rate=5e-4, workers=4):
        print(f'Training on device {device}')

        self.model = model.to(device)

        self.device = device

        learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        scale_factor = 4 if model.four_times_scale else 2

        self.batch_size = batch_size

        self.iterations = iterations

        self.checkpoint_epochs = checkpoint_epochs
        self.checkpoint_path = checkpoint_path

        self.writer = SummaryWriter(summary_path)

        self.train = data_loader(train_path, scale_factor, batch_size, lr_size, workers, iterations)
        self.valid = data_loader(valid_path, scale_factor, batch_size, lr_size, workers, 1)

        self.loss_iterations = loss_iterations
        self.val_iterations = val_iterations
        self.plot_iterations = plot_iterations

        self.lr_size = lr_size
        self.hr_size = scale_factor * lr_size

        self.iterator = None

    def iteration(self):

        data, target = next(self.iterator)
        hr_truth = target.to(self.device)
        lr = data.to(self.device)

        self.optimizer.zero_grad()

        hr_estimate = self.model(lr)

        loss = loss_fn(hr_estimate, hr_truth) / self.batch_size
        loss.backward()

        self.optimizer.step()

        return loss

    def plot_image(self, step):

        lr, hr_truth = next(iter(self.train))
        lr = lr.to(self.device)
        hr_truth = hr_truth.to(self.device)
        hr_estimate = self.model(lr)

        lr_nearest = torchvision.transforms.functional.resize(lr, self.hr_size, torchvision.transforms.InterpolationMode.NEAREST)
        lr_bicubic = torchvision.transforms.functional.resize(lr, self.hr_size, torchvision.transforms.InterpolationMode.BICUBIC)

        self.writer.add_images("Example/HR Truth", hr_truth, step)
        self.writer.add_images("Example/LR Nearest", lr_nearest, step)
        self.writer.add_images("Example/LR Bicubic", lr_bicubic, step)
        self.writer.add_images("Example/HR Estimate", hr_estimate, step)

    def validate(self):

        loss = 0

        np.random.seed()
        for data, target in iter(self.valid):
            hr_truth = target.to(self.device)
            lr = data.to(self.device)

            hr_estimate = self.model(lr)

            loss = loss_fn(hr_estimate, hr_truth) / self.batch_size

        return loss

    def run(self):

        self.model.train(True)

        start = datetime.now()

        print(f'Training starting at {start}')

        self.iterator = iter(self.train)

        for iteration in range(self.iterations):

            loss = self.iteration()

            step = iteration + 1

            if step % self.plot_iterations == 0:
                self.model.train(False)
                self.plot_image(step)
                self.model.train(True)

            if step % self.val_iterations == 0:
                self.model.train(False)
                val_loss = self.validate()
                self.writer.add_scalar("Loss/Val", val_loss, step)
                self.model.train(True)

                print(
                    f'Reached step {step} in {datetime.now() - start}, with train loss {loss} and val loss {val_loss}')

            if step % self.loss_iterations == 0:
                self.writer.add_scalar("Loss/Train", loss, step)
                self.writer.flush()

            if iteration % self.checkpoint_epochs == 0:
                torch.save({
                    "iteration": iteration,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_State_dict": self.optimizer.state_dict(),
                    "loss": loss
                }, f"{self.checkpoint_path}/{iteration}.ckpt")

        print(f'Training finished in {datetime.now() - start}')
