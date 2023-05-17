import argparse

from shuffle_mixer.model import create_shuffle_mixer
import torch

from shuffle_mixer.trainer import Trainer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(dat_folder, output_folder):
    model = create_shuffle_mixer(tiny=False, four_times_scale=True)

    trainer = Trainer(model, device, f"{dat_folder}/train", f"{dat_folder}/valid", f"{output_folder}/tensorboard",
                  f"{output_folder}/checkpoints", lr_size=64, batch_size=16, iterations=250000, loss_iterations=100,
                    val_iterations=200, plot_iterations=1000, checkpoint_epochs=1000, learning_rate=5e-4, workers=2)

    trainer.run()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-r', '--run', choices=['train'], required=True)
    parser.add_argument('-d', '--data')
    parser.add_argument('-o', '--output')

    args = parser.parse_args()

    print('train script called', flush=True)

    if args.run == 'train':

        print('running train func', flush=True)

        train(args.data, args.output)
