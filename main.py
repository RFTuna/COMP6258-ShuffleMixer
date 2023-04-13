import argparse

from shuffle_mixer.model import create_shuffle_mixer
import torch

from shuffle_mixer.train import Train

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(dat_folder, output_folder):

    # model = create_shuffle_mixer(four_times_scale=True)
    model = create_shuffle_mixer(tiny=True, four_times_scale=False)

    train = Train(model, device, f"{dat_folder}/train", f"{dat_folder}/valid", f"{output_folder}/tensorboard", f"{output_folder}/checkpoints", epochs=3, train_batch=64)

    train.run()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-r', '--run', choices=['train'], required=True)
    parser.add_argument('-d', '--data')
    parser.add_argument('-o', '--output')

    args = parser.parse_args()

    if args.run == 'train':

        train(args.data, args.output)





    # main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
