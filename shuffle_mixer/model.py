from torch import nn
import torch

padding = 'same'


# Taken from https://github.com/kuangliu/pytorch-cifar/blob/49b7aa97b0c12fe0d4054e670403a16b6b834ddd/models/shufflenet.py#L31
class ChannelShuffle(nn.Module):

    def __init__(self, groups):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        n, c, h, w = x.shape[-4], x.shape[-3], x.shape[-2], x.shape[-1]
        x = x.view(n, self.groups, c // self.groups, h, w)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape((n, c, h, w))
        return x


class PointWiseMLP(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.conv_input = nn.Conv2d(channels, 2 * channels, 1, padding=padding)
        self.conv_output = nn.Conv2d(2 * channels, channels, 1, padding=padding)

    def forward(self, x):
        x = self.conv_input(x)
        x = nn.functional.silu(x)
        x = self.conv_output(x)
        return x


class ChannelProjection(nn.Module):

    def __init__(self, conv_channels, channels):
        super().__init__()
        self.conv_channels = conv_channels
        self.point_wise_mlp = PointWiseMLP(conv_channels)
        self.shuffle = ChannelShuffle(channels)

    def forward(self, z_0):
        c, h, w = z_0.shape[-3], z_0.shape[-2], z_0.shape[-1]
        z = nn.functional.layer_norm(z_0, [c, h, w])
        z_1, z_2 = torch.split(z, [self.conv_channels, c - self.conv_channels], dim=-3)
        z_1 = self.point_wise_mlp(z_1)
        z = torch.concat((z_1, z_2), dim=-3)
        z = self.shuffle(z)
        return z + z_0


class ShuffleMixerLayer(nn.Module):

    def __init__(self, channels, depth_kernel_size):
        super().__init__()
        self.channel_proj_input = ChannelProjection(channels // 2, channels)
        self.depth_conv = nn.Conv2d(channels, channels, depth_kernel_size, padding=padding, groups=channels)
        self.channel_proj_output = ChannelProjection(channels // 2, channels)

    def forward(self, x):
        x = self.channel_proj_output(x)
        x = self.depth_conv(x)
        x = self.channel_proj_output(x)
        return x


class FusedMBConv(nn.Module):

    def __init__(self, channels, hidden_channels):
        super().__init__()
        self.conv_input = nn.Conv2d(channels, channels + hidden_channels, 3, padding=padding)
        self.conv_output = nn.Conv2d(channels + hidden_channels, channels, 1, padding=padding)

    def forward(self, x_0):
        x = self.conv_input(x_0)
        x = nn.functional.silu(x)
        x = self.conv_output(x)
        return x + x_0


class FeatureMixingBlock(nn.Module):

    def __init__(self, channels, depth_kernel_size):
        super().__init__()
        self.shuffle_a = ShuffleMixerLayer(channels, depth_kernel_size)
        self.shuffle_b = ShuffleMixerLayer(channels, depth_kernel_size)
        self.fmb_conv = FusedMBConv(channels, 16)

    def forward(self, x_0):
        x = self.shuffle_a(x_0)
        x = self.shuffle_b(x)
        x = x_0 + x
        x = self.fmb_conv(x)
        return x


class FeatureExtraction(nn.Module):

    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, 3, padding=padding)

    def forward(self, x):
        x = self.conv(x)
        return x


class Upsampler(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels * 2 * 2, 1, padding=padding)

    def forward(self, x):
        x = self.conv(x)
        x = nn.functional.pixel_shuffle(x, 2)
        x = nn.functional.silu(x)
        return x


class ShuffleMixer(nn.Module):

    def __init__(self, input_channels, features, depth_kernel_size, feature_mixing_block_count, four_times_scale=False):
        super().__init__()
        self.feature_extraction = FeatureExtraction(input_channels, features)
        self.feature_mixing_blocks = nn.ModuleList(
            [FeatureMixingBlock(features, depth_kernel_size) for _ in range(feature_mixing_block_count)]
        )
        self.upsampler = Upsampler(features)
        self.four_times_scale = four_times_scale
        if four_times_scale:
            self.upsampler2 = Upsampler(features)
        self.conv = nn.Conv2d(features, input_channels, 3, padding=padding)

    def forward(self, image):
        scale_factor = 4 if self.four_times_scale else 2
        upscaled = nn.functional.interpolate(image, scale_factor=scale_factor, mode='bilinear')
        residuals = self.feature_extraction(image)
        for feature_mixing_block in self.feature_mixing_blocks:
            residuals = feature_mixing_block(residuals)
        residuals = self.upsampler(residuals)
        if self.four_times_scale:
            residuals = self.upsampler2(residuals)
        residuals = self.conv(residuals)
        upscaled = upscaled + residuals
        return upscaled


def create_shuffle_mixer(tiny=False, four_times_scale=False):
    if tiny:
        return ShuffleMixer(3, 32, 3, 5, four_times_scale)
    else:
        return ShuffleMixer(3, 64, 7, 5, four_times_scale)
