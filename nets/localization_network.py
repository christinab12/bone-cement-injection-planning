import torch
import torch.nn as nn
from . import custom_layers as layers


class ConvDilBlock(nn.Module):
    """3D dilated convolution with default parameters
    kernel=3, stride=1, padding=2, dilation=2, Xavier's init
    followed by Instance Normalization + Leaky ReLU Activation
    """
    def __init__(self, in_chs, out_chs):
        super(ConvDilBlock, self).__init__()
        self.conv = nn.Sequential(
            layers.conv3d_dil_layer(in_chs, out_chs),
            nn.InstanceNorm3d(out_chs, eps=1e-6),
            nn.LeakyReLU(negative_slope=0.2)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class ConvBlock(nn.Module):
    """3D convolution with default parameters
    kernel=1, stride=1, padding=0, Xavier's init
    followed by Instance Normalization + Leaky ReLU Activation
    """
    def __init__(self, in_chs, out_chs):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            layers.conv3d_layer(in_chs, out_chs),
            nn.InstanceNorm3d(out_chs, eps=1e-6),
            nn.LeakyReLU(negative_slope=0.2)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class UpConvBlock(nn.Module):
    """3D transposed convolution with default parameters
    kernel=2, stride=2, Xavier's init
    followed by Leaky ReLU Activation
    """
    def __init__(self, in_chs, out_chs):
        super(UpConvBlock, self).__init__()
        self.upconv = nn.Sequential(
            layers.upconv3d_layer(in_chs, out_chs),
            nn.LeakyReLU(negative_slope=0.2)
        )

    def forward(self, x):
        x = self.upconv(x)
        return x


class SqExBlock(nn.Module):
    """Squeeze and Excitation Block
    Convolutions are 1x1 padding 0
    Refer https://arxiv.org/pdf/1808.08127.pdf
    """
    def __init__(self, in_chs):
        super(SqExBlock, self).__init__()
        self.identity = nn.Identity()
        # Channel squeeze
        self.dense_conv1 = nn.Sequential(
            layers.conv3d_layer(in_chs, int(in_chs // 2)),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.dense_conv2 = nn.Sequential(
            layers.conv3d_layer(int(in_chs // 2), in_chs),
            nn.Sigmoid()
        )
        # Spatial squeeze
        self.spatial_conv1 = nn.Sequential(
            layers.conv3d_layer(in_chs, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_in = self.identity(x)
        # Channel squeeze
        x_pool = layers.reduce_max(x, axis=[2, 3, 4], keepdims=True)
        x_pool = self.dense_conv1(x_pool)
        x_pool = self.dense_conv2(x_pool)
        x_pool = x_pool * x_in
        # Spatial squeeze
        x_space = self.spatial_conv1(x)
        x_space = x_space * x_in
        return x_pool + x_space


class LocalizationNet(nn.Module):

    def __init__(self, in_chs=1, out_chs=1):
        super(LocalizationNet, self).__init__()
        self.k = 4

        self.identity = nn.Identity()
        self.max_pool_3d = nn.MaxPool3d(kernel_size=2, stride=2, ceil_mode=True)
        self.sigmoid = nn.Sigmoid()

        # Encoder with 3 steps
        self.conv_enc_1 = ConvDilBlock(in_chs=in_chs, out_chs=self.k)
        self.sqex_enc_1 = SqExBlock(in_chs=self.k)
        self.conv_enc_2 = ConvDilBlock(in_chs=self.k, out_chs=2 * self.k)
        self.sqex_enc_2 = SqExBlock(in_chs=2 * self.k)
        self.conv_enc_3 = ConvDilBlock(in_chs=2 * self.k, out_chs=4 * self.k)
        self.sqex_enc_3 = SqExBlock(in_chs=4 * self.k)

        # Base
        self.base = ConvBlock(in_chs=4 * self.k, out_chs=4 * self.k)

        # Decoder with 3 steps
        self.upconv_dec_1 = UpConvBlock(in_chs=4 * self.k, out_chs=2 * self.k)
        self.conv_dec_1 = ConvBlock(in_chs=4 * self.k + 2 * self.k, out_chs=2 * self.k)
        self.sqex_dec_1 = SqExBlock(in_chs=2 * self.k)
        self.upconv_dec_2 = UpConvBlock(in_chs=2 * self.k, out_chs=self.k)
        self.conv_dec_2 = ConvBlock(in_chs=2 * self.k + self.k, out_chs=self.k)
        self.sqex_dec_2 = SqExBlock(in_chs=self.k)
        self.upconv_dec_3 = UpConvBlock(in_chs=self.k, out_chs=self.k)
        self.conv_dec_3 = ConvBlock(in_chs=self.k + self.k, out_chs=self.k)
        self.sqex_dec_3 = SqExBlock(in_chs=self.k)

        self.final = layers.conv3d_layer(self.k, out_chs, kernel_size=3, padding=1)

    def forward(self, x):
        x_in = self.identity(x)
        # Encoder step 1
        x = self.conv_enc_1(x)
        x = self.sqex_enc_1(x)
        x_1 = self.identity(x)
        x_p = self.max_pool_3d(x)

        # Encoder step 2
        x = self.conv_enc_2(x_p)
        x = self.sqex_enc_2(x)
        x_2 = self.identity(x)
        x_p = self.max_pool_3d(x)

        # Encoder step 3
        x = self.conv_enc_3(x_p)
        x = self.sqex_enc_3(x)
        x_3 = self.identity(x)
        x_p = self.max_pool_3d(x)

        # Base
        x = self.base(x_p)

        # Decoder step 1
        x = self.upconv_dec_1(x)
        x = layers.crop_concat3d(x, x_3)
        x = self.conv_dec_1(x)
        x = self.sqex_dec_1(x)

        # Decoder step 2
        x = self.upconv_dec_2(x)
        x = layers.crop_concat3d(x, x_2)
        x = self.conv_dec_2(x)
        x = self.sqex_dec_2(x)

        # Decoder step 3
        x = self.upconv_dec_3(x)
        x = layers.crop_concat3d(x, x_1)
        x = self.conv_dec_3(x)
        x = self.sqex_dec_3(x)

        # Scoring
        x = layers.crop3d_to_image(x, x_in)
        x = self.final(x)
        x_sigmoid = self.sigmoid(x)

        return x_sigmoid

    def predict(self, x):

        self.eval()

        with torch.no_grad():
            output = self.forward(x)

        return output
