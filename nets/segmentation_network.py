import torch
import torch.nn as nn
from . import custom_layers as layers


class ConvBlock(nn.Module):
    """2 x 3D convolution with following parameters
    kernel=3, stride=1, padding=1, Xavier's init
    followed by Leaky ReLU Activation and Batch Normalization
    """

    def __init__(self, in_chs, chs_1, chs_2):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            layers.conv3d_layer(in_chs, chs_1, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            layers.conv3d_layer(chs_1, chs_2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm3d(chs_2, eps=0.001, momentum=0.01)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class UpConvBlock(nn.Module):
    """3D transposed convolution with default parameters
    kernel=2, stride=2, Xavier's init
    followed by Leaky ReLU Activation
    """

    def __init__(self, in_chs, skip_chs, chs_1, chs_2):
        super(UpConvBlock, self).__init__()
        self.upconv = nn.Sequential(
            layers.upconv3d_layer(in_chs, in_chs),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv = nn.Sequential(
            ConvBlock(in_chs=in_chs + skip_chs, chs_1=chs_1, chs_2=chs_2),
            nn.BatchNorm3d(chs_2, eps=0.001, momentum=0.01)
        )

    def forward(self, x, x_skip):
        x = self.upconv(x)
        x = layers.crop_concat3d(x, x_skip)
        x = self.conv(x)
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
            layers.conv3d_layer(in_chs, int(in_chs / 2)),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.dense_conv2 = nn.Sequential(
            layers.conv3d_layer(int(in_chs / 2), in_chs),
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


class SegmentationNet(nn.Module):

    def __init__(self):
        super(SegmentationNet, self).__init__()

        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout3d(p=0.5)
        self.max_pool_3d = nn.MaxPool3d(kernel_size=2, stride=2, ceil_mode=True)

        # Contracting Gaussian Maps - 4 Steps
        self.conv_1_g = ConvBlock(in_chs=1, chs_1=1, chs_2=1)
        self.conv_2_g = ConvBlock(in_chs=1, chs_1=1, chs_2=1)
        self.conv_3_g = ConvBlock(in_chs=1, chs_1=1, chs_2=1)
        self.conv_4_g = ConvBlock(in_chs=1, chs_1=1, chs_2=1)

        # Contracting Images - 4 Steps
        self.conv_1_im = ConvBlock(in_chs=1, chs_1=4, chs_2=8)
        self.sqex_1_im = SqExBlock(in_chs=8)
        self.conv_2_im = ConvBlock(in_chs=8, chs_1=8, chs_2=16)
        self.sqex_2_im = SqExBlock(in_chs=16)
        self.conv_3_im = ConvBlock(in_chs=16, chs_1=16, chs_2=32)
        self.sqex_3_im = SqExBlock(in_chs=32)
        self.conv_4_im = ConvBlock(in_chs=32, chs_1=32, chs_2=64)
        self.sqex_4_im = SqExBlock(in_chs=64)

        # Base
        self.base = ConvBlock(in_chs=64, chs_1=64, chs_2=64)

        # Expanding Images - 4 Steps
        self.upconv_1 = UpConvBlock(in_chs=64, skip_chs=64+1, chs_1=64, chs_2=64)
        self.sqex_1_up = SqExBlock(in_chs=64)
        self.upconv_2 = UpConvBlock(in_chs=64, skip_chs=32+1, chs_1=32, chs_2=32)
        self.sqex_2_up = SqExBlock(in_chs=32)
        self.upconv_3 = UpConvBlock(in_chs=32, skip_chs=16+1, chs_1=16, chs_2=16)
        self.sqex_3_up = SqExBlock(in_chs=16)
        self.upconv_4 = UpConvBlock(in_chs=16, skip_chs=8+1, chs_1=8, chs_2=8)
        self.sqex_4_up = SqExBlock(in_chs=8)

        # Final layer
        self.final = layers.conv3d_layer(in_ch=8, out_ch=1, kernel_size=3, padding=1)

    def forward(self, x_im, x_g):
        # Contracting Gaussian Maps - 4 Steps
        conv_b_1_g = self.conv_1_g(x_g)
        conv_p_1_g = self.max_pool_3d(conv_b_1_g)
        conv_b_2_g = self.conv_2_g(conv_p_1_g)
        conv_p_2_g = self.max_pool_3d(conv_b_2_g)
        conv_b_3_g = self.conv_3_g(conv_p_2_g)
        conv_p_3_g = self.max_pool_3d(conv_b_3_g)
        conv_b_4_g = self.conv_4_g(conv_p_3_g)
        conv_p_4_g = self.max_pool_3d(conv_b_4_g)
        # conv_b_4_g = self.max_pool_3d(conv_b_4_g)

        # Contracting Images - 4 Steps
        conv_b_1_im = self.conv_1_im(x_im)
        conv_b_1_im = self.sqex_1_im(conv_b_1_im)
        conv_p_1_im = self.max_pool_3d(conv_b_1_im)

        conv_b_2_im = self.conv_2_im(conv_p_1_im)
        conv_b_2_im = self.sqex_2_im(conv_b_2_im)
        conv_p_2_im = self.max_pool_3d(conv_b_2_im)

        conv_b_3_im = self.conv_3_im(conv_p_2_im)
        conv_b_3_im = self.sqex_3_im(conv_b_3_im)
        conv_p_3_im = self.max_pool_3d(conv_b_3_im)

        conv_b_4_im = self.conv_4_im(conv_p_3_im)
        conv_b_4_im = self.sqex_4_im(conv_b_4_im)
        conv_p_4_im = self.max_pool_3d(conv_b_4_im)

        # Concatenation Images and Gaussian Maps
        conv_b_1 = torch.cat([conv_b_1_im, conv_b_1_g], dim=1)
        conv_b_2 = torch.cat([conv_b_2_im, conv_b_2_g], dim=1)
        conv_b_3 = torch.cat([conv_b_3_im, conv_b_3_g], dim=1)
        conv_b_4 = torch.cat([conv_b_4_im, conv_b_4_g], dim=1)

        # Base
        conv_base = self.base(conv_p_4_im)
        conv_base = self.dropout(conv_base)

        # Expanding Images - 4 Steps
        upconv_b_1 = self.upconv_1(conv_base, conv_b_4)
        upconv_b_1 = self.sqex_1_up(upconv_b_1)
        upconv_b_2 = self.upconv_2(upconv_b_1, conv_b_3)
        upconv_b_2 = self.sqex_2_up(upconv_b_2)
        upconv_b_3 = self.upconv_3(upconv_b_2, conv_b_2)
        upconv_b_3 = self.sqex_3_up(upconv_b_3)
        upconv_b_4 = self.upconv_4(upconv_b_3, conv_b_1)
        upconv_b_4 = self.sqex_4_up(upconv_b_4)

        # Scoring
        upconv_b_4 = layers.crop3d_to_image(upconv_b_4, x_im)
        score = self.final(upconv_b_4)
        sigmoid = self.sigmoid(score)

        return sigmoid

    def predict(self, x_im, x_g):

        self.eval()

        with torch.no_grad():
            output = self.forward(x_im, x_g)

        return output