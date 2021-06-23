import torch
import torch.nn as nn
import sys


class ConvBlock2d(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=(3, 3), padding=(1, 1)):
        super(ConvBlock2d, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size,
                                            stride=(1, 1), padding=padding),
                                  nn.BatchNorm2d(out_dim),
                                  nn.LeakyReLU()
                                  # nn.InstanceNorm3d(out_dim),
                                  )

    def forward(self, x):
        x = self.conv(x)
        return x


class ConvTransposeBlock2d(nn.Module):
    def __init__(self, in_dim, skip_dim, out_dim, kernel_size=(4, 4), padding=(1, 1)):
        super(ConvTransposeBlock2d, self).__init__()
        self.convT = nn.Sequential(nn.ConvTranspose2d(in_dim, in_dim, kernel_size=kernel_size,
                                                      stride=(2, 2), padding=padding),
                                   nn.BatchNorm2d(in_dim),
                                   nn.LeakyReLU())
        self.conv = ConvBlock2d(in_dim + skip_dim, out_dim, kernel_size=(3, 3), padding=(1, 1))

    def crop_cat(self, x, y):
        _, _, hx, wx = x.shape
        _, _, hy, wy = y.shape
        if hx > hy and wx > wy:
            # crop x to y
            x = x[:, :, (hx - hy) // 2: (hx - hy) // 2 + hy, (wx - wy) // 2: (wx - wy) // 2 + wy]
        else:
            print('something\'s off!')
            sys.exit(0)
        return torch.cat([x, y], 1)

    def forward(self, x, x_skip):
        x = self.convT(x)
        x = self.crop_cat(x, x_skip)
        x = self.conv(x)
        return x


class Btrfly(nn.Module):
    def __init__(self, in_chs=1, out_chs=24, factor=16):

        super(Btrfly, self).__init__()

        self.in_chs = in_chs
        self.out_chs = out_chs

        # sagittal
        self.conv1_s = ConvBlock2d(in_chs, factor)
        self.conv2_s = ConvBlock2d(factor, 2 * factor)
        self.conv3_s = ConvBlock2d(2 * factor, 4 * factor)
        self.conv4_s = ConvBlock2d(4 * factor, 8 * factor)

        self.gamma1_s = torch.tensor(1.0, requires_grad=True)
        self.conv5_s = ConvBlock2d(8 * factor + 8 * factor, 16 * factor)  # fused_input

        self.gamma2_s = torch.tensor(1.0, requires_grad=True)
        self.upconv0_s = ConvTransposeBlock2d(16 * factor + 16 * factor, 16 * factor, 8 * factor)  # fused_input
        self.gamma3_s = torch.tensor(1.0, requires_grad=True)
        self.upconv1_s = ConvTransposeBlock2d(8 * factor + 8 * factor, 4 * factor, 8 * factor)  # fused_input
        self.upconv2_s = ConvTransposeBlock2d(8 * factor, 2 * factor, 4 * factor)
        self.upconv3_s = ConvTransposeBlock2d(4 * factor, factor, 2 * factor)

        self.score_s = nn.Conv2d(2 * factor, self.out_chs, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        # coronal
        self.conv1_c = ConvBlock2d(in_chs, factor)
        self.conv2_c = ConvBlock2d(factor, 2 * factor)
        self.conv3_c = ConvBlock2d(2 * factor, 4 * factor)
        self.conv4_c = ConvBlock2d(4 * factor, 8 * factor)

        self.gamma1_c = torch.tensor(1.0, requires_grad=True)
        self.conv5_c = ConvBlock2d(8 * factor + 8 * factor, 16 * factor)  # fused_input

        self.gamma2_c = torch.tensor(1.0, requires_grad=True)
        self.upconv0_c = ConvTransposeBlock2d(16 * factor + 16 * factor, 16 * factor, 8 * factor)  # fused_input
        self.gamma3_c = torch.tensor(1.0, requires_grad=True)
        self.upconv1_c = ConvTransposeBlock2d(8 * factor + 8 * factor, 4 * factor, 8 * factor)  # fused_input
        self.upconv2_c = ConvTransposeBlock2d(8 * factor, 2 * factor, 4 * factor)
        self.upconv3_c = ConvTransposeBlock2d(4 * factor, factor, 2 * factor)

        self.score_c = nn.Conv2d(2 * factor, self.out_chs, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        # common
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
                print('Special initialisation for:', m)
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def fuse_views(self, x_a, x_b, gamma_a, gamma_b):

        x_a_f = torch.cat([x_a, gamma_a * x_b], 1)
        x_b_f = torch.cat([x_b, gamma_b * x_a], 1)
        return x_a_f, x_b_f

    def forward(self, xs, xc):
        B, C, H, W = xs.shape
        # sagittal-encoding
        xs_1 = self.conv1_s(xs)
        xs = self.pool(xs_1)
        xs_2 = self.conv2_s(xs)
        xs = self.pool(xs_2)
        xs_3 = self.conv3_s(xs)
        xs = self.pool(xs_3)
        xs_4 = self.conv4_s(xs)
        # coronal-encoding
        xc_1 = self.conv1_c(xc)
        xc = self.pool(xc_1)
        xc_2 = self.conv2_c(xc)
        xc = self.pool(xc_2)
        xc_3 = self.conv3_c(xc)
        xc = self.pool(xc_3)
        xc_4 = self.conv4_c(xc)
        # fused
        xs_4, xc_4 = self.fuse_views(xs_4, xc_4, self.gamma1_s, self.gamma1_c)
        xs = self.pool(xs_4)
        xc = self.pool(xc_4)
        xs = self.conv5_s(xs)
        xc = self.conv5_c(xc)
        xs, xc = self.fuse_views(xs, xc, self.gamma2_s, self.gamma2_c)
        xs = self.upconv0_s(xs, xs_4)
        xc = self.upconv0_c(xc, xc_4)
        xs, xc = self.fuse_views(xs, xc, self.gamma3_s, self.gamma3_c)
        # sagittal-decoding
        xs = self.upconv1_s(xs, xs_3)
        xs = self.upconv2_s(xs, xs_2)
        xs = self.upconv3_s(xs, xs_1)
        # coronal-decoding
        xc = self.upconv1_s(xc, xc_3)
        xc = self.upconv2_s(xc, xc_2)
        xc = self.upconv3_s(xc, xc_1)
        # scoring layers
        xs = self.score_s(xs)
        xc = self.score_c(xc)
        return torch.sigmoid(xs), torch.sigmoid(xc)

    def predict(self, xs, xc):

        self.eval()
        with torch.no_grad():
            xs, xc = self.forward(xs, xc)
        return xs, xc


class LabellingNet(nn.Module):

    def __init__(self, in_chs=1, out_chs=24, factor_stack=[16, 4]):
        super(LabellingNet, self).__init__()

        self.btrfly1 = Btrfly(in_chs=in_chs, out_chs=out_chs, factor=factor_stack[0])

        self.btrfly2 = Btrfly(in_chs=out_chs, out_chs=out_chs, factor=factor_stack[1])

    def forward(self, xs, xc):
        output_stack = []

        xs, xc = self.btrfly1(xs, xc)
        output_stack.append((xs, xc))
        xs, xc = self.btrfly2(xs, xc)
        output_stack.append((xs, xc))

        return output_stack

    def predict(self, xs, xc):
        self.eval()

        with torch.no_grad():
            output_stack = self.forward(xs, xc)
            xs, xc = output_stack[-1]
        return xs, xc
