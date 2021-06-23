import torch
import torch.nn as nn


def conv3d_layer(in_ch, out_ch, kernel_size=1, stride=1, padding=0, init='xavier'):
    """3D convolution layer with with default Xavier's initialisation
    Parameters
    ----------
    in_ch: channels C from input tensor (N, C, D, H, W)
    out_ch: number of filters
    kernel_size: size of the convolving kernel (default 1)
    stride: stride of the convolution
    padding: zero-padding added to all sides (default 0 to preserve dimensions)
    init: weight initialization scheme, 'xavier' or 'normal'

    Returns
    -------
    conv: the 3D convolution layer
    """
    conv = nn.Conv3d(in_ch, out_ch, kernel_size, stride, padding)

    if init == "xavier":
        nn.init.xavier_uniform_(conv.weight)
    elif init == "normal":
        nn.init.normal_(conv.weight, std=0.2)
    else:
        raise Exception("Invalid Initialization Scheme")
    conv.bias.data.fill_(0.)

    return conv


def conv3d_dil_layer(in_ch, out_ch, kernel_size=3, stride=1, padding=2, dilation=2, init='xavier'):
    """3D dilated convolution layer with Xavier's initialisation
    Parameters
    ----------
    in_ch: channels C from input tensor (N, C, D, H, W)
    out_ch: number of filters
    kernel_size: size of the convolving kernel
    stride: stride of the convolution
    padding: zero-padding added to all sides of the input (default 2 to preserve dimensions)
    dilation: scale of kernel dilation. Refer https://arxiv.org/abs/1511.07122
    init: weight initialization scheme, 'xavier' or 'normal'

    Returns
    -------
    conv: the 3D dilated convolution layer
    """
    conv = nn.Conv3d(in_ch, out_ch, kernel_size, stride, padding, dilation)

    if init == "xavier":
        nn.init.xavier_uniform_(conv.weight)
    elif init == "normal":
        nn.init.normal_(conv.weight, std=0.2)
    else:
        raise Exception("Invalid Initialization Scheme")
    conv.bias.data.fill_(0.)

    return conv


def upconv3d_layer(in_ch, out_ch, kernel_size=2, stride=2, init='xavier'):
    """3D transposed convolution layer with Xavier's initialisation
    Parameters
    ----------
    in_ch: channels C from input tensor (N, C, D, H, W)
    out_ch: number of filters
    kernel_size: recommended kernel shape is an even multiple of stride
                 Refer https://distill.pub/2016/deconv-checkerboard/
    stride: stride which results in upconvolution, scalar or tuple.
    init: weight initialization scheme, 'xavier' or 'normal'

    Returns
    -------
    conv: the 3D deconvolution layer
    """
    upconv = nn.ConvTranspose3d(in_ch, out_ch, kernel_size, stride)

    if init == "xavier":
        nn.init.xavier_uniform_(upconv.weight)
    elif init == "normal":
        nn.init.normal_(upconv.weight, std=0.2)
    upconv.bias.data.fill_(0.)

    return upconv


def crop3d(x1, target_dims):
    """3D center crop, used in U-Net like architectures
    Parameters
    ----------
    x1: tensor to be cropped (_,_,D1,H1,W1)
    target_dims: list [target_d,target_h,target_w]

    Returns
    -------
    x1_cropped: tensor cropped to (_,_,D2,H2,W2)
    """

    shape_diff = torch.tensor(x1.shape[2:]) - torch.tensor(target_dims)
    offsets = (shape_diff / 2).type(torch.LongTensor)
    target_d, target_h, target_w = target_dims
    x1_cropped = x1[:,
                    :,
                    offsets[0]:offsets[0] + target_d,
                    offsets[1]:offsets[1] + target_h,
                    offsets[2]:offsets[2] + target_w]
    return x1_cropped


def crop3d_to_image(x1, x2):
    """3D center crop, used in U-Net like architectures
    Parameters
    ----------
    x1: tensor to be cropped (_,_,D1,H1,W1)
    x2: tensor for reference of shape (_,_,D2,H2,W2)

    Returns
    -------
    x1_cropped: tensor cropped to (_,_,D2,H2,W2)
    """

    # (N, C, D, H, W)
    target_d = x2.shape[2]
    target_h = x2.shape[3]
    target_w = x2.shape[4]

    x1_cropped = crop3d(x1, target_dims=[target_d, target_h, target_w])

    return x1_cropped


def crop_concat3d(x1, x2, axis=1):
    """3D center crop x1 to x2's shape and concatenate cropped x1 to x2.
    Used in U-Net like architectures
    Parameters
    ----------
    x1: tensor to be cropped and concatenated (_,C1,D1,H1,W1)
    x2: tensor for reference of shape and concatenated to (_,C2,D2,H2,W2)
    axis: dimension over which the tensors are concatenated

    Returns
    -------
    x_concat: cropped x1 concatenated to x2, (_,C1+C2,D2,H2,W2)
    """
    x1_crp = crop3d_to_image(x1, x2)
    x_concat = torch.cat([x1_crp, x2], axis)
    return x_concat


def reduce_max(x, axis, keepdims):
    """Reduces input_tensor along the dimensions given in axis
    Parameters
    ----------
    x: tensor to reduce
    axis: dimensions to reduce, python list
    keepdims: if true, retains reduced dimensions with length 1

    Returns
    -------
    x_red: reduced tensor
    """
    x_red = x
    for n in axis:
        x_red = x_red.max(dim=n, keepdim=keepdims).values
    return x_red