import numpy as np
import nibabel as nib
import SimpleITK as sitk
from matplotlib import patches
import scipy.ndimage.filters as fi
from matplotlib import pyplot as plt
from dipy.align.reslice import reslice

plt.rcParams['image.cmap'] = 'gray'


# -------------------------
# Nifti Image Preprocessing
# -------------------------
def load_nib(fpath):
    """
    Load nifti image
    :param fpath: path of nifti file
    """
    im = nib.load(fpath)

    return im


def resample_nib(im, new_spacing=(1, 1, 1), order=0):
    """
    Resample nifti voxel array and corresponding affine
    :param im: nifti image
    :param new_spacing: new voxel size
    :param order: order of interpolation for resampling/reslicing, 0 nearest interpolation, 1 trilinear etc.
    :return new_im: resampled nifti image
    """
    header = im.header
    vox_zooms = header.get_zooms()
    vox_arr = im.get_fdata()
    vox_affine = im.affine
    # resample using DIPY.ALIGN
    if isinstance(new_spacing, int) or isinstance(new_spacing, float):
        new_spacing = (new_spacing[0], new_spacing[1], new_spacing[2])
    new_vox_arr, new_vox_affine = reslice(vox_arr, vox_affine, vox_zooms, new_spacing, order=order)
    # create resampled image
    new_im = nib.Nifti1Image(new_vox_arr, new_vox_affine, header)

    return new_im


def transpose_compatible(arr, direction):
    """
    Transpose array to a compatible direction
    :param arr: numpy array
    :param direction: 'asl_to_np' or 'np_to_asl' only
    :return arr: transposed array
    """
    if direction == 'asl_to_np':
        arr = arr.transpose([1, 0, 2])[:, :, ::-1]
    if direction == 'np_to_asl':
        arr = arr[:, :, ::-1].transpose([1, 0, 2])
    else:
        'Direction can only be ASL to Anjany\'s numpy indexing or the other way around!'

    return arr


# --------------------------
# Preprocessing Segmentation
# --------------------------
def get_vert_lims(loc, off, h, w, d):
    """
    Get vertebra and padding limits for segmentation training
    :param loc: vertebra centroid coordinates
    :param off: offset
    :param h: original image height
    :param w: original image width
    :param d: original image depth
    :return:
    vert_lims: vertebra patch in original full spine image coordinates (for cropping)
    vert_pads: padding to add on 3 dimensions to center the vertebrae in the patch
    """
    # height
    if loc[0] + off[0, 0] < 0:
        h_min = 0
        h_lo_pad = 0 - (loc[0] + off[0, 0])
    else:
        h_min = loc[0] + off[0, 0]
        h_lo_pad = 0

    if loc[0] + off[0, 1] > h:
        h_max = h
        h_hi_pad = (loc[0] + off[0, 1]) - h
    else:
        h_max = loc[0] + off[0, 1]
        h_hi_pad = 0

    # width
    if loc[1] + off[1, 0] < 0:
        w_min = 0
        w_lo_pad = 0 - (loc[1] + off[1, 0])
    else:
        w_min = loc[1] + off[1, 0]
        w_lo_pad = 0

    if loc[1] + off[1, 1] > w:
        w_max = w
        w_hi_pad = (loc[1] + off[1, 1]) - w
    else:
        w_max = loc[1] + off[1, 1]
        w_hi_pad = 0

    # depth
    if loc[2] + off[2, 0] < 0:
        d_min = 0
        d_lo_pad = 0 - (loc[2] + off[2, 0])
    else:
        d_min = loc[2] + off[2, 0]
        d_lo_pad = 0

    if loc[2] + off[2, 1] > d:
        d_max = d
        d_hi_pad = (loc[2] + off[2, 1]) - d
    else:
        d_max = loc[2] + off[2, 1]
        d_hi_pad = 0

    vert_lims = [h_min, h_max, w_min, w_max, d_min, d_max]
    vert_pads = [h_lo_pad, h_hi_pad, w_lo_pad, w_hi_pad, d_lo_pad, d_hi_pad]

    return vert_lims, vert_pads


def rescale(x, min_val, max_val):
    return (max_val - min_val) * (x - np.min(x)) / float(np.max(x) - np.min(x)) + min_val


def gen_gaussian_im(shape, mean, variance):
    """
    Generate a 3D Gaussian kernel array for a single vertebra centroid
    :param shape: full spine image shape 1 mm
    :param mean: gaussian mean
    :param variance: gaussian variance
    :return:
    """
    # create nxn zeros
    gauss = np.zeros(shape)
    # set element at the middle to one, a dirac delta
    gauss[mean[0], mean[1], mean[2]] = 1
    # gaussian-smooth the dirac, resulting in a gaussian filter mask
    return rescale(fi.gaussian_filter(gauss, variance), 0, 1)


def get_channelwise_gaussian(centroids_list, verts_in_im, im_shape):
    """
    Generate a 3D Gaussian kernel array for all vertebrae
    :param centroids_list: centroid coordinates
    :param verts_in_im: vertebrae to patch
    :param im_shape: full spine image shape 1 mm
    :return:
    """
    num_verts = centroids_list.shape[0]
    cent_mask = np.repeat(np.expand_dims(np.zeros(im_shape, dtype='float32'), axis=-1), num_verts, axis=-1)
    for vert_idx in verts_in_im:
        if vert_idx <= num_verts:
            cent_loc = centroids_list[vert_idx - 1].astype(int)
            gauss = gen_gaussian_im(im_shape, cent_loc, variance=2)
            gauss = (gauss - np.amin(gauss)) / np.amax(gauss - np.amin(gauss))
            cent_mask[:, :, :, vert_idx - 1] = gauss
    return cent_mask


def get_gaussian_heatmap(im_shape, cent_loc):
    """
    Generate a 3D Gaussian heatmap for a single vertebrae
    :param im_shape: full spine image shape 1 mm
    :param cent_loc: vertebra centroid coordinates
    :return: cent_mask: heatmap mask
    """
    cent_mask = np.zeros(im_shape, dtype='float32')
    gauss = gen_gaussian_im(im_shape, cent_loc, variance=2)
    gauss = (gauss - np.amin(gauss)) / np.amax(gauss - np.amin(gauss))
    cent_mask[:, :, :] = gauss
    return cent_mask


def get_seg_patch(im, loc, off):
    """
    Generate a vertebra patch for segmentation training
    :param im: original full spine image 1 mm
    :param loc: centroid coordinates 1 mm
    :param gauss_cents: gaussian heatmaps
    :param vert_idx:
    :param off: padding offset
    :return:
    """
    h, w, d = im.shape
    # get patch limits and padding
    lims, pads = get_vert_lims(loc, off, h, w, d)
    gauss_hm = get_gaussian_heatmap(im.shape, loc)
    # crop
    vert_im = im[lims[0]:lims[1], lims[2]:lims[3], lims[4]:lims[5]]
    vert_gauss = gauss_hm[lims[0]:lims[1], lims[2]:lims[3], lims[4]:lims[5]]
    # pad
    vert_im = np.pad(vert_im, pad_width=((pads[0], pads[1]), (pads[2], pads[3]), (pads[4], pads[5])), mode='constant')
    vert_gauss = np.pad(vert_gauss, pad_width=((pads[0], pads[1]), (pads[2], pads[3]), (pads[4], pads[5])),
                        mode='constant')

    return vert_im, vert_gauss, lims, pads


def crop_seg_patch(msk, pads):
    """
    Crop the patch to original size
    :param msk: patch mask
    :param pads: pads to crop
    :return:
    """
    h,w,d = msk.shape
    [h_lo_pad, h_hi_pad, w_lo_pad, w_hi_pad, d_lo_pad, d_hi_pad] = pads
    msk_crop = msk[h_lo_pad:h-h_hi_pad, w_lo_pad:w-w_hi_pad, d_lo_pad:d-d_hi_pad]

    return msk_crop


# ----------------------------
# Postprocessing Localization
# ----------------------------
def clean_hm_prediction(msk, threshold):
    """
    Apply largest 3d connected component for localization
    :param msk: 3d spine localization heatmap
    :param threshold: intensity (probability) threshold
    :return msk_corrected: post-processed mask
    """
    msk[msk < threshold] = 0
    msk_binary = np.copy(msk)
    msk_binary[msk_binary > threshold] = 1
    msk_im = sitk.GetImageFromArray(msk_binary.astype('uint8'))
    msk_im.SetSpacing([5, 5, 5])
    # connected component filter
    connected = sitk.ConnectedComponentImageFilter()
    connected.FullyConnectedOn()
    cc = connected.Execute(msk_im)
    # find largest component
    no_of_cc = connected.GetObjectCount()
    cc_sizes = np.zeros((1, no_of_cc))
    cc_arr = sitk.GetArrayFromImage(cc)
    for i in range(1, no_of_cc + 1):
        cc_sizes[0, i - 1] = np.count_nonzero(cc_arr == i)
    cc_seq = np.argsort(cc_sizes)
    largest_comp = cc_seq[0, -1] + 1
    # remove every other 'component' other than largest component
    cc_arr[cc_arr != largest_comp] = False
    cc_arr[cc_arr == largest_comp] = True
    # return the 'mask' corresponding to the largest connected component
    msk_corrected = np.zeros_like(msk)
    msk_corrected[cc_arr != 0] = msk[cc_arr != 0]

    return msk_corrected


def msk_2_box(msk, threshold):
    """
    Compute the 3d bounding box coordinates from the localization heatmap
    :param msk: 3d spine localization heatmap
    :param threshold: intensity (probability) threshold
    :return: 3d bounding box coordinates
    """
    msk_temp = np.copy(msk)
    msk_temp[msk < threshold] = 0
    nzs = np.nonzero(msk_temp)
    if len(nzs[0]) > 0:
        h_min = np.amin(nzs[0])
        w_min = np.amin(nzs[1])
        d_min = np.amin(nzs[2])
        h_max = np.amax(nzs[0])
        w_max = np.amax(nzs[1])
        d_max = np.amax(nzs[2])
        return [h_min, h_max, w_min, w_max, d_min, d_max]
    else:
        h, w, d = msk_temp.shape

    return [0, h, 0, w, 0, d]


def add_tolerance(box, im_shape, tols):
    """
    Add distance tolerance to the dimensions of the bounding box
    :param box: 3d bounding box
    :param im_shape: image shape where the bounding box is applied
    :param tols: tolerances
    :return: new 3d bounding box coordinates
    """
    h, w, d = im_shape
    [h_min, h_max, w_min, w_max, d_min, d_max] = box
    h_min = h_min - tols[0]
    h_max = h_max + tols[1]
    w_min = w_min - tols[2]
    w_max = w_max + tols[3]
    d_min = d_min - tols[4]
    d_max = d_max + tols[5]
    if h_min < 0:
        h_min = 0
    if h_max > h:
        h_max = h
    if w_min < 0:
        w_min = 0
    if w_max > w:
        w_max = w
    if d_min < 0:
        d_min = 0
    if d_max > d:
        d_max = d

    return h_min, h_max, w_min, w_max, d_min, d_max


def adjust_box(box, im, image=True):
    """
    Adjust bounding box shape
    :param box: 3d bounding box
    :param im: image or mask where the bounding box is applied
    :param image: True if image, False if centroid mask
    :return: new 3d image or
    """
    # first bounding box
    [h_min, h_max, w_min, w_max, d_min, d_max] = box

    # based on first box decide tolerance
    depth = d_max - d_min
    width = w_max - w_min
    max_dim = max(depth, width)

    # tolerance
    tol_h = (50, 50)
    tol_d = (0, 0)

    # add tolerance on sagittal view depending on bounding box
    if max_dim <= 25:
        tol_w = (25, 35)
    elif max_dim <= 45:
        tol_w = (10, 15)
    else:
        tol_w = (5, 5)

    if image:
        im_shape = im.shape
    else:
        im_shape = (im.shape[0], im.shape[1], im.shape[2])

    box_tolerance = add_tolerance(box, im_shape, (tol_h[0], tol_h[1], tol_w[0], tol_w[1], tol_d[0], tol_d[1]))
    [h_min, h_max, w_min, w_max, d_min, d_max] = box_tolerance

    # width and depth must be the same after adding the tolerance
    depth = d_max - d_min
    width = w_max - w_min

    # correct tolerance
    if depth > width:
        diff = depth - width
        if diff % 2 == 0:
            tol_w = (tol_w[0] + diff // 2, tol_w[1] + diff // 2)
        else:
            tol_w = (tol_w[0] + diff // 2, tol_w[1] + diff // 2 + 1)

    elif depth < width:
        diff = width - depth
        if diff % 2 == 0:
            tol_d = (tol_d[0] + diff // 2, tol_d[1] + diff // 2)
        else:
            tol_d = (tol_d[0] + diff // 2, tol_d[1] + diff // 2 + 1)

    # second box with tolerance can get out of image margins
    box_tolerance = add_tolerance(box, im_shape, (tol_h[0], tol_h[1], tol_w[0], tol_w[1], tol_d[0], tol_d[1]))
    [h_min, h_max, w_min, w_max, d_min, d_max] = box_tolerance

    # initialize background
    height = h_max - h_min
    width_depth = 90

    if image:
        background = np.zeros((height, width_depth, width_depth))
    else:
        background = np.zeros((height, width_depth, width_depth, 24))

    # calculate the difference between background shape and bounding box
    w_diff = background.shape[1] - (w_max - w_min)

    if w_diff % 2 == 0:
        w_background = (w_diff // 2, w_diff // 2)
    else:
        w_background = (w_diff // 2, w_diff // 2 + 1)

    d_diff = background.shape[2] - (d_max - d_min)

    if d_diff % 2 == 0:
        d_background = (d_diff // 2, d_diff // 2)
    else:
        d_background = (d_diff // 2, d_diff // 2 + 1)

    # place the cropped image in the center of the background
    background[:, w_background[0]:background.shape[1] - w_background[1], d_background[0]:background.shape[2] - d_background[1]] = im[h_min:h_max, w_min:w_max, d_min:d_max]
    box_background = (w_background[0], background.shape[1] - w_background[1], d_background[0], background.shape[2] - d_background[1])

    return background, box_background, box_tolerance


# ----------------------------
# Postprocessing Labelling
# ----------------------------
def masks_2d_to_3d(msk_s, msk_c):
    """
    Convert 2d vertebrae centroid heatmaps to 2d
    :param msk_s: sagittal 2d centroids mask
    :param msk_c: coronal 2d centroids mask
    :return msk_3d: 3d vertebrae centroid heatmap
    """
    msk_s_3d = np.tile(np.expand_dims(msk_s, 2), reps=[1, 1, np.shape(msk_c)[1], 1])
    msk_c_3d = np.tile(np.expand_dims(msk_c, 1), reps=[1, np.shape(msk_s)[1], 1, 1])
    msk_3d = msk_c_3d * msk_s_3d

    return msk_3d


def mask_to_centroids(msk_3d, verts_in_im):
    """
    Convert 3d vertebrae centroid heatmaps to numpy array (list of coordinates)
    :param msk_3d: 3d vertebrae centroid heatmap
    :param verts_in_im: how many vertebrae were detected
    :return cents: vertebrae centroid array of coordinates
    """
    h, w, d, chs = msk_3d.shape
    cents = np.full((chs, 3), np.nan)
    for vert in verts_in_im:
        cent_loc = np.unravel_index(np.argmax(msk_3d[..., vert - 1]), (h, w, d))
        cents[vert - 1, ...] = cent_loc

    return cents


# ---------------------------
# Segmentation Postprocessing
# ---------------------------
def clean_seg_prediction(msk, threshold):

    msk[msk < threshold] = 0
    msk_binary = np.copy(msk)
    msk_binary[msk_binary > threshold] = 1

    msk_im = sitk.GetImageFromArray(msk_binary.astype('uint8'))
    msk_im.SetSpacing([1, 1, 1])

    # connected component filter
    connected = sitk.ConnectedComponentImageFilter()
    connected.FullyConnectedOn()
    cc = connected.Execute(msk_im)

    # find largest component
    no_of_cc = connected.GetObjectCount()
    cc_sizes = np.zeros((1, no_of_cc))
    cc_arr = sitk.GetArrayFromImage(cc)

    for i in range(1, no_of_cc + 1):
        cc_sizes[0, i - 1] = np.count_nonzero(cc_arr == i)

    cc_seq = np.argsort(cc_sizes)
    largest_comp = cc_seq[0, -1] + 1

    # remove every other 'component' other than largest component
    cc_arr[cc_arr != largest_comp] = False
    cc_arr[cc_arr == largest_comp] = True

    # return the 'mask' corresponding to the largest connected component
    msk_corrected = np.zeros_like(msk)
    msk_corrected[cc_arr != 0] = msk[cc_arr != 0]

    return msk_corrected


def refine_mask(msk, threshold):
    msk[msk < threshold] = 0
    msk_binary = np.copy(msk)
    msk_binary[msk_binary > threshold] = 1

    msk_im = sitk.GetImageFromArray(msk_binary.astype('uint8'))
    msk_im.SetSpacing([1, 1, 1])
    # connected component filter
    connected = sitk.ConnectedComponentImageFilter()
    connected.FullyConnectedOn()
    cc = connected.Execute(msk_im)
    # find largest component
    no_of_cc = connected.GetObjectCount()
    cc_sizes = np.zeros((1, no_of_cc))
    cc_arr = sitk.GetArrayFromImage(cc)

    for i in range(1, no_of_cc + 1):
        cc_sizes[0, i - 1] = np.count_nonzero(cc_arr == i)
    cc_seq = np.argsort(cc_sizes)
    largest_comp = cc_seq[0, -1] + 1
    # remove every other 'component' other than largest component
    cc_arr[cc_arr != largest_comp] = 0
    cc_arr[cc_arr == largest_comp] = 1

    return cc_arr


# ---------------------------
# Visualization Localization
# ---------------------------
def plot_hm(im, msk, threshold):
    """
    Plot image and localization heatmap (coronal and sagittal views)
    :param im: 3d image
    :param msk: 3d localization heatmap
    :param threshold: intensity (probability) threshold
    """
    msk[msk < threshold] = 0
    plt.figure(figsize=(8, 8))
    plt.subplot(121)
    plt.imshow(np.amax(im, -1), cmap="gray")
    plt.imshow(np.sum(msk, -1), cmap='gnuplot2', alpha=0.5)
    plt.xlabel('%.2f - %.2f' % (np.amin(msk), np.amax(msk)))
    plt.subplot(122)
    plt.imshow(np.amax(im, 1), cmap="gray")
    plt.imshow(np.sum(msk, 1), cmap='gnuplot2', alpha=0.5)
    plt.xlabel('%.2f - %.2f' % (np.amin(msk), np.amax(msk)))


def plot_box(im, box):
    """
    Plot image and bounding box (coronal and sagittal views)
    :param im: 3d image
    :param box: bounding box
    """
    [h_min, h_max, w_min, w_max, d_min, d_max] = box

    fig, ax = plt.subplots(1, 2, figsize=(8, 8))
    ax[0].imshow(np.amax(im, -1), cmap="gray")
    rect = patches.Rectangle((w_min, h_min), w_max - w_min, h_max - h_min, linewidth=3, edgecolor='g', facecolor='none')
    ax[0].add_patch(rect)
    ax[1].imshow(np.amax(im, 1), cmap="gray")
    rect = patches.Rectangle((d_min, h_min), d_max - d_min, h_max - h_min, linewidth=3, edgecolor='g', facecolor='none')
    ax[1].add_patch(rect)


# ---------------------------
# Visualization Labelling
# ---------------------------
def plot_labels(im, cents, im_c=None):
    """
    Plot image and bounding box (coronal and sagittal views)
    :param im: 3d image
    :param cents: vertebrae centroid array of coordinates
    """
    if im_c is None:
        # im is 3D
        im_s = np.amax(im, -1)
        im_c = np.amax(im, 1)
    else:
        # im is im_s
        im_s = im

    vert_labels = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7',
                   'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10', 'T11', 'T12',
                   'L1', 'L2', 'L3', 'L4', 'L5']

    font = {'family': 'monospace',
            'color': 'white',
            'style': 'normal',
            'weight': 'bold',
            'size': 9}

    plt.subplot(121)
    plt.imshow(im_s)
    chs_active = np.argwhere(~np.isnan(cents[:, 0]))
    for [ch] in chs_active:
        loc = cents[ch, [0, 1]].astype(int)
        plt.plot([loc[1]], [loc[0]], marker='x', markersize=5)
        plt.text(loc[1], loc[0] - 4, vert_labels[ch], fontdict=font)
    plt.subplot(122)
    plt.imshow(im_c)
    chs_active = np.argwhere(~np.isnan(cents[:, 0]))
    for [ch] in chs_active:
        loc = cents[ch, [0, 2]].astype(int)
        plt.plot([loc[1]], [loc[0]], marker='x', markersize=5)
        plt.text(loc[1], loc[0] - 4, vert_labels[ch], fontdict=font)
    plt.show()


# ---------------------------
# Visualization Localization
# ---------------------------
def plot_segmentation(im, msk):
    """
    Plot image and segmentation
    :param im: image
    :param msk: segmentation mask
    :return:
    """
    plt.subplot(121)
    plt.imshow(np.amax(im, -1))
    msk_mip = np.amax(msk, -1)
    plt.imshow(msk_mip, cmap='summer')
    plt.subplot(122)
    plt.imshow(np.amax(im, 1))
    msk_mip = np.amax(msk, 1)
    plt.imshow(msk_mip, cmap='winter')
    plt.show()

