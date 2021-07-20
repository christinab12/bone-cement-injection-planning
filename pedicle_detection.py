import nilearn.image
import numpy as np
import pandas as pd
import gc
import matplotlib.pyplot as plt
import SimpleITK as sitk
import scipy.ndimage
from sklearn.decomposition import PCA
from skimage.measure import block_reduce
from nilearn import image
from scipy.interpolate import interp1d
from scipy.ndimage import affine_transform
from scipy.misc import derivative
import sys
import cv2
import math

import matplotlib.patches
import matplotlib.pyplot as plt

import math
import sys

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches
import SimpleITK as sitk
import cv2
from skimage.feature import canny

from data_utilities import *


class PedicleDetection():

    def __init__(self, ):
        self.SLICES_PER_VERTEBRA = 5

    def apply(self, img_iso, msk_iso, sub_iso, ctd_iso, scan, COUNTER, metrics_file,
                                     only_eval=False):
        """
        General pipeline for spine straightening and hough transform
        """
        detected_valid_shapes = True
        if only_eval:
            msk_np = msk_iso.get_data()
        else:
            ctd_iso_df = pd.DataFrame(ctd_iso[1:], columns=['vert', ctd_iso[0][0], ctd_iso[0][1], ctd_iso[0][2]])
            f = self.interpolation(img_iso, msk_iso, ctd_iso, 'sag')
            principal_components = self.pca(ctd_iso_df)
            principal_components = self.rescale(principal_components, ctd_iso_df, img_iso.shape)
            new_centroids = self.create_centroids(img_iso, msk_iso, ctd_iso, principal_components)
            diff = self.create_diff(ctd_iso, new_centroids)
            msk_np, COUNTER, detected_valid_shapes = self.transform_hough(img_iso, msk_iso, ctd_iso, new_centroids, diff, f,
                                                                     scan, COUNTER, metrics_file)
        return msk_np

    """
    STRAIGHTENING-HOUGH
    """

    def pca(self, ctd_iso_df):
        """
        Get first principal component from positions of the centroids
        """
        vert = ctd_iso_df['vert']
        # Input data for the PCA: only coordinates, no vertebra labels
        ctd_iso_df_pca = ctd_iso_df.iloc[:, 1:]
        # Perform PCA
        pca = PCA(n_components=1)
        principal_components = pca.fit_transform(ctd_iso_df_pca)
        principal_components = pd.DataFrame(data=principal_components, columns=['principal component 1'])
        principal_components.insert(0, 'vert', vert)
        return principal_components

    def rescale(self, principal_components, ctd_iso_df, shape):
        """
        Re-scale the principal component positions to match the axis of the scan
        """
        # Position in z-axis of first and last centroid in original scan
        first, last = ctd_iso_df.iloc[0][1], ctd_iso_df.iloc[-1][1]
        original_diff = last - first
        # Position in z-axis of first and last centroid in principal components
        first_pca, last_pca = principal_components.iloc[0][1], principal_components.iloc[-1][1]
        pca_diff = last_pca - first_pca
        # Proportion
        prop = original_diff / pca_diff
        # Re-scale
        principal_components['pos'] = [0.0] * principal_components.shape[0]
        principal_components.loc[0, 'pos'] = first
        for i in range(1, principal_components.shape[0]):
            diff = principal_components.loc[i, 'principal component 1'] - principal_components.loc[
                i - 1, 'principal component 1']
            principal_components.loc[i, 'pos'] = principal_components.loc[i - 1, 'pos'] + diff * prop
        return principal_components

    def create_centroids(self, img_iso, msk_iso, ctd_iso, principal_components):
        """
        Create list of new centroids aligned according to the PCA on the axial plane and in the middle of
        the sagittal plane
        """
        # Get voxel data
        im_np = img_iso.get_fdata()
        mid_sag = int(im_np.shape[2] / 2)
        mid_cor = int(im_np.shape[1] / 2)
        pos = principal_components['pos']
        new_centroids = np.array(ctd_iso.copy()[1:]).astype('float')
        new_centroids[:, 1] = pos
        new_centroids[:, 2] = mid_cor
        new_centroids[:, 3] = mid_sag
        new_centroids = new_centroids.astype('int')
        new_centroids = new_centroids.tolist()
        new_centroids.insert(0, ('I', 'P', 'L'))
        return new_centroids

    def create_diff(self, ctd_iso, new_centroids):
        """
        Create array of differences between original centroids and PCA centroids
        """
        ctd_iso = np.array(ctd_iso[1:])
        new_centroids = np.array(new_centroids[1:])
        diff = ctd_iso - new_centroids
        return np.delete(diff, 0, 1)

    def create_translation(self, diff_arr):
        """
        Translation based on offsets
        """
        aff = np.identity(4)
        aff[:3, 3] = diff_arr
        return aff

    def apply_affine_to_numpy(self, np_arr, aff, lower, upper, interpolation='linear'):
        """
        Apply an affine transformation to a 3-D-numpy-array
        """
        aff_inv = np.linalg.inv(aff)
        new_arr = np.zeros_like(np_arr).astype(np.int)
        for i in range(int(lower), int(upper) + 1):
            for j in range(new_arr.shape[1]):
                for k in range(new_arr.shape[2]):
                    i_p, j_p, k_p, _ = aff_inv.dot([i, j, k, 1])
                    if interpolation == 'nn':
                        i_p, j_p, k_p = round(i_p), round(j_p), round(k_p)
                        if i_p >= int(upper):
                            i_p = int(upper) - 1
                        if i_p < 0:
                            i_p = 0
                        if j_p >= np_arr.shape[1]:
                            j_p = np_arr.shape[1] - 1
                        if j_p < 0:
                            j_p = 0
                        if k_p >= np_arr.shape[2]:
                            k_p = np_arr.shape[2] - 1
                        if k_p < 0:
                            k_p = 0
                        if np_arr[i_p][j_p][k_p] != 0:
                            new_arr[i][j][k] = np_arr[i_p][j_p][k_p]
                    if interpolation == 'linear':
                        if i_p >= int(upper) - 1:
                            i_p = int(upper) - 1 - math.exp(-6)
                        if i_p < 0:
                            i_p = 0
                        if j_p >= np_arr.shape[1] - 1:
                            j_p = np_arr.shape[1] - 1 - math.exp(-6)
                        if j_p < 0:
                            j_p = 0
                        if k_p >= np_arr.shape[2] - 1:
                            k_p = np_arr.shape[2] - 1 - math.exp(-6)
                        if k_p < 0:
                            k_p = 0

                        x_0 = math.floor(i_p)
                        x_1 = math.ceil(i_p)
                        if x_1 == x_0:
                            x_1 += 1
                        y_0 = math.floor(j_p)
                        y_1 = math.ceil(j_p)
                        if y_1 == y_0:
                            y_1 += 1
                        z_0 = math.floor(k_p)
                        z_1 = math.ceil(k_p)
                        if z_1 == z_0:
                            z_1 += 1

                        # Difference between point and coordinates on the lattice
                        x_d = (i_p - x_0) / (x_1 - x_0)
                        y_d = (j_p - y_0) / (y_1 - y_0)
                        z_d = (k_p - z_0) / (z_1 - z_0)

                        # Interpolate along x
                        c_000 = np_arr[x_0][y_0][z_0]
                        c_001 = np_arr[x_0][y_0][z_1]
                        c_010 = np_arr[x_0][y_1][z_0]
                        c_011 = np_arr[x_0][y_1][z_1]
                        c_100 = np_arr[x_1][y_0][z_0]
                        c_101 = np_arr[x_1][y_0][z_1]
                        c_110 = np_arr[x_1][y_1][z_0]
                        c_111 = np_arr[x_1][y_1][z_1]

                        if c_000 == 0 and c_001 == 0 and c_010 == 0 and c_011 == 0 \
                                and c_100 == 0 and c_101 == 0 and c_110 == 0 and c_111 == 0:
                            continue

                        c_00 = c_000 * (1 - x_d) + c_100 * x_d
                        diff = int(c_100 - c_000)
                        if diff != 0:
                            c_00 = round(c_00 / diff) * diff
                        else:
                            c_00 = round(c_00)
                        c_01 = c_001 * (1 - x_d) + c_101 * x_d
                        diff = int(c_101 - c_001)
                        if diff != 0:
                            c_01 = round(c_01 / diff) * diff
                        else:
                            c_01 = round(c_01)
                        c_10 = c_010 * (1 - x_d) + c_110 * x_d
                        diff = int(c_110 - c_010)
                        if diff != 0:
                            c_10 = round(c_10 / diff) * diff
                        else:
                            c_10 = round(c_10)
                        c_11 = c_011 * (1 - x_d) + c_111 * x_d
                        diff = int(c_111 - c_011)
                        if diff != 0:
                            c_11 = round(c_11 / diff) * diff
                        else:
                            c_11 = round(c_11)

                        # Interpolate along y
                        c_0 = c_00 * (1 - y_d) + c_10 * y_d
                        diff = int(c_10 - c_00)
                        if diff != 0:
                            c_0 = round(c_0 / diff) * diff
                        else:
                            c_0 = round(c_0)
                        c_1 = c_01 * (1 - y_d) + c_11 * y_d
                        diff = int(c_11 - c_01)
                        if diff != 0:
                            c_1 = round(c_1 / diff) * diff
                        else:
                            c_1 = round(c_1)

                        # Interpolate along z
                        c = c_0 * (1 - z_d) + c_1 * z_d
                        diff = int(c_1 - c_0)
                        if diff != 0:
                            c = round(c / diff) * diff
                        else:
                            c = round(c)

                        new_arr[i][j][k] = c
        return new_arr

    def create_rotation(self, f, pos, pivot, plane):
        """
        Rotation based on gradient of interpolation polynomial
        """
        d = derivative(f, pos, n=1)
        # Convert derivative to angle in radian
        rad = math.atan(d / 1)
        if plane == 'sag':
            trans_1 = np.identity(4)
            trans_1[:2, 3] = -pivot
            rot = np.array([[math.cos(-rad), -math.sin(-rad), 0, 0],
                            [math.sin(-rad), math.cos(-rad), 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])
            trans_2 = np.identity(4)
            trans_2[:2, 3] = pivot
        elif plane == 'cor':
            trans_1 = np.identity(4)
            trans_1[[0, 2], 3] = -pivot
            rot = np.array([[math.cos(-rad), 0, math.sin(-rad), 0],
                            [0, 1, 0, 0],
                            [-math.sin(-rad), 0, math.cos(-rad), 0],
                            [0, 0, 0, 1]])
            trans_2 = np.identity(4)
            trans_2[[0, 2], 3] = pivot
        return trans_2.dot(rot.dot(trans_1))

    def get_slices(self, img, lower, upper):
        """
        Get a certain number of axial slices for a given vertebra (based on volume per slice)
        """
        # Store number of found voxels per axial slide
        num_pixels = []
        # Loop over all axial slices
        for i in range(int(lower), int(upper) + 1):
            slice = self.crop_2D_image(img, i)
            voxel_count = np.count_nonzero(slice)
            if voxel_count > 0:
                num_pixels.append((i, voxel_count))
        # Get top x slides in terms of voxel number
        num_pixels.sort(key=lambda x: x[1], reverse=True)
        num_pixels = [x[0] for x in num_pixels[:self.SLICES_PER_VERTEBRA]]
        return num_pixels

    def evaluate(self, a, b, c, d):
        """
        Check if given parameters for body/canal are valid
        :return: True if valid, False if invalid
        """
        if c == 0 or d == 0:
            return False
        return True

    def get_downsampling_factor(self, height, width):
        """
        Get appropriate downsampling-factor (width should not be bigger than 30)
        :param width:
        :param height:
        :return:
        """
        if width <= 35:
            return 1
        return int(width / 35)

    def hough(self, list_slices, img_mask, vert, lower, upper, COUNTER):
        """
        Perform Hough transform for a number of axial slices of a given vertebra
        """
        # Parameters for the found ellipses (distinguished between body and canal detection)
        params_body = []
        params_canal = []
        for index_slice, x in enumerate(list_slices):
            # Create 2D-slices
            tmp = self.crop_2D_image(img_mask, x)
            # Fit bounding box around vertebra
            bound_x, bound_y, bound_w, bound_h, bound_c_x, bound_c_y = self.bounding_box(tmp)
            downsampling_factor = self.get_downsampling_factor(bound_h, bound_w)
            tmp = block_reduce(tmp, block_size=(downsampling_factor, downsampling_factor), func=np.max)
            bound_x = int(bound_x / downsampling_factor)
            bound_y = int(bound_y / downsampling_factor)
            bound_w = int(bound_w / downsampling_factor)
            bound_h = int(bound_h / downsampling_factor)
            bound_c_x = int(bound_c_x / downsampling_factor)
            bound_c_y = int(bound_c_y / downsampling_factor)
            # Crop images to bounding box
            tmp_ROI = tmp[bound_y:bound_y + bound_h, bound_x:bound_x + bound_w]
            # Width and height of ROI
            tmp_width = tmp_ROI.shape[1]
            tmp_height = tmp_ROI.shape[0]
            # Utility variables
            perc_x = tmp_width / 100
            perc_y = tmp_height / 100
            middle_x = round(tmp_width / 2)
            middle_y = round(tmp_height / 2)
            """
            Parameters for body
            """
            if vert in list(range(4)):  # Cervical
                params_body_ab = round(tmp_width * 0.01 / 2), round(tmp_width * 0.93 / 2), round(
                    tmp_height * 0.01 / 2), round(
                    (tmp_height * 0.53) / 2)
                params_body_cd = round(bound_c_x - 10 * perc_x), round(bound_c_x + 10 * perc_x), \
                                 1, round(bound_c_y * (7 / 8))
            elif vert in list(range(4, 8)):  # Cervical
                params_body_ab = round(tmp_width * 0.24 / 2), round(tmp_width * 0.93 / 2), round(
                    tmp_height * 0.06 / 2), round(
                    (tmp_height * 0.53) / 2)
                params_body_cd = round(bound_c_x - 10 * perc_x), round(bound_c_x + 10 * perc_x), \
                                 1, round(bound_c_y * (7 / 8))
            elif vert in list(range(8, 10)):  # Thoracic 8-9
                params_body_ab = round(tmp_width * 0.27 / 2), round(tmp_width * 0.82 / 2), round(
                    tmp_height * 0.10 / 2), round(
                    (tmp_height * 0.70) / 2)
                params_body_cd = round(bound_c_x - 10 * perc_x), round(bound_c_x + 10 * perc_x), round(
                    1 / 3 * bound_c_y), round(
                    bound_c_y * (7 / 8))
            elif vert in list(range(10, 18)):  # Thoracic 10-17
                params_body_ab = round(tmp_width * 0.23 / 2), round(tmp_width * 0.85 / 2), round(
                    tmp_height * 0.13 / 2), round(
                    (tmp_height * 0.70) / 2)
                params_body_cd = round(bound_c_x - 10 * perc_x), round(bound_c_x + 10 * perc_x), round(
                    1 / 3 * bound_c_y), round(
                    bound_c_y * (7 / 8))
            elif vert in list(range(18, 20)):  # Thoracic 18-19
                params_body_ab = round(tmp_width * 0.47 / 2), round(tmp_width * 0.96 / 2), round(
                    tmp_height * 0.26 / 2), round(
                    (tmp_height * 0.56) / 2)
                params_body_cd = round(bound_c_x - 10 * perc_x), round(bound_c_x + 10 * perc_x), round(
                    1 / 3 * bound_c_y), round(
                    bound_c_y * (7 / 8))
            elif vert in list(range(20, 24)):  # Lumbar 20-23
                params_body_ab = round(tmp_width * 0.28 / 2), round(tmp_width * 0.98 / 2), round(
                    tmp_height * 0.14 / 2), round(
                    (tmp_height * 0.62) / 2)
                params_body_cd = round(bound_c_x - 10 * perc_x), round(bound_c_x + 10 * perc_x), round(
                    1 / 3 * bound_c_y), round(
                    bound_c_y * (7 / 8))
            elif vert >= 24:  # Lumbar 24
                params_body_ab = round(tmp_width * 0.37 / 2), round(tmp_width * 0.78 / 2), round(
                    tmp_height * 0.15 / 2), round(
                    (tmp_height * 0.57) / 2)
                params_body_cd = round(bound_c_x - 10 * perc_x), round(bound_c_x + 10 * perc_x), round(
                    1 / 3 * bound_c_y), round(
                    bound_c_y * (7 / 8))
            else:
                params_body_ab = round(tmp_width * (1 / 4) / 2), round(tmp_width / 2), 1, round(
                    (tmp_height * (3 / 4)) / 2)
                params_body_cd = round(bound_c_x - 10 * perc_x), round(bound_c_x + 10 * perc_x), round(
                    1 / 3 * bound_c_y), round(
                    bound_c_y * (7 / 8))
            # Additional constraint
            params_body_ab = max(params_body_ab[0], 1), max(params_body_ab[1], 1), max(params_body_ab[2], 1), max(
                params_body_ab[3], 1)
            """
            Find body
            """
            hough_space, d_body, c_body, a_body, b_body = self.hough_transform_ellipse_filled(tmp_ROI, params_body_ab,
                                                                                         params_body_cd, tmp_width,
                                                                                         tmp_height, vert,
                                                                                         to_detect='body',
                                                                                         c_x=bound_c_x, c_y=bound_c_y)
            """
            Parameters for canal
            """
            if vert in list(range(8)):  # Cervical
                params_canal_ab = round((tmp_width * 0.18) / 2), round((tmp_width * 0.45) / 2), round(
                    (tmp_height * 0.11) / 2), round((tmp_height * 0.42) / 2)
                params_canal_cd = round(c_body - 20 * perc_x), round(c_body + 20 * perc_x), \
                                  round(d_body + b_body + 1), round(bound_c_y + (tmp_height - 2 - bound_c_y) * (3 / 5))
            elif vert in list(range(8, 10)):  # Thoracic 8-9
                params_canal_ab = round((tmp_width * 0.09) / 2), round((tmp_width * 0.45) / 2), round(
                    (tmp_height * 0.14) / 2), round((tmp_height * 0.33) / 2)
                params_canal_cd = round(c_body - 8 * perc_x), round(c_body + 8 * perc_x), \
                                  round(d_body + b_body + 1), round(bound_c_y + (tmp_height - 2 - bound_c_y) * (3 / 5))
            elif vert in list(range(10, 18)):  # Thoracic 10-17
                params_canal_ab = round((tmp_width * 0.09) / 2), round((tmp_width * 0.47) / 2), round(
                    (tmp_height * 0.08) / 2), round((tmp_height * 0.38) / 2)
                params_canal_cd = round(c_body - 8 * perc_x), round(c_body + 8 * perc_x), \
                                  round(d_body + b_body + 1), round(bound_c_y + (tmp_height - 2 - bound_c_y) * (4 / 5))
            elif vert in list(range(18, 20)):  # Thoracic 18-19
                params_canal_ab = round((tmp_width * 0.16) / 2), round((tmp_width * 0.46) / 2), round(
                    (tmp_height * 0.08) / 2), round((tmp_height * 0.33) / 2)
                params_canal_cd = round(c_body - 8 * perc_x), round(c_body + 8 * perc_x), \
                                  round(d_body + b_body + 1), round(bound_c_y + (tmp_height - 2 - bound_c_y) * (4 / 5))
            elif vert in list(range(20, 24)):  # Lumbar 20-23
                params_canal_ab = round((tmp_width * 0.05) / 2), round((tmp_width * 0.46) / 2), round(
                    (tmp_height * 0.06) / 2), round((tmp_height * 0.41) / 2)
                params_canal_cd = round(c_body - 8 * perc_x), round(c_body + 8 * perc_x), \
                                  round(d_body + b_body + 1), round(bound_c_y + (tmp_height - 2 - bound_c_y) * (4 / 5))
            elif vert >= 24:  # Lumbar 24
                params_canal_ab = round((tmp_width * 0.09) / 2), round((tmp_width * 0.36) / 2), round(
                    (tmp_height * 0.05) / 2), round((tmp_height * 0.48) / 2)
                params_canal_cd = round(c_body - 8 * perc_x), round(c_body + 8 * perc_x), \
                                  round(d_body + b_body + 1), round(bound_c_y + (tmp_height - 2 - bound_c_y) * (4 / 5))
            else:
                params_canal_ab = 1, round((tmp_width * 0.8) / 2), round((tmp_height * (1 / 15)) / 2), round(
                    (tmp_height * 0.5) / 2)
            params_canal_ab = max(params_canal_ab[0], 1), max(params_canal_ab[1], 1), max(params_canal_ab[2], 1), max(
                params_canal_ab[3], 1)

            """
            Find canal
            """
            hough_space, d_canal, c_canal, a_canal, b_canal = self.hough_transform_ellipse_filled(tmp_ROI, params_canal_ab,
                                                                                             params_canal_cd, tmp_width,
                                                                                             tmp_height, vert,
                                                                                             inverted=True,
                                                                                             lowest_body=d_body + b_body,
                                                                                             body_height=b_body * 2,
                                                                                             to_detect='canal')
            """
            Show found ellipses
            """
            self.show_ellipses([(a_body, b_body, c_body, d_body), (a_canal, b_canal, c_canal, d_canal)], tmp_ROI,
                          pedicle_detection=False, save=True, dir=f'run/visualized/{COUNTER}',
                          center=(bound_c_x, bound_c_y))

            COUNTER += 1
            self.show_ellipses([(a_body, b_body, c_body, d_body), (a_canal, b_canal, c_canal, d_canal)], tmp_ROI,
                          pedicle_detection=True, save=True, dir=f'run/visualized/{COUNTER}',
                          center=(bound_c_x, bound_c_y))
            COUNTER += 1
            if self.evaluate(a_body, b_body, c_body, d_body) and \
                    self.evaluate(a_canal, b_canal, c_canal, d_canal):
                # Parameters are considered valid -> store them and multiply by downsampling-factor
                params_body.append(((
                    a_body * downsampling_factor, b_body * downsampling_factor,
                    (bound_x + c_body) * downsampling_factor,
                    (bound_y + d_body) * downsampling_factor)))
                params_canal.append((a_canal * downsampling_factor, b_canal * downsampling_factor,
                                     (bound_x + c_canal) * downsampling_factor,
                                     (bound_y + d_canal) * downsampling_factor))
        return params_body, params_canal, COUNTER

    def bounding_box(self, img):
        """
        Bounding box around vertebra in axial slice
        Function taken from https://stackoverflow.com/questions/21104664/extract-all-bounding-boxes-using-opencv-python
        """
        src = np.where(img == 0, 255, 0)
        src = src.astype(np.uint8)
        original = src.copy()
        thresh = cv2.threshold(src, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # Find contours, obtain bounding boxes
        cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        # Find bounding box surrounding all objects
        glob_ul, glob_ur, glob_ll, glob_lr = None, None, None, None
        # Filter out bounding boxes that are too small
        cnts_new = [c for c in cnts if cv2.boundingRect(c)[2] > 10]
        if len(cnts_new) != 0:
            cnts = cnts_new
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            ul = x, y
            ur = x + w, y
            ll = x, y + h
            lr = x + w, y + h
            if glob_ul is None:
                glob_ul = ul[0], ul[1]
                glob_ur = ur[0], ur[1]
                glob_ll = ll[0], ll[1]
                glob_lr = lr[0], lr[1]
            else:
                glob_ul = min(ul[0], glob_ul[0]), min(ul[1], glob_ul[1])
                glob_ur = max(ur[0], glob_ur[0]), min(ur[1], glob_ur[1])
                glob_ll = min(ll[0], glob_ll[0]), max(ll[1], glob_ll[1])
                glob_lr = max(lr[0], glob_lr[0]), max(lr[1], glob_lr[1])
        glob_x, glob_y = min(glob_ul[0], glob_ll[0]), min(glob_ul[1], glob_ur[1])
        glob_x0, glob_y0 = max(glob_ur[0], glob_lr[0]), max(glob_ll[1], glob_lr[1])
        glob_w, glob_h = glob_x0 - glob_x, glob_y0 - glob_y
        ROI = original[glob_y:glob_y + glob_h, glob_x:glob_x + glob_w]

        thresh = cv2.threshold(ROI, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        # Calculate moments of binary image
        M = cv2.moments(thresh)

        # Calculate x,y coordinate of center
        centroid_X = int(M['m10'] / M['m00'])
        centroid_Y = int(M['m01'] / M['m00'])

        return glob_x, glob_y, glob_w, glob_h, centroid_X, centroid_Y

    def fit_cylinder(self, ellipse_params):
        """
        Fit elliptical cylinder to a set of ellipses
        """
        # Simple method to fit elliptical cylinder (minimizes mean-squared distances between lowermost point
        # and rightmost point of target cylinder and all ellipses)
        sum_m = np.zeros(2)
        sum_n = np.zeros(2)
        for a, b, c, d in ellipse_params:
            m = np.array([c + a, d])
            n = np.array([c, d + b])
            sum_m += m
            sum_n += n
        sum_m /= len(ellipse_params)
        sum_n /= len(ellipse_params)

        c = sum_n[0]
        d = sum_m[1]
        a = sum_m[0] - c
        b = sum_n[1] - d
        # Return parameters of cylinder
        return a, b, c, d

    def relabel(self, msk, vert_index, lower, upper, valid, m=None, b=None):
        """
        Re-label mask according to the plain detected during pedicle detection
        """
        for i in range(int(lower), int(upper) + 1):
            slice = self.crop_2D_image(msk, i)
            if valid:
                res = np.fromfunction(lambda j, k: k * m + b < j, (msk.shape[1], msk.shape[2]))
                res = np.where(res == True, 5, vert_index)
                bool_mask = np.where(slice == 0, False, True)
                res = np.where(bool_mask == False, 0, res)
                res = np.where(res == 5, 0, res)
                msk[i] = res
            else:
                msk[i] = np.where(slice != 0, vert_index, 0)

    def count_pixels_body(self, msk, type):
        """
        Counts the number of pixels belonging to vertebral bodies
        :param msk:
        :param type: type of input msk (as labeling differs)
        'sub': ground truth mask from Anduin
        'gen': generated mask from pipeline
        :return:
        """
        if type == 'sub':
            return len(np.where(msk == 49 or msk == 50))
        elif type == 'gen':
            return len(np.where(msk == 50))

    def count_pixels_processes(self, msk, type):
        """
        Counts the number of pixels belonging to spinal processes
        :param msk:
        :param type: type of input msk (as labeling differs)
        'sub': ground truth mask from Anduin
        'gen': generated mask from pedicle detection pipeline
        :return:
        """
        if type == 'sub':
            return len(np.where(msk in list(range(41, 49))))
        elif type == 'gen':
            return len(np.where(msk == 41))

    def count_pixels_no_spine(self, msk):
        """
        Counts the number of pixels that do not belong to the spine (in the case of masks, background images)
        :param msk:
        :param type: type of input msk (as labeling differs)
        'sub': ground truth mask from Anduin
        'gen': generated mask from pipeline
        :return:
        """
        return msk.shape[0] * msk.shape[1] * msk.shape[2] - np.count_nonzero(msk)

    def performance_metrics(self, pred, ground_truth):
        """
        Calculates performance metrics based on the vertebral bodies
        :param pred: predicted segmentation mask from pedicle detection pipeline
        :param ground_truth: segmentation mask from Anduin
        :return:
        """
        pred = np.where(pred != 0, 50, 0)
        ground_truth = np.where((ground_truth == 49) | (ground_truth == 50), 50, 100)
        matching = np.count_nonzero(ground_truth == pred)
        only_pred = (pred == 50).sum()
        only_gt = (ground_truth == 50).sum()
        dice = (2 * matching) / (only_gt + only_pred)
        intersection_over_union = matching / (only_gt + only_pred - matching)
        return dice, intersection_over_union

    def scale_centroids(self, ctr_list, scaling_factor):
        """
        Scale list of centroids according to scaling factor
        """
        for i in range(1, len(ctr_list)):
            ctr_list[i] = [ctr_list[i][0]] + [x * scaling_factor for x in ctr_list[i][1:]]

    def transform_hough(self, img_iso, msk_iso, ctd_iso, new_centroids, diff, f, scan, COUNTER, metrics_file):
        """
        Transformation (Rotation) of each vertebra and hough transform
        """
        # List to store transformation for each vertebra
        transformations = []
        org_shape = msk_iso.shape
        zooms = img_iso.header.get_zooms()
        # Get voxel data
        im_np = img_iso.get_fdata()
        mid_sag = int(im_np.shape[2] / 2)
        mid_cor = int(im_np.shape[1] / 2)
        # Did the algorithm succeed?
        detected_valid_shapes = True
        for v_index, v in enumerate(ctd_iso[1:]):
            detected_valid_shapes = False
            vert = v[0]
            pos = v[1]
            # Filter mask to current vertebra
            msk_iso_new = nib.Nifti1Image(np.where(msk_iso.get_fdata() == vert, vert, 0), msk_iso.affine,
                                          msk_iso.header)
            # Rotation to straighten vertebra
            rot_1 = self.create_rotation(f, pos, np.array([v[1], v[2]]), 'sag')
            # Optimization: Set upper and lower border for the transformation
            if v_index == 0:
                upper = ctd_iso[v_index + 3][1]
            elif v_index >= len(ctd_iso) - 3:
                upper = msk_iso_new.get_fdata().shape[0] - 1
            else:
                upper = ctd_iso[v_index + 3][1]
            if v_index in [0, 1]:
                lower = 0
            else:
                lower = ctd_iso[v_index - 1][1]
            # Define 4x4-affine matrix
            affine = rot_1
            inv_affine = np.linalg.inv(affine)

            msk_np_new = msk_iso_new.get_data()
            """
            Visualization of vertebra before rotation (not re-labeled)
            """
            # Get the mid-slice of the scan and mask in both sagittal and coronal planes
            msk_np_sag = msk_np_new[:, :, mid_sag]
            msk_np_cor = msk_np_new[:, mid_cor, :]
            # Plot
            fig, axs = create_figure(96, msk_np_sag, msk_np_cor)
            axs[0].imshow(msk_np_sag, cmap=cm_itk, alpha=0.3, vmin=1, vmax=64)
            plot_sag_centroids(axs[0], new_centroids, zooms)
            axs[1].imshow(msk_np_cor, cmap=cm_itk, alpha=0.3, vmin=1, vmax=64)
            plot_cor_centroids(axs[1], new_centroids, zooms)
            fig.savefig(f'run/visualized/{COUNTER}')
            COUNTER += 1

            msk_iso_new = nib.Nifti1Image(msk_np_new, msk_iso.affine, msk_iso.header)
            msk_iso_new = nilearn.image.resample_img(msk_iso_new, target_affine=msk_iso_new.affine.dot(inv_affine),
                                                     target_shape=msk_iso_new.shape, interpolation='nearest')
            # Current vertebra volume
            msk_np_new = msk_iso_new.get_data()
            self.print_vert_volume(msk_np_new, vert, metrics_file)
            msk_iso_new_hough = nib.Nifti1Image(np.where(msk_np_new == vert, 1, 0), msk_iso_new.affine,
                                                msk_iso_new.header)

            c = 0
            hole_filling_factor = 0.05
            while True:
                # Search for ellipses until enough valid assignments were found
                if c != 0:
                    # If there was already an unsuccessful run of hough transforms, there are likely holes in the ellipse
                    # -> hole-filling procedure
                    width = im_np.shape[2]
                    tmp = sitk.GetImageFromArray(msk_iso_new_hough.get_data())
                    mc = sitk.BinaryMorphologicalClosingImageFilter()
                    radius = round((width * hole_filling_factor) / 2)
                    mc.SetKernelRadius(radius)
                    tmp = mc.Execute(tmp)

                    tmp = sitk.GetArrayFromImage(tmp)
                    msk_iso_new_hough = nib.Nifti1Image(tmp, msk_iso_new_hough.affine, msk_iso_new_hough.header)

                """
                Hough transform
                """
                # Slices to perform the hough transform
                list_slices = self.get_slices(msk_iso_new_hough.get_data(), lower, upper)
                # Hough transform
                params_body, params_canal, COUNTER = self.hough(list_slices, msk_iso_new_hough.get_data(), vert, lower,
                                                           upper,
                                                           COUNTER)
                if len(params_body) >= 2:
                    # We successfully found an assignment with the hough transform
                    # Constraint can be varied: for instance, we could demand at least 2 valid ellipses
                    detected_valid_shapes = True
                    break
                if hole_filling_factor > 0.10:
                    # We did not find a successful assignment, even after morphological closing
                    break
                COUNTER -= 10
                if c != 0:
                    hole_filling_factor += 0.01
                c += 1
            if detected_valid_shapes:
                a_body, b_body, c_body, d_body = self.fit_cylinder(params_body)
                a_canal, b_canal, c_canal, d_canal = self.fit_cylinder(params_canal)
                m, b = self.pedicle_detection([(a_body, b_body, c_body, d_body), (a_canal, b_canal, c_canal, d_canal)],
                                         msk_np_new)
                self.relabel(msk_np_new, vert, lower, upper, True, m=m, b=b)
            else:
                self.relabel(msk_np_new, vert, lower, upper, False)

            msk_np_new = msk_iso_new.get_data()
            """
            Visualization of the vertebra after rotation (re-labeled)
            """
            # Get the mid-slice of the scan and mask in both sagittal and coronal planes
            msk_np_sag = msk_np_new[:, :, mid_sag]
            msk_np_cor = msk_np_new[:, mid_cor, :]
            # Plot
            fig, axs = create_figure(96, msk_np_sag, msk_np_cor)
            axs[0].imshow(msk_np_sag, cmap=cm_itk, alpha=0.3, vmin=1, vmax=64)
            plot_sag_centroids(axs[0], new_centroids, zooms)
            axs[1].imshow(msk_np_cor, cmap=cm_itk, alpha=0.3, vmin=1, vmax=64)
            plot_cor_centroids(axs[1], new_centroids, zooms)
            fig.savefig(f'run/visualized/{COUNTER}')
            COUNTER += 1

            """
            Rotate vertebra back to original position
            """
            msk_iso_new = nib.Nifti1Image(msk_np_new, msk_iso_new.affine, msk_iso_new.header)
            msk_iso_new = nilearn.image.resample_img(msk_iso_new, target_affine=msk_iso_new.affine.dot(affine),
                                                     target_shape=msk_iso_new.shape, interpolation='nearest')
            msk_np_new = msk_iso_new.get_data().astype(np.uint8)
            """
            Visualization of the vertebra in original position (re-labeled)
            """
            # Get the mid-slice of the scan and mask in both sagittal and coronal planes
            msk_np_sag = msk_np_new[:, :, mid_sag]
            msk_np_cor = msk_np_new[:, mid_cor, :]
            # Plot
            fig, axs = create_figure(96, msk_np_sag, msk_np_cor)
            axs[0].imshow(msk_np_sag, cmap=cm_itk, alpha=0.3, vmin=1, vmax=64)
            plot_sag_centroids(axs[0], new_centroids, zooms)
            axs[1].imshow(msk_np_cor, cmap=cm_itk, alpha=0.3, vmin=1, vmax=64)
            plot_cor_centroids(axs[1], new_centroids, zooms)
            fig.savefig(f'run/visualized/{COUNTER}')
            COUNTER += 1

            # Append re-labeled array to global array
            transformations.append(msk_np_new)
            sys.stdout.write(f'Transformed vertebra {vert}\n')
        transformations = np.array(transformations)
        transformed = np.sum(transformations, axis=0)
        """
        Visualization of global mask
        """
        # Get the mid-slice of the scan and mask in both sagittal and coronal planes
        msk_np_sag = transformed[:, :, mid_sag]
        msk_np_cor = transformed[:, mid_cor, :]
        # Plot
        fig, axs = create_figure(96, msk_np_sag, msk_np_cor)
        axs[0].imshow(msk_np_sag, cmap=cm_itk, alpha=0.3, vmin=1, vmax=64)
        plot_sag_centroids(axs[0], new_centroids, zooms)
        axs[1].imshow(msk_np_cor, cmap=cm_itk, alpha=0.3, vmin=1, vmax=64)
        plot_cor_centroids(axs[1], new_centroids, zooms)
        fig.savefig(f'run/visualized/{COUNTER}')
        COUNTER += 1
        return transformed, COUNTER, True

    def interpolation(self, img_iso, msk_iso, ctd_iso, plane, kind=1):
        """
        Interpolate through centroids to compute rotation
        """
        # Get "middle" in the sagittal/coronal plane
        msk_np = msk_iso.get_data()
        mid_sag = int(msk_np.shape[2] / 2)
        mid_cor = int(msk_np.shape[1] / 2)
        # Distances from centroids to middle in the coronal/sagittal direction
        if plane == 'sag':
            ctd_cor = np.array(ctd_iso[1:])[:, 2]
            dist = ctd_cor - mid_cor
        elif plane == 'cor':
            ctd_sag = np.array(ctd_iso[1:])[:, 3]
            dist = ctd_sag - mid_sag
        x = np.array(ctd_iso[1:])[:, 1]
        # Create interpolation function
        f = interp1d(x, dist, kind=kind, fill_value='extrapolate')
        return f

    def non_zero_entries(self, arr):
        """
        Non-zero entries of a numpy-array
        """
        return np.count_nonzero(arr)

    def print_vert_volume(self, msk_np, vert, metrics_file):
        """
        Print the volume of a given vertebra in mm^3
        """
        volume = self.non_zero_entries(msk_np)
        sys.stdout.write('-----------------------------------------------\n')
        metrics_file.write('-----------------------------------------------\n')
        sys.stdout.write(f'\t\t\tVolume of vertebra {vert}: {volume}\n')
        metrics_file.write(f'\t\t\tVolume of vertebra {vert}: {volume}\n')
        sys.stdout.write('-----------------------------------------------\n')
        metrics_file.write('-----------------------------------------------\n')

    def compare_volumes(self, msk_np, sub_np, metrics_file):
        """
        Compare volumes of the spine between two masks
        """
        after = self.non_zero_entries(msk_np)
        before = self.non_zero_entries(sub_np)
        sys.stdout.write('-----------------------------------------------\n')
        metrics_file.write('-----------------------------------------------\n')
        sys.stdout.write(f'Number of voxels before transformation: {before}\n')
        metrics_file.write(f'Number of voxels before transformation: {before}\n')
        sys.stdout.write(f'Number of voxels after transformation: {after}\n')
        metrics_file.write(f'Number of voxels after transformation: {after}\n')
        relative_error = (max(before, after) - min(before, after)) / max(before, after)
        sys.stdout.write(f'Relative error: {relative_error}\n')
        metrics_file.write(f'Relative error: {relative_error}\n')
        sys.stdout.write('-----------------------------------------------\n')
        metrics_file.write('-----------------------------------------------\n')
        return relative_error

    def print_performance_metrics(self, dice, IoU, metrics_file):
        """
        Print performance metrics (dice coefficient, IoU)
        """
        sys.stdout.write('-----------------------------------------------\n')
        metrics_file.write('-----------------------------------------------\n')
        sys.stdout.write('Performance metrics:\n')
        metrics_file.write('Performance metrics:\n')
        sys.stdout.write(f'Dice coefficient: {dice:.2f} ({(dice * 100):.2f} %)\n')
        metrics_file.write(f'Dice coefficient: {dice:.2f} ({(dice * 100):.2f} %)\n')
        sys.stdout.write(f'IoU: {IoU:.2f} ({(IoU * 100):.2f} %)\n')
        metrics_file.write(f'IoU: {IoU:.2f} ({(IoU * 100):.2f} %)\n')
        sys.stdout.write('-----------------------------------------------\n')
        metrics_file.write('-----------------------------------------------\n')
        return dice, IoU

    """
    HOUGH TRANSFORM
    """

    def convert_to_binary(self, path, threshold):
        """
        Converts input image in arbitrary format to binary images containing only black and white pixels
        :return:
        """
        img = cv2.imread(path, 2)
        ret, bw_img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
        return bw_img

    def hough_transform_ellipse_filled(self, input, hyperparams_ab, hyperparams_cd, img_width, img_height, vert,
                                       inverted=False, lowest_body=None, body_height=None, to_detect='body', c_x=None,
                                       c_y=None):
        """
        Performs a hough transform for ellipse detection
        """
        min_a, max_a, min_b, max_b = hyperparams_ab
        min_c, max_c, min_d, max_d = hyperparams_cd

        def hough_accumulation():
            # Initialize Hough space
            hough_space = [[[[0] * (max_b - min_b + 2) for _ in range(max_a - min_a + 2)] for _ in range(img_width)] for
                           _ in range(img_height)]
            # Loop over permitted ellipse shapes
            for a in range(min_a, max_a + 1):
                for b in range(min_b, max_b + 1):
                    # For every ellipse shape, loop over all possible center points
                    for c in range(min_c, max_c + 1):
                        # Break if ellipse is invalid with respect to c and a
                        if c - a < 1 or c + a > img_width - 2:
                            continue
                        for d in range(min_d, max_d + 1):
                            # Break if ellipse is invalid with respect to d and b
                            if d - b < 1 or d + b > img_height - 2:
                                continue
                            if to_detect == 'canal':
                                """
                                If we currently detect spinal canal, ensure that uppermost point is not too far away
                                from lowermost point of body
                                """
                                upper_canal = d - b
                                # Distinguish between different spine regions
                                if vert in list(range(2)):  # Cervical
                                    pass
                                elif vert in list(range(2, 8)):
                                    if upper_canal - lowest_body > body_height * 2 / 5:
                                        continue
                                elif vert in list(range(8, 10)):  # Thoracic 8-9
                                    if upper_canal - lowest_body > body_height * 2 / 5:
                                        continue
                                elif vert in list(range(10, 18)):  # Thoracic 10-17
                                    if upper_canal - lowest_body > body_height * 1 / 5:
                                        continue
                                elif vert in list(range(18, 20)):  # Thoracic 18-19
                                    if upper_canal - lowest_body > body_height * 1 / 5:
                                        continue
                                elif vert in list(range(20, 24)):  # Lumbar 20-23
                                    if upper_canal - lowest_body > body_height * 1 / 5:
                                        continue
                                elif vert >= 24:  # Lumbar 24
                                    if upper_canal - lowest_body > body_height * 1 / 5:
                                        continue
                            # Check if every point on the ellipse has the same color
                            if self.check_ellipse_is_plain(input, a, b, c, d, inverted):
                                if inverted:
                                    # Loop over every point contained in a bounding box defined by the ellipse parameter
                                    # we increase the current accumulator for every point we find that is "black" and in
                                    # the ellipse
                                    white = 0
                                    total = 0
                                    for x in range(c - a, c + a + 1):
                                        for y in range(d - b, d + b + 1):
                                            if self.check_point_in_ellipse(x, y, a, b, c, d):
                                                if input[y][x] == 0:
                                                    hough_space[d][c][a - min_a][b - min_b] += 1
                                                else:
                                                    white += 1
                                                total += 1
                                    # Amount of white pixels inside the ellipse can't exceed a certain threshold
                                    if white / total >= 0.1:
                                        hough_space[d][c][a - min_a][b - min_b] = 0
                                else:
                                    # Loop over every point contained in a bounding box defined by the ellipse parameter
                                    # we increase the current accumulator for every point we find that is "white" and in
                                    # the ellipse
                                    black_above_centroid = 0
                                    black_below_centroid = 0
                                    total = 0
                                    for x in range(c - a, c + a + 1):
                                        for y in range(d - b, d + b + 1):
                                            if self.check_point_in_ellipse(x, y, a, b, c, d):
                                                if input[y][x] != 0:
                                                    hough_space[d][c][a - min_a][b - min_b] += 1
                                                # Only count black pixels below the middle of the ellipse
                                                elif c_y is not None:
                                                    if y >= c_y:
                                                        black_below_centroid += 1
                                                    else:
                                                        black_above_centroid += 1
                                                total += 1
                                    # Amount of black pixels inside the ellipse can't exceed a certain threshold
                                    if black_above_centroid / total >= 0.1 or black_below_centroid / total >= 0.05:
                                        hough_space[d][c][a - min_a][b - min_b] = 0

            return hough_space

        def find_most_promising(hough_space):
            """
            Find the parameters in the Hough space that got most "votes"
            """
            maximum = 0
            max_d_index = 0
            max_c_index = 0
            max_a_index = 0
            max_b_index = 0
            for d_index in range(img_height):
                for c_index in range(img_width):
                    for a_index in range(max_a - min_a):
                        for b_index in range(max_b - min_b):
                            if hough_space[d_index][c_index][a_index][b_index] > maximum:
                                # Determine width and height
                                a = a_index + min_a
                                b = b_index + min_b
                                if vert in list(range(8)):  # Cervical
                                    criterion = False
                                elif vert in list(range(8, 10)):  # Thoracic 8-9
                                    criterion = False
                                elif vert in list(range(10, 18)):  # Thoracic 10-17
                                    criterion = False
                                elif vert in list(range(18, 20)):  # Thoracic 18-19
                                    criterion = False
                                elif vert in list(range(20, 24)):  # Lumbar 20-23
                                    criterion = a < b / 1.00
                                elif vert >= 24:  # Lumbar 24
                                    criterion = a < b / 1.00
                                # Put constraint for form of canal (approximately round)
                                if not (to_detect == 'canal' and criterion):
                                    maximum = hough_space[d_index][c_index][a_index][b_index]
                                    max_d_index = d_index
                                    max_c_index = c_index
                                    max_a_index = a_index
                                    max_b_index = b_index
            return max_d_index, max_c_index, max_a_index + min_a, max_b_index + min_b

        hough_space = hough_accumulation()
        d, c, a, b = find_most_promising(hough_space)
        return hough_space, d, c, a, b

    def show_current_pixel(self, input, x, y):
        fig, ax = plt.subplots()
        fig.set_size_inches(4, 4)
        plt.imshow(input, cmap=plt.cm.Greys_r, aspect='auto')

        ax.plot(x, y, 'or')
        plt.title('Currently examined edge point')
        plt.show()

    def show_all_votes(self, input, votes):
        fig, ax = plt.subplots()
        fig.set_size_inches(4, 4)
        plt.imshow(input, cmap=plt.cm.Greys_r, aspect='auto')
        for x, y in votes:
            ax.plot(x, y, 'or', markersize=1)
        plt.title('Currently examined edge point')
        plt.show()

    def show_current_ellipse(self, input, x, y, d, c, a, b):
        fig, ax = plt.subplots()
        fig.set_size_inches(4, 4)
        plt.imshow(input, cmap=plt.cm.Greys_r, aspect='auto')
        Drawing_uncolored_ellipse = matplotlib.patches.Ellipse((c, d),
                                                               a * 2, b * 2,
                                                               color='r', fill=False)

        ax.add_artist(Drawing_uncolored_ellipse)
        ax.plot(x, y, 'or')
        plt.title(f'Found ellipse with x = {c}, y = {d}, a = {a}, b = {b}')
        plt.show()

    def check_ellipse_is_plain(self, input, a, b, c, d, inverted):
        if a > b:
            # Loop over all possible values of x (those range from d to d+b)
            for x in range(c, c + a + 1):
                # Calculate corresponding value of y
                y = round(d - (b * math.sqrt(a ** 2 - c ** 2 + 2 * c * x - x ** 2)) / a)
                # Check if the whole ellipse if plain on the input image
                if inverted:
                    if input[y][x] != 0 or input[y][c - (x - c)] != 0 or input[d + (d - y)][x] != 0 or \
                            input[d + (d - y)][c - (x - c)] != 0:
                        return False
                else:
                    if input[y][x] == 0 or input[y][c - (x - c)] == 0 or input[d + (d - y)][x] == 0 or \
                            input[d + (d - y)][c - (x - c)] == 0:
                        return False
            return True
        else:
            for y in range(d, d + b + 1):
                # Calculate corresponding value of y
                x = round(c - (a * math.sqrt(b ** 2 - d ** 2 + 2 * d * y - y ** 2)) / b)
                # Check if the whole ellipse if plain on the input image
                if inverted:
                    if input[y][x] != 0 or input[y][c - (x - c)] != 0 or input[d + (d - y)][x] != 0 or \
                            input[d + (d - y)][c - (x - c)] != 0:
                        return False
                else:
                    if input[y][x] == 0 or input[y][c - (x - c)] == 0 or input[d + (d - y)][x] == 0 or \
                            input[d + (d - y)][c - (x - c)] == 0:
                        return False
            return True

    def check_point_in_ellipse(self, x, y, a, b, c, d):
        """
        Determines if a given point x,y is within an ellipse specified by the given parameters a,b,c,d
        """
        if (x - c) ** 2 / a ** 2 + (y - d) ** 2 / b ** 2 <= 1:
            return True
        else:
            return False

    """
    SITK UTILS
    """

    def read_image(self, path, mod):
        itk_img = sitk.ReadImage(path)
        if mod == 'no_mod':
            itk_img = sitk.PermuteAxes(itk_img, [2, 0, 1])
        elif mod in ['mod_1', 'mod_2']:
            itk_img = sitk.PermuteAxes(itk_img, [0, 1, 2])
        # image_viewer.Execute(itk_img)
        return itk_img


    def print_image_information(self, itk_img):
        sys.stdout.write('Information about SimpleITK-image\n')
        sys.stdout.write(f'origin: {str(itk_img.GetOrigin())}\n')
        sys.stdout.write(f'size: {str(itk_img.GetSize())}\n')
        sys.stdout.write(f'spacing: {str(itk_img.GetSpacing())}\n')
        sys.stdout.write(f'direction: {str(itk_img.GetDirection())}\n')
        sys.stdout.write(f'pixel type: {str(itk_img.GetPixelIDTypeAsString())}\n')
        sys.stdout.write(f'number of pixel components: {str(itk_img.GetNumberOfComponentsPerPixel())}\n')


    def convert_img_to_numpy(self, itk_img):
        return sitk.GetArrayFromImage(itk_img)


    def extract_single_vertebra(self, img, label):
        return np.where(img == label, 1, 0)


    def crop_2D_image(self, img, slice_nb):
        # Crop particular 2D image
        img_cropped = img[slice_nb]
        return img_cropped


    def detect_edges(self, img, SHOW_ATLAS=False):
        edges = canny(img, sigma=2.0,
                      low_threshold=0.55, high_threshold=0.8)
        if SHOW_ATLAS:
            fig = plt.figure()
            fig.set_size_inches(4, 4)
            plt.imshow(edges, cmap=plt.cm.Greys_r, aspect='auto')
            plt.show()
        return edges


    def show_ellipses(self, params, msk, pedicle_detection=False, save=False, dir=None, center=None):
        if center is not None:
            # Center of mass
            bound_c_x = center[0]
            bound_c_y = center[1]
        fig, ax = plt.subplots()
        # fig.set_size_inches(4, 8)
        plt.imshow(msk, cmap=plt.cm.Greys_r, aspect='auto')
        # Plot center of mass of bounding box
        plt.plot(bound_c_x, bound_c_y, marker='o', markersize=3, color='red')
        for i in range(len(params)):
            a, b, c, d = params[i]

            Drawing_uncolored_ellipse = patches.Ellipse((c, d),
                                                                   a * 2, b * 2,
                                                                   color='r', fill=False)

            ax.add_artist(Drawing_uncolored_ellipse)
        # We want to plot the lines for pedicle detection
        if pedicle_detection:
            assert len(params) == 2
            # Plot line that connects the two centroids
            x_values = [params[0][2], params[1][2]]
            y_values = [params[0][3], params[1][3]]
            plt.plot(x_values, y_values)
            # Find perpendicular line that goes through centroid of canal
            # Find slope
            if x_values[1] - x_values[0] != 0:
                m = (y_values[0] - y_values[1]) / (x_values[0] - x_values[1])
            else:
                m = math.inf
            # Find negative reciprocal of slope
            nr = -1 / m
            # Find points of canal that serve as intersection point
            a_ellipse, b_ellipse, _, _ = params[1]
            if m == math.inf:
                b = 0
                a = b_ellipse
            else:
                x_1 = - (math.sqrt(a_ellipse ** 2 * b_ellipse ** 2 * (a_ellipse ** 2 * m ** 2 + b_ellipse ** 2))) / (a_ellipse ** 2 * m ** 2 + b_ellipse ** 2)
                x_2 = (math.sqrt(a_ellipse ** 2 * b_ellipse ** 2 * (a_ellipse ** 2 * m ** 2 + b_ellipse ** 2 - 0 ** 2)) - a_ellipse ** 2 * 0 * m) / (a_ellipse ** 2 * m ** 2 + b_ellipse ** 2)
                if m < 0:
                    b = max(x_1, x_2)
                else:
                    b = - min(x_1, x_2)
                a = abs(m * b)
            # Evaluate two points: leftmost point and rightmost point
            x_1, x_2 = 0, msk.shape[1] - 1
            # y_1 = params[1][3] + nr * (x_1 - params[1][2]) - params[1][1]
            # y_2 = params[1][3] + nr * (x_2 - params[1][2]) - params[1][1]
            y_1 = round((params[1][3] - b) + nr * (x_1 - params[1][2]) - a)
            y_2 = round((params[1][3] - b) + nr * (x_2 - params[1][2]) - a)
            x_values, y_values = [x_1, x_2], [y_1, y_2]
            plt.plot(x_values, y_values)
        plt.title('Ellipse')
        if save:
            plt.savefig(dir)
            plt.close('all')
        else:
            plt.show()


    def pedicle_detection(self, params, msk):
        # Plot line that connects the two centroids
        x_values = [params[0][2], params[1][2]]
        y_values = [params[0][3], params[1][3]]
        # Find perpendicular line tangent to canal
        # Find slope
        if x_values[1] - x_values[0] != 0:
            m = (y_values[0] - y_values[1]) / (x_values[0] - x_values[1])
        else:
            m = math.inf
        # Find negative reciprocal of slope
        nr = -1 / m
        # Find points of canal that serve as intersection point
        a_ellipse, b_ellipse, _, _ = params[1]
        if m == math.inf:
            b = 0
            a = b_ellipse
        else:
            x_1 = - (math.sqrt(a_ellipse ** 2 * b_ellipse ** 2 * (a_ellipse ** 2 * m ** 2 + b_ellipse ** 2))) / (
                        a_ellipse ** 2 * m ** 2 + b_ellipse ** 2)
            x_2 = (math.sqrt(a_ellipse ** 2 * b_ellipse ** 2 * (
                        a_ellipse ** 2 * m ** 2 + b_ellipse ** 2 - 0 ** 2)) - a_ellipse ** 2 * 0 * m) / (
                              a_ellipse ** 2 * m ** 2 + b_ellipse ** 2)
            if m < 0:
                b = max(x_1, x_2)
            else:
                b = min(x_1, x_2)
            a = -m * b
        # Evaluate two points: leftmost point and rightmost point
        x_1, x_2 = 0, msk.shape[2] - 1
        # y_1 = params[1][3] + nr * (x_1 - params[1][2]) - params[1][1]
        # y_2 = params[1][3] + nr * (x_2 - params[1][2]) - params[1][1]
        y_1 = (params[1][3] + b) + nr * (x_1 - params[1][2]) - a
        y_2 = (params[1][3] + b) + nr * (x_2 - params[1][2]) - a
        sol = np.linalg.solve(np.array([[x_1, 1], [x_2, 1]]), np.array([y_1, y_2]))
        return sol[0], sol[1]


    def show_circle(self, r, c, d, edges):

        sys.stdout.write(f'We predict the following circle: x = {c}, y = {d}, r = {r}\n')

        fig, ax = plt.subplots()
        fig.set_size_inches(4, 4)
        plt.imshow(edges, cmap=plt.cm.Greys_r, aspect='auto')

        Drawing_uncolored_circle = plt.Circle((c, d),
                                              r,
                                              color='r', fill=True)

        ax.add_artist(Drawing_uncolored_circle)
        plt.title('Circle')
        plt.show()


    def show_image(self, img):
        fig = plt.figure()
        fig.set_size_inches(3, 8)
        plt.imshow(img, cmap=plt.cm.Greys_r, aspect='auto')
        plt.show()


    def dilation(self, img):
        img = img.astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        return cv2.dilate(img, kernel, iterations=1)