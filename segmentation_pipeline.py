import os
import torch
import argparse
import numpy as np
import nibabel as nib
import utils_segmentation as utils
from matplotlib import pyplot as plt
from dipy.align.reslice import reslice
from nets.localization_network import LocalizationNet
from nets.labelling_network import LabellingNet
from nets.segmentation_network import SegmentationNet
from helper import reorient_nib


class SpineSegmentation(object):

    def __init__(self, model_dir, pat_path, save_path=None):
        """
        Initialize models and directories
        :param model_dir: dir of models for all 3 stages
        :param pat_path: dir of a single patient nifti scan
        :param save_path: dir to save the segmentation nifti mask
        """
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cpu')

        self.localization_net = LocalizationNet().to(self.device)
        self.labelling_net = LabellingNet().to(self.device)
        self.segmentation_net = SegmentationNet().to(self.device)

        state_dict_loc = torch.load(os.path.join(model_dir, 'localization_model.pt'),
                                    map_location=self.device)
        state_dict_lab = torch.load(os.path.join(model_dir, 'labelling_model.pt'),
                                    map_location=self.device)
        state_dict_seg = torch.load(os.path.join(model_dir, 'segmentation_model.pt'),
                                    map_location=self.device)

        self.localization_net.load_state_dict(state_dict_loc)
        self.labelling_net.load_state_dict(state_dict_lab)
        self.segmentation_net.load_state_dict(state_dict_seg)

        self.pat_dir = pat_path
        self.save_dir = save_path

        # downsample to new voxel size in mm
        self.scale_loc = 4.
        self.scale_lab = 2.
        self.scale_seg = 1.

        #print('Models initialized')

    def load_img(self, scale):
        """
        Load, reorient, resample, transpose image for inference
        :param scale: downsampling to new voxel size in mm
        :return: preprocessed image array
        """
        im = utils.load_nib(self.pat_dir)
        #self.orig_orientation = nib.aff2axcodes(im.affine)
        im = reorient_nib(im, new_orient='PIR')
        im = utils.resample_nib(im, new_spacing=(scale, scale, scale), order=3)
        im_arr = im.get_fdata()
        im_arr = utils.transpose_compatible(im_arr, direction='asl_to_np')
        im_arr = im_arr / 2048.
        im_arr[im_arr < -1] = -1
        im_arr[im_arr > 1] = 1

        return im_arr, im.affine, im.header

    def save_mask_nib(self, im_orig, mask_1mm):
        """
        Save segmentation mask based on the original image nifti attributes
        :param im_orig: original image nifti
        :param im_1mm: image nifti at 1 mm
        :param msk_1mm: segmentation mask array at 1 mm
        """
        new_spacing = im_orig.header.get_zooms()
        mask_arr = mask_1mm.get_fdata()
        mask_zooms = mask_1mm.header.get_zooms()        
        msk_affine = mask_1mm.affine
        #msk_arr = utils.transpose_compatible(msk_1mm, direction='np_to_asl')
        #msk_zooms = (1., 1., 1.)
        # resample using DIPY.ALIGN
        new_vox_arr, new_vox_affine = reslice(mask_arr, msk_affine, mask_zooms, new_spacing, order=0)
        # adjust for differences in last dimension
        if im_orig.get_fdata().shape != new_vox_arr.shape:
            if im_orig.get_fdata().shape[2] < new_vox_arr.shape[2]:
                new_vox_corrected = new_vox_arr[:,:,1:]
            elif im_orig.get_fdata().shape[2] > new_vox_arr.shape[2]:
                new_vox_corrected = np.zeros(im_orig.get_fdata().shape)
                new_vox_corrected[:,:,1:] = new_vox_arr
            new_vox_arr = new_vox_corrected

        new_vox_arr = new_vox_arr.astype(np.int8)
        # create resampled image
        im_orig.set_sform(new_vox_affine)
        new_im = nib.Nifti1Image(new_vox_arr, new_vox_affine, im_orig.header)
        nib.save(new_im, self.save_dir)
        print('Segmentation saved at: ', self.save_dir)


    def localize(self, im):
        """
        Localize spine in a 3d CT scan
        :param im: 3d CT scan voxel size 4 mm
        :return: 3d bounding box coordinates voxel size 4 mm
        """
        im_tensor = torch.FloatTensor(im)
        # (N,H,W,D) -> (N,C,H,W,D) -> (N,C,D,H,W)
        im_tensor = im_tensor.unsqueeze(0).unsqueeze(1).permute(0, 1, 4, 2, 3).to(self.device).type(torch.float32)
        # inference
        hm_tensor = self.localization_net.predict(im_tensor)
        # (N,C,D,H,W) -> (H,W,D)
        im = im_tensor.permute(0, 1, 3, 4, 2)[0, 0, :, :].detach().cpu().numpy()
        hm = hm_tensor.permute(0, 1, 3, 4, 2)[0, 0, :, :].detach().cpu().numpy()
        # 3d largest connected component
        hm_clean = utils.clean_hm_prediction(hm, 0.5)
        # heatmap prediction to bounding box
        box = utils.msk_2_box(hm_clean, 0.5)

        print('Spine localized')

        return box #box_4mm

    @staticmethod
    def plot_localization(im, box):
        """
        Visualize bounding box
        """
        utils.plot_box(im, box)

    def label(self, im, box_pred):
        """
        Localize and label vertebrae centroids
        :param im: 3d CT scan voxel size 2 mm
        :param box: 3d bounding box voxel size 4 mm
        :return: vertebrae centroids coordinates at 2 mm
        """
        # upsample bounding box to 2 mm
        [h_min, h_max, w_min, w_max, d_min, d_max] = tuple([2 * x for x in box_pred])
        box_pred = [h_min, h_max, w_min, w_max, d_min, d_max]
        # crop volume to adjusted bounding box
        im_cropped, box_background, box_tolerance = utils.adjust_box(box_pred, im)
        # MIPs sagittal and coronal
        im_s = np.amax(im_cropped, -1).astype(float)
        im_c = np.amax(im_cropped, 1).astype(float)
        im_s_tensor = torch.FloatTensor(im_s)
        im_c_tensor = torch.FloatTensor(im_c)
        # (N,H,W) -> (N,C,H,W)
        im_s_tensor = im_s_tensor.unsqueeze(0).unsqueeze(1).to(self.device).type(torch.float32)
        im_c_tensor = im_c_tensor.unsqueeze(0).unsqueeze(1).to(self.device).type(torch.float32)
        # inference
        pred_s, pred_c = self.labelling_net.predict(im_s_tensor, im_c_tensor)
        # (N,C,H,W) -> (H,W)
        im_s = im_s_tensor[0, 0, :, :].detach().cpu().numpy()
        im_c = im_c_tensor[0, 0, :, :].detach().cpu().numpy()
        # (N,C,H,W) -> (N,H,W,C) -> (H,W,C)
        pred_s = pred_s.permute(0, 2, 3, 1)[0, ...].detach().cpu().numpy()
        pred_c = pred_c.permute(0, 2, 3, 1)[0, ...].detach().cpu().numpy()
        # threshold
        pred_s[pred_s < 0.1] = 0
        pred_c[pred_c < 0.1] = 0
        # convert mask to centroid list
        pred_3d = utils.masks_2d_to_3d(pred_s, pred_c)
        cents_2mm = utils.mask_to_centroids(pred_3d, verts_in_im=np.arange(1, 25))
        #cents_pred[cents_pred == 0] = np.nan
        cents_2mm[cents_2mm == 0] = np.nan
        print('Spine labelled')

        return cents_2mm, box_background, box_tolerance

    @staticmethod
    def plot_labelling(im, pred_cents):
        """
        Visualize centroids
        """
        plt.figure(figsize=(10, 10))
        utils.plot_labels(im=im, cents=pred_cents)

    def segment(self, im, box_background, box_tolerance, cents):
        """
        Segment vertebrae
        :param im: 3d CT scan voxel size 1 mm
        :param box: 3d bounding box voxel size 4 mm
        :return: vertebrae segmentations voxel size 1 mm
        """
        # initial mask with zeros
        h, w, d = im.shape
        final_mask = np.zeros((h, w, d)).astype(int)
        # upsample cents to 1 mm
        cents_1mm = cents * 2

        # upsample background box to 1 mm
        [w_min, w_max, d_min, d_max] = [2 * x for x in box_background] #tuple([2 * x for x in box_background])
        # translate centroids to original cropped image
        for cent in cents_1mm:
            cent[1] -= w_min
            cent[2] -= d_min

        # upsample bounding box to 1 mm
        [h_min, h_max, w_min, w_max, d_min, d_max] = [2 * x for x in box_tolerance] #tuple([2 * x for x in box_tolerance])
        # translate centroid coordinates to original image
        for cent in cents_1mm:
            cent[0] += h_min
            cent[1] += w_min
            cent[2] += d_min

        # calculate padding area (offsets)
        c_off = np.array([[-50, 50], [-50, 80], [-50, 50]])
        t_off = np.array([[-50, 50], [-50, 80], [-50, 50]])
        l_off = np.array([[-50, 50], [-50, 80], [-70, 70]])

        verts_in_im = np.argwhere(~np.isnan(cents_1mm[:, 0])) + 1

        # for every vertebrae
        for vert_idx in verts_in_im:

            # vertebrae centroid
            loc = cents_1mm[vert_idx - 1, :].astype(int)[0]

            if loc[0] < 2 or loc[0] > (h-2):
                continue
            if loc[1] < 2 or loc[1] > (w-2):
                continue
            if loc[2] < 2 or loc[2] > (d-2):
                continue

            # cervical offset
            if vert_idx < 8:
                off = c_off
            # thoracic offset
            elif vert_idx < 20:
                off = t_off
            # lumbar offset
            else:
                off = l_off

            # get patch, gaussian, limits and pads
            patch, gauss, lims, pads = utils.get_seg_patch(im, loc, off)
            # transform for inference
            patch = patch.astype(float)
            gauss = gauss.astype(float)
            patch = torch.FloatTensor(patch)
            gauss = torch.FloatTensor(gauss)
            # (N,H,W,D) -> (N,C,H,W,D) -> (N,C,D,H,W)
            patch = patch.unsqueeze(0).unsqueeze(1).permute(0, 1, 4, 2, 3).to(self.device).type(torch.float32)
            gauss = gauss.unsqueeze(0).unsqueeze(1).permute(0, 1, 4, 2, 3).to(self.device).type(torch.float32)
            # inference
            pred = self.segmentation_net.predict(patch, gauss)
            # (N,C,D,H,W) -> (H,W,D)
            patch = patch.permute(0, 1, 3, 4, 2)[0, 0, :, :].detach().cpu().numpy()
            pred = pred.permute(0, 1, 3, 4, 2)[0, 0, :, :].detach().cpu().numpy()
            # postprocessing
            msk = utils.refine_mask(pred, 0.5)
            msk = msk.astype(int)
            msk_cropped = utils.crop_seg_patch(msk, pads)
            # translate to full spine mask
            [h_min, h_max, w_min, w_max, d_min, d_max] = lims
            mask_temp = np.zeros(final_mask.shape)
            mask_temp[h_min:h_max, w_min:w_max, d_min:d_max] = msk_cropped
            final_mask[mask_temp == 1] = vert_idx

        print('Spine segmented')

        return final_mask

    def apply(self):
        """
        Apply localization, labelling and segmentation pipeline
        :return: image and segmentation array at 1 mm
        """
        # downsample image to 4 mm for localization
        im_localize, _, _ = self.load_img(self.scale_loc)
        # get spine bounding box at 4 mm
        box_4mm = self.localize(im_localize)
        # downsample image to 2 mm for labelling
        im_label, _, _ = self.load_img(self.scale_lab)
        # get centroid coordinates at 2 mm
        cents_2mm, box_background, box_tolerance = self.label(im_label, box_4mm)
        # downsample image to 1 mm for segmentation
        im_seg, im_seg_affine, im_seg_header = self.load_img(self.scale_seg)
        # get final mask
        mask_1mm = self.segment(im_seg, box_background, box_tolerance, cents_2mm)
        # get original nifti information
        im_original = utils.load_nib(self.pat_dir)
        orient_orig = nib.aff2axcodes(im_original.affine)
        mask_1mm = utils.transpose_compatible(mask_1mm, direction='np_to_asl')
        new_mask = nib.Nifti1Image(mask_1mm, im_seg_affine, im_seg_header)
        new_mask = reorient_nib(new_mask, new_orient=''.join(orient_orig))
        #im_1mm = reorient_nib(im_original, new_orient='')
        #im_1mm = utils.resample_nib(im_1mm, new_spacing=(self.scale_seg, self.scale_seg, self.scale_seg), order=3)
        if self.save_dir:
            self.save_mask_nib(im_original, new_mask) #, im_1mm, mask_1mm)
        else:
            self.save_dir = './temp_seg_mask.nii.gz'
            self.save_mask_nib(im_original, new_mask) #im_1mm, mask_1mm)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Spine Segmentation Pipeline')
    parser.add_argument('--model_dir', default='models')
    parser.add_argument('--pat_dir', default='sub-kypho005/post_fracture/ct.nii.gz')
    parser.add_argument('--save_dir', default='sub-kypho005/post_fracture/mask_auto.nii.gz')
    args = parser.parse_args()
    args_dict = vars(args)

    segment = SpineSegmentation(args_dict['model_dir'], args_dict['pat_dir'], args_dict['save_dir'])
    _ = segment.apply()
