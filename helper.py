import os
import numpy as np
import nibabel as nib
from  dcmstack import reorder_voxels
from dipy.align.reslice import reslice

def load_scan(path_name, orientation='RIP', resample=False):
    '''
    Load a scan, reorder to the defined orientation and return the image array along with the header and affine matrix
    :param path_name: The path to the scan we want to load
    :param orientation: The orientation to which scan will be re-oriented
    :param resample: If true also resample the scan so it has the same resolution in all three dims - sample as lateral resolution
    :return img_data: The numpy array of the image
    :return header: The header information
    :return affine: The affine matrix of the scan
    :return original_zooms: If resample is True return also the original resolution of the scan
    '''
    img = nib.load(path_name)
    imgc = reorient_nib(img, orientation)
    original_zooms = imgc.header.get_zooms() #RIP so lateral, axial, coronal
    if resample:
        img_resampled = resample_nib(imgc, new_spacing=(original_zooms[1], original_zooms[1], original_zooms[1]))
        return img_data.get_fdata(), img_resampled.header, img_resampled.affine, original_zooms
    else:
        return imgc.get_fdata(), imgc.header, imgc.affine

def return_scan_to_orig(img_array, img_affine, img_header, img_zooms, datatype=np.float32):
    '''
    Given a scan which haas been resampled and reoriented return it to the original orientation and resolution
    :param img_array: numpy array of image
    :param img_affine: the affine matrix of the scan
    :param img_header: the header of the image
    :param img_zooms: the resolution of the scan in x,y,z we want to have
    :param datatype: the data type of the image data
    :return img: nib scan re
    '''
    img = nib.Nifti1Image(img_array, img_affine, img_header)
    img_resampled = resample_nib(img, new_spacing=img_zooms)
    img_resampled.header.set_data_dtype(datatype)
    img = reorient_nib(img_resampled, new_orient='PIL')
    return img

def compute_vol(mask, vertebra, zooms):
    '''
    Given a segmentation mask, the pixel spacing and vertebra of interest, this function calculates 
    the volume of that vertebra and returns the computed value in cm^3
    :param mask: numpy array of mask
    :param vertebra: id of vertebra for which we want to compute volume
    :param zooms: resolution of scan in x,y,z
    :return vertebra_volume: volume of vertebra uder consideration
    '''
    voxel_vol = zooms[0] * zooms[1] * zooms[2] # in mm
    vert_pixels = np.where(mask==vertebra)
    num_pixels = vert_pixels[0].shape[0]
    vertebra_volume = num_pixels * voxel_vol # in mm^3
    vertebra_volume = vertebra_volume/1000 # in cm^3
    return vertebra_volume

def get_vert_range(mask, vertebra_fracture_id):
    """
    Compute the cement injected during surgery
    :param mask: numpy array of mask
    :param vertebra_fracture_id: int, fractured vertebra label which has undergone kyphoplasty
    :return vert_range: a list with the vertebrae we want to crop for straightening and inpainting
                        must be at least five and if possible to above and two below fractured vertebra
    """
    mask_vertebrae = np.unique(mask)
    if ((vertebra_fracture_id-2) in mask_vertebrae) and ((vertebra_fracture_id+2) in mask_vertebrae):
        vert_range = [vertebra_fracture_id-2, vertebra_fracture_id-1, vertebra_fracture_id, vertebra_fracture_id+1, vertebra_fracture_id+2]    
    elif ((vertebra_fracture_id-3)in mask_vertebrae) and ((vertebra_fracture_id+1) in mask_vertebrae):
        vert_range = [vertebra_fracture_id-3, vertebra_fracture_id-2, vertebra_fracture_id-1, vertebra_fracture_id, vertebra_fracture_id+1]    
    elif ((vertebra_fracture_id-1) in mask_vertebrae) and ((vertebra_fracture_id+3)in mask_vertebrae):
        vert_range = [vertebra_fracture_id-1, vertebra_fracture_id, vertebra_fracture_id+1, vertebra_fracture_id+2, vertebra_fracture_id+3]    
    elif (vertebra_fracture_id-4) in mask_vertebrae:
        vert_range = [vertebra_fracture_id-4, vertebra_fracture_id-3, vertebra_fracture_id-2, vertebra_fracture_id-1, vertebra_fracture_id]    
    elif (vertebra_fracture_id+4) in mask_vertebrae:
        vert_range = [vertebra_fracture_id, vertebra_fracture_id+1, vertebra_fracture_id+2, vertebra_fracture_id+3, vertebra_fracture_id+4]    
    else:
        raise Exception("Not a range of five vertebra including fracture have been found in scan. The inpainting algorithm \
                        requires at least five vertebrae present in image. Please check the scan or the generated segmentation \
                        mask.")
    return vert_range

def compute_cement(ct_data, mask_data, fracture, ct_zooms):
    """
    Compute the cement injected during surgery
    :param ct_data: numpy array of post-op ct scan
    :param mask_data: numpy array of post-op mask
    :param fracture: int, fractured vertebra label which has undergone kyphoplasty
    :param ct_zooms: resolution of scan in x,y,z
    :return vol_cement: the volume of cement in cm^3
    """
    where_mask = np.where(mask_data==fracture)
    max_x, min_x = np.max(where_mask[0])+50, np.min(where_mask[0])-50
    max_y, min_y = np.max(where_mask[1])+50, np.min(where_mask[1])-50
    max_z, min_z = np.max(where_mask[2])+50, np.min(where_mask[2])-50
    ct_data = ct_data[min_x:max_x, min_y:max_y, min_z:max_z]
    cement_mask = np.zeros(ct_data.shape)
    cement_mask[ct_data>1600] = 1
    vol_cement = compute_vol(cement_mask, 1, ct_zooms)
    return vol_cement

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
    header.set_zooms(new_spacing)
    # create resampled image
    new_im = nib.Nifti1Image(new_vox_arr, new_vox_affine, header)
    return new_im

def reorient_nib(im, new_orient='RIP', datatype = np.float64):
    """
    Reorient nifti voxel array and corresponding affine
    ----------
    :param im: nifti image
    :param new_orient: A three character code specifying the desired starting point for rows, columns, and slices
                       in terms of the orthogonal axes of patient space:
                       (L)eft, (R)ight, (A)nterior, (P)osterior, (S)uperior, and (I)nferior
    :param datatype: specify the data type of return image
    :return new_im: reoriented nifti image
    """
    header = im.header
    vox_arr = im.get_fdata()
    vox_affine = im.affine
    # reorient using DCMStack
    new_vox_arr, new_vox_affine, _, _ = reorder_voxels(vox_arr, vox_affine, new_orient)
    # specify datatype
    new_vox_arr = new_vox_arr.astype(datatype)
    header.set_data_dtype(datatype)
    # create reoriented NIB image
    new_im = nib.Nifti1Image(new_vox_arr, new_vox_affine, header)

    return new_im