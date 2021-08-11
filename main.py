import os
import argparse
import numpy as np
import napari

from helper import compute_vol, load_scan, get_vert_range, compute_cement
from segmentation_pipeline import SpineSegmentation
from spine_straighten import SpineStraighten
from inpaint import Inpaint
from pedicle_detection import PedicleDetection

def get_args():
    parser = argparse.ArgumentParser(description='Bone Cement Planning Pipeline')
    parser.add_argument('--patient_dir', help='The patient directory including the fractured scan')
    parser.add_argument('--fracture', type=int, help='The label of the fractured vertebra')
    parser.add_argument('--healthy', action='store_true', help='Set to true if a pre-fracture scan is also \
    available in the patient directory')
    parser.add_argument('--post_op', action='store_true', help='Set to true if a post-op scan is also available \
    in the patient directory')
    parser.add_argument('--height_scale', default=1, type=int, help='The height scale used for the spine straightening step.')
    parser.add_argument('--visualize', action='store_true', help='Set to true if you want to visualize outputs')
    parser.add_argument('--save', action='store_true', help='If set to true intermediate scans, e.g. straightened, inpainted are saved')
    return parser.parse_args()

def run(args):
    patient_dir = args.patient_dir
    vertebra_fracture_id = args.fracture

    list_dir = [file for file in os.listdir(patient_dir) if os.path.isdir(os.path.join(patient_dir, file))]
    # directories, pre-fractured, fractured, post-op must be organised data-wise
    dates = []
    for sub_dir in list_dir:
        name_split = sub_dir.split('_')[0]
        name_split = name_split.split('-')[-1]
        dates.append(int(name_split))

    list_dir=[x for _,x in sorted(zip(dates,list_dir))]
    if not args.healthy:
        fractured_dir = os.path.join(patient_dir, list_dir[0])
    else:
        fractured_dir = os.path.join(patient_dir, list_dir[1])

    files = [file for file in os.listdir(fractured_dir) if (file.endswith('.nii.gz') or file.endswith('.nii'))]
    ct_file = [file for file in files if 'ct' in file][0]
    patient_scan_path = os.path.join(fractured_dir, ct_file)

    # ________SEGMENTATION PART________ #
    vertebra_mask_path = os.path.join(fractured_dir, 'vert-mask.nii')
    # if this step has not be done already compute vertebra segmentation
    if not os.path.isfile(vertebra_mask_path):
        segmentation = SpineSegmentation('./models', patient_scan_path, vertebra_mask_path)
        segmentation.apply()

    # load vertebra mask
    img, img_header, _ = load_scan(patient_scan_path)
    img_spacing = img_header.get_zooms()
    vertebra_mask, mask_header, _ = load_scan(vertebra_mask_path)
    mask_spacing = mask_header.get_zooms()

    # ________PEDICLE DETECTION PART (FRACTURED) ________ #

    # define path to saved pedicle detected inpainted mask
    segmentation_pd_mask_path = os.path.join(patient_dir, f'segmentation_pd_{vertebra_fracture_id}.nii')
    segmentation_pd_cent_path = f'{patient_scan_path[:-10]}_seg-subreg_ctd.json'
    if not os.path.isfile(segmentation_pd_mask_path):
        pedicle_detection = PedicleDetection(patient_dir, patient_scan_path, vertebra_mask_path,
                                             segmentation_pd_cent_path, vertebra_fracture_id, mode='segmentation')
        pedicle_detection.apply()

    # load vertebra mask with pedicle detection applied and compute volume
    vertebra_pd_mask, mask_pd_header, _ = load_scan(segmentation_pd_mask_path)
    pd_mask_spacing = mask_pd_header.get_zooms()
    fractured_volume = compute_vol(vertebra_pd_mask, vertebra_fracture_id, pd_mask_spacing)

    # ________SPINE STRAIGHTENING PART________ #  
    
    # to generate the displacement field and do resampling, the healthy_scan must be provided                                             
    h_mask_path = './data/healthy_ref_mask.nii.gz'
    # specify the vertebrae to include in crop
    vert_range = get_vert_range(vertebra_mask, vertebra_fracture_id)
    print('Cropping scan in vertebra range', vert_range)
    # define path to save straightened scan and mask
    straighten_scan_path = os.path.join(patient_dir, 'str_scan.nii.gz')
    straighten_mask_path = os.path.join(patient_dir, 'str_mask.nii.gz')
    # if this step has not be done already perform straightening
    if not os.path.isfile(straighten_scan_path):
        spine_straighten = SpineStraighten(h_mask_path, vertebra_mask_path, patient_scan_path, vertebra_fracture_id, vert_range, scale_factor=args.height_scale)
        final_displacement_field = spine_straighten.run()
        spine_straighten.straighten_spine(final_displacement_field=final_displacement_field,
                                        straight_scan_name=straighten_scan_path,
                                        straight_mask_name=straighten_mask_path,
                                        whole_spine=True)
    # for visualizing the straightened spine
    if args.visualize:
        straighten_scan_arr, straighten_scan_header, _ = load_scan(straighten_scan_path)
        straighten_scan_spacing = straighten_scan_header.get_zooms()
        straighten_mask_arr, straighten_mask_header, _ = load_scan(straighten_mask_path)
        straighten_mask_spacing = straighten_mask_header.get_zooms()
    
    # ________INPAINTING PART________ # 

    # define path to save inpainted scan and mask
    inpaint_mask_path = os.path.join(patient_dir, 'inpaint_mask_fuse_%s.nii.gz'%(vertebra_fracture_id))
    inpaint_img_path =  os.path.join(patient_dir, 'inpaint_img_%s.nii.gz'%(vertebra_fracture_id)) 
    # if this step has not be done already perform inpainting
    if not os.path.isfile(inpaint_mask_path):
        inpainting = Inpaint(straighten_scan_path, straighten_mask_path, vertebra_fracture_id, inpaint_img_path,
                             inpaint_mask_path)
        inpainting.apply(mode='fuse') # fuse is for using lateral and coronal models and fusing results

    # ________PEDICLE DETECTION PART (INPAINTED) ________ #

    # define path to saved pedicle detected inpainted mask
    inpaint_pd_mask_path = os.path.join(patient_dir, f'inpaint_pd_{vertebra_fracture_id}.nii')
    inpaint_pd_cent_path = f'{patient_dir}/{patient_dir[2:]}_inpaint_seg-subreg_ctd.json'
    if not os.path.isfile(inpaint_pd_mask_path):
        pedicle_detection = PedicleDetection(patient_dir, inpaint_img_path, inpaint_mask_path,
                                             inpaint_pd_cent_path, vertebra_fracture_id, mode='inpaint')
        pedicle_detection.apply()

    inpainted_mask, inpaint_mask_header, _ = load_scan(inpaint_pd_mask_path)
    inpaint_mask_spacing = inpaint_mask_header.get_zooms()
    inpainted_volume = compute_vol(inpainted_mask, vertebra_fracture_id, inpaint_mask_spacing)

    # compare inpainted volume with pre-fracture vertebra volume
    print('Computed volume of fractured vertebra: ', round(fractured_volume, 3), 'cm^3')
    print('Computed volume of inpainted vertebra: ', inpainted_volume, ' cm^3')
    print('Difference of fractured and inpainted: ', inpainted_volume  - fractured_volume, 'cm^3')
    
    # calculate cement upped bound
    max_cement_volume = inpainted_volume - fractured_volume
    print('Upper bound of cement to be injected is: ', max_cement_volume)
    
    # if the scan before the fracture is available compare it with inpainting result
    if args.healthy:
        healthy_dir = os.path.join(patient_dir, list_dir[0])
        
        files = [file for file in os.listdir(healthy_dir) if (file.endswith('.nii.gz') or file.endswith('.nii'))]
        ct_file = [file for file in files if 'ct' in file][0]

        # define paths and compute mask
        healthy_img_path = os.path.join(healthy_dir, ct_file)
        healthy_vertebra_mask_path = os.path.join(healthy_dir, 'vert-mask.nii')

        if not os.path.isfile(healthy_vertebra_mask_path):
            segmentation = SpineSegmentation('./models', healthy_img_path, healthy_vertebra_mask_path)
            segmentation.apply()

        # define path to saved pedicle detected healthy mask
        healthy_pd_mask_path = os.path.join(patient_dir, f'healthy_pd_{vertebra_fracture_id}.nii')
        healthy_pd_cent_path = f'{healthy_img_path[:-10]}_seg-subreg_ctd.json'
        if not os.path.isfile(healthy_pd_mask_path):
            pedicle_detection = PedicleDetection(patient_dir, healthy_img_path, healthy_vertebra_mask_path,
                                                 healthy_pd_cent_path, vertebra_fracture_id, mode='healthy')
            pedicle_detection.apply()
        
        # load img and mask
        healthy_img, healthy_img_header, _ = load_scan(healthy_img_path)
        healthy_img_spacing = healthy_img_header.get_zooms()
        healthy_vertebra_mask, healthy_mask_header, _ = load_scan(healthy_pd_mask_path)
        healthy_mask_spacing = healthy_mask_header.get_zooms()
        # compute vertebra volume
        healthy_vertebra_volume = compute_vol(healthy_vertebra_mask, vertebra_fracture_id, healthy_mask_spacing)
        print('Computed volume of healthy (pre-fractured) vertebra: ', round(healthy_vertebra_volume,3), 'cm^3')
        print('Difference of healthy and inpainted: ', (inpainted_volume - healthy_vertebra_volume), 'cm^3')
        #print('Difference of healthy and fractured: ', (healthy_vertebra_volume - fractured_volume), 'cm^3') 

    # if the scan after the fracture is available compare cement volume with computed upper bound
    if args.post_op: 
        if not args.healthy:
            postop_dir = os.path.join(patient_dir, list_dir[1])
        else:
            postop_dir = os.path.join(patient_dir, list_dir[2])
        
        files = [file for file in os.listdir(postop_dir) if (file.endswith('.nii.gz') or file.endswith('.nii'))]
        ct_file = [file for file in files if 'ct' in file][0]

        # define paths and compute mask
        postop_img_path = os.path.join(postop_dir, ct_file)
        postop_vertebra_mask_path = os.path.join(postop_dir, 'vert-mask.nii')

        if not os.path.isfile(postop_vertebra_mask_path):
            segmentation = SpineSegmentation('./models', postop_img_path, postop_vertebra_mask_path)
            segmentation.apply()

        # load img and mask and compute cement
        postop_img, postop_img_header, _ = load_scan(postop_img_path)
        postop_img_spacing = postop_img_header.get_zooms()
        postop_vertebra_mask, postop_mask_header, _ = load_scan(postop_vertebra_mask_path)
        postop_mask_spacing = postop_mask_header.get_zooms()
        cement_vol = compute_cement(postop_img, postop_vertebra_mask, vertebra_fracture_id, postop_img_spacing)
        print('Computed amount of cement in post-op image: ', cement_vol, 'cm^3')
    
    # ________VISUALIZATION PART________ #
    '''
    Warning: This may be too heavy to run - consider removing some scans from the visualisation
    '''
    if args.visualize:    
        
        with napari.gui_qt():
    
            viewer = napari.Viewer()
            viewer.add_image(img, name='fracture', scale=img_spacing)
            viewer.add_labels(vertebra_mask.astype(int), name='fractured_segmentation', scale=mask_spacing)
            viewer.add_image(straighten_scan_arr, name='straightened', scale=straighten_scan_spacing)
            viewer.add_labels(straighten_mask_arr.astype(int), name='straightened_segmentation', scale=straighten_mask_spacing)
            viewer.add_labels(inpainted_mask.astype(int), name='inpainted_segmentation', scale=inpaint_mask_spacing)
            
            if args.healthy:
                viewer.add_image(healthy_img, name='healthy', scale=healthy_img_spacing)
                viewer.add_labels(healthy_vertebra_mask.astype(int), name='healthy-segmentation', scale=healthy_mask_spacing)
            if args.post_op: 
                viewer.add_image(postop_img, name='post-op', scale=postop_img_spacing)
                viewer.add_labels(postop_vertebra_mask.astype(int), name='post-op-segmentation', scale=postop_mask_spacing)
    
    # remove any temporarily created files
    if not args.save:
        #os.remove(vertebra_mask_path)
        os.remove(straighten_scan_path)
        os.remove(straighten_mask_path)
        os.remove(inpaint_mask_path)
        os.remove(inpaint_img_path)
        if args.healthy:
            os.remove(healthy_vertebra_mask_path)
        if args.post_op:
            os.remove(postop_vertebra_mask_path)


if __name__ == '__main__':
    args = get_args()
    run(args)


