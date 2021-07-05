import numpy as np
import os
import gc
import SimpleITK as sitk
import nibabel as nib
from helper import reorient_nib
from typing import List, Union, Tuple, Optional
from helper import preprocess_image, resample_sitk, get_bbox, get_shape_from_bbox


class SpineStraighten():
    def __init__(self, h_mask_path: str, p_mask_path: str, p_scan_path: str, fracture_id: int = None,
                 vert_range: List[int] = None,
                 scale_factor: float = None):
        """
        Initialize SpineStraighten class
        :param h_mask_path: str, path of healthy mask
        :param p_mask_path: str, path of patient mask
        :param p_scan_path: str, path of patient scan
        :param fracture_id: int, the id of fractured vertebra
        :param vert_range: list, the range of interest vertebrae
        :param scale_factor: float, the scaling factor of healthy mask
        """

        h_mask_path = preprocess_image(h_mask_path, "h_mask.nii.gz", new_orient="PIL", datatype=np.int8)
        p_mask_path = preprocess_image(p_mask_path, "p_mask.nii.gz", new_orient="PIL", datatype=np.int8)
        p_scan_path = preprocess_image(p_scan_path, "p_scan.nii.gz", new_orient="PIL", datatype=np.float64)

        p_scan = sitk.ReadImage(p_scan_path, sitk.sitkFloat32)
        p_mask = sitk.ReadImage(p_mask_path)

        h_mask = sitk.ReadImage(h_mask_path)
        if scale_factor is not None:
            new_spacing = tuple(np.array(h_mask.GetSpacing()) * scale_factor)
            h_mask.SetSpacing(new_spacing)

        # resample healthy spine to be same spacing as the patient
        h_mask = resample_sitk(h_mask, p_scan.GetSpacing(), order=0)

        # print("Original Size:", "\nhealthy:", h_mask.GetSize(), "\npatient:", p_scan.GetSize())
        if vert_range is None:
            self.vert_range = None
            self.get_vert_range(h_mask, p_mask)
        else:
            self.vert_range = np.array(vert_range)
        if fracture_id is not None:
            self.vert_range = np.delete(self.vert_range, np.argwhere(self.vert_range == fracture_id))

        # print("vert_range:", self.vert_range)

        self.p_shape = None
        p_scan, p_mask = self.crop_spine(p_scan, p_mask, patient=1)
        _, h_mask = self.crop_spine(None, h_mask, patient=0)

        # sitk.WriteImage(h_mask,"h_mask.nii.gz")

        print("Cropped Size:", "\nhealthy:", h_mask.GetSize(), "\npatient:", p_mask.GetSize())

        self.h_mask = h_mask
        self.p_mask = p_mask
        self.p_scan = p_scan

        self.h_mask_array = None
        self.p_mask_array = None
        self.p_scan_array = None

        # save the header information of healthy and patient
        self.h_origin = h_mask.GetOrigin()
        self.h_direction = h_mask.GetDirection()
        self.h_spacing = h_mask.GetSpacing()

        self.p_origin = p_scan.GetOrigin()
        self.p_direction = p_scan.GetDirection()
        self.p_spacing = p_scan.GetSpacing()

        self.registration_method = None
        self.final_displacement_field = None

    def get_vert_range(self, h_mask: sitk.Image, p_mask: sitk.Image):
        """
        Get the intersection of vertebrae range between healthy and patient if vert_range is not specified
        :param h_mask: sitk.Image, mask of healthy atlas
        :param p_mask: sitk.Image, mask of patient spine
        :return:
        """
        h_mask_array = sitk.GetArrayFromImage(h_mask)
        p_mask_array = sitk.GetArrayFromImage(p_mask)

        h_vert_range = np.unique(h_mask_array)
        h_vert_range = np.delete(h_vert_range, h_vert_range.argmin())
        p_vert_range = np.unique(p_mask_array)
        p_vert_range = np.delete(p_vert_range, p_vert_range.argmin())

        self.vert_range = np.intersect1d(h_vert_range, p_vert_range, assume_unique=True)

    def crop_scan(self, scan: sitk.Image, bbox: np.ndarray, vert_id: int = 0) -> sitk.Image:
        """
        Crop 3D scan without messing up the Physical Point coordinate
        :param scan: sitk.Image, the image to be cropped
        :param bbox: the cropping bounding box
        :param vert_id: 0 - cropping spine mode, else - cropping vertebra mode
        :return: cropped scan
        """
        scan_array = sitk.GetArrayFromImage(scan)
        if vert_id != 0:
            scan_array = np.where(scan_array == vert_id, 1, -1).astype(np.int8)

        crop_array = scan_array[bbox[0]:bbox[1] + 1, bbox[2]:bbox[3] + 1, bbox[4]:bbox[5] + 1]

        new_origin = scan.TransformIndexToPhysicalPoint((int(bbox[4]), int(bbox[2]), int(bbox[0])))

        cropped_scan = sitk.GetImageFromArray(crop_array)
        cropped_scan.SetOrigin(new_origin)
        cropped_scan.SetSpacing(scan.GetSpacing())
        cropped_scan.SetDirection(scan.GetDirection())

        return cropped_scan

    def crop_spine(self, scan: Optional[sitk.Image], mask: sitk.Image, patient: int = 1,
                   **kwargs):
        """
        Crop out the spine according to the segmentation mask and vert_range
        :param scan: sitk.Image, the scan of spine
        :param mask: sitk.Image, the mask of spine
        :param patient: bool/int, True: patient spine, False: healthy atlas
        :param kwargs:
                    border: tuple(int,int,int), the border spacing of bounding box
        :return:
        """

        vert_min = self.vert_range.min()
        vert_max = self.vert_range.max()

        mask_array = sitk.GetArrayFromImage(mask)
        mask_array = np.where((mask_array >= vert_min) & (mask_array <= vert_max), mask_array, 0)
        bbox = get_bbox(mask_array, **kwargs)

        if patient:
            bbox = self.get_square_bbox(bbox, mask_array.shape)
            self.p_shape = get_shape_from_bbox(bbox)
        else:
            bbox = self.get_square_bbox(bbox, mask_array.shape)
            # bbox = self.get_square_bbox(bbox, mask_array.shape)

        if scan:
            cropped_scan = self.crop_scan(scan, bbox, vert_id=0)
        else:
            cropped_scan = None

        crop_mask = self.crop_scan(mask, bbox, vert_id=0)

        return cropped_scan, crop_mask

    def get_square_bbox(self, bbox: Union[np.ndarray, List[int]], orig_shape: Tuple[int, int, int]):
        """
        Make the bounding box to have square sagittal slices
        :param bbox: np.ndarray, bounding box
        :param orig_shape: tuple, the original shape
        :return:
        """
        bbox = np.array(bbox, dtype=np.int).reshape(-1, 2)
        bbox_shape = get_shape_from_bbox(bbox)

        # get the max number of coronal slices or lateral slices
        # (z, y, x)
        max_num_slices = bbox_shape[1:].max()
        if self.p_shape is not None:
            # max_num_slices = max(max_num_slices, max(self.p_shape))
            max_num_slices = max(max_num_slices, max(self.p_shape[1:]))
        # max_dim = bbox_shape[1:].argmax() + 1
        min_num_slices = bbox_shape[1:].min()
        min_dim = bbox_shape[1:].argmin() + 1

        # print(max_num_slices, min_dim, min_num_slices)

        delta_slices = max_num_slices - min_num_slices
        half_slices = delta_slices // 2

        bbox_dim_min = bbox[min_dim, 0] - half_slices
        bbox_dim_max = bbox_dim_min + max_num_slices

        if bbox_dim_min < 0:
            bbox_dim_max = bbox_dim_max - bbox_dim_min
        elif bbox_dim_max >= orig_shape[min_dim]:
            bbox_dim_min = bbox_dim_min - (bbox_dim_max - (orig_shape[min_dim] - 1))

        bbox[min_dim, 0] = max(0, bbox_dim_min)
        bbox[min_dim, 1] = min(bbox_dim_max, orig_shape[min_dim] - 1)

        return bbox.flatten()

    def get_vertebra(self, mask: sitk.Image, vert_id: int, **kwargs) -> sitk.Image:
        """
        Crop out the selected vertebra without messing up the physical point coordinate
        :param mask: sitk.Image, mask of spine
        :param vert_id: int, selected vertebra id
        :param kwargs:
                    border: tuple(int,int,int), the border spacing of bounding box
        :return:
        """
        mask_array = sitk.GetArrayFromImage(mask)
        vert_array = np.where(mask_array == vert_id, 1, 0).astype(np.int8)

        bbox = get_bbox(vert_array, **kwargs)

        crop_mask = self.crop_scan(mask, bbox, vert_id)

        return crop_mask

    def generate_fixed_moving_image_list(self, **kwargs) -> Tuple[List[sitk.Image], List[sitk.Image]]:
        """
        Generate list of fixed and moving image
        :param kwargs
                    border: tuple(int, int, int), border spacing of bounding box
        :return:
        """
        print("Generating lists of fixed images and moving images.")

        if self.vert_range is None:
            self.get_vert_range()

        fixed_image_list = []
        moving_image_list = []

        for i in self.vert_range:
            fixed_image = self.get_vertebra(self.h_mask, vert_id=i, **kwargs)
            moving_image = self.get_vertebra(self.p_mask, vert_id=i, **kwargs)

            fixed_image_list.append(fixed_image)
            moving_image_list.append(moving_image)
            # print("Fixed image and moving image of vertebra %d is generated." % i)

        del fixed_image
        del moving_image
        gc.collect()

        return fixed_image_list, moving_image_list

    def _initialize_registration(self):
        """
        Initialize the registration method
        :return:
        """
        registration_method = sitk.ImageRegistrationMethod()
        registration_method.SetMetricAsMeanSquares()
        registration_method.SetMetricSamplingStrategy(registration_method.NONE)
        registration_method.SetInterpolator(sitk.sitkLinear)

        registration_method.SetOptimizerAsRegularStepGradientDescent(learningRate=1,
                                                                     minStep=1e-4,
                                                                     relaxationFactor=0.9,
                                                                     numberOfIterations=50)

        registration_method.SetOptimizerScalesFromPhysicalShift()

        # Setup for the multi-resolution framework.
        registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
        registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
        registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

        self.registration_method = registration_method

    def generate_registration_transform_list(self, fixed_image_list: List[sitk.Image],
                                             moving_image_list: [sitk.Image]) -> List[sitk.Transform]:
        """
        Do the registration of each vertebra and store the result transform.
        Note: these transformation are mapping from fixed image to moving image.
        :param fixed_image_list: list[sitk.Image], list of fixed images
        :param moving_image_list: list[sitk.Image], list of moving images
        :return: list of transforms given by registration, the transform maps from f to m
        see https://github.com/SimpleITK/ISBI2020_TUTORIAL/blob/master/04_basic_registration.ipynb
        """

        print("Doing registration and generating transform list.")

        if self.registration_method is None:
            self._initialize_registration()

        transform_list = []

        for i, fixed_image, moving_image in zip(self.vert_range, fixed_image_list, moving_image_list):
            fixed_image = sitk.Cast(fixed_image, sitk.sitkFloat32)
            moving_image = sitk.Cast(moving_image, sitk.sitkFloat32)
            initial_transform = sitk.CenteredTransformInitializer(fixed_image, moving_image,
                                                                  sitk.VersorRigid3DTransform(),
                                                                  sitk.CenteredTransformInitializerFilter.MOMENTS)
            self.registration_method.SetInitialTransform(initial_transform, inPlace=False)
            transform = self.registration_method.Execute(fixed_image, moving_image)
            transform_list.append(transform)

        # print("Vertebrae registered.")

        del transform
        gc.collect()

        return transform_list

    def generate_distance_map_list(self, mask: sitk.Image) -> List[sitk.Image]:
        """
        Generate the distance map of each vertebra in the crop spine
        :param mask: sitk.Image, mask of crop spine
        :return:
                distancemap_list: list[sitk.Image], list of distance map
        """
        distancemapFilter = sitk.DanielssonDistanceMapImageFilter()
        distancemapFilter.UseImageSpacingOn()
        distancemap_list = []
        mask_array = sitk.GetArrayFromImage(mask)
        mask_origin = mask.GetOrigin()
        mask_spacing = mask.GetSpacing()
        mask_direction = mask.GetDirection()
        for i in self.vert_range:
            vert_array = np.where(mask_array == i, 1, 0).astype(np.uint16)
            vert_image = sitk.GetImageFromArray(vert_array)
            vert_image.SetOrigin(mask_origin)
            vert_image.SetSpacing(mask_spacing)
            vert_image.SetDirection(mask_direction)

            distancemap = distancemapFilter.Execute(vert_image)
            # print(distancemap.GetSize())
            distancemap = sitk.GetArrayFromImage(distancemap)
            distancemap_list.append(distancemap)
            # print("Distance map of vertebra %d is generated." % i)

        del distancemap
        gc.collect()

        return distancemap_list

    def generate_displacementfield_list(self, ref_image: sitk.Image, transform_list: List[sitk.Transform]) -> List[
        sitk.Image]:
        """
        Turn registration transform into displacement field
        :param ref_image: the reference image for displacement field filter
        :param transform_list: the list of registration transforms, which are mapping from fixed image to moving image
        :return: list of displacement fields
        """

        toDisplacementFilter = sitk.TransformToDisplacementFieldFilter()
        toDisplacementFilter.SetReferenceImage(ref_image)
        displacement_field_list = []

        for i, transform in zip(self.vert_range, transform_list):
            # print("Generate {} displacement field.".format(str(i)))
            displacement_field = toDisplacementFilter.Execute(transform)
            # print(displacement_field.GetSize())
            displacement_field = sitk.GetArrayFromImage(displacement_field)
            displacement_field_list.append(displacement_field)

        del displacement_field
        gc.collect()

        return displacement_field_list

    def combine_displacement_field(self, mask: sitk.Image, distancemap_list: List[sitk.Image],
                                   displacement_field_list: List[sitk.Image]) -> sitk.Image:
        """
        Use distance map as weight to combine displacement fields
        :param mask: sitk.Image, mask of spine
        :param distancemap_list: list[sitk.Image], list of distance map of each vertebra
        :param displacement_field_list: list[sitk.Image], list of displacement field of each vertebra
        :return:
        """
        print("Combining displacement fields")
        mask_array = sitk.GetArrayFromImage(mask)
        mask_origin = mask.GetOrigin()
        mask_direction = mask.GetDirection()
        mask_spacing = mask.GetSpacing()

        sum_of_weights_map = np.zeros_like(mask_array, dtype=np.float64)
        sum_of_displacement_field = np.zeros((3, *mask_array.shape), dtype=np.float64)
        for id, distance_map, displacement_field in zip(self.vert_range, distancemap_list, displacement_field_list):
            # distance_map = sitk.GetArrayFromImage(distance_map)
            # displacement_field = sitk.GetArrayFromImage(displacement_field)
            assert (distance_map.shape == displacement_field.shape[0:3])

            # weights_map = np.where(distance_map != 0.0, 1.0 / distance_map, 0.0)
            weights_map = np.where(abs(distance_map) >= 1e-5, 1.0 / distance_map, 0.0)

            # transpose for broadcasting
            reweighted_displacement_field = weights_map * (displacement_field.transpose(3, 0, 1, 2))

            sum_of_weights_map += weights_map

            sum_of_displacement_field += reweighted_displacement_field

        print("Displacement fields added.")

        # sum_of_weights_map = np.where(sum_of_weights_map != 0.0, 1.0 / sum_of_weights_map, 1)
        sum_of_weights_map = np.where(abs(sum_of_weights_map) >= 1e-5, 1.0 / sum_of_weights_map, 0.0)
        sum_of_displacement_field *= sum_of_weights_map

        sum_of_displacement_field = sum_of_displacement_field.transpose(1, 2, 3, 0)
        for id, displacement_field in zip(self.vert_range, displacement_field_list):
            # displacement_field = sitk.GetArrayFromImage(displacement_field)
            sum_of_displacement_field[mask_array == id] = displacement_field[mask_array == id]

        # demean the displacement field
        # sum_of_displacement_field -= np.mean(sum_of_displacement_field, axis=(0, 1, 2))
        combined_displacement_field = sitk.GetImageFromArray(sum_of_displacement_field)
        combined_displacement_field.SetOrigin(mask_origin)
        combined_displacement_field.SetSpacing(mask_spacing)
        combined_displacement_field.SetDirection(mask_direction)

        return combined_displacement_field

    def run(self):
        fixed_image_list, moving_image_list = self.generate_fixed_moving_image_list(border=(5, 5, 5))
        transform_list = self.generate_registration_transform_list(fixed_image_list, moving_image_list)
        del fixed_image_list
        del moving_image_list
        gc.collect()
        distancemap_list = self.generate_distance_map_list(self.h_mask)
        displacement_field_list = self.generate_displacementfield_list(self.h_mask, transform_list)
        del transform_list
        gc.collect()
        final_displacement_field = self.combine_displacement_field(self.h_mask, distancemap_list,
                                                                   displacement_field_list)
        del distancemap_list
        del displacement_field_list
        gc.collect()

        # self.final_displacement_field = final_displacement_field

        return final_displacement_field

    def straighten_spine(self,
                         scan: sitk.Image = None,
                         mask: sitk.Image = None,
                         final_displacement_field: sitk.Image = None,
                         straight_scan_name: str = 'temp_straight_scan.nii.gz',
                         straight_mask_name: str = 'temp_straight_mask.nii.gz',
                         whole_spine=False):
        """
        Compute the combined displacement field and resample patient spine scan and mask
        :param scan: spine scan to be resampled
        :param mask: spine mask to be resampled
        :param final_displacement_field: the combined displacement filed
        :param straight_scan_name: str, the output scan file
        :param straight_mask_name: str, the output mask file
        :param whole_spine: bool, whether to resample the whole spine
        :return:
        """

        if self.final_displacement_field is None and final_displacement_field is None:
            self.final_displacement_field = self.run()
        elif self.final_displacement_field is None:
            self.final_displacement_field = final_displacement_field

        if scan is None and self.p_scan is None:
            raise ValueError("No patient scan.")
        elif self.p_scan is None:
            self.p_scan = scan

        if mask is not None:
            self.p_mask = mask

        final_displacement_field_array = sitk.GetArrayFromImage(self.final_displacement_field)
        final_displacement_field = sitk.GetImageFromArray(final_displacement_field_array)
        final_displacement_field.SetOrigin(self.h_origin)
        final_displacement_field.SetDirection(self.h_direction)
        final_displacement_field.SetSpacing(self.h_spacing)
        # sitk.WriteImage(final_displacement_field, "final_disp_field.nii.gz")

        if whole_spine:
            output_size = final_displacement_field.GetSize()
        else:
            max_slices = max(max(self.p_scan.GetSize()[:1]), max(self.h_mask.GetSize()[:1]))
            output_size = (max_slices, max_slices, self.h_mask.GetSize()[2])

        straighten_spine = sitk.Resample(self.p_scan, output_size,
                                         sitk.DisplacementFieldTransform(final_displacement_field),
                                         sitk.sitkBSpline,
                                         self.h_origin,
                                         self.p_spacing,
                                         self.h_direction,
                                         -1024.0,
                                         self.p_scan.GetPixelID())

        final_displacement_field = sitk.GetImageFromArray(final_displacement_field_array)
        final_displacement_field.SetOrigin(self.h_origin)
        final_displacement_field.SetDirection(self.h_direction)
        final_displacement_field.SetSpacing(self.h_spacing)

        straighten_mask = sitk.Resample(self.p_mask, output_size,
                                        sitk.DisplacementFieldTransform(final_displacement_field),
                                        sitk.sitkNearestNeighbor,
                                        self.h_origin,
                                        self.p_spacing,
                                        self.h_direction,
                                        0.0,
                                        self.p_mask.GetPixelID())

        sitk.WriteImage(straighten_spine, straight_scan_name)
        sitk.WriteImage(straighten_mask, straight_mask_name)
        print('Saved straighten scan and mask at: ', straight_scan_name, 'and: ', straight_mask_name)


if __name__ == "__main__":
    h_mask_path = "./data/healthy_ref_mask.nii"
    p_mask_path = "./data/"
    p_scan_path = "./data/"
    fracture_id = 20 # check
    vert_range = []
    spine_str = SpineStraighten(h_mask_path=h_mask_path, p_mask_path=p_mask_path, p_scan_path=p_scan_path,
                                fracture_id=fracture_id, vert_range=vert_range, scale_factor=1.04)
    spine_str.straighten_spine(whole_spine=False)