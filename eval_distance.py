import os
import numpy as np
import json
from sklearn.decomposition import PCA
import SimpleITK as sitk
from typing import List, Union, Tuple


class EvalDist():
    def __init__(self, h_mask_path: str, h_ctd_path: str, p_mask_path: str, p_ctd_path: str,
                 eval_frac_id: List[int], n_components: int = 3):
        """
        Evaluate pairwise vertebrae centroids distance and compute the scaling facotr of healthy atlas.
        :param h_mask_path: str, path of healthy atlas mask
        :param h_ctd_path: str, path of healthy atlas centroids
        :param p_mask_path: str, path of patient spine mask
        :param p_ctd_path: str, path of patient spine centroids
        :param eval_frac_id: list of centroids id which are fractured
        :param n_components: int, the number of pca components to keep
        """
        self.h_ctd_index, self.h_ctd_pp, self.h_label, _ = self.read_centroids(h_mask_path, h_ctd_path)
        self.p_ctd_index, self.p_ctd_pp, self.p_label, _ = self.read_centroids(p_mask_path, p_ctd_path)
        self.eval_frac_id = eval_frac_id
        self.n_components = n_components

    def read_centroids(self, img_path: str, ctd_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
        """
        Read centroids index from the output json file from segmentation network and transform to physical point.
        :param img_path: str, path of scan/mask
        :param ctd_path: str, path of centroids file
        :return: tuple(np.ndarray, np.ndarray, np.ndarray, str)
                centroids_index: voxel index coordinates of centroids
                centroids_pp: physical point coordinates of centroids
                label: label value of each centroid
                orientation: the orientation of scan/mask
        """
        img = sitk.ReadImage(img_path, sitk.sitkFloat32)
        centroids_index = json.load(open(ctd_path, "r"))
        orientation = "".join(centroids_index[0]['direction'])
        label = np.array([c["label"] for c in centroids_index[1:]], dtype=int)
        centroids_index = [[c['X'], c['Y'], c['Z']] for c in centroids_index[1:]]
        centroids_index = np.array(centroids_index, dtype=float)
        centroids_pp = np.array([img.TransformContinuousIndexToPhysicalPoint(pnt) for pnt in centroids_index],
                                dtype=float)
        return centroids_index, centroids_pp, label, orientation

    def compute_physical_distance(self, ctd_pp: np.ndarray, n_components: int = 3) -> np.ndarray:
        """
        Get PCA components of centroids physical point and compute the distance in between.
        :param ctd_pp: np.ndarray, physical point coordinates of centroids
        :param n_components: int, number of PCA components
        :return:
                dist: np.ndarray, distance between vertebrae
        """
        pca = PCA(n_components)
        ctd_pp = pca.fit_transform(ctd_pp)
        diff = np.diff(ctd_pp, axis=0)
        dist = np.linalg.norm(diff, axis=1)
        return dist

    def compute_scale_factor(self, h_dist: np.ndarray, h_label: np.ndarray, p_dist: np.ndarray,
                             p_label: np.ndarray) -> float:
        """
        Compute the scaling factor of healthy atlas
        :param h_dist: np.ndarray, pairwise centroid distance of healthy atlas
        :param h_label: np.ndarray, centroid labels of healthy atlas
        :param p_dist: np.ndarray, pairwise centroid distance of patient spine
        :param p_label: np.ndarray, centroid labels of patient spine
        :return:
                scale_factor: float, scale facotr of healthy atlas
        """
        vert_range, h_slice_index, p_slice_index = np.intersect1d(h_label, p_label, assume_unique=True,
                                                                  return_indices=True)
        h_dist = h_dist[h_slice_index[:-1]]
        p_dist = p_dist[p_slice_index[:-1]]
        assert h_dist.shape == p_dist.shape
        if len(self.eval_frac_id) > 0:
            sel_mask = np.ones(vert_range.shape[0] - 1, dtype=bool)
            for i in self.eval_frac_id:
                sel_mask[np.argwhere(vert_range == i).item()] = False
            h_dist = h_dist[sel_mask]
            p_dist = p_dist[sel_mask]
        scale_factor = np.mean(p_dist / h_dist, axis=-1)
        return scale_factor

    def run(self):
        h_dist = self.compute_physical_distance(self.h_ctd_pp, self.n_components)
        p_dist = self.compute_physical_distance(self.p_ctd_pp, self.n_components)
        scale_factor = self.compute_scale_factor(h_dist, self.h_label, p_dist, self.p_label)
        return scale_factor


if __name__ == "__main__":
    h_mask_path = "./data/01_healthy_1491_1989_seg_AB.nii"
    h_ctd_path = "./data/01_healthy_1491_1989_ctd.json"
    p_ctd_path = "./data/sub-kypho001_ses-20160502_dir-sag_sequ-wirbelule20sag5_seg-subreg_ctd.json"
    p_mask_path = "./data/sub-kypho001_ses-20160502_dir-sag_sequ-wirbelule20sag5_seg-vert_msk.nii.gz"
    eval_dist = EvalDist(h_mask_path=h_mask_path, h_ctd_path=h_ctd_path, p_mask_path=p_mask_path, p_ctd_path=p_ctd_path,
                         eval_frac_id=[19, 20, 22, 23], n_components=2)
    print(eval_dist.run())
