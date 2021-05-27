from pathlib import Path

from typing import Union

import numpy as np

from torch.utils.data import Dataset

import nrrd

from skimage.morphology import opening
from skimage.measure import label, regionprops

from dpipe.im import pad_to_shape
from dpipe.im.axes import broadcast_to_axis
from dpipe.im.box import mask2bounding_box, limit_box, get_centered_box

PathLike = Union[Path, str]


class Autoimplant(Dataset):
    def __init__(self, root: PathLike, ids: list):
        super().__init__()

        self.root = root
        self.ids = ids
        self.complete_skulls, self.defective_skulls, self.complete_regions, self.defective_regions = [], [], [], []

        zone = 'bilateral'

        for idx in self.ids:
            complete_skull = nrrd.read(self.root / 'complete_skull' / '{:03d}.nrrd'.format(idx))[0]
            defective_skull = nrrd.read(self.root / 'defective_skull' / zone / '{:03d}.nrrd'.format(idx))[0]
            implant = nrrd.read(self.root / 'implant' / zone / '{:03d}.nrrd'.format(idx))[0]

            center = mask2bounding_box(implant).sum(axis=0) // 2
            box = limit_box(get_centered_box(center=center, box_size=np.array([384, 384, 320])), limit=(512, 512, 512))
            complete_region = complete_skull[tuple([slice(start, stop) for start, stop in zip(*box)])]
            defective_region = defective_skull[tuple([slice(start, stop) for start, stop in zip(*box)])]

            complete_region, defective_region = map(self.pad_to_shape, (complete_region, defective_region))

            self.complete_skulls.append(complete_skull[None, :].astype('bool'))
            self.defective_skulls.append(defective_skull[None, :].astype('bool'))
            self.complete_regions.append(complete_region[None, :].astype('bool'))
            self.defective_regions.append(defective_region[None, :].astype('bool'))

    def __getitem__(self, idx):
        return self.complete_skulls[idx], self.defective_skulls[idx], \
               self.complete_regions[idx], self.defective_regions[idx]

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def pad_to_shape(x, axis=(-3, -2, -1), shape=(384, 384, 320), padding_values=0, ratio=.5):
        shape, ratio = broadcast_to_axis(axis, shape, ratio)

        x = pad_to_shape(x, shape, axis, padding_values, ratio)
        return x


def cutImplantRegion(healthy_skull_predicted, corrupted_skull):
    """
    Find voxel occupancy grid of skull implant

    Parameters
    ----------
        healthy_skull_predicted: np.ndarray
            Voxel occupancy grid with shape NxNxN of predicted skull without hole
        corrupted_skull: np.ndarray
            Voxel occupancy grid with shape NxNxN of skull with hole

    Returns
    -------
        np.ndarray
    """
    implant = healthy_skull_predicted - corrupted_skull
    implant[implant < 0] = 0

    implant = opening(implant.astype(np.uint8))

    # https://stackoverflow.com/questions/53730029/get-biggest-coherent-area
    l = label(implant)
    implant = (l == np.bincount(l.ravel())[1:].argmax() + 1).astype(int)

    return implant


def getBoundingBox(implant, lx=None, ly=None, lz=None):
    l = label(implant)
    r = regionprops(l)
    min_x, min_y, min_z, max_x, max_y, max_z = r[0].bbox
    if not lx is None:
        if max_x - min_x > lx:
            raise ValueError(
                f"Actual size of bounding box larger than is specified: expected {lx}, but given {max_x - min_x}")
        min_x -= (lx - max_x + min_x) // 2
        max_x += (lx - max_x + min_x) // 2 + (lx - max_x + min_x) % 2
    if not ly is None:
        if max_y - min_y > ly:
            raise ValueError(
                f"Actual size of bounding box larger than is specified: expected {ly}, but given {max_y - min_y}")
        min_y -= (ly - max_x + min_x) // 2
        max_y += (ly - max_x + min_x) // 2 + (ly - max_x + min_x) % 2
    if not lz is None:
        if max_z - min_z > lz:
            raise ValueError(
                f"Actual size of bounding box larger than is specified: expected {lz}, but given {max_z - min_z}")
        min_z -= (lz - max_x + min_x) // 2
        max_z += (lz - max_x + min_x) // 2 + (lz - max_x + min_x) % 2
    return min_x, min_y, min_z, max_x, max_y, max_z
