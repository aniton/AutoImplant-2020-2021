from pathlib import Path

import torch

import nrrd

from torch.utils.data import Dataset

import numpy as np
from skimage.morphology import opening
from skimage.measure import label, regionprops


class Autoimplant(Dataset):
    def __init__(self, root: str, part: str, use_cache=False):
        super().__init__()

        assert part == 'train' or part == 'test', 'Bad dataset part. Must be `train` or `test`.'

        self.root = Path(root) / f'{part}set2021'

        self.use_cache = use_cache
        self.complete_skull_cache, self.defective_skull_cache, self.implant_cache = [], [], []

    def set_use_cache(self, use_cache):
        self.use_cache = use_cache

        if not self.use_cache:
            self.complete_skull_cache, self.defective_skull_cache, self.implant_cache = [], [], []

    def __getitem__(self, idx):
        if not self.use_cache:
            # TODO: add other defects; only bilateral for now
            zone = 'bilateral'

            complete_skull = nrrd.read(self.root / 'complete_skull' / '{:03d}.nrrd'.format(idx))[0]
            defective_skull = nrrd.read(self.root / 'defective_skull' / zone / '{:03d}.nrrd'.format(idx))[0]
            implant = nrrd.read(self.root / 'implant' / zone / '{:03d}.nrrd'.format(idx))[0]

            def _cast_to_uint(x):
                return torch.tensor(x, dtype=torch.uint8)

            complete_skull, defective_skull, implant = map(_cast_to_uint, (complete_skull, defective_skull, implant))

            self.complete_skull_cache.append(complete_skull)
            self.defective_skull_cache.append(defective_skull)
            self.implant_cache.append(implant)

        complete_skull = self.complete_skull_cache[idx]
        defective_skull = self.defective_skull_cache[idx]
        implant = self.implant_cache[idx]

        return tuple(map(lambda x: x.float().unsqueeze(0), (complete_skull, defective_skull, implant)))

    def __len__(self):
        return len(list((self.root / 'complete_skull').glob('*.nrrd')))


def cutImplantRegion(healthy_skull_predicted, corrupted_skull):
    """
    Find voxel occupancy grid of skull implant

    Parameters
    ----------
        healthy_skull_predicted: np.ndarray
            Vovel occupancy grid with shape NxNxN of predicted skull without hole
        corrupted_skull: np.ndarray
            Vovel occupancy grid with shape NxNxN of skull with hole

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
