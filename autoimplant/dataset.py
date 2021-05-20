from pathlib import Path

from tqdm import tqdm

import nrrd

from torch.utils.data import Dataset

import numpy as np
from skimage.morphology import opening
from skimage.measure import label, regionprops, block_reduce

from dpipe.im.box import mask2bounding_box, add_margin


class Autoimplant(Dataset):
    def __init__(self, root: str, part: str):
        super().__init__()

        assert part == 'train' or part == 'test', 'Bad dataset part. Must be `train` or `test`.'
        self.root = Path(root) / f'{part}set2021'

        self.complete_skulls, self.defective_skulls, self.complete_regions, self.defective_regions = [], [], [], []

        zone = 'bilateral'  # TODO: add other defects; only bilateral for now

        print(f'Initializing {part} dataset...\n', flush=True)
        for idx in tqdm(range(self.__len__())):
            complete_skull = nrrd.read(self.root / 'complete_skull' / '{:03d}.nrrd'.format(idx))[0]
            defective_skull = nrrd.read(self.root / 'defective_skull' / zone / '{:03d}.nrrd'.format(idx))[0]
            implant = nrrd.read(self.root / 'implant' / zone / '{:03d}.nrrd'.format(idx))[0]

            box = add_margin(mask2bounding_box(implant), margin=(5, 5, 5))
            complete_region = complete_skull[tuple([slice(start, stop) for start, stop in zip(*box)])]
            defective_region = defective_skull[tuple([slice(start, stop) for start, stop in zip(*box)])]

            complete_skull = block_reduce(complete_skull, block_size=(8, 8, 8))
            defective_skull = block_reduce(defective_skull, block_size=(8, 8, 8))

            self.complete_skulls.append(complete_skull[None, :].astype('bool'))
            self.defective_skulls.append(defective_skull[None, :].astype('bool'))
            self.complete_regions.append(complete_region[None, :].astype('bool'))
            self.defective_regions.append(defective_region[None, :].astype('bool'))

    def __getitem__(self, idx):
        return self.complete_skulls[idx], self.defective_skulls[idx], \
               self.complete_regions[idx], self.defective_regions[idx]

    def __len__(self):
        return len(list((self.root / 'complete_skull').glob('*.nrrd')))


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
