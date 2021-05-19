from pathlib import Path

import torch

import nrrd

from torch.utils.data import Dataset


class Autoimplant(Dataset):
    def __init__(self, root: str, part: str):
        super().__init__()

        assert part == 'train' or part == 'test', 'Bad dataset part. Must be `train` or `test`.'

        self.root = Path(root) / f'{part}set2021'

    def __getitem__(self, item):
        # TODO: add other defects; only bilateral for now
        zone = 'bilateral'
        complete_skull = nrrd.read(self.root / 'complete_skull' / '{:03d}.nrrd'.format(item))
        defective_skull = nrrd.read(self.root / 'defective_skull' / zone / '{:03d}.nrrd'.format(item))
        implant = nrrd.read(self.root / 'implant' / zone / '{:03d}.nrrd'.format(item))

        sample = complete_skull, defective_skull, implant

        return tuple(map(lambda x: torch.tensor(x[0], dtype=torch.float32).unsqueeze(0), sample))

    def __len__(self):
        return len(list((self.root / 'complete_skull').glob('*.nrrd')))
