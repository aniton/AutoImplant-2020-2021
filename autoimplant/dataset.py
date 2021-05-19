from pathlib import Path

import torch

import nrrd

from torch.utils.data import Dataset


class Autoimplant(Dataset):
    def __init__(self, root: str):
        super().__init__()

        self.root = Path(root)

    def __getitem__(self, item):
        # TODO: add other defects; only bilateral for now
        zone = 'bilateral'
        complete_skull = nrrd.read(self.root / 'complete_skull' / '{:03d}.nrrd'.format(item))[0]
        defective_skull = nrrd.read(self.root / 'defective_skull' / zone / '{:03d}.nrrd'.format(item))[0]
        implant = nrrd.read(self.root / 'implant' / zone / '{:03d}.nrrd'.format(item))[0]

        return map(lambda x: torch.tensor(x, dtype=torch.float16), (complete_skull, defective_skull, implant))

    def __len__(self):
        return len(list((self.root / 'complete_skull').glob('*/*.nrrd')))
