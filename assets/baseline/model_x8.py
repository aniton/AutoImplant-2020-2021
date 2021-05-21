from pathlib import Path

from torch.utils.data import DataLoader

import torch
import torch.nn as nn

import numpy as np

from autoimplant.dataset import Autoimplant
from autoimplant.evaluate import evaluate
from autoimplant.model import ModelX8
from autoimplant.train import train
from autoimplant.predict import predict

from dpipe.io import save
from dpipe.split import train_val_test_split

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True


def run_experiment(exp_name, exp_root='/gpfs/gpfs0/a.shevtsov/experiments/autoimplant/',
                   data_root='/gpfs/gpfs0/a.shevtsov/data/trainset2021',
                   num_epochs=10, batch_size=1, lr=1e-3):
    exp_root, data_root = map(Path, (exp_root, data_root))

    exp_dir = exp_root / exp_name
    exp_dir.mkdir(exist_ok=True)

    n_samples = len(list((data_root / 'complete_skull').glob('*.nrrd')))
    train_ids, val_ids, test_ids = train_val_test_split(np.arange(n_samples), val_size=10, n_splits=3)[0]
    save(train_ids, exp_root / 'train_ids.json')
    save(val_ids, exp_root / 'val_ids.json')
    save(test_ids, exp_root / 'test_ids.json')

    print('Initializing dataloaders...\n', flush=True)
    train_dataloader = DataLoader(Autoimplant(root=data_root, ids=train_ids), batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(Autoimplant(root=data_root, ids=val_ids), batch_size=1, shuffle=False)
    test_dataloader = DataLoader(Autoimplant(root=data_root, ids=test_ids), batch_size=1, shuffle=False)

    model_x8 = ModelX8()
    optimizer = torch.optim.Adam(model_x8.parameters(), lr=lr)
    criterion = nn.BCELoss()

    print('Training the model...\n', flush=True)
    train(num_epochs, (train_dataloader, val_dataloader), model_x8, optimizer, criterion, exp_dir)

    print('Making predictions...\n', flush=True)
    predict(test_dataloader, model_x8, exp_dir)

    print('Evaluating predictions...\n', flush=True)
    evaluate(test_dataloader, exp_dir)
