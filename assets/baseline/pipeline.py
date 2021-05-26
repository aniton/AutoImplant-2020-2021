from torch.utils.data import DataLoader

import torch

import numpy as np

from autoimplant.dataset import Autoimplant
from autoimplant.evaluate import evaluate
from autoimplant.train import train
from autoimplant.predict import predict

from dpipe.io import save
from dpipe.split import train_val_test_split

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True


def pipeline(exp_name, exp_root, data_root, model, optimizer, batch_size, num_epochs, criterion):
    exp_dir = exp_root / exp_name
    exp_dir.mkdir(exist_ok=True)

    n_samples = len(list((data_root / 'complete_skull').glob('*.nrrd')))
    train_ids, val_ids, test_ids = train_val_test_split(np.arange(n_samples), val_size=10, n_splits=3)[0]
    save(train_ids, exp_dir / 'train_ids.json')
    save(val_ids, exp_dir / 'val_ids.json')
    save(test_ids, exp_dir / 'test_ids.json')

    print('Initializing dataloaders...\n', flush=True)
    train_dataloader = DataLoader(Autoimplant(root=data_root, ids=train_ids), batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(Autoimplant(root=data_root, ids=val_ids), batch_size=1, shuffle=False)
    test_dataloader = DataLoader(Autoimplant(root=data_root, ids=test_ids), batch_size=1, shuffle=False)

    print('Training the model...\n', flush=True)
    train(exp_name, num_epochs, (train_dataloader, val_dataloader), model, optimizer, criterion, exp_dir)

    print('Making predictions...\n', flush=True)
    predict(exp_name, test_dataloader, model, exp_dir)

    print('Evaluating predictions...\n', flush=True)
    evaluate(exp_name, test_dataloader, exp_dir)
