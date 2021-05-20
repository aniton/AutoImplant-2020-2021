from pathlib import Path

from torch.utils.data import DataLoader

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from autoimplant.dataset import Autoimplant
from autoimplant.model import ModelX8
from autoimplant.train import train
from autoimplant.evaluate import evaluate

from dpipe.batch_iter.pipeline import combine_pad


def run_experiment(exp_name,
                   exp_root='/gpfs/gpfs0/a.shevtsov/experiments/autoimplant/', data_root='/gpfs/gpfs0/a.shevtsov/data/',
                   num_epochs=10, batch_size=25, lr=1e-3):
    exp_dir = Path(exp_root) / exp_name
    exp_dir.mkdir()

    torch.cuda.manual_seed(42)

    train_dataset = Autoimplant(root=data_root, part='train')
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=combine_pad)
    test_dataset = Autoimplant(root=data_root, part='test')
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model_x8 = ModelX8()

    optimizer = torch.optim.Adam(model_x8.parameters(), lr=lr)
    criterion = nn.BCELoss()

    logs_dir = exp_dir / 'logs'
    logs_dir.mkdir()
    writer = SummaryWriter(logs_dir)

    train(exp_dir / 'model_x8.pth', num_epochs, train_dataloader, model_x8, optimizer, criterion, writer)

    evaluate(exp_dir / 'predictions', test_dataloader, model_x8, criterion, writer)
