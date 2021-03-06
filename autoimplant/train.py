from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm

import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from dpipe.im.metrics import dice_score


def train(exp_name, num_epochs, dataloaders, model, optimizer, criterion, exp_dir, device='cuda'):
    # scheduler = MultiStepLR(optimizer, milestones=(10,), gamma=.2)

    model.to(device)

    logs_dir = exp_dir / 'logs'
    logs_dir.mkdir(exist_ok=True)
    writer = SummaryWriter(logs_dir)

    for epoch in range(num_epochs):
        loss, dice_score_ = run_epoch(exp_name, dataloaders, model, optimizer, criterion, device)

        writer.add_scalar('loss/train', loss, epoch)
        writer.add_scalar('metrics/val/dice_score', dice_score_, epoch)

        torch.save(model.state_dict(), exp_dir / f'{exp_name}.pth')
        # scheduler.step()


def run_epoch(exp_name, dataloaders, model, optimizer, criterion, device='cuda'):
    train_dataloader, val_dataloader = dataloaders

    train_loss = 0
    dice_scores = []

    model.train()
    with torch.set_grad_enabled(True):
        for complete_skull, defective_skull, complete_region, defective_region in tqdm(train_dataloader):
            complete = complete_skull if 'ss' not in exp_name else complete_region
            defective = defective_skull if 'ss' not in exp_name else defective_region

            complete = complete.float().to(device)
            defective = defective.float().to(device)

            # reconstructed = model.forward(complete)
            reconstructed = model.forward(defective)

            loss = criterion(reconstructed, F.max_pool3d(complete, kernel_size=8))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item()

    model.eval()
    with torch.set_grad_enabled(False):
        for complete_skull, defective_skull, complete_region, defective_region in tqdm(val_dataloader):
            complete = complete_skull if 'ss' not in exp_name else complete_region
            defective = defective_skull if 'ss' not in exp_name else defective_region

            defective = defective.float().to(device)

            reconstructed = model(defective).detach().cpu().numpy() > .5
            complete = complete.bool().numpy()

            dice_scores.append(dice_score(reconstructed, complete))

    return train_loss / len(train_dataloader), np.mean(dice_scores)
