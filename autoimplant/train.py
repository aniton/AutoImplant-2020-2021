import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from dpipe.im.metrics import dice_score, hausdorff_distance
from tqdm import tqdm


def train(exp_name, num_epochs, dataloaders, model, optimizer, criterion, exp_dir, device='cuda'):
    model.to(device)

    logs_dir = exp_dir / 'logs'
    logs_dir.mkdir(exist_ok=True)
    writer = SummaryWriter(logs_dir)

    for epoch in range(num_epochs):
        loss, dice_score_, hausdorff_distance_ = run_epoch(exp_name, dataloaders, model, optimizer, criterion, device)

        writer.add_scalar('loss/train', loss, epoch)
        writer.add_scalar('metrics/val/dice_score', dice_score_, epoch)
        writer.add_scalar('metrics/val/hausdorff_distance', hausdorff_distance_, epoch)

        torch.save(model.state_dict(), exp_dir / f'{exp_name}.pth')


def run_epoch(exp_name, dataloaders, model, optimizer, criterion, device):
    train_dataloader, val_dataloader = dataloaders

    train_loss = 0
    dice_scores, hausdorff_distances = [], []

    model.train()
    with torch.set_grad_enabled(True):
        for complete_skull, _, complete_region, _ in tqdm(train_dataloader):
            complete = complete_skull if exp_name == 'model_x8' else complete_region

            complete = complete.float().to(device)

            reconstructed = model.forward(complete)

            loss = criterion(reconstructed, F.max_pool3d(complete, kernel_size=8))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item()

    model.eval()
    with torch.set_grad_enabled(False):
        for complete_skull, defective_skull, complete_region, defective_region in tqdm(val_dataloader):
            complete = complete_skull if exp_name == 'model_x8' else complete_region
            defective = defective_skull if exp_name == 'model_x8' else defective_region

            defective = defective.float().to(device)

            reconstructed = model(defective).detach().cpu().numpy() > .5
            complete = complete.bool().numpy()

            dice_scores.append(dice_score(reconstructed, complete))

            # hausdorff_distances.append(hausdorff_distance(reconstructed, complete))
            hausdorff_distances.append(100)
    return train_loss / len(train_dataloader), np.mean(dice_scores), np.mean(hausdorff_distances)
