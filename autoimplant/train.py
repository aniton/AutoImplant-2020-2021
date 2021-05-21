import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter

from dpipe.im.metrics import dice_score, hausdorff_distance


def train(num_epochs, dataloaders, model, optimizer, criterion, exp_dir, device='cuda'):
    model.to(device)

    logs_dir = exp_dir / 'logs'
    logs_dir.mkdir(exist_ok=True)
    writer = SummaryWriter(logs_dir)

    for epoch in range(num_epochs):
        train_loss, dice_score_, hausdorff_distance_ = run_epoch(dataloaders, model, optimizer, criterion, device)

        writer.add_scalar('loss/train', train_loss, epoch)
        writer.add_scalar('metrics/val/dice_score', dice_score_, epoch)
        writer.add_scalar('metrics/val/hausdorff_distance', hausdorff_distance_, epoch)

        torch.save(model.state_dict(), exp_dir / 'model_x8.pth')


def run_epoch(dataloaders, model, optimizer, criterion, device):
    train_dataloader, val_dataloader = dataloaders

    train_loss = 0
    dice_scores, hausdorff_distances = [], []

    model.train()
    with torch.set_grad_enabled(True):
        for complete_skulls, _, _, _ in train_dataloader:
            complete_skulls = complete_skulls.float().to(device)

            reconstructed_skulls = model(complete_skulls)

            loss = criterion(reconstructed_skulls, complete_skulls)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item()

    model.eval()
    with torch.set_grad_enabled(False):
        for complete_skull, defective_skull, _, _ in val_dataloader:
            defective_skull = defective_skull.float().to(device)

            reconstructed_skull = model(defective_skull).detach().cpu().numpy() > .5
            complete_skull = complete_skull.numpy()

            dice_scores.append(dice_score(reconstructed_skull, complete_skull))
            hausdorff_distances.append(hausdorff_distance(reconstructed_skull, complete_skull))

    return train_loss / len(train_dataloader), np.mean(dice_scores), np.mean(hausdorff_distances)
