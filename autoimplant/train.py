import torch


def run_epoch(model, dataloader, optimizer, criterion, device='cuda'):
    model.train()

    epoch_loss = 0

    with torch.set_grad_enabled(True):
        for i, batch in enumerate(dataloader):
            complete_skulls, _, _ = batch
            complete_skulls.to(device)

            reconstructed_skulls = model(complete_skulls)

            loss = criterion(reconstructed_skulls, torch.max_pool3d(complete_skulls, kernel_size=8))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()

    average_loss = epoch_loss / len(dataloader)

    return average_loss
