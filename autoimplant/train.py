import torch
from tqdm import tqdm


def train(model_save_path, num_epochs, dataloader, model, optimizer, criterion, writer):
    model.train()

    for epoch in range(num_epochs):
        train_loss = run_epoch(dataloader, model, optimizer, criterion)

        writer.add_scalar('loss/train', train_loss, epoch)

        if epoch == 0:
            dataloader.dataset.set_use_cache(use_cache=True)
            dataloader.num_workers = 8

    torch.save(model.state_dict(), model_save_path)
    dataloader.dataset.set_use_cache(use_cache=False)


def run_epoch(dataloader, model, optimizer, criterion, device='cuda'):
    epoch_loss = 0

    with torch.set_grad_enabled(True):
        for complete_skulls, _, _ in tqdm(dataloader):
            complete_skulls.to(device)

            reconstructed_skulls = model(complete_skulls)

            loss = criterion(reconstructed_skulls, torch.max_pool3d(complete_skulls, kernel_size=8))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()

    average_loss = epoch_loss / len(dataloader)

    return average_loss
