import torch
from tqdm import tqdm


def train(model_save_path, num_epochs, dataloader, model, optimizer, criterion, writer, device='cuda'):
    model.train()
    model.to(device)

    print('\n\nTraining...', flush=True)
    for epoch in range(num_epochs):
        train_loss = run_epoch(dataloader, model, optimizer, criterion, device)

        writer.add_scalar('loss/train', train_loss, epoch)

        torch.save(model.state_dict(), model_save_path)


def run_epoch(dataloader, model, optimizer, criterion, device):
    epoch_loss = 0

    with torch.set_grad_enabled(True):
        for complete_skulls, _, _, _ in tqdm(dataloader):
            complete_skulls = torch.tensor(complete_skulls, device=device, dtype=torch.float32)

            reconstructed_skulls = model(complete_skulls)

            loss = criterion(reconstructed_skulls, complete_skulls)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()

    average_loss = epoch_loss / len(dataloader)

    return average_loss
