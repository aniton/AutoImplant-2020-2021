import torch

from dpipe.io import save


def evaluate(preds_save_path, dataloader, model, criterion, writer, device='cuda'):
    model.eval()

    test_loss = 0

    preds_save_path.mkdir(exist_ok=True)

    with torch.no_grad():
        for idx, (complete_skull, _, _) in enumerate(dataloader):
            complete_skull.to(device)

            reconstructed_skull = model(complete_skull)
            save(reconstructed_skull, preds_save_path / '{:03d}.nii.gz'.format(idx))

            test_loss += criterion(reconstructed_skull, torch.max_pool3d(complete_skull, kernel_size=8)).detach().item()

        writer.add_scalar('loss/test', test_loss / len(dataloader))
