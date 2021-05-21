import torch

from dpipe.io import save


def predict(dataloader, model, exp_dir, device='cuda'):
    model.load_state_dict(torch.load(exp_dir / 'model_x8.pth', map_location=device))
    model.eval()
    model.to(device)

    test_predictions_dir = exp_dir / 'test_predictions'
    test_predictions_dir.mkdir(exist_ok=True)

    with torch.no_grad():
        for idx, (_, defective_skull, _, _) in zip(dataloader.dataset.ids, dataloader):
            defective_skull = defective_skull.float().to(device)

            reconstructed_skull = model(defective_skull).detach().cpu().squeeze(0).numpy()

            save(reconstructed_skull, test_predictions_dir / '{:03d}.npy.gz'.format(idx), compression=1, timestamp=0)
