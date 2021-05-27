import torch

from dpipe.io import save


def predict(exp_name, dataloader, model, exp_dir, device='cuda'):
    model.load_state_dict(torch.load(exp_dir / f'{exp_name}.pth', map_location=device))
    model.eval()
    model.to(device)

    test_predictions_dir = exp_dir / 'test_predictions'
    test_predictions_dir.mkdir(exist_ok=True)

    model.eval()
    with torch.no_grad():
        for idx, (_, defective_skull, _, defective_region) in zip(dataloader.dataset.ids, dataloader):
            defective = defective_skull if 'ss' not in exp_name else defective_region

            defective = defective.float().to(device)

            reconstructed = model(defective).detach().cpu().squeeze(0).numpy()

            save(reconstructed, test_predictions_dir / '{:03d}.npy.gz'.format(idx), compression=1, timestamp=0)
