import numpy as np

from dpipe.io import load, save
from dpipe.im.metrics import dice_score, hausdorff_distance


def evaluate(exp_name, dataloader, exp_dir):
    dice_scores, hausdorff_distances = {}, {}

    for idx, (complete_skull, _, complete_region, _) in zip(dataloader.dataset.ids, dataloader):
        reconstructed = load(exp_dir / 'test_predictions' / '{:03d}.npy.gz'.format(idx)) > .5
        complete = complete_skull if exp_name == 'model_x8' else complete_region
        complete = complete[0].numpy()

        dice_scores[str(idx)] = dice_score(reconstructed, complete)
        hausdorff_distances[str(idx)] = hausdorff_distance(reconstructed, complete)

    test_metrics_dir = exp_dir / 'test_metrics'
    test_metrics_dir.mkdir(exist_ok=True)

    save(dice_scores, test_metrics_dir / 'dice_scores.json')
    save(np.mean(list(dice_scores.values())), test_metrics_dir / 'mean_dice_score.json')

    save(hausdorff_distances, test_metrics_dir / 'hausdorff_distances.json')
    save(np.mean(list(hausdorff_distances.values())), test_metrics_dir / 'mean_hausdorff_distance.json')
