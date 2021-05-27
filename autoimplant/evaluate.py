from tqdm import tqdm

import numpy as np
from dpipe.im.box import mask2bounding_box

from dpipe.io import load, save

from autoimplant.metrics import hd, hd95, dc, bdc


def evaluate(exp_name, dataloader, exp_dir, th=.5):
    dice_scores, hausdorff_distances, hausdorff_distances95, border_dice_scores = {}, {}, {}, {}

    for idx, (complete_skull, defective_skull, complete_region, _) in tqdm(zip(dataloader.dataset.ids, dataloader)):
        reconstructed = load(exp_dir / 'test_predictions' / '{:03d}.npy.gz'.format(idx))[0] > th
        complete = complete_skull if 'model_x8' in exp_name else complete_region
        complete = complete[0][0].numpy()
        defective = defective_skull[0][0].numpy()

        implant, generated_implant = complete & ~defective, reconstructed & ~defective
        box = mask2bounding_box(implant)

        mask = np.zeros_like(implant, dtype=bool)
        mask[tuple([slice(start, stop) for start, stop in zip(*box)])] = True
        generated_implant = ~generated_implant | mask

        dice_scores[str(idx)] = dc(reconstructed, complete)
        hausdorff_distances[str(idx)] = hd(reconstructed, complete, voxelspacing=(.4, .4, .4))
        # hausdorff_distances95[str(idx)] = hd95(reconstructed, complete, voxelspacing=(.4, .4, .4))
        border_dice_scores[str(idx)] = bdc(implant, generated_implant, defective)

    test_metrics_dir = exp_dir / 'test_metrics'
    test_metrics_dir.mkdir(exist_ok=True)

    save(dice_scores, test_metrics_dir / 'dice_scores.json')
    save(np.mean(list(dice_scores.values())), test_metrics_dir / 'mean_dice_score.json')

    save(hausdorff_distances, test_metrics_dir / 'hausdorff_distances.json')
    save(np.mean(list(hausdorff_distances.values())), test_metrics_dir / 'mean_hausdorff_distance.json')

    # save(hausdorff_distances95, test_metrics_dir / 'hausdorff_distances95.json')
    # save(np.mean(list(hausdorff_distances95.values())), test_metrics_dir / 'mean_hausdorff_distance95.json')

    save(border_dice_scores, test_metrics_dir / 'border_dice_scores.json')
    save(np.mean(list(border_dice_scores.values())), test_metrics_dir / 'mean_border_dice_score.json')
