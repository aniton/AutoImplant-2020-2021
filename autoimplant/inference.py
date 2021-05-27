import torch
import numpy as np
from skimage.measure import block_reduce
from autoimplant.model.baseline import ModelX8, LowRes
from autoimplant.dataset import cutImplantRegion, getBoundingBox

def _np_ar(tensor):
    return tensor.detach().cpu().numpy()

def _add_two_dim(tensor):
     tensor = torch.unsqueeze(tensor, 0)
     tensor = torch.unsqueeze(tensor, 0)
     return tensor

def inferenceSkull(low_res_model, high_res_model, damaged_skull_high_res, device, downsample_scale=8, padding=10, thresh=0.5):
    '''
    Inference step in baseline model.

    Parameters
    ----------
    `low_res_model`: `nn.Module`
        Pretrained NN for low resolution.

    `high_res_model`:`nn.Module`
        Pretrained NN for high resolution.

    `damaged_skull_high_res`:`torch.tensor`
        Voxel occupancy grid for the damaged skull in high resolution. Shape is `[N, N, N]`.

    `device`: `torch.device`
        

    `downsample_scale`: `int`
        Scale factor for downsampling input skull in high resolution.

    `padding`:`int`
        Margin in bounding box. 

    `thresh`:`float`
        Threshold used to obtain binary voxel occupancy grid from tesor with floats

    Return
    ------
    torch.tensor
        Output is voxel occupance grid of 'cured' damaged area. Shape is [S, S, S] 
        
    '''
    low_res_model.eval()
    high_res_model.eval()

    skull_low_res = block_reduce(_np_ar(damaged_skull_high_res), block_size=(downsample_scale, downsample_scale, downsample_scale), func=np.max)
    skull_low_res = skull_low_res.float().to(device)

    with torch.set_grad_enabled(False):
        # Damaged scull in low resolution pass through the first net
        pred_skull_low_res = low_res_model(_add_two_dim(skull_low_res))[0, 0]
        pred_skull_low_res[pred_skull_low_res < thresh] = 0
        pred_skull_low_res[pred_skull_low_res >= thresh] = 1
        
        # Localize trauma
        implant_low_res = cutImplantRegion(_np_ar(pred_skull_low_res), _np_ar(skull_low_res))
        bbox = getBoundingBox(implant_low_res)
        bbox = np.array(bbox)
        bbox *= downsample_scale

        assert bbox[0] - padding > 0 and bbox[3] + padding < damaged_skull_high_res.shape[0] and \
            bbox[1] - padding > 0 and bbox[4] + padding < damaged_skull_high_res.shape[1] and \
                bbox[2] - padding > 0 and bbox[5] + padding < damaged_skull_high_res.shape[2]

        corrupted_region_high_res = damaged_skull_high_res[(bbox[0] - padding):(bbox[3] + padding), 
                                                    (bbox[1] - padding):(bbox[4] + padding),
                                                    (bbox[2] - padding):(bbox[5] + padding)]
        corrupted_region_high_res = corrupted_region_high_res.to(device)
        
        # Damaged region in high resolution pass through the second net
        region_pred_high_res = high_res_model(_add_two_dim(corrupted_region_high_res))[0, 0]
        region_pred_high_res[region_pred_high_res < thresh] = 0
        region_pred_high_res[region_pred_high_res >= thresh] = 1

        return region_pred_high_res


            

