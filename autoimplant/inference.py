import torch
import numpy as np
from skimage.measure import block_reduce
from autoimplant.model.baseline import ModelX8, LowRes
from autoimplant.dataset import cutImplantRegion, getBoundingBox

def _np_ar(tensor):
    return tensor.detach().cpu().numpy() 

def inferenceSkull(low_res_model, high_res_model, skull_high_res, device, downsample_scale=8, padding=10):
    low_res_model.eval()
    high_res_model.eval()
    skull_high_res = skull_high_res.float().to(device)
    skull_low_res = block_reduce(skull_high_res, block_size=(downsample_scale, downsample_scale, downsample_scale), func=np.max)
    with torch.set_grad_enabled(False):
        pred_skull_low_res = low_res_model(torch.unsqueeze(skull_low_res, 0))[0]
        implant_low_res = cutImplantRegion(_np_ar(skull_low_res), _np_ar(pred_skull_low_res))
        bbox = getBoundingBox(implant_low_res)

        bbox *= downsample_scale

        assert bbox[0] - padding > 0 and bbox[3] + padding < skull_high_res.shape[0] and \
            bbox[1] - padding > 0 and bbox[4] + padding < skull_high_res.shape[1] and \
                bbox[2] - padding > 0 and bbox[5] + padding < skull_high_res.shape[2]

        corrupted_region_high_res = skull_high_res[(bbox[0] - padding):(bbox[3] + padding), 
                                                    (bbox[1] - padding):(bbox[4] + padding),
                                                    (bbox[2] - padding):(bbox[5] + padding)]
        
        
        region_pred_high_res = high_res_model(torch.unsqueeze(corrupted_region_high_res, 0))[0]

        return region_pred_high_res


            

