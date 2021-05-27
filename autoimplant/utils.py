import numpy as np

from scipy.ndimage import distance_transform_edt


def dt_weights(mask, _max=30, _min=1):
    if np.any(mask):
        inner = _max - distance_transform_edt(mask)
        outer = _max - distance_transform_edt(~mask)
        return np.maximum(inner * mask + outer * (~mask), _min)
    else:
        return np.ones_like(mask)
