from typing import List

import torch
from torchvision.transforms import RandomCrop, functional as F


def random_crop_tensors(height: int, width: int, *tensors) -> List[torch.Tensor]:
    """
    Crops given tensors at the same locations.
    Args:
        height: Expected output height of the crop.
        width: Expected output width of the crop.
        *tensors: Tensors of shape [..., H, W]
    Returns:
        List of cropped tensors.
    """
    i, j, h, w = RandomCrop.get_params(img=tensors[0], output_size=(height, width))
    return [F.crop(t, i, j, h, w).contiguous() for t in tensors]
