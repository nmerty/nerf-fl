from typing import List, Tuple

import torch
from torchvision.transforms import RandomCrop, functional as F


def random_crop_tensors(output_size: Tuple[int, int], *tensors) -> List[torch.Tensor]:
    """
    Crops given tensors at the same locations.
    Args:
        output_size: Expected output size of the crop.
        *tensors: Tensors of shape [..., H, W]
    Returns:
        List of cropped tensors.
    """
    i, j, h, w = RandomCrop.get_params(img=tensors[0], output_size=output_size)
    return [F.crop(t, i, j, h, w) for t in tensors]
