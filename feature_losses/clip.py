# Semantic consistency loss as in DietNeRF
# https://github.com/openai/CLIP
"""TODO: DietNeRF renders images for semantic consistency loss at half-precision and evaluates CLIP embeddings at
half-precision. As backpropagation through rendering is memory intensive with reverse-mode automatic differentiation,
we render images for LSC with mixed precision computation and eval- uate φ(·) at half-precision. We delete
intermediate MLP activations during rendering and rematerialize them during the backward pass."""

# Mixed precision:
# https://pytorch.org/docs/stable/amp.html
# https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html
# https://pytorch.org/docs/master/notes/amp_examples.html#gradient-penalty

# Half precision
# https://discuss.pytorch.org/t/training-with-half-precision/11815
# https://pytorch.org/docs/stable/amp.html

# Gradient checkpointing:
# https://pytorch.org/docs/stable/checkpoint.html
# https://qywu.github.io/2019/05/22/explore-gradient-checkpointing.html
# https://github.com/prigoyal/pytorch_memonger/blob/master/tutorial/Checkpointing_for_PyTorch_models.ipynb
from typing import Union

import clip
import kornia as K
import kornia.augmentation as KA
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


def encode_from_file(filepath, device):
    model, preprocess = clip.load("ViT-B/32", device=device)
    image = preprocess(Image.open(filepath)).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
    return image_features


def semantic_consistency_loss(embedding_0, embedding_1):
    # calculate cosine similarity up to a constant
    # Convert to unit vectors
    normalized_0 = F.normalize(embedding_0)
    normalized_1 = F.normalize(embedding_1)
    return torch.einsum("bi,bi->b", normalized_0, normalized_1)  # batch dot product


class CLIPWrapper:
    def __init__(
        self,
        name: str = "ViT-B/32",
        device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
        jit=True,
    ):
        super().__init__()
        self.device = device
        self.model, self.preprocess_image = clip.load(name, device=device, jit=jit)

        if not jit:
            input_size = self.model.visual.input_resolution
        else:
            input_size = self.model.input_resolution.item()

        # image processing for tensors (with autograd) corresponding to clip._transform
        self.process_tensor = nn.Sequential(
            K.K.Resize(input_size, interpolation="bicubic"),
            KA.CenterCrop(input_size),
            KA.Normalize(
                mean=torch.tensor([0.48145466, 0.4578275, 0.40821073]),
                std=torch.tensor([0.26862954, 0.26130258, 0.27577711]),
            ),
        )

    def cache_embedding(self, image_path, cache_out_path):
        pil_image = Image.open(image_path)
        image = self.preprocess_image(pil_image).unsqueeze(0).to(self.model.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image)
        image_features = image_features.cpu().numpy()
        np.save(cache_out_path, image_features)

    @staticmethod
    def load_embedding(cache_path):
        image_features = torch.from_numpy(np.load(cache_path))
        return image_features

    def get_embedding(self, image: Union[Image.Image, torch.Tensor]):
        if isinstance(image, Image.Image):
            image = self.preprocess_image(image).unsqueeze(0).to(self.device)
        else:
            image = self.process_tensor(image).to(self.device)
        image_features = self.model.encode_image(image)
        return image_features
