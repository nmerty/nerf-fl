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
import numpy as np
import torch
from PIL import Image


def encode_from_file(filepath, device):
    model, preprocess = clip.load("ViT-B/32", device=device)
    image = preprocess(Image.open(filepath)).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
    return image_features


def semantic_consistency_loss(embedding_0, embedding_1):
    # CLIP produces normalized image embeddings

    # Calculate cosine similarity up to a constant
    return torch.matmul(embedding_0, embedding_1)


class CLIPWrapper:
    def __init__(
        self, name: str, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu", jit=True
    ):
        super().__init__()
        self.model, self.preprocess = clip.load(name, device=device, jit=jit)

    def cache_embedding(self, image_path, cache_out_path):
        pil_image = Image.open(image_path)
        image = self.preprocess(pil_image).unsqueeze(0).to(self.model.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image)
        image_features = image_features.cpu().numpy()
        np.save(cache_out_path, image_features)

    @staticmethod
    def load_embedding(cache_path):
        image_features = torch.from_numpy(np.load(cache_path))
        return image_features

    def get_embedding(self, image):
        image = self.preprocess(image).unsqueeze(0).to(self.model.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image)
        return image_features
