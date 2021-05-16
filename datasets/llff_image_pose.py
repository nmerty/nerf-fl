import glob
import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from .llff import center_poses
from .ray_utils import *


class LLFFImagePose(Dataset):
    """
    Dataset for getting rgb images and poses to be used by the feature losses.
    """

    def __init__(
        self,
        root_dir,
        split="train",
        img_wh=(504, 378),
        transform=None,
    ):
        self.root_dir = root_dir
        self.split = split
        self.img_wh = img_wh
        if transform is None:
            self.transform = T.ToTensor()
        else:
            self.transform = transform

        self.read_meta()

    def read_meta(self):
        poses_bounds = np.load(os.path.join(self.root_dir, "poses_bounds.npy"))  # (N_images, 17)
        self.image_paths = sorted(glob.glob(os.path.join(self.root_dir, "images/*")))
        # load full resolution image then resize
        if self.split in ["train", "val"]:
            assert len(poses_bounds) == len(
                self.image_paths
            ), "Mismatch between number of images and number of poses! Please rerun COLMAP!"
        poses = poses_bounds[:, :15].reshape(-1, 3, 5)  # (N_images, 3, 5)
        self.bounds = poses_bounds[:, -2:]  # (N_images, 2)

        # Step 1: rescale focal length according to training resolution
        H, W, self.focal = poses[0, :, -1]  # original intrinsics, same for all images
        # assert H*self.img_wh[0] == W*self.img_wh[1], \
        #     f'You must set @img_wh to have the same aspect ratio as ({W}, {H}) !'

        self.focal *= self.img_wh[0] / W

        # Step 2: correct poses
        # Original poses has rotation in form "down right back", change to "right up back"
        # See https://github.com/bmild/nerf/issues/34
        poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
        # (N_images, 3, 4) exclude H, W, focal
        self.poses, self.pose_avg = center_poses(poses)
        # distances_from_center = np.linalg.norm(self.poses[..., 3], axis=1)
        # center image

        # Step 3: correct scale so that the nearest depth is at a little more than 1.0
        # See https://github.com/bmild/nerf/issues/34
        near_original = self.bounds.min()
        scale_factor = near_original * 0.75  # 0.75 is the default parameter
        # the nearest depth is at 1/0.75=1.33
        self.bounds /= scale_factor
        self.poses[..., 3] /= scale_factor

    def __getitem__(self, idx):
        if self.split == "train":
            # use first N_images-1 to train, the LAST is val
            image_path = self.image_paths[idx]
            # if i == val_idx:  # exclude the val image
            #     continue
            c2w = torch.FloatTensor(self.poses[idx])

            img = Image.open(image_path).convert("RGB")
            assert (
                img.size[1] * self.img_wh[0] == img.size[0] * self.img_wh[1]
            ), f"""{image_path} has different aspect ratio than img_wh, 
                           please check your data!"""
            img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img)  # (3, h, w)
            sample = {"img": img, "c2w": c2w}
            return sample
        else:
            raise ValueError(f"Unsupported split: {self.split}")

    def __len__(self):
        if self.split == "train":
            return len(self.poses)
        else:
            raise ValueError(f"Unsupported split: {self.split}")
