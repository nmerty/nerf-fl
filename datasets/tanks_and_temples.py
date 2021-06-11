import cv2
import torch
from torch.utils.data import Dataset
import json
import numpy as np
import os
from PIL import Image
from torchvision import transforms as T

from .ray_utils import *


class TanksAndTemplesDataset(Dataset):
    def __init__(self, root_dir, split='train', img_wh=(800, 800)):
        self.root_dir = root_dir
        self.split = split
        self.img_wh = img_wh
        self.define_transforms()

        self.read_meta()
        self.white_back = True

    def read_meta(self):
        def parse_txt(filename):  # https://github.com/Kai-46/nerfplusplus/blob/master/data_loader_split.py
            assert os.path.isfile(filename)
            nums = open(filename).read().split()
            return np.array([float(x) for x in nums]).reshape([4, 4]).astype(np.float32)

        self.real_split = 'test' if self.split == 'val' else self.split
        self.filenames = sorted([file[:-len('.png')] for file in os.listdir(os.path.join(self.root_dir, self.real_split, 'rgb')) if file.endswith('.png')])
        self.num_cams = len(self.filenames)

        # assume shared
        intrinsics = parse_txt(os.path.join(self.root_dir, self.real_split, 'intrinsics', f'{self.filenames[0]}.txt'))
        poses = np.stack([parse_txt(os.path.join(self.root_dir, self.real_split, 'pose', f'{filename}.txt')) for filename in self.filenames])
        poses = np.concatenate([poses[..., 0:1], -poses[..., 1:3], poses[..., 3:4]], -1)

        img = cv2.imread(os.path.join(self.root_dir, self.real_split, 'rgb', f'{self.filenames[0]}.png'))
        w, h = self.img_wh
        H, W, *C = img.shape
        self.focal = intrinsics[0,0] * self.img_wh[0]/W

        # bounds, common for all scenes todo fix
        self.near = 0.0
        self.far = 6.0

        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = get_ray_directions(h, w, self.focal) # (h, w, 3)
        directions = self.directions.view(-1, 3)
        if self.split == 'train': # create buffer of all rays and rgb data
            self.image_paths = []
            self.poses = []
            self.all_rays = []
            self.all_rgbs = []
            for i, pose in enumerate(poses):
                self.poses += [pose]

                image_path = os.path.join(self.root_dir, self.real_split, 'rgb', f"{self.filenames[i]}.png")
                self.image_paths += [image_path]
                img = Image.open(image_path)
                assert img.size[1]*self.img_wh[0] == img.size[0]*self.img_wh[1], f"{image_path} has different aspect ratio than img_wh, please check your data!"
                img = img.resize(self.img_wh, Image.LANCZOS)
                img = self.transform(img) # (3, h, w)
                img = img.view(3, -1).permute(1, 0) # (h*w, 3)
                self.all_rgbs += [img]
                rays_t = i * torch.ones(len(directions), 1)

                self.all_rays += [torch.cat([directions,
                                             self.near*torch.ones_like(directions[:, :1]),
                                             self.far*torch.ones_like(directions[:, :1]),
                                             rays_t],
                                             1)] # (h*w, 6)

            self.poses = np.stack(self.poses)[:, :3, :4]
            self.poses_dict = {int(self.filenames[i]): self.poses[i] for i in range(self.poses.shape[0])}

            self.all_rays = torch.cat(self.all_rays, 0) # (len(self.meta['frames])*h*w, 3)
            self.all_rgbs = torch.cat(self.all_rgbs, 0) # (len(self.meta['frames])*h*w, 3)

        if self.split == 'val':
            self.val_poses = []
            for idx in range(self.__len__()):
                c2w = torch.tensor(poses[idx])[:3, :4]
                self.val_poses += [c2w]
            self.val_poses = torch.stack(self.val_poses)

    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        if self.split == 'train':
            return len(self.all_rays)
        if self.split == 'val':
            return 8 # only validate 8 images (to support <=8 gpus)
        return self.num_cams

    def __getitem__(self, idx):
        if self.split == 'train': # use data in the buffers
            sample = {'rays': self.all_rays[idx],
                      'ts': self.all_rays[idx, -1].long(),
                      'rgbs': self.all_rgbs[idx]}

        else: # create data for each image separately
            c2w = self.val_poses[idx][:3, :4]

            img = Image.open(os.path.join(self.root_dir, self.real_split, 'rgb', f"{self.filenames[idx]}.png"))
            img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img)  # (3, h, w)
            img = img.view(3, -1).permute(1, 0)  # (h*w, 3)

            directions = self.directions.view(-1, 3)
            rays = torch.cat([directions,
                              self.near*torch.ones_like(directions[:, :1]),
                              self.far*torch.ones_like(directions[:, :1])],
                              1) # (H*W, 8)

            sample = {'rays': rays,
                      'rgbs': img,
                      'c2w': c2w,
                      'ts': idx}

        return sample