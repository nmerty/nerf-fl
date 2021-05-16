import glob
import os
import pickle

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from .colmap_utils import read_images_binary, read_points3d_binary
from .ray_utils import *


class PhototourismImagePose(Dataset):
    """
    Dataset for getting rgb images and poses to be used by the feature losses.
    """

    def __init__(
        self,
        root_dir,
        split="train",
        img_downscale=1,
        val_num=1,
        use_cache=False,
        refine_pose=False,
    ):
        self.root_dir = root_dir
        self.split = split
        self.refine_pose = refine_pose
        assert img_downscale >= 1, "image can only be downsampled, please set img_downscale>=1!"
        self.img_downscale = img_downscale
        if split == "val":  # image downscale=1 will cause OOM in val mode
            self.img_downscale = max(2, self.img_downscale)
        self.val_num = max(1, val_num)  # at least 1
        self.use_cache = use_cache
        self.transform = T.ToTensor()

        self.read_meta()

    def read_meta(self):
        # read all files in the tsv first (split to train and test later)
        tsv = glob.glob(os.path.join(self.root_dir, "*.tsv"))[0]
        self.scene_name = os.path.basename(tsv)[:-4]
        self.files = pd.read_csv(tsv, sep="\t")
        self.files = self.files[~self.files["id"].isnull()]  # remove data without id
        self.files.reset_index(inplace=True, drop=True)

        # Step 1. load image paths
        # Attention! The 'id' column in the tsv is BROKEN, don't use it!!!!
        # Instead, read the id from images.bin using image file name!
        if self.use_cache:
            with open(os.path.join(self.root_dir, f"cache/img_ids.pkl"), "rb") as f:
                self.img_ids = pickle.load(f)
            with open(os.path.join(self.root_dir, f"cache/img_to_cam_id.pkl"), "rb") as f:
                self.image_to_cam = pickle.load(f)
            with open(os.path.join(self.root_dir, f"cache/image_paths.pkl"), "rb") as f:
                self.image_paths = pickle.load(f)
        else:
            imdata = read_images_binary(os.path.join(self.root_dir, "dense/sparse/images.bin"))
            img_path_to_id = {}
            self.image_to_cam = {}  # {id: image id}

            for v in imdata.values():
                img_path_to_id[v.name] = v.id
                self.image_to_cam[v.id] = v.camera_id
            self.img_ids = []
            self.image_paths = {}  # {id: filename}
            for filename in list(self.files["filename"]):
                id_ = img_path_to_id[filename]
                self.image_paths[id_] = filename
                self.img_ids += [id_]

        # Step 3: read c2w poses (of the images in tsv file only) and correct the order
        if self.use_cache:
            self.poses = np.load(os.path.join(self.root_dir, "cache/poses.npy"))
        else:
            w2c_mats = []
            bottom = np.array([0, 0, 0, 1.0]).reshape(1, 4)
            for id_ in self.img_ids:
                im = imdata[id_]
                R = im.qvec2rotmat()
                t = im.tvec.reshape(3, 1)
                w2c_mats += [np.concatenate([np.concatenate([R, t], 1), bottom], 0)]
            w2c_mats = np.stack(w2c_mats, 0)  # (N_images, 4, 4)
            self.poses = np.linalg.inv(w2c_mats)[:, :3]  # (N_images, 3, 4)
            # Original poses has rotation in form "right down front", change to "right up back"
            self.poses[..., 1:3] *= -1

        # Step 4: correct scale
        if self.use_cache:
            self.xyz_world = np.load(os.path.join(self.root_dir, "cache/xyz_world.npy"))
            with open(os.path.join(self.root_dir, f"cache/nears.pkl"), "rb") as f:
                self.nears = pickle.load(f)
            with open(os.path.join(self.root_dir, f"cache/fars.pkl"), "rb") as f:
                self.fars = pickle.load(f)
        else:
            pts3d = read_points3d_binary(os.path.join(self.root_dir, "dense/sparse/points3D.bin"))
            self.xyz_world = np.array([pts3d[p_id].xyz for p_id in pts3d])
            xyz_world_h = np.concatenate([self.xyz_world, np.ones((len(self.xyz_world), 1))], -1)
            # Compute near and far bounds for each image individually
            self.nears, self.fars = {}, {}  # {id_: distance}
            for i, id_ in enumerate(self.img_ids):
                xyz_cam_i = (xyz_world_h @ w2c_mats[i].T)[:, :3]  # xyz in the ith cam coordinate
                xyz_cam_i = xyz_cam_i[xyz_cam_i[:, 2] > 0]  # filter out points that lie behind the cam
                self.nears[id_] = np.percentile(xyz_cam_i[:, 2], 0.1)
                self.fars[id_] = np.percentile(xyz_cam_i[:, 2], 99.9)

            max_far = np.fromiter(self.fars.values(), np.float32).max()
            scale_factor = max_far / 5  # so that the max far is scaled to 5
            self.poses[..., 3] /= scale_factor
            for k in self.nears:
                self.nears[k] /= scale_factor
            for k in self.fars:
                self.fars[k] /= scale_factor
            self.xyz_world /= scale_factor
        self.poses_dict = {id_: self.poses[i] for i, id_ in enumerate(self.img_ids)}
        # Step 5. split the img_ids (the number of images is verfied to match that in the paper)
        self.img_ids_train = [id_ for i, id_ in enumerate(self.img_ids) if self.files.loc[i, "split"] == "train"]
        self.img_ids_test = [id_ for i, id_ in enumerate(self.img_ids) if self.files.loc[i, "split"] == "test"]
        self.N_images_train = len(self.img_ids_train)
        self.N_images_test = len(self.img_ids_test)

    def __getitem__(self, idx):
        if self.split == "train":
            id_ = self.img_ids_train[idx]
            img = Image.open(os.path.join(self.root_dir, "dense/images", self.image_paths[id_])).convert("RGB")
            img_w, img_h = img.size
            if self.img_downscale > 1:
                img_w = img_w // self.img_downscale
                img_h = img_h // self.img_downscale
                img = img.resize((img_w, img_h), Image.LANCZOS)
            img = self.transform(img)  # (3, h, w)

            sample = {"img": img, "c2w": torch.FloatTensor(self.poses_dict[id_])}
            return sample
        else:
            raise ValueError(f"Unsupported split: {self.split}")

    def __len__(self):
        if self.split == "train":
            return len(self.poses)
        else:
            raise ValueError(f"Unsupported split: {self.split}")
