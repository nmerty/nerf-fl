"""
MIT License https://github.com/ActiveVisionLab/nerfmm/blob/main/models/poses.py
"""
from typing import List, Optional

import torch
import torch.nn as nn

from utils.lie_group_helper import make_c2w


class LearnPose(nn.Module):
    def __init__(self, num_cams, learn_R, learn_t, init_c2w=None, perturb_sigma=0):
        """
        :param num_cams:
        :param learn_R:  True/False
        :param learn_t:  True/False
        :param init_c2w: (N, 4, 4) torch tensor
        """
        super(LearnPose, self).__init__()
        self.num_cams = num_cams
        self.init_c2w = None
        if init_c2w is not None:
            self.init_c2w = nn.Parameter(init_c2w, requires_grad=False)

        self.perturbation = nn.Parameter(torch.normal(0, perturb_sigma, [num_cams, 6]),
                                         requires_grad=False) if perturb_sigma > 0 else None

        self.r = nn.Parameter(torch.zeros(size=(num_cams, 3), dtype=torch.float32), requires_grad=learn_R)  # (N, 3)
        self.t = nn.Parameter(torch.zeros(size=(num_cams, 3), dtype=torch.float32), requires_grad=learn_t)  # (N, 3)

    def forward(self, cam_id):
        r = self.r[cam_id]  # (3, ) axis-angle
        t = self.t[cam_id]  # (3, )
        if self.perturbation is not None:
            r = r + self.perturbation[cam_id, :3]
            t = t + self.perturbation[cam_id, 3:]

        c2w = make_c2w(r, t)  # (4, 4)

        # learn a delta pose between init pose and target pose, if a init pose is provided
        if self.init_c2w is not None:
            c2w = c2w @ self.init_c2w[cam_id]

        return c2w

    def cam_grad(self, cam_id):
        """
        Get the gradient for the pose parameters.
        """
        return self.r.grad[cam_id], self.t.grad[cam_id]


class LearnPosePartial(nn.Module):
    """
    LearnPose wrapper which allows only learning some poses while fixing others.
    """

    def __init__(self, learn_poses: LearnPose, fix_poses: LearnPose,
                 learn_ids: List[int], fix_ids: Optional[List[int]] = None):
        """
        Args:
            learn_poses: LearnPose for cameras to optimize.
            fix_poses: LearnPose for cameras to fix.
            learn_ids: Camera ids to optimize.
            fix_ids: Camera ids to fix.
        """
        super().__init__()
        assert learn_poses.num_cams == fix_poses.num_cams
        self.num_cams = learn_poses.num_cams
        self.learn_ids = set(learn_ids)
        assert len(self.learn_ids) > 0
        all_ids = set(range(self.num_cams))
        if fix_ids is None:
            self.fix_ids = all_ids.difference(self.learn_ids)
        else:
            self.fix_ids = set(fix_ids)
        assert all_ids == self.learn_ids.union(self.fix_ids)

        # Define two separate LearnPose instances one fixed, other to train
        self.learn_poses = learn_poses
        self.fix_poses = fix_poses

    def forward(self, cam_id):
        if cam_id in self.learn_ids:
            return self.learn_poses(cam_id)
        else:
            return self.fix_poses(cam_id)

    def cam_grad(self, cam_id):
        if cam_id in self.learn_ids:
            return self.learn_poses.cam_grad(cam_id)
        else:
            return self.fix_poses.cam_grad(cam_id)
