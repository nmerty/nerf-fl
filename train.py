import os

from datasets.ray_utils import get_rays, get_ndc_rays
from models.poses import LearnPose
from opt import get_opts
import torch
from collections import defaultdict

from torch.utils.data import DataLoader
from datasets import dataset_dict

# models
from models.nerf import *
from models.rendering import *

# optimizer, scheduler, visualization
from utils import *

# losses
from losses import loss_dict

# metrics
from metrics import *

# pytorch-lightning
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.loggers import TestTubeLogger

from utils.align_traj import align_ate_c2b_use_a2b
from utils.comp_ate import compute_ate
from utils.lie_group_helper import convert3x4_4x4
import matplotlib.pyplot as plt


def save_pose_plot(poses, gt, current_epoch, dataset_name):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    bounds = 1 if dataset_name == 'llff' else 5
    ax.axes.set_xlim3d(left=-bounds, right=bounds)
    ax.axes.set_ylim3d(bottom=-bounds, top=bounds)
    ax.axes.set_zlim3d(bottom=0, top=bounds)
    ax.plot(gt[:, 0, 3], gt[:, 1, 3], gt[:, 2, 3], 'k.', label='GT')
    ax.plot(poses[:, 0, 3], poses[:, 1, 3], poses[:, 2, 3], 'r.', label='pred')
    ax.legend()
    ax.set_title(f"Epoch: {current_epoch}")
    return fig, ax


class NeRFSystem(LightningModule):
    def __init__(self, hparams):
        super(NeRFSystem, self).__init__()
        self.hparams = hparams

        self.loss = loss_dict['color'](coef=1)

        self.embedding_xyz = Embedding(3, hparams.N_emb_xyz, epoch_start=hparams.barf_start, epoch_end=hparams.barf_end)
        self.embedding_dir = Embedding(3, hparams.N_emb_dir)
        self.embeddings = {'xyz': self.embedding_xyz,
                           'dir': self.embedding_dir}

        self.nerf_coarse = NeRF(hparams.N_layers, hparams.N_hidden_units, skips=hparams.skip_connections)
        self.models = {'coarse': self.nerf_coarse}
        load_ckpt(self.nerf_coarse, hparams.weight_path, 'nerf_coarse')
        self.automatic_optimization = False

        if hparams.N_importance > 0:
            self.nerf_fine = NeRF(hparams.N_layers, hparams.N_hidden_units, skips=hparams.skip_connections)
            self.models['fine'] = self.nerf_fine
            load_ckpt(self.nerf_fine, hparams.weight_path, 'nerf_fine')

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def forward(self, rays):
        """Do batched inference on rays using chunk."""
        kwargs = {'current_epoch': self.global_step // self.hparams.N_images}
        B = rays.shape[0]
        results = defaultdict(list)
        for i in range(0, B, self.hparams.chunk):
            rendered_ray_chunks = \
                render_rays(self.models,
                            self.embeddings,
                            rays[i:i+self.hparams.chunk],
                            self.hparams.N_samples,
                            self.hparams.use_disp,
                            self.hparams.perturb,
                            self.hparams.noise_std,
                            self.hparams.N_importance,
                            self.hparams.chunk, # chunk size is effective in val mode
                            self.train_dataset.white_back,
                            **kwargs)

            for k, v in rendered_ray_chunks.items():
                results[k] += [v]

        for k, v in results.items():
            results[k] = torch.cat(v, 0)
        return results

    def setup(self, stage):
        dataset = dataset_dict[self.hparams.dataset_name]
        kwargs = {'root_dir': self.hparams.root_dir,
                  'img_wh': tuple(self.hparams.img_wh)}
        if self.hparams.dataset_name == 'llff':
            kwargs['spheric_poses'] = self.hparams.spheric_poses
            kwargs['val_num'] = self.hparams.num_gpus
        self.train_dataset = dataset(split='train', **kwargs)
        self.val_dataset = dataset(split='val', **kwargs)

        N_images = len(self.train_dataset.poses_dict.keys())
        assert N_images == hparams.N_images, f"Set number of images in args to {N_images}"
        c2ws = convert3x4_4x4(torch.from_numpy(self.train_dataset.poses))
        refine_pose = self.hparams.refine_pose

        initial_poses = c2ws.float() if not refine_pose or hparams.pose_init in ['original', 'perturb'] else None
        self.learn_poses = LearnPose(len(self.train_dataset.poses_dict.keys()), refine_pose, refine_pose,
                                     init_c2w=initial_poses, perturb_sigma=self.hparams.pose_sigma)#.to(self.device)
        # load_ckpt(self.learn_poses, hparams...)

    def configure_optimizers(self):
        optimizer = get_optimizer(self.hparams, self.models)
        scheduler = get_scheduler(self.hparams, optimizer)

        optimizer_pose = Adam(get_parameters(self.learn_poses), lr=self.hparams.lr_pose, eps=1e-8,
                                   weight_decay=self.hparams.weight_decay)
        scheduler_pose = MultiStepLR(optimizer_pose, milestones=self.hparams.decay_step_pose, gamma=self.hparams.decay_gamma_pose)

        return ({'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'}},
                {'optimizer': optimizer_pose, 'lr_scheduler': {'scheduler': scheduler_pose, 'interval': 'step'}},)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          shuffle=True,
                          num_workers=4,
                          batch_size=self.hparams.batch_size,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          shuffle=False,
                          num_workers=4,
                          batch_size=1, # validate one image (H*W rays) at a time
                          pin_memory=True)
    
    def training_step(self, batch, batch_nb, optimizer_idx):
        opt1, opt2 = self.optimizers()
        opt1.zero_grad()
        opt2.zero_grad()

        rays, rgbs, ts = batch['rays'], batch['rgbs'], batch['ts']

        self.learn_poses.train()

        poses = {img_id: self.learn_poses(i) for i, img_id in enumerate(self.train_dataset.poses_dict.keys())}
        c2ws = torch.stack([poses[int(img_id)] for img_id in ts])[:, :3]

        rays_o, rays_d = get_rays(rays[:, :3], c2ws)
        if self.hparams.dataset_name == 'llff':
            rays_o, rays_d = get_ndc_rays(self.train_dataset.img_wh[1], self.train_dataset.img_wh[0],
                                          self.train_dataset.focal, 1.0, rays_o, rays_d)
        # reassemble ray data struct
        rays_ = torch.cat([rays_o, rays_d, rays[:, 3:]], 1)
        results = self(rays_)
        loss = self.loss(results, rgbs)

        with torch.no_grad():
            typ = 'fine' if 'rgb_fine' in results else 'coarse'
            psnr_ = psnr(results[f'rgb_{typ}'], rgbs)

        self.log('lr', get_learning_rate(opt1))
        if self.hparams.refine_pose:
            self.log('lr_pose', get_learning_rate(opt2))
        self.log('train/loss', loss)
        self.log('train/psnr', psnr_, prog_bar=True)

        self.manual_backward(loss)
        opt1.step()
        opt2.step()

        # before 1.3 automatically called https://pytorch-lightning.readthedocs.io/en/latest/common/optimizers.html#learning-rate-scheduling-manual
        # scheduler1, scheduler2 = self.lr_schedulers()
        # scheduler1.step()
        # scheduler2.step()
        return loss

    def validation_step(self, batch, batch_nb):
        rays, rgbs = batch['rays'], batch['rgbs']
        rays = rays.squeeze() # (H*W, 8)
        rgbs = rgbs.squeeze() # (H*W, 3)
        results = self(rays)
        log = {'val_loss': self.loss(results, rgbs)}
        typ = 'fine' if 'rgb_fine' in results else 'coarse'
    
        if batch_nb == 0:
            W, H = self.hparams.img_wh

            if self.hparams.refine_pose:
                self.learn_poses.eval()
                poses = torch.stack([self.learn_poses(i) for i, img_id in enumerate(self.train_dataset.poses_dict.keys())])
                gt = torch.from_numpy(np.array(list(self.train_dataset.poses_dict.values())))

                '''Align est traj to gt traj'''
                c2ws_est_aligned = align_ate_c2b_use_a2b(poses, gt)  # (N, 4, 4)

                # compute ate (absolute trajectory error)
                stats_tran_est, stats_rot_est, _ = compute_ate(c2ws_est_aligned, gt, align_a2b=None)
                log['val_tr'] = torch.tensor(stats_tran_est['mean'])
                log['val_rot'] = torch.tensor(stats_rot_est['mean'])

                fig, ax = save_pose_plot(c2ws_est_aligned.cpu().numpy(), gt.cpu().numpy(), self.global_step // hparams.N_images, self.hparams.dataset_name)
                self.logger.experiment.add_figure('val/path', fig, self.global_step)

            img = results[f'rgb_{typ}'].view(H, W, 3).permute(2, 0, 1).cpu() # (3, H, W)
            img_gt = rgbs.view(H, W, 3).permute(2, 0, 1).cpu() # (3, H, W)
            depth = visualize_depth(results[f'depth_{typ}'].view(H, W)) # (3, H, W)
            stack = torch.stack([img_gt, img, depth]) # (3, 3, H, W)
            self.logger.experiment.add_images('val/GT_pred_depth', stack, self.global_step)

        psnr_ = psnr(results[f'rgb_{typ}'], rgbs)
        log['val_psnr'] = psnr_

        ssim_ = ssim(results[f'rgb_{typ}'].view(1, *self.hparams.img_wh, 3), rgbs.view(1, *self.hparams.img_wh, 3))
        log['val_ssim'] = ssim_

        return log

    def validation_epoch_end(self, outputs):
        mean_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        mean_psnr = torch.stack([x['val_psnr'] for x in outputs]).mean()
        mean_ssim = torch.stack([x['val_ssim'] for x in outputs]).mean()
        mean_tr = torch.stack([x['val_tr'] for x in outputs]).mean()
        mean_rot = torch.stack([x['val_rot'] for x in outputs]).mean()

        self.log('val/loss', mean_loss)
        self.log('val/psnr', mean_psnr, prog_bar=True)
        self.log('val/ssim', mean_ssim)
        self.log('val/tr', mean_tr)
        self.log('val/rot', mean_rot)


def main(hparams):
    N_iter_in_epoch = hparams.N_images
    max_iter = N_iter_in_epoch * hparams.num_epochs

    step_lr_iter = hparams.N_images  # change lr every n steps
    hparams.decay_step = list(range(0, max_iter, step_lr_iter))
    hparams.decay_step_pose = list(range(0, max_iter, step_lr_iter))

    seed_everything(hparams.random_seed)

    system = NeRFSystem(hparams)
    checkpoint_callback = \
        ModelCheckpoint(dirpath=os.path.join(hparams.save_path, 'ckpts', hparams.exp_name),
                        filename='{epoch:d}',
                        monitor='val/psnr',
                        mode='max',
                        save_top_k=-1)

    logger = TestTubeLogger(save_dir=os.path.join(hparams.save_path, 'logs'),
                            name=hparams.exp_name,
                            debug=False,
                            create_git_tag=False,
                            log_graph=False)

    trainer = Trainer(max_steps=max_iter,
                      val_check_interval=N_iter_in_epoch * 5,
                      checkpoint_callback=True,
                      callbacks=[checkpoint_callback],
                      resume_from_checkpoint=hparams.ckpt_path,
                      logger=logger,
                      weights_summary=None,
                      progress_bar_refresh_rate=1,
                      gpus=hparams.num_gpus,
                      accelerator='ddp' if hparams.num_gpus>1 else None,
                      num_sanity_val_steps=1,
                      benchmark=True,
                      profiler="simple" if hparams.num_gpus==1 else None)

    trainer.fit(system)


if __name__ == '__main__':
    hparams = get_opts()
    main(hparams)
