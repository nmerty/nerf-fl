import os
from collections import defaultdict

import matplotlib.pyplot as plt
from pytorch_lightning import LightningModule, Trainer, seed_everything
# pytorch-lightning
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TestTubeLogger
from torch.utils.data import DataLoader

from datasets import dataset_dict
from datasets.dataset_utils import dataset_with_img_rays_together
from datasets.ray_utils import get_ndc_rays, get_rays
from feature_losses.vgg_loss import VGGLoss
from losses import loss_dict
# metrics
from metrics import *
# models
from models.nerf import *
from models.poses import LearnPose
from models.rendering import *
from opt import get_opts
# optimizer, scheduler, visualization
from utils import *
from utils.align_traj import align_ate_c2b_use_a2b
from utils.comp_ate import compute_ate
from utils.lie_group_helper import convert3x4_4x4


def save_pose_plot(poses, gt, val_poses, current_epoch, dataset_name, title=""):
    fig = plt.figure(figsize=plt.figaspect(2))#
    fig.suptitle(title)

    ax = fig.add_subplot(211, projection='3d')
    ax2 = fig.add_subplot(212)
    # ax3 = fig.add_subplot(223, projection='3d')
    # ax4 = fig.add_subplot(224, projection='3d')
    bounds = {'llff': 0.5,
              'blender': 5,
              't&t': 0.8, }[dataset_name]

    fw = np.array([0,0,-1])
    rotGT = np.array(gt[:,:3,:3]@fw, dtype=np.float32)
    rotPose = np.array(poses[:,:3,:3]@fw, dtype=np.float32)
    rotVal = np.array(val_poses[:,:3,:3]@fw, dtype=np.float32)
    arrow_length = bounds / 15

    ax.axes.set_xlim3d(left=-bounds, right=bounds)
    ax.axes.set_ylim3d(bottom=-bounds, top=bounds)
    # ax.axes.set_zlim3d(bottom=-0.15, top=bounds)
    ax.quiver(gt[:, 0, 3], gt[:, 1, 3], gt[:, 2, 3], rotGT[:, 0], rotGT[:, 1], rotGT[:, 2], length=arrow_length, normalize=True, colors='k', label='GT')
    ax.quiver(val_poses[:, 0, 3], val_poses[:, 1, 3], val_poses[:, 2, 3], rotVal[:, 0], rotVal[:, 1], rotVal[:, 2], length=arrow_length, normalize=True, colors='b', label='val')
    ax.quiver(poses[:, 0, 3], poses[:, 1, 3], poses[:, 2, 3], rotPose[:, 0], rotPose[:, 1], rotPose[:, 2], length=arrow_length, normalize=True, colors='r', label='pred')
    ax.set_title(f"Epoch: {current_epoch}")

    ax2.plot(gt[:, 0, 3], gt[:, 1, 3], 'k.', label='GT')
    ax2.plot(poses[:, 0, 3], poses[:, 1, 3], 'r.', label='pred')
    ax2.plot(val_poses[:, 0, 3], val_poses[:, 1, 3], 'b.', label='val')
    ax2.legend()
    plt.tight_layout()

    # fig.show()
    return fig, ax


class NeRFSystem(LightningModule):
    def __init__(self, hparams):
        super(NeRFSystem, self).__init__()
        self.hparams = hparams

        self.loss = loss_dict['nerfw'](coef=1)

        self.models_to_train = []
        self.embedding_xyz = PosEmbedding(hparams.N_emb_xyz-1, hparams.N_emb_xyz, epoch_start=hparams.barf_start, epoch_end=hparams.barf_end)
        self.embedding_dir = PosEmbedding(hparams.N_emb_dir-1, hparams.N_emb_dir, epoch_start=hparams.barf_start, epoch_end=hparams.barf_end)
        self.embeddings = {'xyz': self.embedding_xyz,
                           'dir': self.embedding_dir}

        if hparams.encode_a:
            self.embedding_a = torch.nn.Embedding(hparams.N_vocab, hparams.N_a)
            self.embeddings['a'] = self.embedding_a
            self.models_to_train += [self.embedding_a]
        if hparams.encode_t:
            self.embedding_t = torch.nn.Embedding(hparams.N_vocab, hparams.N_tau)
            self.embeddings['t'] = self.embedding_t
            self.models_to_train += [self.embedding_t]

        self.nerf_coarse = NeRF('coarse',
                                in_channels_xyz=6 * hparams.N_emb_xyz + 3,
                                in_channels_dir=6 * hparams.N_emb_dir + 3)
        self.models = {'coarse': self.nerf_coarse}
        load_ckpt(self.nerf_coarse, hparams.weight_path, 'nerf_coarse')
        self.automatic_optimization = False

        self.use_feature_loss: bool = hparams.feature_loss is not None

        if hparams.feature_loss_updates != 'both' and not hparams.apply_feature_loss_exclusively:
            raise ValueError('Feature loss should be applied separately if it only updates scene or pose.')

        if hparams.N_importance > 0:
            self.nerf_fine = NeRF('fine',
                                  in_channels_xyz=6 * hparams.N_emb_xyz + 3,
                                  in_channels_dir=6 * hparams.N_emb_dir + 3,
                                  encode_appearance=hparams.encode_a,
                                  in_channels_a=hparams.N_a,
                                  encode_transient=hparams.encode_t,
                                  in_channels_t=hparams.N_tau,
                                  beta_min=hparams.beta_min)
            self.models['fine'] = self.nerf_fine
            load_ckpt(self.nerf_fine, hparams.weight_path, 'nerf_fine')
        self.models_to_train += [self.models]

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def forward(self, rays, ts):
        """Do batched inference on rays using chunk."""
        kwargs = {'current_epoch': self.global_step // self.hparams.N_images}
        B = rays.shape[0]
        results = defaultdict(list)
        for i in range(0, B, self.hparams.chunk):
            rendered_ray_chunks = \
                render_rays(self.models,
                            self.embeddings,
                            rays[i:i+self.hparams.chunk],
                            ts[i:i+self.hparams.chunk],
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
        kwargs = {'root_dir': self.hparams.root_dir}
        if self.hparams.dataset_name == 'phototourism':
            kwargs['img_downscale'] = self.hparams.img_downscale
            kwargs['val_num'] = self.hparams.num_gpus
            kwargs['use_cache'] = self.hparams.use_cache
        elif self.hparams.dataset_name == 'blender':
            kwargs['img_wh'] = tuple(self.hparams.img_wh)
            kwargs['perturbation'] = self.hparams.data_perturb
        else:
            kwargs['img_wh'] = tuple(self.hparams.img_wh)
        self.train_dataset = dataset(split='train', **kwargs)
        self.val_dataset = dataset(split='val', **kwargs)

        if self.use_feature_loss:
            fl_kwargs = kwargs.copy()
            # We might use lower resolution images
            downsample_factor = self.hparams.feature_img_downsample
            w, h = tuple(self.hparams.img_wh)
            # assert w % downsample_factor == 0 and h % downsample_factor == 0
            fl_img_wh = (w // downsample_factor, h // downsample_factor)
            if not hparams.feature_img_crop:
                # resize original image
                fl_kwargs['img_wh'] = fl_img_wh
            else:  # Crop instead of resize
                # keep original image size and take crop
                fl_kwargs['img_wh'] = self.hparams.img_wh
                fl_kwargs['feature_loss_crop_size'] = fl_img_wh

            print(f'fl_kwargs: {fl_kwargs}')
            self.train_fl_dataset = dataset(split='train', feature_loss=True, **fl_kwargs)
            self.train_fl_loader = DataLoader(
                self.train_fl_dataset,
                shuffle=True,
                num_workers=2,
                batch_size=self.hparams.fl_batch_size,
                pin_memory=True
            )
            self.train_fl_iter = iter(iter_cycle(self.train_fl_loader))

        N_images = len(self.train_dataset.poses_dict.keys())
        assert N_images == hparams.N_images, f"Set number of images in args to {N_images}"
        c2ws = convert3x4_4x4(torch.from_numpy(self.train_dataset.poses))
        refine_pose = self.hparams.refine_pose

        initial_poses = c2ws.float() if not refine_pose or hparams.pose_init in ['original', 'perturb'] else None
        self.learn_poses = LearnPose(len(self.train_dataset.poses_dict.keys()), refine_pose, refine_pose,
                                     init_c2w=initial_poses, perturb_sigma=self.hparams.pose_sigma)#.to(self.device)
        # load_ckpt(self.learn_poses, hparams...)

    def configure_optimizers(self):
        optimizer = get_optimizer(self.hparams, self.models_to_train)
        scheduler = get_scheduler(self.hparams, optimizer)

        if self.hparams.refine_pose:
            optimizer_pose = Adam(get_parameters(self.learn_poses), lr=self.hparams.lr_pose, eps=1e-8,
                                  weight_decay=self.hparams.weight_decay)
            scheduler_pose = MultiStepLR(optimizer_pose, milestones=self.hparams.decay_step_pose,
                                         gamma=self.hparams.decay_gamma_pose)

            return ({'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'}},
                    {'optimizer': optimizer_pose, 'lr_scheduler': {'scheduler': scheduler_pose, 'interval': 'step'}},)
        return ({'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'}})

    def train_dataloader(self):
        if self.hparams.img_rays_together:
            ds = dataset_with_img_rays_together(self.train_dataset,
                                                self.hparams.img_wh,
                                                self.hparams.batch_size,
                                                self.hparams.num_imgs_in_batch)
            shuffle = False  # already shuffled
        else:
            ds = self.train_dataset
            shuffle = True

        return DataLoader(ds,
                          shuffle=shuffle,
                          num_workers=4,
                          batch_size=self.hparams.batch_size,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          shuffle=False,
                          num_workers=4,
                          batch_size=1, # validate one image (H*W rays) at a time
                          pin_memory=True)

    def get_feature_loss(self, content_weight=1.0, style_weight=None, ):
        if self.hparams.feature_loss == 'vgg':
            vgg_loss = VGGLoss(
                style_weight=style_weight,
                content_weight=content_weight,
                content_layers=["conv_1", "conv_2", "conv_3", "conv_4", "conv_5"],
                style_layers=[],
                device=self.device
            )
            return vgg_loss
        else:
            raise NotImplementedError

    def training_step(self, batch, batch_nb, optimizer_idx):
        opt1, opt2 = self.optimizers()

        # Apply feature loss every n batch
        _apply_feature_loss = self.use_feature_loss and batch_nb % self.hparams.fl_every_n_batch == 0

        rays, rgbs, ts = batch['rays'], batch['rgbs'], batch['ts']

        self.learn_poses.train()

        poses = [self.learn_poses(i) for i in range(self.learn_poses.num_cams)]
        c2ws = torch.stack([poses[int(img_id)] for img_id in ts])[:, :3]

        rays_o, rays_d = get_rays(rays[:, :3], c2ws)
        if self.hparams.dataset_name == 'llff':
            rays_o, rays_d = get_ndc_rays(self.train_dataset.img_wh[1], self.train_dataset.img_wh[0],
                                          self.train_dataset.focal, 1.0, rays_o, rays_d)
        # reassemble ray data struct
        rays_ = torch.cat([rays_o, rays_d, rays[:, 3:]], 1)
        results = self(rays_, ts)
        loss_d = self.loss(results, rgbs)
        loss = sum(l for l in loss_d.values())

        typ = 'fine' if 'rgb_fine' in results else 'coarse'
        with torch.no_grad():
            psnr_ = psnr(results[f'rgb_{typ}'], rgbs)

        self.log('lr', get_learning_rate(opt1))
        if self.hparams.refine_pose:
            self.log('lr_pose', get_learning_rate(opt2))
        self.log('train/loss', loss)
        self.log('train/psnr', psnr_, prog_bar=True)

        opt1.zero_grad()
        opt2.zero_grad()

        # Use same compute graph for both losses if they share the pose input
        retain_graph = _apply_feature_loss and not hparams.apply_feature_loss_exclusively
        self.manual_backward(loss, retain_graph=retain_graph)  # Backward pass for NeRF loss

        if _apply_feature_loss and not hparams.apply_feature_loss_exclusively:
            # Combined update step for NeRF and feature loss
            # Done later
            pass
        elif _apply_feature_loss and hparams.apply_feature_loss_exclusively:
            # Step for NeRF loss and reset gradients for feature loss backward pass
            opt1.step()
            opt2.step()
            opt1.zero_grad()
            opt2.zero_grad()
        elif not _apply_feature_loss:
            # Step for NeRF loss -> done
            opt1.step()
            opt2.step()

        # Apply feature loss
        if _apply_feature_loss:
            print('_apply_feature_loss')
            # Do inference again if we don't want to use the same compute graph
            poses = poses if retain_graph else [self.learn_poses(i) for i in range(self.learn_poses.num_cams)]
            feature_loss = self.feature_forward(poses)
            self.manual_backward(feature_loss)
            if hparams.feature_loss_updates == 'scene':
                # Feature loss only updates scene
                opt1.step()
            elif hparams.feature_loss_updates == 'pose':
                # Feature loss only updates pose
                opt2.step()
            else:
                opt1.step()
                opt2.step()

        # before 1.3 automatically called https://pytorch-lightning.readthedocs.io/en/latest/common/optimizers.html
        # #learning-rate-scheduling-manual
        # scheduler1, scheduler2 = self.lr_schedulers()
        # scheduler1.step()
        # scheduler2.step()
        # return loss

    def feature_forward(self, poses):
        """
        Do a forward pass for the feature loss.

        Returns: Feature loss result
        """

        feature_loss_ = self.get_feature_loss(content_weight=1.0, style_weight=0.0).to(self.device)
        fl_batch = next(self.train_fl_iter)

        fl_rays, fl_ts, fl_imgs_gt = fl_batch['rays'].to(self.device), fl_batch['ts'].to(self.device), fl_batch[
            'img'].to(self.device)
        fl_c2ws = torch.stack([poses[int(img_id)] for img_id in fl_ts])[:, :3]  # (N_images, 3)

        # Merge first two dimensions i.e. use batch of rays instead of batch of images for inference
        # (N_images, h * w, 6) -> (N_images * h * w, 6)
        fl_n_images, fl_n_rays_per_image = fl_rays.shape[0], fl_rays.shape[1]
        fl_rays = fl_rays.view(fl_n_images * fl_n_rays_per_image, -1)

        # Repeat c2ws for rays of the same image (N_images, 3) -> (N_images*h*w, 3)
        fl_c2ws = torch.repeat_interleave(fl_c2ws, repeats=fl_n_rays_per_image, dim=0)

        # Same as NeRF inference
        fl_rays_o, fl_rays_d = get_rays(fl_rays[:, :3], fl_c2ws)
        if hparams.feature_img_crop:
            fl_w, fl_h = self.train_fl_dataset.feature_loss_crop_size
        else:
            fl_w, fl_h = self.train_fl_dataset.img_wh

        if self.hparams.dataset_name == 'llff':
            fl_rays_o, fl_rays_d = get_ndc_rays(fl_h, fl_w, self.train_fl_dataset.focal, 1.0, fl_rays_o, fl_rays_d)
        # reassemble ray data struct
        fl_rays_ = torch.cat([fl_rays_o, fl_rays_d, fl_rays[:, 3:]], 1)
        fl_results = self(fl_rays_)

        typ = 'fine' if 'rgb_fine' in fl_results else 'coarse'
        # Unmerge images dimension
        fl_imgs = fl_results[f'rgb_{typ}'] \
            .view(fl_n_images, fl_h, fl_w, 3) \
            .permute(0, 3, 1, 2)  # (B, 3, H, W)

        # Put a feature loss
        # for img_idx in range(len(fl_imgs)):
        #     img_gt = fl_imgs_gt[img_idx]
        #     img = fl_imgs[img_idx].unsqueeze(0)  # fake batch dimension
        #     feature_loss += self.feature_loss(img, img_gt, img_gt)
        feature_loss = feature_loss_(fl_imgs, fl_imgs_gt, fl_imgs_gt)
        self.log(f'train/{self.hparams.feature_loss}_loss', feature_loss)
        feature_loss = feature_loss * hparams.feature_loss_coeff
        return feature_loss

    def validation_step(self, batch, batch_nb):
        log = {}
        rays, rgbs, ts, c2w = batch['rays'], batch['rgbs'], batch['ts'], batch['c2w']
        rays = rays.squeeze() # (H*W, 6)
        rgbs = rgbs.squeeze() # (H*W, 3)
        ts = ts.squeeze()  # val id

        self.learn_poses.eval()

        if self.hparams.refine_pose or (self.current_epoch == 0 and batch_nb == 0):
            poses = torch.stack([self.learn_poses(i) for i in range(self.learn_poses.num_cams)])
            gt = torch.from_numpy(self.train_dataset.poses)

            '''Align est traj to gt traj'''
            val_pose_aligned = align_ate_c2b_use_a2b(gt, poses, c2w)  # (N, 4, 4) gt val pose aligned to pred
            if batch_nb == 0:
                c2ws_est_aligned = align_ate_c2b_use_a2b(poses, gt)  # (N, 4, 4)
                # compute ate for training poses (absolute trajectory error)
                stats_tran_est, stats_rot_est, _ = compute_ate(c2ws_est_aligned, gt, align_a2b=None)
                log['val_tr'] = torch.tensor(stats_tran_est['mean'])
                log['val_rot'] = torch.tensor(stats_rot_est['mean'])

                # gt coord system, val pose already there
                val_poses2plot = self.val_dataset.val_poses
                fig, ax = save_pose_plot(c2ws_est_aligned.cpu().numpy(), gt.cpu().numpy(), val_poses2plot, self.global_step // hparams.N_images, self.hparams.dataset_name, "GT space")
                self.logger.experiment.add_figure('val/path', fig, self.global_step)
                # fig, ax = save_pose_plot(poses.cpu().numpy(), align_ate_c2b_use_a2b(gt.cpu(), poses.cpu()).numpy(), val_pose_aligned.cpu().numpy(), self.global_step // hparams.N_images, self.hparams.dataset_name, "pred space")
                # self.logger.experiment.add_figure('val/path_estimate', fig, self.global_step)
        else:
            val_pose_aligned = c2w

        rays_o, rays_d = get_rays(rays[:, :3], val_pose_aligned[:, :3].to(rays.device))
        if self.hparams.dataset_name == 'llff':
            rays_o, rays_d = get_ndc_rays(self.train_dataset.img_wh[1], self.train_dataset.img_wh[0],
                                          self.train_dataset.focal, 1.0, rays_o, rays_d)
        # reassemble ray data struct
        rays_ = torch.cat([rays_o, rays_d, rays[:, 3:]], 1)
        results = self(rays_, ts)  # run inference

        typ = 'fine' if 'rgb_fine' in results else 'coarse'
        loss_d = self.loss(results, rgbs)
        log['val_loss'] = sum(l for l in loss_d.values())

        W, H = self.hparams.img_wh
        img = results[f'rgb_{typ}'].view(H, W, 3).permute(2, 0, 1)  # (3, H, W)
        img_gt = rgbs.view(H, W, 3).permute(2, 0, 1)  # (3, H, W)

        # plot validation image
        if batch_nb == 0:
            depth = visualize_depth(results[f'depth_{typ}'].view(H, W))  # (3, H, W)
            stack = torch.stack([img_gt.cpu(), img.cpu(), depth])  # (3, 3, H, W)
            self.logger.experiment.add_images('val/GT_pred_depth', stack, self.global_step)

        psnr_ = psnr(results[f'rgb_{typ}'], rgbs)
        log['val_psnr'] = psnr_

        ssim_ = ssim(results[f'rgb_{typ}'].view(1, *self.hparams.img_wh, 3), rgbs.view(1, *self.hparams.img_wh, 3))
        log['val_ssim'] = ssim_

        if self.use_feature_loss:
            feature_loss_ = self.get_feature_loss().to(self.device)
            img = img.unsqueeze(0)
            img_gt = img_gt.unsqueeze(0)
            feature_loss = feature_loss_(img, img_gt, img_gt)
            log[f'val_{self.hparams.feature_loss}_loss'] = feature_loss
            log[f'val_total_loss'] = feature_loss + log['val_loss']

        return log

    def validation_epoch_end(self, outputs):
        mean_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        mean_psnr = torch.stack([x['val_psnr'] for x in outputs]).mean()
        mean_ssim = torch.stack([x['val_ssim'] for x in outputs]).mean()

        self.log('val/loss', mean_loss)
        self.log('val/psnr', mean_psnr, prog_bar=True)
        self.log('val/ssim', mean_ssim)

        if self.hparams.refine_pose:
            mean_tr = outputs[0]['val_tr']
            mean_rot = outputs[0]['val_rot']
            self.log('val/tr', mean_tr)
            self.log('val/rot', mean_rot)


def main(hparams):
    N_iter_in_epoch = hparams.N_images
    max_iter = N_iter_in_epoch * hparams.num_epochs

    if hparams.lr_scheduler == 'steplr':
        step_lr_iter = hparams.N_images  # change lr every n steps

        hparams.decay_step = list(range(0, max_iter, step_lr_iter))
        hparams.decay_step_pose = list(range(0, max_iter, step_lr_iter))

        hparams.decay_gamma = np.power(hparams.lr_end/hparams.lr, 1./hparams.num_epochs)
        hparams.decay_gamma_pose = np.power(hparams.lr_pose_end/hparams.lr_pose, 1./hparams.num_epochs)

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
                            debug=hparams.debug,
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
                      profiler="simple" if hparams.num_gpus == 1 else None,
                      reload_dataloaders_every_epoch=hparams.img_rays_together)  # needed if img_rays_together

    trainer.fit(system)


if __name__ == '__main__':
    hparams = get_opts()
    main(hparams)
