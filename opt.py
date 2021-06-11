import argparse


def get_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument('--root_dir', type=str,
                        default='/home/ubuntu/data/nerf_example_data/nerf_synthetic/lego',
                        help='root directory of dataset')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Set random seed everywhere')
    parser.add_argument('--N_images', type=int, required=True,
                        help='Number of images in dataset')
    parser.add_argument('--dataset_name', type=str, default='blender',
                        choices=['blender', 'llff', 't&t'],
                        help='which dataset to train/val')
    parser.add_argument('--img_wh', nargs="+", type=int, default=[800, 800],
                        help='resolution (img_w, img_h) of the image')
    parser.add_argument('--spheric_poses', default=False, action="store_true",
                        help='whether images are taken in spheric poses (for llff)')

    parser.add_argument('--N_emb_xyz', type=int, default=10,
                        help='number of xyz embedding frequencies')
    parser.add_argument('--N_emb_dir', type=int, default=4,
                        help='number of direction embedding frequencies')
    parser.add_argument('--N_samples', type=int, default=64,
                        help='number of coarse samples')
    parser.add_argument('--N_importance', type=int, default=128,
                        help='number of additional fine samples')
    parser.add_argument('--N_hidden_units', type=int, default=256,
                        help='number of hidden units in each layer')
    parser.add_argument('--N_layers', type=int, default=8,
                        help='number of layers')
    parser.add_argument('--skip_connections', nargs='+', type=int, default=[4],
                        help='skip connections for pose embedding')

    parser.add_argument('--use_disp', default=False, action="store_true",
                        help='use disparity depth sampling')
    parser.add_argument('--perturb', type=float, default=0.0,
                        help='factor to perturb depth sampling points')
    parser.add_argument('--noise_std', type=float, default=1.0,
                        help='std dev of noise added to regularize sigma')

    # Unknown/inaccurate camera poses
    parser.add_argument('--refine_pose', default=False, action="store_true",
                        help='whether to refine input camera poses')
    parser.add_argument('--pose_init', type=str, choices=['identity', 'perturb', 'original'], default='original',
                        help='How to initialize poses when optimizing for them too')
    parser.add_argument('--pose_sigma', type=float, default=0,
                        help='Perturb initial pose by additive noise sampled from normal dist with this sigma')
    # BARF https://arxiv.org/pdf/2104.06405.pdf
    parser.add_argument('--barf_start', type=int, default=-1,
                        help='Set alpha between start and end for pos encoding')
    parser.add_argument('--barf_end', type=int, default=-1,
                        help='Set alpha between start and end for pos encoding')

    # parser.add_argument('--decay_step_pose', nargs='+', type=int, default=list(range(0,370000,3700)),
    #                     help='scheduler decay step for pose params')
    # parser.add_argument('--decay_gamma_pose', type=float, default=0.9,
    #                     help='learning rate decay amount for pose params')
    parser.add_argument('--lr_pose', type=float, default=1e-3,
                        help='learning rate for pose')

    parser.add_argument('--batch_size', type=int, default=1024,
                        help='batch size')
    parser.add_argument('--chunk', type=int, default=32 * 1024,
                        help='chunk size to split the input to avoid OOM')
    parser.add_argument('--num_epochs', type=int, default=10000,
                        help='number of training epochs')
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='number of gpus')

    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='pretrained checkpoint to load (including optimizers, etc)')
    parser.add_argument('--prefixes_to_ignore', nargs='+', type=str, default=['loss'],
                        help='the prefixes to ignore in the checkpoint state dict')
    parser.add_argument('--weight_path', type=str, default=None,
                        help='pretrained model weight to load (do not load optimizers, etc)')

    parser.add_argument('--optimizer', type=str, default='adam',
                        help='optimizer type',
                        choices=['sgd', 'adam', 'radam', 'ranger'])
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='learning rate momentum')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--lr_scheduler', type=str, default='steplr',
                        help='scheduler type',
                        choices=['steplr', 'cosine', 'poly'])
    #### params for warmup, only applied when optimizer == 'sgd' or 'adam'
    parser.add_argument('--warmup_multiplier', type=float, default=1.0,
                        help='lr is multiplied by this factor after --warmup_epochs')
    parser.add_argument('--warmup_epochs', type=int, default=0,
                        help='Gradually warm-up(increasing) learning rate in optimizer')
    ###########################
    #### params for steplr ####
    parser.add_argument('--lr_end', type=float, default=5e-4,
                        help='learning rate at last epoch')
    parser.add_argument('--lr_pose_end', type=float, default=1e-3,
                        help='learning rate for pose')
    # parser.add_argument('--decay_step', nargs='+', type=int, default=list(range(0,370000,370)),
    #                     help='[SET AUTOMATICALLY] scheduler decay step')
    parser.add_argument('--decay_gamma', type=float, default=0.9954,
                        help='learning rate decay amount')
    ###########################
    #### params for poly ####
    parser.add_argument('--poly_exp', type=float, default=0.9,
                        help='exponent for polynomial learning rate decay')
    ###########################

    parser.add_argument('--save_path', type=str, default='./',
                        help='paths to save checkpoints and logs to')
    parser.add_argument('--exp_name', type=str, default='exp',
                        help='experiment name')

    # Feature Losses
    parser.add_argument('--feature_loss',
                        help='Feature loss to use.',
                        default=None,
                        type=str,
                        choices=['vgg'],
                        )
    parser.add_argument('--fl_batch_size',
                        help='batch size for the feature loss i.e. number of images',
                        default=1,
                        type=int,
                        )
    parser.add_argument('--fl_every_n_batch',
                        help='frequency of applying feature loss',
                        type=int,
                        )
    parser.add_argument('--feature_img_downsample', type=int, default=4,
                        help='downsampling factor for the resolution of the image to be used for the feature loss')
    parser.add_argument('--apply_feature_loss_exclusively', action='store_true',
                        help='Do not apply NeRF when applying feature loss.')
    parser.add_argument('--feature_loss_coeff', type=float, default=1e-3,
                        help='Multiplier of feature loss contribution to total loss.')
    parser.add_argument('--feature_loss_updates', choices=['scene', 'pose', 'both'], default='both',
                        help='Which parameters to update with the feature loss.')
    parser.add_argument('--debug', '-D', action='store_true')

    return parser.parse_args()
