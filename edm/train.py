# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Train diffusion-based generative model using the techniques described in the
paper "Elucidating the Design Space of Diffusion-Based Generative Models"."""

from __future__ import annotations
import os
import re
import json
import click
import torch
import dnnlib
# import wandb

from torch_utils import distributed as dist
from training import training_loop

import warnings
warnings.filterwarnings('ignore', 'Grad strides do not match bucket view strides') # False warning printed by PyTorch 1.12.

#----------------------------------------------------------------------------
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]


def parse_tuple(t: tuple[str, ...]):
    return [int(e) for e in t]


def wandb_init(project, group, log, runname):
    if os.access("/home/tikai103", os.R_OK):
        os.environ["WANDB_API_KEY"] = "a9efb3b6cddc090dbf125d4c5d0dff12b178eb36"
    wandb.init(project=project, entity="hhu-mmbs", group=group)
    wandb.run.name = runname
    wandb.config.update(log)


def get_dataset_path(dataset):
    # get dataset path
    if dataset == 'cifar10':
        dataset_folder = '/home/shared/DataSets/cifar-10'
        dataset_path = f'{dataset_folder}/cifar10-32x32.zip'
        n_train = 50000
    elif dataset == 'cifar100':
        dataset_folder = '/home/shared/DataSets/cifar-100'
        dataset_path = f'{dataset_folder}/cifar100-32x32.zip'
        n_train = 50000
    elif dataset == 'ffhq':
        dataset_folder = '/home/shared/DataSets/vision_benchmarks/FFHQ-i'
        dataset_path = f'{dataset_folder}/ffhq-64x64.zip'
        n_train = 70000
    elif dataset == 'imagenet':
        dataset_folder = '/home/shared/DataSets/vision_benchmarks/IN_64x64_karras'
        dataset_path = f'{dataset_folder}/imagenet-64x64.zip'
        n_train = 1281167
    elif dataset == 'mnist':
        dataset_folder = '/home/shared/DataSets/MNIST'
        dataset_path = f'{dataset_folder}/mnist-32x32.zip'
        n_train = 60000
    elif dataset == 'afhqv2':
        dataset_folder = '/home/shared/DataSets/vision_benchmarks/AFHQ-v2'
        dataset_path = f'{dataset_folder}/afhqv2-64x64.zip'
        n_train = 15803
    elif dataset == 'cifar100-coarse':
        dataset_folder = '/home/shared/DataSets/cifar-100'
        dataset_path = f'{dataset_folder}/cifar100-coarse-32x32.zip'
        n_train = 50000
    elif dataset == 'cifar10_incp_804_512':
        dataset_folder = '/home/shared/DataSets/cifar-10'
        dataset_path = f'{dataset_folder}/cifar10-32x32_incp_804_512.zip'
        n_train = 44803
    elif dataset == 'cifar10_dino_804_512':
        dataset_folder = '/home/shared/DataSets/cifar-10'
        dataset_path = f'{dataset_folder}/cifar10-32x32_dino_804_512.zip'
        n_train = 39195
    elif dataset == 'cifar10_incp_dino_804_512':
        dataset_folder = '/home/shared/DataSets/cifar-10'
        dataset_path = f'{dataset_folder}/cifar10-32x32_incp_dino_804_512.zip'
        n_train = 48085
    else:
        assert False, f'Dataset {dataset} not found.'
    return dataset_path, dataset_folder, n_train


def parse_int_list(s):
    """Parse a comma separated list of numbers or ranges and return a list of ints.
       Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]"""
    if isinstance(s, list): return s
    if s == "none": return []
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2)) + 1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------

@click.command()
# Main options.
@click.option('--outdir',           help='Where to save the results', metavar='DIR',                    type=str, required=True)
@click.option('--dataset',          help='Dataset Name', metavar='STR',                                 type=str, required=True)
@click.option('--cond',             help='Train class-conditional model', metavar='BOOL',               type=bool, default=True, show_default=True)
@click.option('--arch',             help='Network architecture', metavar='ddpmpp|ncsnpp|adm',           type=click.Choice(['ddpmpp', 'ncsnpp', 'adm']), default='ddpmpp', show_default=True)
@click.option('--n_val',            help='Size of validation set', metavar='INT',                       type=click.IntRange(min=0), default=0, show_default=True)
@click.option('--n_train',          help='Limit the size of the training dataset', metavar='INT',       type=click.IntRange(min=0), default=0, show_default=True)
@click.option('--overfit',          help='Overfit to a single image', metavar='BOOL',                   type=bool, default=False, show_default=True)
@click.option('--subclass',         help='Restrict dataset to one label subclass', metavar='INT',       type=int, default=-1, show_default=True)

# Loss options
@click.option('--loss_fn',          help='Loss function', metavar='str',                                type=click.Choice(['karras', 'ours', 'eff']), show_default=True)
@click.option('--update_eff_ema',   help='How often to update the ema for efficient loss', metavar='TICKS', type=click.IntRange(min=1), default=1, show_default=True)
@click.option('--eff_ema_weight',   help='Weight for the efficient loss EMA', metavar='FLOAT',         type=click.FloatRange(min=0), default=0.5, show_default=True)
@click.option('--net_neg_pkl',      help='Pretrained negative network for finetuning loss',             type=str, default=None, show_default=True)
@click.option('--range_alpha',     help='Use range weight for residual loss', metavar='BOOL',          type=float, default=0, show_default=True)
@click.option('--sigma_eps',        help='Sigma_alpha distribution hyperparameter', metavar='FLOAT',    type=click.FloatRange(min=0), show_default=True)
@click.option('--sigma_od_min',     help='Minimal value for OD', metavar='FLOAT',                       type=click.FloatRange(min=0), show_default=True)
@click.option('--sigma_od_max',     help='Maximal value for OD', metavar='FLOAT',                       type=click.FloatRange(min=0), show_default=True)
@click.option('--gamma',            help='Exponent of asymptotic correction', metavar='FLOAT',          type=click.FloatRange(min=0), show_default=True)
@click.option('--p_uncond',         help='probability to drop condition',                               type=float, default=0, show_default=True)

# Hyperparameters.
@click.option('--duration',      help='Training duration', metavar='KIMG',                          type=click.FloatRange(min=0, min_open=True), default=200, show_default=True)
@click.option('--batch',         help='Total batch size', metavar='INT',                            type=click.IntRange(min=1), default=512, show_default=True)
@click.option('--batch-gpu',     help='Limit batch size per GPU', metavar='INT',                    type=click.IntRange(min=1))
@click.option('--cbase',         help='Channel multiplier  [default: varies]', metavar='INT',       type=int)
@click.option('--cres',          help='Channels per resolution  [default: varies]', metavar='LIST', type=parse_int_list)
@click.option('--num_blocks',    help='Number of residual blocks per resolution.', metavar='INT',   type=int, default=None, show_default=True)
@click.option('--lr',            help='Learning rate', metavar='FLOAT',                             type=click.FloatRange(min=0, min_open=True), default=1e-3, show_default=True)
@click.option('--lr_rampup',     help='Learning rate rampup', metavar='KIMG',                       type=click.FloatRange(min=0, min_open=True), default=10000, show_default=True)  # duration / 20
@click.option('--weight_decay',  help='Weight decay', metavar='FLOAT',                              type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--ema',           help='EMA half-life', metavar='KIMG',                              type=click.FloatRange(min=0), default=0.5, show_default=True)  # was 0.5 MIMG, duration / 400
@click.option('--dropout',       help='Dropout probability', metavar='FLOAT',                       type=click.FloatRange(min=0, max=1), default=0.13, show_default=True)
@click.option('--augment',       help='Augment probability', metavar='FLOAT',                       type=click.FloatRange(min=0, max=1), default=0.12, show_default=True)
@click.option('--xflip',         help='Enable dataset x-flips', metavar='BOOL',                     type=bool, default=False, show_default=True)

# Performance-related.
@click.option('--fp16',          help='Enable mixed-precision training', metavar='BOOL',            type=bool, default=False, show_default=True)
@click.option('--ls',            help='Loss scaling', metavar='FLOAT',                              type=click.FloatRange(min=0, min_open=True), default=1, show_default=True)
@click.option('--bench',         help='Enable cuDNN benchmarking', metavar='BOOL',                  type=bool, default=True, show_default=True)
@click.option('--cache',         help='Cache dataset in CPU memory', metavar='BOOL',                type=bool, default=True, show_default=True)
@click.option('--workers',       help='DataLoader worker processes', metavar='INT',                 type=click.IntRange(min=1), default=1, show_default=True)

# I/O-related.
@click.option('--desc',          help='String to include in result dir name', metavar='STR',        type=str)
@click.option('--nosubdir',      help='Do not create a subdirectory for results',                   type=bool, default=True)
@click.option('--tick',          help='How often to print progress', metavar='KIMG',                type=click.IntRange(min=1), default=50, show_default=True)
@click.option('--snap',          help='How often to save snapshots', metavar='TICKS',               type=click.IntRange(min=1), default=50, show_default=True)
@click.option('--dump',          help='How often to dump state', metavar='TICKS',                   type=click.IntRange(min=1), default=500, show_default=True)
@click.option('--seed',          help='Random seed  [default: random]', metavar='INT',              type=int)
@click.option('--transfer',      help='Transfer learning from network pickle', metavar='PKL|URL',   type=str)
@click.option('--resume',        help='Resume from previous training state', metavar='PT',          type=str)
@click.option('-n', '--dry-run', help='Print training options and exit',                            is_flag=True)
@click.option('--wandb',         help='Enable Weights & Biases logging', metavar='BOOL',            type=bool, default=False, show_default=True)
def main(**kwargs):
    """Train diffusion-based generative model using the techniques described in the
    paper "Elucidating the Design Space of Diffusion-Based Generative Models".

    Examples:

    \b
    # Train DDPM++ model for class-conditional CIFAR-10 using 8 GPUs
    torchrun --standalone --nproc_per_node=8 train.py --outdir=training-runs \\
        --data=datasets/cifar10-32x32.zip --cond=1 --arch=ddpmpp
    """
    opts = dnnlib.EasyDict(kwargs)
    torch.multiprocessing.set_start_method('spawn')
    dist.init()

    dataset_path = get_dataset_path(opts.dataset)[0] if not os.path.isdir(opts.dataset) else opts.dataset

    # Initialize config dict.
    c = dnnlib.EasyDict()
    c.dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=dataset_path,
                                       use_labels=opts.cond, xflip=opts.xflip,
                                       cache=opts.cache, subclass=opts.subclass,
                                       p_uncond=opts.p_uncond, return_idx=True)
    c.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, num_workers=opts.workers, prefetch_factor=2)
    c.network_kwargs = dnnlib.EasyDict()
    c.loss_kwargs = dnnlib.EasyDict()
    c.optimizer_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', lr=opts.lr, betas=[0.9,0.999], eps=1e-8,
                                         weight_decay=opts.weight_decay)
    c.lr_rampup_kimg = opts.lr_rampup

    try:
        dataset_obj = dnnlib.util.construct_class_by_name(**c.dataset_kwargs)
        n_train = len(dataset_obj) if opts.n_train == 0 else opts.n_train
        n_train = 1 if opts.overfit else n_train
        if opts.batch > n_train:
            opts.batch = n_train
            dist.print0(f'UserWarning: Batch size {opts.batch} cannot exceed dataset size {n_train}, reducing to {n_train}.')
        dataset_name = dataset_obj.name
        assert opts.n_train + opts.n_val <= dataset_obj.raw_shape[0], f'--n_train + --n_val cannot exceed dataset size ({dataset_obj.raw_shape[0]} samples)'
        if n_train != 0 and n_train != dataset_obj.raw_shape[0]:
            dist.print0(f'Limiting dataset size to {n_train} samples.')
        c.dataset_kwargs.max_size = n_train
        # c.dataset_kwargs.seq_len = dataset_obj.seq_len # be explicit about dataset resolution
        # c.dataset_kwargs.n_train = len(dataset_obj) # be explicit about dataset size
        del dataset_obj  # conserve memory
    except IOError as err:
        raise click.ClickException(f'--data: {err}')

    # Validation set
    c.n_val = opts.n_val

    # Network architecture.
    if opts.arch == 'ddpmpp':
        c.network_kwargs.update(model_type='SongUNet', embedding_type='positional', encoder_type='standard', decoder_type='standard')
        c.network_kwargs.update(channel_mult_noise=1, resample_filter=[1,1], model_channels=128, channel_mult=[2,2,2])
    elif opts.arch == 'ncsnpp':
        c.network_kwargs.update(model_type='SongUNet', embedding_type='fourier', encoder_type='residual', decoder_type='standard')
        c.network_kwargs.update(channel_mult_noise=2, resample_filter=[1,3,3,1], model_channels=128, channel_mult=[2,2,2])
    else:
        assert opts.arch == 'adm'
        c.network_kwargs.update(model_type='DhariwalUNet', model_channels=192, channel_mult=[1,2,3,4])

    # Preconditioning & loss function.
    c.network_kwargs.class_name = 'training.networks.EDMPrecond'
    if opts.loss_fn == 'karras':
        c.loss_kwargs.class_name = 'training.loss.EDMLoss'
    elif opts.loss_fn == 'ours':
        c.loss_kwargs.class_name = 'training.loss.ResidualLossCont'
        # Sigma distribution
        c.loss_kwargs.sigma_eps = opts.sigma_eps
        c.loss_kwargs.sigma_od_min = opts.sigma_od_min
        c.loss_kwargs.sigma_od_max = opts.sigma_od_max
        c.loss_kwargs.gamma = opts.gamma
        c.loss_kwargs.range_alpha = opts.range_alpha
        if opts.range_alpha:
            assert opts.resume is not None, f"Currently only supports finetuning"
            assert opts.net_neg_pkl is not None, f"Currently only supports finetuning"
        c.net_neg_pkl = opts.net_neg_pkl
    elif opts.loss_fn == 'eff':
        c.loss_kwargs.class_name = 'training.loss.EfficientLoss'
        assert opts.resume is not None, f"Currently only supports finetuning"
        # Sigma distribution
        c.loss_kwargs.sigma_od_min = opts.sigma_od_min
        c.loss_kwargs.sigma_od_max = opts.sigma_od_max
        c.loss_kwargs.eff_ema_weight = opts.eff_ema_weight
        c.eff_ema_ticks = opts.update_eff_ema
    else:
        raise click.ClickException(f'--loss_fn: invalid loss function {opts.loss_fn}')

    # Network options.
    if opts.cbase is not None:
        c.network_kwargs.model_channels = opts.cbase
    if opts.cres is not None:
        c.network_kwargs.channel_mult = opts.cres
    if opts.num_blocks is not None:
        c.network_kwargs.num_blocks = opts.num_blocks
    if opts.augment:
        c.augment_kwargs = dnnlib.EasyDict(class_name='training.augment.AugmentPipe', p=opts.augment)
        c.augment_kwargs.update(xflip=1e8, yflip=1, scale=1, rotate_frac=1, aniso=1, translate_frac=1)
        c.network_kwargs.augment_dim = 9
    c.network_kwargs.update(dropout=opts.dropout, use_fp16=opts.fp16)

    # Training options.
    c.total_kimg = max(int(opts.duration * 1000), 1)
    c.ema_halflife_kimg = int(opts.ema * 1000)
    c.update(batch_size=opts.batch, batch_gpu=opts.batch_gpu)
    c.update(loss_scaling=opts.ls, cudnn_benchmark=opts.bench)
    c.update(kimg_per_tick=opts.tick, snapshot_ticks=opts.snap, state_dump_ticks=opts.dump)

    # Random seed.
    if opts.seed is not None:
        c.seed = opts.seed
    else:
        seed = torch.randint(1 << 31, size=[], device=torch.device('cuda'))
        torch.distributed.broadcast(seed, src=0)
        c.seed = int(seed)

    # Transfer learning and resume.
    if opts.transfer is not None:
        if opts.resume is not None:
            raise click.ClickException('--transfer and --resume cannot be specified at the same time')
        c.resume_pkl = opts.transfer
        c.ema_rampup_ratio = None
    elif opts.resume is not None:
        match = re.fullmatch(r'training-state-(\d+).pt', os.path.basename(opts.resume))
        if not match or not os.path.isfile(opts.resume):
            raise click.ClickException('--resume must point to training-state-*.pt from a previous training run')
        c.resume_pkl = os.path.join(os.path.dirname(opts.resume), f'network-snapshot-{match.group(1)}.pkl')
        c.resume_kimg = int(match.group(1))
        c.resume_state_dump = opts.resume

    # Description string.
    cond_str = 'cond' if c.dataset_kwargs.use_labels else 'uncond'
    desc = f'{dataset_name:s}-{cond_str:s}'
    if opts.desc is not None:
        desc = f'{opts.desc}_{desc}'

    # Pick output directory.
    if dist.get_rank() != 0:
        c.run_dir = None
    elif opts.nosubdir:
        c.run_dir = opts.outdir
    else:
        prev_run_dirs = []
        if os.path.isdir(opts.outdir):
            prev_run_dirs = [x for x in os.listdir(opts.outdir) if os.path.isdir(os.path.join(opts.outdir, x))]
        prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
        prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
        cur_run_id = max(prev_run_ids, default=-1) + 1
        c.run_dir = os.path.join(opts.outdir, f'{cur_run_id:05d}-{desc}')
        assert not os.path.exists(c.run_dir)

    if opts.wandb and dist.get_rank() == 0:
        runname = "-".join(c.run_dir.split('/')[6:])
        wandb_init("Residual_CIFAR", "RESIDUAL-CIFAR10", opts, runname)
        c.use_wandb = True
    else:
        c.use_wandb = False

    # Print options.
    dist.print0()
    dist.print0('Training options:')
    dist.print0(json.dumps(c, indent=2))
    dist.print0()
    dist.print0(f'Output directory:        {c.run_dir}')
    dist.print0(f'Dataset path:            {c.dataset_kwargs.path}')
    dist.print0(f'Class-conditional:       {c.dataset_kwargs.use_labels}')
    dist.print0(f'Network architecture:    {opts.arch}')
    dist.print0(f'Loss:                    {opts.loss_fn}')
    dist.print0(f'Number of GPUs:          {dist.get_world_size()}')
    dist.print0(f'Batch size:              {c.batch_size}')
    dist.print0(f'Mixed-precision:         {c.network_kwargs.use_fp16}')
    dist.print0(f'Seed:                    {c.seed}')
    dist.print0()

    # Dry run?
    if opts.dry_run:
        dist.print0('Dry run; exiting.')
        return

    # Create output directory.
    dist.print0('Creating output directory...')
    if dist.get_rank() == 0:
        os.makedirs(c.run_dir, exist_ok=True)
        with open(os.path.join(c.run_dir, 'training_options.json'), 'wt') as f:
            json.dump(c, f, indent=2)
        dnnlib.util.Logger(file_name=os.path.join(c.run_dir, 'log.txt'), file_mode='a', should_flush=True)

    # Train.
    training_loop.training_loop(**c)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
