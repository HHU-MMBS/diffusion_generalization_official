# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Generate random images using the given model."""

import os
import re
import warnings
import click
import tqdm
import pickle
import numpy as np
import torch
import PIL.Image
import dnnlib
import wandb
import json

from pathlib import Path
from torch_utils import distributed as dist
from frechet_utils import calc_python, parse_metric_list

warnings.filterwarnings('ignore', '`resume_download` is deprecated')

#----------------------------------------------------------------------------
# Configuration presets.

model_root = 'https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions'

config_presets = {
    'edm2-img512-xs-fid':        dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xs-2147483-0.135.pkl'),  # fid = 3.53
    'edm2-img512-s-fid':         dnnlib.EasyDict(net=f'{model_root}/edm2-img512-s-2147483-0.130.pkl'),   # fid = 2.56
    'edm2-img512-m-fid':         dnnlib.EasyDict(net=f'{model_root}/edm2-img512-m-2147483-0.100.pkl'),   # fid = 2.25
    'edm2-img512-l-fid':         dnnlib.EasyDict(net=f'{model_root}/edm2-img512-l-1879048-0.085.pkl'),   # fid = 2.06
    'edm2-img512-xl-fid':        dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xl-1342177-0.085.pkl'),  # fid = 1.96
    'edm2-img512-xxl-fid':       dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xxl-0939524-0.070.pkl'), # fid = 1.91
    'edm2-img64-s-fid':          dnnlib.EasyDict(net=f'{model_root}/edm2-img64-s-1073741-0.075.pkl'),    # fid = 1.58
    'edm2-img64-m-fid':          dnnlib.EasyDict(net=f'{model_root}/edm2-img64-m-2147483-0.060.pkl'),    # fid = 1.43
    'edm2-img64-l-fid':          dnnlib.EasyDict(net=f'{model_root}/edm2-img64-l-1073741-0.040.pkl'),    # fid = 1.33
    'edm2-img64-xl-fid':         dnnlib.EasyDict(net=f'{model_root}/edm2-img64-xl-0671088-0.040.pkl'),   # fid = 1.33
    'edm2-img512-xs-dino':       dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xs-2147483-0.200.pkl'),  # fd_dinov2 = 103.39
    'edm2-img512-s-dino':        dnnlib.EasyDict(net=f'{model_root}/edm2-img512-s-2147483-0.190.pkl'),   # fd_dinov2 = 68.64
    'edm2-img512-m-dino':        dnnlib.EasyDict(net=f'{model_root}/edm2-img512-m-2147483-0.155.pkl'),   # fd_dinov2 = 58.44
    'edm2-img512-l-dino':        dnnlib.EasyDict(net=f'{model_root}/edm2-img512-l-1879048-0.155.pkl'),   # fd_dinov2 = 52.25
    'edm2-img512-xl-dino':       dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xl-1342177-0.155.pkl'),  # fd_dinov2 = 45.96
    'edm2-img512-xxl-dino':      dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xxl-0939524-0.150.pkl'), # fd_dinov2 = 42.84
    'edm2-img512-xs-guid-fid':   dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xs-2147483-0.045.pkl', net_neg=f'{model_root}/edm2-img512-xs-uncond-2147483-0.045.pkl', guidance=1.4), # fid = 2.91
    'edm2-img512-s-guid-fid':    dnnlib.EasyDict(net=f'{model_root}/edm2-img512-s-2147483-0.025.pkl', net_neg=f'{model_root}/edm2-img512-xs-uncond-2147483-0.025.pkl', guidance=1.4), # fid = 2.23
    'edm2-img512-m-guid-fid':    dnnlib.EasyDict(net=f'{model_root}/edm2-img512-m-2147483-0.030.pkl', net_neg=f'{model_root}/edm2-img512-xs-uncond-2147483-0.030.pkl', guidance=1.2), # fid = 2.01
    'edm2-img512-l-guid-fid':    dnnlib.EasyDict(net=f'{model_root}/edm2-img512-l-1879048-0.015.pkl', net_neg=f'{model_root}/edm2-img512-xs-uncond-2147483-0.015.pkl', guidance=1.2), # fid = 1.88
    'edm2-img512-xl-guid-fid':   dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xl-1342177-0.020.pkl', net_neg=f'{model_root}/edm2-img512-xs-uncond-2147483-0.020.pkl', guidance=1.2), # fid = 1.85
    'edm2-img512-xxl-guid-fid':  dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xxl-0939524-0.015.pkl', net_neg=f'{model_root}/edm2-img512-xs-uncond-2147483-0.015.pkl', guidance=1.2), # fid = 1.81
    'edm2-img512-xs-guid-dino':  dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xs-2147483-0.150.pkl', net_neg=f'{model_root}/edm2-img512-xs-uncond-2147483-0.150.pkl', guidance=1.7), # fd_dinov2 = 79.94
    'edm2-img512-s-guid-dino':   dnnlib.EasyDict(net=f'{model_root}/edm2-img512-s-2147483-0.085.pkl', net_neg=f'{model_root}/edm2-img512-xs-uncond-2147483-0.085.pkl', guidance=1.9), # fd_dinov2 = 52.32
    'edm2-img512-m-guid-dino':   dnnlib.EasyDict(net=f'{model_root}/edm2-img512-m-2147483-0.015.pkl', net_neg=f'{model_root}/edm2-img512-xs-uncond-2147483-0.015.pkl', guidance=2.0), # fd_dinov2 = 41.98
    'edm2-img512-l-guid-dino':   dnnlib.EasyDict(net=f'{model_root}/edm2-img512-l-1879048-0.035.pkl', net_neg=f'{model_root}/edm2-img512-xs-uncond-2147483-0.035.pkl', guidance=1.7), # fd_dinov2 = 38.20
    'edm2-img512-xl-guid-dino':  dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xl-1342177-0.030.pkl', net_neg=f'{model_root}/edm2-img512-xs-uncond-2147483-0.030.pkl', guidance=1.7), # fd_dinov2 = 35.67
    'edm2-img512-xxl-guid-dino': dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xxl-0939524-0.015.pkl', net_neg=f'{model_root}/edm2-img512-xs-uncond-2147483-0.015.pkl', guidance=1.7), # fd_dinov2 = 33.09
}

#----------------------------------------------------------------------------
# EDM sampler from the paper
# "Elucidating the Design Space of Diffusion-Based Generative Models",
# extended to support classifier-free guidance.

def edm_sampler(
    net_pos, noise,
    labels=None, net_neg=None,
    early_stop=-1,
    num_steps=32, sigma_min=0.002, sigma_max=80, rho=7, guidance=1,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
    dtype=torch.float32, randn_like=torch.randn_like, **g_kwargs,
):
    # Guided denoiser.
    def denoise(x, t):
        Dx = net_pos(x, t, labels).to(dtype)
        if guidance == 1:
            return Dx
        ref_Dx = net_neg(x, t).to(dtype)
        return ref_Dx.lerp(Dx, guidance)

    def compute_score(x, t, i, labels, num_steps, g_weight, g_method, swg_sizes, swg_steps, g_interval):
        """Compute the score function at the point x and time t."""

        y_pred = net_pos(x, t, labels).to(dtype)

        if g_interval == [] or g_interval is None:
            g_interval = [0, num_steps - 1]

        if g_interval[0] <= i <= g_interval[-1] and g_weight:
            if g_method == 'regular':  # CFG
                y_neg = net_neg(x, t, class_labels=None).to(dtype)
            elif g_method == 'wmg':
                y_neg = net_neg(x, t, class_labels=labels).to(dtype)
            elif g_method == 'swg':
                initial_res = net_pos.img_resolution
                bs, c, img_size, _ = x.shape
                y_neg = torch.zeros_like(x)
                w = torch.zeros_like(x)
                assert swg_steps in [1, 2, 3, 5], f'Invalid crop steps: {swg_steps}'
                net_crop = net_neg if net_neg is not None else net_pos
                for crop_size in swg_sizes:
                    net_crop.img_resolution = crop_size
                    stride = (img_size - crop_size) // (swg_steps - 1) if swg_steps > 1 else 0
                    for i in range(swg_steps):
                        for j in range(swg_steps):
                            le, re = stride * i, stride * i + crop_size  # left, right
                            te, be = stride * j, stride * j + crop_size  # top, bottom
                            xc = x[:, :, te:be, le:re]  # [bc, c, crop_size, crop_size]
                            y_crop = net_crop(xc, t, labels).to(dtype)
                            y_neg[:, :, te:be, le:re] += y_crop
                            w[:, :, te:be, le:re] += torch.ones_like(y_crop)
                y_neg = y_neg / w
                net_pos.img_resolution = initial_res # reset resolution for next iteration
            else:
                raise NotImplementedError(f'Invalid CFG method: {g_method}')

            y_pred = y_pred + g_weight * (y_pred - y_neg)
        return y_pred

    g_kwargs['num_steps'] = num_steps

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=dtype, device=noise.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # t_N = 0

    assert early_stop <= num_steps, f'early_stop must be <= num_steps, but got early_stop={early_stop} and num_steps={num_steps}'

    # Main sampling loop.
    x_next = noise.to(dtype) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        if i == early_stop:  # Stop here and do a one-shot Euler step to noise 0.
            t_next = 0

        # Increase noise temporarily.
        if S_churn > 0 and S_min <= t_cur <= S_max:
            gamma = min(S_churn / num_steps, np.sqrt(2) - 1)
            t_hat = t_cur + gamma * t_cur
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)
        else:
            t_hat = t_cur
            x_hat = x_cur

        # Euler step.
        d_cur = (x_hat - compute_score(x_hat, t_hat, i, labels, **g_kwargs)) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if t_next != 0:
            d_prime = (x_next - compute_score(x_next, t_next, i, labels, **g_kwargs)) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
        else:
            break

    return x_next

#----------------------------------------------------------------------------
# Wrapper for torch.Generator that allows specifying a different random seed
# for each sample in a minibatch.

class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])

#----------------------------------------------------------------------------
# Generate images for the given seeds in a distributed fashion.
# Returns an iterable that yields
# dnnlib.EasyDict(images, labels, noise, batch_idx, num_batches, indices, seeds)

def generate_images(
    net_pos_pkl,                                # Main network. Path, URL, or torch.nn.Module.
    net_neg_pkl         = None,                 # Reference network for guidance. None = same as main network.
    encoder             = None,                 # Instance of training.encoders.Encoder. None = load from network pickle.
    outdir              = None,                 # Where to save the output images. None = do not save.
    subdirs             = False,                # Create subdirectory for every 1000 seeds?
    seeds               = range(16, 24),        # List of random seeds.
    class_idx           = None,                 # Class label. None = select randomly.
    all_labels          = None,                 # Class frequencies, tensor with number of images for each class.
    max_batch_size      = 32,                   # Maximum batch size for the diffusion model.
    encoder_batch_size  = 4,                   # Maximum batch size for the encoder. None = default.
    verbose             = True,                 # Enable status prints?
    device              = torch.device('cuda'), # Which compute device to use.
    sampler_fn          = edm_sampler,          # Which sampler function to use.
    **sampler_kwargs, # Additional arguments for the sampler function.
):
    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier(device_ids=[dist.get_rank()])

    # Load main network.
    if isinstance(net_pos_pkl, str):
        if "512" in net_pos_pkl:
            img_resolution = 512
        elif "64" in net_pos_pkl:
            img_resolution = 64
        if verbose:
            dist.print0(f'Loading positive network from {net_pos_pkl} ...')
        with dnnlib.util.open_url(net_pos_pkl, verbose=(verbose and dist.get_rank() == 0)) as f:
            data = pickle.load(f)
        net_pos = data['ema'].to(device)
        if encoder is None:
            encoder = data.get('encoder', None)
            if encoder is None:
                encoder = dnnlib.util.construct_class_by_name(class_name='training.encoders.StandardRGBEncoder')
    assert net_pos is not None

    # Load negative network.
    net_neg = None
    if sampler_kwargs["g_weight"] != 0:
        if net_neg_pkl is None or net_neg_pkl == 'none':
            if sampler_kwargs['g_method'] != 'none':
                dist.print0(f"Using positive network as negative.")
                net_neg = net_pos
            else:
                net_neg = None
        elif isinstance(net_neg_pkl, str):
            if verbose:
                dist.print0(f'Loading negative network from {net_neg_pkl} ...')
            with dnnlib.util.open_url(net_neg_pkl, verbose=(verbose and dist.get_rank() == 0)) as f:
                net_neg = pickle.load(f)['ema'].to(device)
        else:
            raise ValueError(f'Invalid net_neg_pkl: {net_neg_pkl}')


    # Initialize encoder.
    assert encoder is not None
    if verbose:
        dist.print0(f'Setting up {type(encoder).__name__}...')
    encoder.init(device)
    if encoder_batch_size is not None and hasattr(encoder, 'batch_size'):
        encoder.batch_size = encoder_batch_size

    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier(device_ids=[dist.get_rank()])

    # Divide seeds into batches.
    num_batches = max((len(seeds) - 1) // (max_batch_size * dist.get_world_size()) + 1, 1) * dist.get_world_size()
    rank_batches = np.array_split(np.arange(len(seeds)), num_batches)[dist.get_rank() :: dist.get_world_size()]
    if verbose:
        dist.print0(f'Generating {len(seeds)} images...')

    # Generate class labels if specified
    if net_pos.label_dim > 0 and all_labels is not None:
        all_labels = torch.load(all_labels, weights_only=True)
        os.makedirs(outdir, exist_ok=True)
        torch.save(all_labels, os.path.join(os.path.dirname(os.path.normpath(outdir)), "labels.pt"))  # Go one dir up
        assert len(all_labels) == len(seeds), f'Expected {len(seeds)} labels, got {len(all_labels)}'

    # Return an iterable over the batches.
    class ImageIterable:
        def __len__(self):
            return len(rank_batches)

        def __iter__(self):
            # Loop over batches.
            for batch_idx, indices in enumerate(rank_batches):
                r = dnnlib.EasyDict(images=None, labels=None, noise=None, batch_idx=batch_idx, num_batches=len(rank_batches), indices=indices)
                r.seeds = [seeds[idx] for idx in indices]
                if len(r.seeds) > 0:
                    # Pick noise and labels.
                    rnd = StackedRandomGenerator(device, r.seeds)
                    r.noise = rnd.randn([len(r.seeds), net_pos.img_channels, net_pos.img_resolution, net_pos.img_resolution], device=device)
                    r.labels = None
                    # Set the class labels for the mini-batch.
                    if net_pos.label_dim > 0:
                        assert not (class_idx is not None and all_labels is not None), 'Specify either class_idx or all_labels, not both.'
                        r.labels = torch.zeros(len(indices), net_pos.label_dim, device=device, dtype=bool)
                        if all_labels is not None:
                            batch_labels = all_labels[indices]  # labels for this batch
                            r.labels[:, :] = 0
                            r.labels[np.arange(len(batch_labels)), batch_labels] = 1
                        elif class_idx is not None:
                                r.labels[:, :] = 0
                                r.labels[:, class_idx] = 1
                        else:
                            r.labels = torch.eye(net_pos.label_dim, device=device)[rnd.randint(net_pos.label_dim, size=[len(r.seeds)], device=device)]  # uniform random

                    # Generate images.
                    latents = dnnlib.util.call_func_by_name(func_name=sampler_fn, net_pos=net_pos, noise=r.noise,
                                                                 labels=r.labels, net_neg=net_neg, randn_like=rnd.randn_like, **sampler_kwargs)
                    r.images = encoder.decode(latents)

                    # Save images.
                    if outdir is not None:
                        for seed, image in zip(r.seeds, r.images.permute(0, 2, 3, 1).cpu().numpy()):
                            assert image.shape[0] == img_resolution, f'Expected image resolution {img_resolution}, got {image.shape[0]}'
                            image_dir = os.path.join(outdir, f'{seed//1000*1000:07d}') if subdirs else outdir
                            os.makedirs(image_dir, exist_ok=True)
                            PIL.Image.fromarray(image, 'RGB').save(os.path.join(image_dir, f'{seed:07d}.png'))

                # Yield results.
                torch.distributed.barrier(device_ids=[dist.get_rank()]) # keep the ranks in sync
                yield r

    return ImageIterable()

#----------------------------------------------------------------------------
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]

def parse_int_list(s):
    if isinstance(s, list): return s
    if s == "none": return []
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------
# Command line interface.

@click.command()
@click.option('--preset',                   help='Configuration preset', metavar='STR',                             type=str, default=None)
@click.option('--net_pos_pkl',              help='Network pickle filename', metavar='PATH|URL',                     type=str, default=None)
@click.option('--net_neg_pkl',              help='Reference network for guidance', metavar='PATH|URL',              type=str, default=None)
@click.option('--outdir',                   help='Where to save the output images', metavar='DIR',                  type=str, required=True)
@click.option('--subdirs',                  help='Create subdirectory for every 1000 seeds',                        is_flag=True)
@click.option('--seeds',                    help='List of random seeds (e.g. 1,2,5-10)', metavar='LIST',            type=parse_int_list, default='16-19', show_default=True)
@click.option('--class', 'class_idx',       help='Class label  [default: random]', metavar='INT',                   type=click.IntRange(min=0), default=None)
@click.option('--batch', 'max_batch_size',  help='Maximum batch size', metavar='INT',                               type=click.IntRange(min=1), default=32, show_default=True)

@click.option('--steps', 'num_steps',       help='Number of sampling steps', metavar='INT',                         type=click.IntRange(min=1), default=32, show_default=True)
@click.option('--sigma_min',                help='Lowest noise level', metavar='FLOAT',                             type=click.FloatRange(min=0, min_open=True), default=0.002, show_default=True)
@click.option('--sigma_max',                help='Highest noise level', metavar='FLOAT',                            type=click.FloatRange(min=0, min_open=True), default=80, show_default=True)
@click.option('--rho',                      help='Time step exponent', metavar='FLOAT',                             type=click.FloatRange(min=0, min_open=True), default=7, show_default=True)
@click.option('--guidance',                 help='Guidance strength  [default: 1; no guidance]', metavar='FLOAT',   type=float, default=None)
@click.option('--S_churn', 'S_churn',       help='Stochasticity strength', metavar='FLOAT',                         type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_min', 'S_min',           help='Stoch. min noise level', metavar='FLOAT',                         type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_max', 'S_max',           help='Stoch. max noise level', metavar='FLOAT',                         type=click.FloatRange(min=0), default='inf', show_default=True)
@click.option('--S_noise', 'S_noise',       help='Stoch. noise inflation', metavar='FLOAT',                         type=float, default=1, show_default=True)

@click.option('--g_weight',                 type=float, default=0., help="CFG weight. 0 = no guidance")
@click.option('--g_method',                 type=click.Choice(['regular', 'wmg', 'swg', 'none']), default='none', help="CFG method")
@click.option('--swg_sizes',                type=parse_int_list, default="16,16", help="Crop size")
@click.option('--swg_steps',                type=int, default=2, help="Crop stride")
@click.option('--g_interval',               type=parse_int_list, default=None, help="Interval for CFG guidance")
@click.option('--use_wandb',                help='Enable Weights & Biases logging', metavar='BOOL',  type=bool, default=False, show_default=True)
@click.option('--wandb_group',              help='Group name for wandb', type=str, default=None)
@click.option('--wandb_runname',            help='Run name for wandb', type=str, default=None)
@click.option('--early_stop',               help="Stop inference at this index and do a one-shot Euler step to sigma=0", type=int, default=-1)

@click.option('--all_labels',               help="Path to tensor specifying a label (int) for every single image", )
@click.option('--metrics',                  help='List of metrics to compute', metavar='LIST',            type=parse_metric_list, default='fid,fd_dinov2', show_default=True)
@click.option('--ref_path',                 help='Path to reference for FID/FDD calculation', metavar='PATH', type=str, default=None)
def cmdline(preset, metrics, ref_path, use_wandb, wandb_group, wandb_runname, **opts):
    """Generate random images using the given model.

    Examples:

    \b
    # Generate a couple of images and save them as out/*.png
    python generate_images.py --preset=edm2-img512-s-guid-dino --outdir=out

    \b
    # Generate 50000 images using 8 GPUs and save them as out/*/*.png
    torchrun --standalone --nproc_per_node=8 generate_images.py \\
        --preset=edm2-img64-s-fid --outdir=out --subdirs --seeds=0-49999
    """
    opts = dnnlib.EasyDict(opts)

    # Apply preset.
    if preset is not None:
        if preset not in config_presets:
            raise click.ClickException(f'Invalid configuration preset "{preset}"')
        for key, value in config_presets[preset].items():
            if opts[key] is None:
                opts[key] = value

    # Validate options.
    if opts.net_pos_pkl is None:
        raise click.ClickException('Please specify either --preset or --net_pos_pkl')
    # if opts.guidance is None or opts.guidance == 1:
    #     opts.guidance = 1
    #     opts.net_neg_pkl = None
    # elif opts.net_neg_pkl is None:
    #     raise click.ClickException('Please specify --net_neg when using guidance')

    assert not ((opts.g_method == 'short' or opts.g_method == 'shallow') and opts.net_neg_pkl is None), 'WMG requires a shallow network'
    assert not (opts.g_method == 'regular' and opts.net_neg_pkl is None), 'g_weight requires an uncond network'

    # Generate.
    dist.init()

    # Wandb initialization
    all_arguments = click.get_current_context().params
    all_arguments["seeds"] = len(opts.seeds)
    if use_wandb and dist.get_rank() == 0:
        if os.access("/home/tikai103", os.R_OK):
            os.environ["WANDB_API_KEY"] = "a9efb3b6cddc090dbf125d4c5d0dff12b178eb36"
        wandb.init(project="Inductive-bias", entity="hhu-mmbs", group=wandb_group)
        wandb.run.name = wandb_runname
        wandb.config.update(all_arguments)

    # Save arguments
        os.makedirs(opts.outdir, exist_ok=True)
        json_filename = Path(opts.outdir) / 'generate_args.json'
        # Write arguments to JSON file
        with open(json_filename, 'w') as jsonfile:
            json.dump(all_arguments, jsonfile, indent=4)

    if len(opts.seeds) > 0:  # Otherwise jump to metric computation
        if opts.g_method == 'regular':
            method_str = 'regular CFG'
        elif opts.g_method == 'wmg':
            method_str = 'WMG'
        elif opts.g_method == 'swg':
            method_str = 'ILG'
        elif opts.g_method == 'none':
            method_str = 'karras inference'
            opts.g_weight = 0
        else:
            raise NotImplementedError(f'Unknown method: {opts.g_method}')

        dist.print0(f'Generating {len(opts.seeds)} images to "{opts.outdir}" with {method_str}...')

        image_iter = generate_images(**opts)
        for _r in tqdm.tqdm(image_iter, unit='batch', disable=(dist.get_rank() != 0)):
            pass

    if metrics:
        metric_kwargs = {'image_path': opts.outdir,
                         'ref_path': ref_path,
                         'metrics': metrics,
                         'num_images': 50000,
                         'seed': 0,
                         'max_batch_size': min(1024, opts.max_batch_size)}
        results = calc_python(**metric_kwargs)
        if dist.get_rank() == 0 and use_wandb:
            log = {}
            for metric in metrics:
                if metric == 'fid':
                    name = "FID"
                elif metric == 'fd_dinov2':
                    name = "FDD"
                else:
                    raise NotImplementedError(f'Invalid metric: {metric}')
                log[name] = results[metric]
            wandb.config.update(log)

    torch.cuda.empty_cache()


#----------------------------------------------------------------------------

if __name__ == "__main__":
    cmdline()

#----------------------------------------------------------------------------
