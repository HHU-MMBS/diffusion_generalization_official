# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Generate random images using the techniques described in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import os
import click
import pickle
import numpy as np
import torch
import PIL.Image
import dnnlib
import json
import zipfile
# import wandb

from train import wandb_init, parse_int_list
from pathlib import Path
from tqdm import tqdm
from torch_utils import distributed as dist
from torch_utils.frechet_utils import calc_python, parse_metric_list


def load_training_images(idx, use_tqdm=False):
    images = []
    with zipfile.ZipFile('/home/shared/DataSets/cifar-10/cifar10-32x32.zip') as z:
        for i in tqdm(idx, disable=not use_tqdm):
            with z.open(f'{int(i / 1000):05d}/img{i:08d}.png', 'r') as f:
                image = np.array(PIL.Image.open(f))
            images.append(torch.from_numpy(image).float().permute(2, 0, 1))  # [3, 32, 32]
    return images


def heun_step(net_pos, net_neg, x_cur, t_cur, t_next, i, num_steps,
              S_churn, S_min, S_max, S_noise, randn_like, **score_kwargs):
    """One Heun-step."""
    # Increase noise temporarily.
    gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
    t_hat = net_pos.round_sigma(t_cur + gamma * t_cur)
    x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

    score_kwargs.update(i=i, randn_like=randn_like)

    # Euler step.
    dt = t_next - t_hat
    d_eul, y_eul, _ = compute_score(net_pos, net_neg, x_hat, t_hat, **score_kwargs)
    x_eul = x_hat + dt * d_eul

    # Apply 2nd order correction (Heun).
    if i < num_steps - 1:
        d_prime, y_heun, _ = compute_score(net_pos, net_neg, x_eul, t_next, **score_kwargs)
        x_next = x_hat + dt * (0.5 * d_eul + 0.5 * d_prime)
    else:
        x_next = x_eul

    return x_next, y_eul


def our_step(net_pos, net_neg, x_cur, t_cur, t_next, t_prev, g_prev, i, num_steps, cfg_weight,
              **score_kwargs):
    """"""
    score_kwargs.update(i=i, cfg_weight=cfg_weight)

    # Preparation
    dt = t_next - t_cur  # < 0
    # de = t_prev - t_cur if i != 0 else None # > 0

    # Euler step
    d_eul, y_eul, g_eul = compute_score(net_pos, net_neg, x_cur, t_cur, **score_kwargs)
    x_eul = x_cur + dt * d_eul

    # # Flow field correction
    # x_eul += (g_prev - g_eul) * dt / de if i != 0 else x_eul

    # Apply 2nd order correction (Heun)
    if i < num_steps - 1:
        d_heun, y_heun, g_heun = compute_score(net_pos, net_neg, x_eul, t_next, **score_kwargs)
        x_next = x_cur + dt * (0.5 * d_eul + 0.5 * d_heun)

        # Flow field correction
        x_next += (g_heun - g_eul) if i != 0 else x_next
        # dist.print0(i, f'{t_cur:.2e}', f'{(dy_prev - dy_heun).abs().mean().item():.2e}')
    else:
        x_next, g_heun = x_eul, g_eul

    return x_next, y_eul, (0.5 * g_eul + 0.5 * g_heun)  # g_prev for next step


def compute_score(net_pos, net_neg, x, t, i, class_labels, high_precision, dtype, renoise,
                  cfg_interval, cfg_weight, cfg_method, cfg_swg_sizes, cfg_swg_steps, augment_scale, randn_like):
    """Compute the score function at the point x and time t."""
    net_pos.img_resolution = x.shape[-1]
    y_pos = net_pos(x, t, class_labels, force_fp32=high_precision).to(dtype)

    # Renoising
    if renoise:
        x_re = y_pos + t * randn_like(x)
        y_pred = net_pos(x_re, t, class_labels, force_fp32=high_precision).to(dtype)
    else:
        x_re = x
        y_pred = y_pos

    # Guidance
    gy = None
    if cfg_interval[0] <= i <= cfg_interval[-1] and cfg_weight:
        if cfg_method == 'regular':  # CFG
            y_neg = net_neg(x_re, t, class_labels=None, force_fp32=high_precision).to(dtype)
        elif cfg_method == 'wmg':  # CFG++
            augment_labels = torch.ones(x.shape[0], 9, device=x.device) * augment_scale if augment_scale else None
            y_neg = net_neg(x_re, t, class_labels=class_labels, augment_labels=augment_labels,
                            force_fp32=high_precision).to(dtype)
        elif cfg_method == 'swg':
            bs, c, img_size, _ = x.shape
            y_neg = torch.zeros_like(x)
            w = torch.zeros_like(x)
            assert cfg_swg_steps in [1,2,3,5], f'Invalid crop steps: {cfg_swg_steps}'
            net_crop = net_neg if net_neg is not None else net_pos
            for crop_size in cfg_swg_sizes:
                net_crop.img_resolution = crop_size
                stride = (img_size - crop_size) // (cfg_swg_steps - 1) if cfg_swg_steps > 1 else 0
                for i in range(cfg_swg_steps):
                    for j in range(cfg_swg_steps):
                        le, re = stride * i, stride * i + crop_size  # left, right
                        te, be = stride * j, stride * j + crop_size  # top, bottom
                        xc = x_re[:, :, te:be, le:re]  # [bc, c, crop_size, crop_size]
                        y_crop = net_crop(xc, t, class_labels, force_fp32=high_precision).to(dtype)
                        y_neg[:, :, te:be, le:re] += y_crop
                        w[:, :, te:be, le:re] += torch.ones_like(y_crop)
                net_crop.img_resolution = img_size
            y_neg = y_neg / w
        y_pred = y_pos + cfg_weight * (y_pred - y_neg)
        gy = 1.0 * y_pos - y_neg

    score = (x - y_pred) / t  # score = force / t^2
    return score, y_pred, gy


def y_cfgpp(y_pos, y_neg, weight):
    """One step of CFG++, given a negative model and cfg weight
    Params:
        y_pos: positive model
        y_neg: negative model
        weight: cfg weight
    """
    # y_neg = brightness_correction(y_neg, y_pos)
    dy = y_pos - y_neg
    # dy = dy / dy.var((1, 2, 3), keepdim=True) * dy.var((0, 1, 2, 3), keepdim=True)  # Variance correction
    y_pred = y_pos + weight * dy
    # y_pred = brightness_correction(y_pred, y_pos)
    return y_pred, dy


def brightness_correction(input, target):
    """Adjusts the image-wise mean and std of the input to match the target."""
    return ((input - input.mean((2, 3), keepdim=True)) / input.std((2, 3), keepdim=True)
            * target.std((2, 3), keepdim=True) + target.mean((2, 3), keepdim=True))


def edm_sampler(
    net_pos, net_neg, latents, num_steps, sigma_min, sigma_max, rho, S_churn, S_min, S_max, S_noise,
    cfg_weight, cfg_method, cfg_swg_sizes, cfg_swg_steps, augment_scale,
    class_labels=None, randn_like=torch.randn_like, return_hist=False, high_precision=True,
    cfg_interval=None, our_guidance=False, renoise=False,
):
    """Proposed EDM sampler (Algorithm 2)."""
    dtype = torch.float64 if high_precision else torch.float32

    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net_pos.sigma_min)
    sigma_max = min(sigma_max, net_pos.sigma_max)

    if cfg_interval == [] or cfg_interval is None:
        cfg_interval = [0, num_steps-1]

    if not our_guidance:
        heun_kwargs = {'num_steps': num_steps, 'S_churn': S_churn, 'S_min': S_min, 'class_labels': class_labels,
                       'S_max': S_max, 'S_noise': S_noise, 'randn_like': randn_like, 'high_precision': high_precision,
                       'dtype': dtype, 'cfg_weight': cfg_weight, 'cfg_method': cfg_method, 'cfg_interval': cfg_interval,
                       'augment_scale': augment_scale, 'renoise': renoise,
                       'cfg_swg_sizes': cfg_swg_sizes, 'cfg_swg_steps': cfg_swg_steps, 'randn_like': randn_like}
    else:
        our_kwargs = {'num_steps': num_steps, 'class_labels': class_labels, 'high_precision': high_precision,
                       'dtype': dtype, 'cfg_method': cfg_method, 'cfg_interval': cfg_interval,
                       'augment_scale': augment_scale,
                       'cfg_swg_sizes': cfg_swg_sizes, 'cfg_swg_steps': cfg_swg_steps}

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=dtype, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net_pos.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])  # t_N = 0

    # Main sampling loop.
    x_next = latents.to(dtype) * t_steps[0]
    if return_hist:
        x_hist, pred_hist = [x_next], []
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  # 0, ..., N-1
        x_cur = x_next
        if not our_guidance:
            x_next, y_pred = heun_step(net_pos, net_neg, x_cur, t_cur, t_next, i, **heun_kwargs)
        else:
            if i == 0:
                t_prev, g_prev = None, None
            our_kwargs['cfg_weight'] = cfg_weight * (1 - torch.exp(-1 * t_cur)).item()
            # dist.print0(i, f'{t_cur:.2e}', our_kwargs['cfg_weight'])
            x_next, y_pred, g_prev = our_step(net_pos, net_neg, x_cur, t_cur, t_next, t_prev, g_prev, i, **our_kwargs)
            t_prev = t_cur  # Store for next iteration

        if return_hist:
            x_hist.append(x_next)
            pred_hist.append(y_pred)

    if return_hist:  # Doesn't work atm
        dist.print0(len(x_hist), len(pred_hist), len(t_steps))
        return x_hist, pred_hist, t_steps
    else:
        return x_next



class StackedRandomGenerator:
    """Wrapper for torch.Generator that allows specifying a different random seed for each sample in a minibatch."""
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

    def multinomial(self, input, replacement=False, device='cuda', **kwargs):
        return torch.stack([torch.multinomial(input, 1, replacement, generator=gen, **kwargs).to(device) for gen in self.generators])

    #def studentT(self, input, replacement=False, device='cuda', **kwargs):


@click.command()
@click.option('--net_pos_pkl',              help='Network pickle filename', metavar='PATH|URL',                  type=str, required=False)
@click.option('--net_neg_pkl',              help='Network pickle filename', metavar='PATH|URL',          type=str, default="None")
@click.option('--outdir',                   help='Where to save the output images', metavar='DIR',               type=str, required=True)
@click.option('--seeds',                    help='Random seeds (e.g. 1,2,5-10)', metavar='LIST',                 type=parse_int_list, default='0', show_default=True)
@click.option('--subdirs',                  help='Create subdirectory for every 1000 seeds',                     is_flag=True)
@click.option('--class', 'class_idx',       help='Class label  [default: random]', metavar='INT',                type=click.IntRange(min=0), default=None)
@click.option('--batch', 'max_batch_size',  help='Maximum batch size', metavar='INT',                            type=click.IntRange(min=1), default=16, show_default=True)
@click.option('--metrics',                  help='List of metrics to compute', metavar='LIST',            type=parse_metric_list, default='fid,fd_dinov2', show_default=True)
@click.option('--ref_path',                 help='Path to reference for FID/FDD calculation', metavar='PATH', type=str, default=None)
# @click.option('--compute_fid',              help='Compute FID of generated images', metavar='BOOL', type=bool, default=False)
# @click.option('--ref_path_fid',             help='Path to reference for FID calculation', metavar='PATH', type=str, default=None)
# @click.option('--compute_fdd',              help='Compute FDD of generated images', metavar='BOOL', type=bool, default=False)
# @click.option('--ref_path_fdd',             help='Path to reference for FID calculation', metavar='PATH', type=str, default=None)
#
@click.option('--steps', 'num_steps',       help='Number of sampling steps', metavar='INT',                          type=click.IntRange(min=1), default=21, show_default=True)
@click.option('--sigma_min',                help='Lowest noise level  [default: varies]', metavar='FLOAT',           type=click.FloatRange(min=0, min_open=True), default=2e-3)
@click.option('--sigma_max',                help='Highest noise level  [default: varies]', metavar='FLOAT',          type=click.FloatRange(min=0, min_open=True), default=80)
@click.option('--rho',                      help='Time step exponent', metavar='FLOAT',                              type=click.FloatRange(min=-1000000, min_open=True), default=7, show_default=True)
@click.option('--S_churn', 'S_churn',       help='Stochasticity strength', metavar='FLOAT',                          type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_min', 'S_min',           help='Stoch. min noise level', metavar='FLOAT',                          type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_max', 'S_max',           help='Stoch. max noise level', metavar='FLOAT',                          type=click.FloatRange(min=0), default='inf', show_default=True)
@click.option('--S_noise', 'S_noise',       help='Stoch. noise inflation', metavar='FLOAT',                          type=float, default=1, show_default=True)
#
@click.option('--save_stats',               type=bool, default=False, help="Compute and save inference stats")
@click.option('--cfg_weight',               type=float, default=0., help="CFG weight. 0 = no guidance")
@click.option('--cfg_method',               type=click.Choice(['regular', 'wmg', 'swg', 'none']), default='none', help="CFG method")
@click.option('--cfg_swg_sizes',            type=parse_int_list, default="16,16", help="Crop size")
@click.option('--cfg_swg_steps',            type=int, default=2, help="Crop stride")
@click.option('--cfg_interval',             type=parse_int_list, default=None, help="Interval for CFG guidance")
@click.option('--augment_scale',            type=float, default=0.0, help="Augment scale for augment guidance")
@click.option('--our_guidance',             help='Use our guidance method', type=bool, default=False)
@click.option('--renoise',                  help='Renoise during inference', type=bool, default=False)
@click.option('--load_seeds',               type=str, default=None)
@click.option('--ignore_cond',              type=bool, default=False)
@click.option('--use_wandb',                help='Enable Weights & Biases logging', metavar='BOOL',  type=bool, default=False, show_default=True)
@click.option('--wandb_group',              help='Group name for wandb', type=str, default=None)
@click.option('--wandb_runname',            help='Group name for wandb', type=str, default=None)

@click.option('--all_labels',               help="Path to tensor specifying a label (int) for every single image", default=None)
def main(net_pos_pkl, net_neg_pkl, outdir, subdirs, seeds, class_idx, max_batch_size,
         cfg_weight, cfg_method,  cfg_swg_sizes, cfg_swg_steps, cfg_interval,
         save_stats, load_seeds, ignore_cond, use_wandb, wandb_group, wandb_runname,
         metrics, ref_path, all_labels,
         device=torch.device('cuda'), **sampler_kwargs):
    """Generate random images using the techniques described in the paper
    "Elucidating the Design Space of Diffusion-Based Generative Models".

    Examples:

    \b
    # Generate 64 images and save them as out/*.png
    python generate.py --outdir=out --seeds=0-63 --batch=64 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl

    \b
    # Generate 1024 images using 2 GPUs
    torchrun --standalone --nproc_per_node=2 generate.py --outdir=out --seeds=0-999 --batch=64 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl
    """


    assert not ((cfg_method == 'short' or cfg_method == 'shallow') and net_neg_pkl is None), 'WMG requires a shallow network'
    assert not (cfg_method == 'regular' and net_neg_pkl is None), 'cfg_weight requires an uncond network'

    dist.init()
    rank, world_size = dist.get_rank(), dist.get_world_size()

    # Wandb initialization
    all_arguments = click.get_current_context().params
    all_arguments["seeds"] = f"{seeds[0]}-{seeds[-1]}" if seeds != [] else "none"
    if use_wandb and dist.get_rank() == 0:
        wandb_init("Inductive-bias", wandb_group, all_arguments, wandb_runname)

    if len(seeds) > 1:  # Otherwise jump to FID/FDD
        # Load cherry-picked seeds
        if load_seeds is not None:
            raw_seeds = torch.load(load_seeds, weights_only=False)
            # Fill up seeds to 50k
            raw_seeds = raw_seeds[raw_seeds != -1]
            raw_seeds = raw_seeds[:len(seeds)]
            unique = raw_seeds.unique()
            dist.print0(f'Loaded {unique.shape[0]} seeds from {load_seeds}')
            raw_seeds = torch.zeros(len(seeds), dtype=torch.int64)
            raw_seeds[-len(unique):] = unique
            seed = 0
            for i in range(len(seeds) - len(unique)):
                while seed in unique:
                    seed += 1
                raw_seeds[i] = seed
                seed += 1
            seeds = list(range(len(seeds)))
        else:
            raw_seeds = torch.tensor(seeds)
            assert torch.all(raw_seeds.unique() == raw_seeds.sort()[0]), 'Non-unique seeds are not supported'

        num_batches = ((len(seeds) - 1) // (max_batch_size * world_size) + 1) * world_size
        all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
        rank_batches = all_batches[rank :: world_size]  # seeds, batched for each GPU
        idx_batches = torch.arange(len(seeds)).tensor_split(num_batches)[rank :: world_size]  # indices, batched for each GPU

        # Rank 0 goes first.
        if rank != 0:
            torch.distributed.barrier(device_ids=[dist.get_rank()])

        # Load network.
        dist.print0(f'Loading positive network from "{net_pos_pkl}"...')
        with dnnlib.util.open_url(net_pos_pkl, verbose=(rank == 0)) as f:
            data = pickle.load(f)
            net_pos = data['ema'].to(device)
        del data

        cond_gen = net_pos.label_dim

        net_neg = None
        if net_neg_pkl != "None" and cfg_weight != 0 and net_neg_pkl != net_pos_pkl:
            dist.print0(f'Loading negative network from "{net_neg_pkl}"...')
            with dnnlib.util.open_url(net_neg_pkl, verbose=(rank == 0)) as f2:
                data = pickle.load(f2)
                net_neg = data['ema'].to(device)
            del data
            cond_gen = cond_gen or net_neg.label_dim
        elif net_neg_pkl == net_pos_pkl and cfg_weight != 0:
            dist.print0('Using positive network as negative model.')
            net_neg = net_pos

        # Other ranks follow.
        if rank == 0:
            torch.distributed.barrier(device_ids=[dist.get_rank()])

        if cfg_method == 'regular':
            method_str = 'regular CFG'
        elif cfg_method == 'wmg':
            method_str = 'WMG'
        elif cfg_method == 'swg':
            method_str = 'ILG'
        elif cfg_method == 'none':
            method_str = 'karras inference'
            cfg_weight = 0
        else:
            raise NotImplementedError(f'Unknown method: {cfg_method}')

        dist.print0(f'Generating {len(seeds)} images to "{outdir}" with {method_str}...')

        cfg_interval = None if cfg_interval == "none" else cfg_interval
        sampler_kwargs.update(cfg_weight=cfg_weight, cfg_method=cfg_method, cfg_interval=cfg_interval,
                              cfg_swg_sizes=cfg_swg_sizes, cfg_swg_steps=cfg_swg_steps)
        dist.print0(f'Sampler kwargs: {sampler_kwargs}')

        total_counter = 0
        if cond_gen:
            label_dict = {}
            # Generate class labels if specified
            if all_labels is not None:
                all_labels = torch.load(all_labels, weights_only=True).to(device)
                os.makedirs(outdir, exist_ok=True)
                assert len(all_labels) == len(seeds), f'Expected {len(seeds)} labels, got {len(all_labels)}'
            else:
                all_labels = torch.zeros(len(seeds), dtype=torch.int32, device=device)
                if rank == 0:  # Load label dictionary from previous runs
                    label_dict_path = os.path.join(outdir, 'label_dict.pt')
                    if os.path.exists(label_dict_path):
                        dist.print0(f'Loading labels from {label_dict_path}')
                        label_dict = torch.load(label_dict_path, weights_only=False)

        # Save arguments
        os.makedirs(outdir, exist_ok=True)
        json_filename = Path(outdir) / 'generate_args.json'
        # Write arguments to JSON file
        with open(json_filename, 'w') as jsonfile:
            json.dump(all_arguments, jsonfile, indent=4)

        # Loop over batches.
        # batch seeds are the seeds for each each, batch_idx are the indices of the seeds in the original list of seeds
        for c, (batch_seeds, batch_idx) in enumerate(tqdm(zip(rank_batches, idx_batches), unit='batch', disable=(rank != 0),
                                                     ascii=True, total=len(rank_batches))):
            torch.distributed.barrier(device_ids=[dist.get_rank()])
            batch_size = len(batch_seeds)
            if batch_size == 0:
                continue
            rnd = StackedRandomGenerator(device, batch_seeds)
            latents = rnd.randn([batch_size, net_pos.img_channels, net_pos.img_resolution, net_pos.img_resolution], device=device)

            class_labels = None

            # Conditional image generation
            if cond_gen:
                if ignore_cond:
                    labels = torch.ones(batch_size, device=device) * -1
                    class_labels = None
                else:
                    labels = rnd.randint(net_pos.label_dim, size=[batch_size], device=device)  # uniform label-distribution
                    # labels = batch_seeds.to(device) % net_pos.label_dim  # Equal amount of labels for each class
                    class_labels = torch.eye(net_pos.label_dim, device=device)[labels]
                all_labels[batch_idx] = labels.to(torch.int32)

                if class_idx is not None:
                    class_labels[:, :] = 0
                    class_labels[:, class_idx] = 1

            # Generate images.
            sampler_kwargs = {key: value for key, value in sampler_kwargs.items() if value is not None}
            sampler_kwargs.update(randn_like=rnd.randn_like, class_labels=class_labels)
            with torch.no_grad():
                out = edm_sampler(net_pos, net_neg, latents, return_hist=save_stats, **sampler_kwargs)

            for r in range(world_size):
                total_counter += all_batches[world_size * c: world_size * (c + 1)][r].shape[0]  # number of generated images

            # Save images.
            images_np = (out * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
            for i, (seed, image_np) in enumerate(zip(batch_seeds, images_np)):
                # Save samples
                #dist.print0(devsc, cutoff)
                #if i < 32:
                #    continue
                image_dir = os.path.join(outdir, f'{seed-seed%1000:07d}') if subdirs else outdir
                os.makedirs(image_dir, exist_ok=True)
                image_path = os.path.join(image_dir, f'{seed:07d}.png')
                # assert image_np.shape == (net_pos.img_resolution, net_pos.img_resolution, net_pos.img_channels), f'{image_np.shape}'
                if image_np.shape[2] == 1:
                    PIL.Image.fromarray(image_np[:, :, 0], 'L').save(image_path)
                else:
                    PIL.Image.fromarray(image_np, 'RGB').save(image_path)

            if cond_gen and ((c+1) % 25 == 0 or total_counter == len(seeds)):
                torch.distributed.reduce(all_labels, dst=0, op=torch.distributed.ReduceOp.SUM)  # Collect all labels on GPU 0
                if rank == 0:
                    for label, seed in zip(all_labels[:total_counter].cpu(), raw_seeds[:total_counter].cpu()):
                        label_dict[seed.item()] = label.item()
                    torch.save(label_dict, os.path.join(os.path.dirname(os.path.normpath(outdir)), "label_dict.pt"))
                    torch.save(all_labels, os.path.join(os.path.dirname(os.path.normpath(outdir)), "labels.pt"))

        del net_pos, net_neg#, out
        torch.cuda.empty_cache()

    # if compute_fid:
    #     fid_kwargs = {'image_path': outdir,
    #                   'model': 'inception',
    #                   'ref_path': ref_path_fid,
    #                   'num_expected': 50000,
    #                   'seed': 0,
    #                   'batch': min(1024, max_batch_size),
    #                   'calc_is': False,
    #                   'calc_mss': False,
    #                   'calc_vendi': False}
    #     fid, is_score, mss_score, vendi_score = calc_python(**fid_kwargs)
    #     if rank == 0 and use_wandb:
    #         wandb.config.update({'FID': fid})
    #
    # torch.cuda.empty_cache()

    # fd_kwargs = {'image_path': outdir,
    #              'num_expected': 50000,
    #              'seed': 0,
    #              'batch': min(1024, max_batch_size),
    #              'calc_is': False,
    #              'calc_mss': False,
    #              'calc_vendi': False}
    # if compute_fdd:
    #     new_kwargs = { 'model': 'dinov2',
    #                    'ref_path': ref_path_fdd}
    #
    #     fd_kwargs.update(new_kwargs)
    #
    #     fdd = calc_python(**fd_kwargs)[0]
    #     if rank == 0 and use_wandb:
    #         wandb.config.update({'FDD': fdd})
    #
    # if rank == 0 and use_wandb:
    #     df = pd.DataFrame([wandb.config.as_dict()])
    #     df.to_csv(Path(outdir) / 'wandb_config.csv', index=False)

    if metrics:
        metric_kwargs = {'image_path': outdir,
                         'ref_path': ref_path,
                         'metrics': metrics,
                         'num_images': 50000,
                         'seed': 0,
                         'max_batch_size': min(1024, max_batch_size)}
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

    # Done.
    torch.distributed.barrier(device_ids=[dist.get_rank()])
    dist.print0('Done.')

#----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
#----------------------------------------------------------------------------

