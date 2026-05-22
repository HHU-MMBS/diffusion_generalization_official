# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Loss functions used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import torch
import dnnlib
import pickle
import os
from torch_utils import persistence
from torch_utils import distributed as dist
from generate import StackedRandomGenerator
from tqdm import tqdm


@persistence.persistent_class
class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5, **kwargs):  # kwargs for compatibility
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, images, labels=None, augment_pipe=None, sigma=None, return_stats=False, **kwargs):
        # Log normal
        if sigma is None:
            rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
            sigma = (rnd_normal * self.P_std + self.P_mean).exp()
            # sigma = torch.rand([images.shape[0], 1, 1, 1], device=images.device) * 90 + 0.2

        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        eps = (y - D_yn) ** 2  # [bs, C, H, W]
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2

        if return_stats:
            return weight * eps, weight, eps
        return weight * eps


@persistence.persistent_class
class ResidualLossCont:
    def __init__(self, sigma_od_min, sigma_od_max, sigma_eps, gamma,
                 sigma_data=0.5, range_alpha=0, device='cuda'):
        """
        Args:
            sigma_od_min: Minimum value of OD (our sigma distribution)
            sigma_od_max: Maximum value of OD (our sigma distribution)
            sigma_data: Average std of the normalized dataset
            sigma_eps: Ground truth noise level
            gamma: Exponent for asymptotic behavior for sigma -> 0
            device: Training device
        """

        self.device = device
        self.sigma_data = sigma_data

        self.sigma_od_min = sigma_od_min
        self.sigma_od_max = sigma_od_max
        self.gamma = gamma
        self.sigma_eps = sigma_eps

        self.net_neg = None
        self.range_alpha = range_alpha
        self.eps = 1e-8

    def __call__(self, net, images, labels=None, augment_pipe=None, idx=None, sigma=None, return_stats=False, **kwargs):
        """One training step.
        Args:
            net: Diffusion model
            images: batch of targets
            labels: batch of labels
            augment_pipe: karras augmentation pipeline
            sigma: batch of sigmas, if None, sample from sigma-distribution
            return_stats: whether to additionally return eps and weight
        """
        bs = images.shape[0]
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        sigma = self.sample_sigma(bs) if sigma is None else sigma  # [bs, 1, 1, 1]
        x = y + sigma * torch.randn_like(y)  # [bs, C, H, W]

        # Compute loss residual
        D_x = net(x, sigma, labels, augment_labels=augment_labels)

        # Compute weight
        if self.range_alpha:
            with torch.no_grad():
                D_w = self.net_neg(x, sigma, labels, augment_labels=augment_labels)
                range_alpha = (D_x.detach() - D_w + self.eps).abs() ** self.range_alpha
                range_alpha /= range_alpha.mean(dim=(1, 2, 3), keepdim=True)  # Normalize
                assert torch.isnan(range_alpha).sum() == 0, f"Range alpha has at least one nan"
        else:
            range_alpha = 1.0

        eps = (y - D_x) ** 2  # [bs, C, H, W], eq. (10)
        weight = sigma ** -2  # [bs, 1, 1, 1]
        assert torch.isnan(eps).sum() == 0, f"Objective has at least one nan."
        if return_stats:
            return weight * eps, weight, eps
        return range_alpha * weight * eps

    @torch.no_grad()
    def sample_sigma(self, bs):
        """Sample sigma from our distribution ~ sigma / (sigma^2 + eps^2) via inverse transform sampling"""
        return self.sample_sigma_pdf((bs, 1, 1, 1), self.sigma_od_min, self.sigma_od_max, self.sigma_eps, self.gamma)

    #------------------------------------------------------------------------
    # Functions for analytical calculations with the sigma_alpha distribution
    @staticmethod
    def sigma_f(x, eps, gamma):
        """Our unnormalized sigma distribution function f(x,c)."""
        return x ** (gamma - 1) / (x ** gamma + eps ** gamma)

    @staticmethod
    def sigma_F(x, eps, gamma):
        """Integral of f(x, c)"""
        if type(eps) != torch.Tensor:
            eps = torch.tensor(eps)
        if type(x) != torch.Tensor:
            x = torch.tensor(x)
        return torch.log(eps**gamma + x**gamma) / gamma

    def cdf(self, x, a, b, eps, gamma):
        """CDF of our sigma distribution for domain [a,b] and sigma_eps"""
        Fa = self.sigma_F(a, eps, gamma)
        return (self.sigma_F(x, eps, gamma) - Fa) / (self.sigma_F(b, eps, gamma) - Fa)

    def sample_sigma_pdf(self, shape, a, b, eps, gamma):
        """Sample from our sigma distribution via inverse transform sampling"""
        rand = torch.rand(shape, device=self.device)
        if gamma != 0:
            Z = self.sigma_F(b, eps, gamma) - self.sigma_F(a, eps, gamma)  # Normalization constant
            e = torch.exp(gamma * Z * rand)
            return ((e - 1) * eps ** gamma + e * a ** gamma) ** (1 / gamma)  # More numerically stable formula for rand = 0
        else:
            if type(a) != torch.Tensor:
                a = torch.tensor(a)
            if type(b) != torch.Tensor:
                b = torch.tensor(b)
            return torch.exp((torch.log(b) - torch.log(a)) * rand) * a  # dist: p(x) = 1/x


@persistence.persistent_class
class EfficientLoss:
    def __init__(self, sigma_od_min, sigma_od_max, resume_pkl, dataset_obj, eff_ema_weight,
                 batch_gpu, device='cuda'):
        """
        Args:
            sigma_od_min: Minimum value of OD (our sigma distribution)
            sigma_od_max: Maximum value of OD (our sigma distribution)
            device: Training device
        """

        self.device = device

        self.sigma_od_min = sigma_od_min
        self.sigma_od_max = sigma_od_max

        # Efficient weight
        dist.print0(f"Initializing EMA with snaps")
        resume_dir = os.path.dirname(resume_pkl)
        snaps = sorted([s for s in os.listdir(resume_dir) if 'network-snapshot' in s])[1:]  # Exclude 0M

        # Load dataset.
        self.n_bins = 128   # Number of bins for sigma
        self.num_samples = 512  # how many samples to run per noise step
        idx = torch.arange(self.num_samples).repeat_interleave(self.n_bins)  # Same images for each noise step

        # Split the data into batches across GPUs
        self.rank_batches, self.range_batches, num_batches = self.assign_batches(idx, batch_gpu)

        # Use the rank batches to tell the dataloader which batches to sample, different for every GPU
        self.eff_dataset_loader = torch.utils.data.DataLoader(dataset=dataset_obj, batch_sampler=self.rank_batches,
                                                              pin_memory=True,
                                                              num_workers=0,  # num_workers 0 hast the best performance here
                                                              prefetch_factor=2)

        assert idx.max() < len(dataset_obj), f'Index out of bounds. {idx.max()}, {len(dataset_obj)}'

        # Initialize sigma bins
        start = float(torch.log(torch.tensor([sigma_od_min])) / torch.log(torch.tensor([10])))  # log(sigma_min) base 10
        end = float(torch.log(torch.tensor([sigma_od_max])) / torch.log(torch.tensor([10])))  # log(sigma_max) base 10
        self.sigma_bins = torch.logspace(start, end, self.n_bins, device=device)

        # Initialize EMA and eff weight
        self.err_ema = None
        self.eff_weight = None
        self.ema_weight = eff_ema_weight

        pbar = tqdm(snaps, disable=(dist.get_rank() != 0))
        for snap in pbar:
            pbar.set_description(f"{snap}")

            # Load network.
            with dnnlib.util.open_url(f"{resume_dir}/{snap}", verbose=(dist.get_rank() == 0)) as f:
                data = pickle.load(f)
                net = data['ema'].to(device)
            del data

            self.update_efficient_weight(net, snap, first=snap == snaps[0])

    @torch.no_grad()
    def update_efficient_weight(self, net, snap, first=False):
        # Initialize results tensors
        all_eps = torch.zeros(self.n_bins, self.num_samples, dtype=torch.float32, device=self.device)
        # all_counter = torch.zeros(self.n_bins, self.num_samples, dtype=torch.float32, device=self.device)

        # Loop over all batches and collect results
        for i, (img_idx, tensors, range_idx) in enumerate(zip(self.rank_batches, self.eff_dataset_loader, self.range_batches)):
            if len(img_idx) == 0:  # img_idx: e.g. 0, 0, 0, 0, 1, 1, 1, 1, ...
                continue
            images, labels = tensors[:2]
            images = images.to(self.device).to(torch.float32) / 127.5 - 1
            labels = labels.to(self.device)

            # Collect residual from forward pass
            sigma_idx = range_idx % self.n_bins  # e.g. 0, 1, 2, 3, 0, 1, 2, 3, ...

            rnd = StackedRandomGenerator(self.device, img_idx)  # The same noise for the same images
            sigma = self.sigma_bins[sigma_idx][:, None, None, None].to(self.device)
            y = images  # No augmentation
            D_yn = net(y + rnd.randn_like(y) * sigma, sigma, labels, augment_labels=None)
            eps = (y - D_yn) ** 2  # [bs, C, H, W]
            eps = eps.mean(dim=(1, 2, 3))  # [bs]

            # Collect results
            all_eps[sigma_idx, img_idx] = eps
            # all_counter[sigma_idx, img_idx] = 1

        # Aggregate results across GPUs
        torch.distributed.all_reduce(all_eps, op=torch.distributed.ReduceOp.SUM)

        # Average over images
        all_eps = all_eps.sum(dim=1)

        # Update EMA
        self.err_ema = all_eps if first else all_eps * self.ema_weight + self.err_ema * (1 - self.ema_weight)

        # Update efficient weight
        self.eff_weight = (self.err_ema - all_eps).abs() / all_eps

        # # For testing purposes
        # if dist.get_rank() == 0:
        #     torch.save(self.eff_weight.cpu(), f'eff_weight_{snap}.pt')

        # Sync processes
        torch.distributed.barrier()

    def __call__(self, net, images, labels=None, augment_pipe=None, idx=None, sigma=None, return_stats=False, **kwargs):
        """One training step.
        Args:
            net: Diffusion model
            images: batch of targets
            labels: batch of labels
            augment_pipe: karras augmentation pipeline
            sigma: batch of sigmas, if None, sample from sigma-distribution
            return_stats: whether to additionally return eps and weight
        """
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        sigma = self.sample_sigma(images.shape[0]) if sigma is None else sigma  # [bs, 1, 1, 1]
        x = y + sigma * torch.randn_like(y)  # [bs, C, H, W]

        # Compute loss residual
        D_x = net(x, sigma, labels, augment_labels=augment_labels)

        eps = (y - D_x) ** 2  # [bs, C, H, W], eq. (10)
        weight = sigma ** -2  # [bs, 1, 1, 1]

        if return_stats:
            return weight * eps, weight, eps
        return weight * eps

    @torch.no_grad()
    def sample_sigma(self, bs):
        """Sample sigma for each image in the batch from our efficient weight distribution. Sample a bin according to
        the distribution first and then uniformly sample a sigma in that bin."""
        # Sample a bin for each image
        bin_values = (self.eff_weight[1:] + self.eff_weight[:-1]) / 2  # [n_bins - 1], average of endpoints for each bin
        bin_values /= bin_values.sum()  # Normalize
        bin_idx = torch.multinomial(bin_values, bs, replacement=True)  # [bs]

        # Sample a sigma for each bin
        variation = torch.rand(bs, device=self.device)
        sigma = self.sigma_bins[bin_idx] + variation * (self.sigma_bins[bin_idx + 1] - self.sigma_bins[bin_idx])  # [bs]
        return sigma[:, None, None, None]  # [bs, 1, 1, 1]


    @staticmethod
    def assign_batches(idx, batch_gpu):
        """
        Takes in a list of indices and returns a list of batches of indices for the local rank.
        Args:
            idx: list of indices
            batch_gpu: batch size per GPU
        returns:
            rank_batches: list of batches of indices from idx
            range_batches: list of batches of indices from arange(len(idx))
            num_batches: total number of batches
        """
        n_struc = len(idx)
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        num_batches = ((n_struc - 1) // (batch_gpu * world_size) + 1) * world_size
        num_batches = max(num_batches, world_size)  # Make sure we have at least one batch per GPU.
        all_batches = torch.as_tensor(idx).tensor_split(num_batches)
        rank_batches = all_batches[rank:: world_size]
        range_batches = torch.arange(n_struc).tensor_split(num_batches)[
                        rank:: world_size]  # for placing results in the right places later
        return rank_batches, range_batches, num_batches


@persistence.persistent_class
class RangeLoss:
    def __init__(self, sigma_od_min, sigma_od_max, neg_snap, device='cuda'):
        """
        Args:
            sigma_od_min: Minimum value of bin range
            sigma_od_max: Maximum value of bin range
            device: Training device
        """

        self.device = device
        self.net_neg = None
        self.n_bins = 128  # Number of bins for sigma
        self.sigma_bins = torch.logspace(sigma_od_min, sigma_od_max, self.n_bins, device=device)
        self.eps = 1e-8
        self.alpha = 1.0
        self.neg_snap = neg_snap

    def __call__(self, net, images, labels=None, augment_pipe=None, idx=None, sigma=None, return_stats=False, kimg=None):
        """One training step.
        Args:
            net: Diffusion model
            images: batch of targets
            labels: batch of labels
            augment_pipe: karras augmentation pipeline
            sigma: batch of sigmas, if None, sample from sigma-distribution
            return_stats: whether to additionally return eps and weight
        """
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        sigma = self.sample_sigma(images.shape[0]) if sigma is None else sigma  # [bs, 1, 1, 1]
        x = y + sigma * torch.randn_like(y)  # [bs, C, H, W]

        # Compute loss residual
        D_x = net(x, sigma, labels, augment_labels=augment_labels)

        # Compute weight
        if kimg > self.neg_snap:
            with torch.no_grad():
                D_s = D_x.detach()
                D_w = self.net_neg(x, sigma, labels, augment_labels=augment_labels)
                range_alpha = (D_s - D_w + self.eps).abs()
                range_alpha /= range_alpha.mean(dim=(1,2,3), keepdim=True)  # Normalize
        else:
            range_alpha = 1.0

        eps = (y - D_x) ** 2  # [bs, C, H, W], eq. (10)
        weight = sigma ** -2  # [bs, 1, 1, 1]

        if return_stats:
            return weight * eps, weight, eps
        return range_alpha * weight * eps

    @torch.no_grad()
    def sample_sigma(self, bs):
        """Sample sigma for each image in the batch from our efficient weight distribution. Sample a bin according to
        the distribution first and then uniformly sample a sigma in that bin."""
        # Sample a bin for each image
        bin_idx = torch.randint(self.n_bins - 1, (bs,))  # [bs]

        # Sample a sigma for each bin
        variation = torch.rand(bs, device=self.device)
        sigma = self.sigma_bins[bin_idx] + variation * (self.sigma_bins[bin_idx + 1] - self.sigma_bins[bin_idx])  # [bs]
        return sigma[:, None, None, None]  # [bs, 1, 1, 1]


    @staticmethod
    def assign_batches(idx, batch_gpu):
        """
        Takes in a list of indices and returns a list of batches of indices for the local rank.
        Args:
            idx: list of indices
            batch_gpu: batch size per GPU
        returns:
            rank_batches: list of batches of indices from idx
            range_batches: list of batches of indices from arange(len(idx))
            num_batches: total number of batches
        """
        n_struc = len(idx)
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        num_batches = ((n_struc - 1) // (batch_gpu * world_size) + 1) * world_size
        num_batches = max(num_batches, world_size)  # Make sure we have at least one batch per GPU.
        all_batches = torch.as_tensor(idx).tensor_split(num_batches)
        rank_batches = all_batches[rank:: world_size]
        range_batches = torch.arange(n_struc).tensor_split(num_batches)[
                        rank:: world_size]  # for placing results in the right places later
        return rank_batches, range_batches, num_batches
