import os
import dnnlib
import torch
import pickle

from torch_utils import distributed as dist
from torch_utils import misc


def load_data(dataset, model_type, split='train', get_loader=True, get_iter=True, batch_gpu=64, pseudo_label_path=None, max_size=None, verbose=True):
    if 'in' in dataset:
        dataset_str = None
        if dataset == 'in64':
            dataset_str = 'IN_64x64_karras/imagenet-64x64'
        elif dataset == 'in512-raw':
            dataset_str = 'IN_512x512_karras/img512'
        elif dataset == 'in512':
            dataset_str = 'IN_512x512_karras/img512-sd'
        dataset_path = f"/home/shared/DataSets/vision_benchmarks/{dataset_str}{'-val' if split == 'val' else ''}.zip"
        if dataset_str is None:
            if verbose:
                dist.print0(f"Warning: Interpreting 'dataset' arg as full dataset path.")
            dataset_path = dataset
    elif 'cifar' in dataset:
        dataset_path = f"/home/shared/DataSets/{dataset.replace('cifar', 'cifar-')}/{dataset}-32x32{'-val' if split == 'val' else ''}.zip"
    else:
        if verbose:
            dist.print0(f"Warning: Interpreting 'dataset' arg as full dataset path.")
        dataset_path = dataset

    if 'kmeans' in model_type:  # model trained with kmeans pseudo labels (https://github.com/HHU-MMBS/cedm-official-wavc2025)
        n_clusters = int(model_type.split('-')[-1])
        pseudo_label_path = f"/home/shared/generative_models/cluster_ids/{dataset}/kmeans-dino_vitb16/clusters-{n_clusters}/cluster_ids{'_test' if split == 'val' else '_train'}.pt"
        if verbose:
            dist.print0(f'Loading {split if split is not None else ""} dataset from "{dataset_path}" with pseudo labels from "{pseudo_label_path}"...')
    else:
        n_clusters = None
        if verbose:
            dist.print0(f'Loading {split} dataset from "{dataset_path}"...')

    dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=dataset_path,
                                     use_labels=True, pseudo_label_path=pseudo_label_path, label_dim=n_clusters,
                                     max_size=max_size)
    dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs)

    dataset_loader = None
    if get_loader:
        data_loader_kwargs = dict(class_name='torch.utils.data.DataLoader', pin_memory=True, num_workers=2,
                                  prefetch_factor=2)

        state = dnnlib.EasyDict(cur_nimg=0, total_elapsed_time=0)
        dataset_sampler = misc.InfiniteSampler(dataset=dataset_obj, rank=dist.get_rank(), num_replicas=dist.get_world_size(), seed=0,
                                               start_idx=state.cur_nimg)
        dataset_loader = dnnlib.util.construct_class_by_name(dataset=dataset_obj, sampler=dataset_sampler,
                                                             batch_size=batch_gpu, **data_loader_kwargs)
    return dataset_obj, dataset_loader, iter(dataset_loader) if get_iter else None


def load_models(pos_pkl=None, neg_pkl=None, device='cuda'):
    """Load a model given a full path in pos_pkl. Optionally, load a negative model from neg_pkl."""

    def open_data(pkl):
        if os.path.exists(pkl):
            with open(pkl, 'rb') as f:
                data = pickle.load(f)
            return data
        else:
            raise FileNotFoundError(f"File not found: {pkl}")

    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier(device_ids=[dist.get_rank()])

    if pos_pkl is not None:
        data = open_data(pos_pkl)
        net_pos = data['ema'].to(device)
        encoder = data.get('encoder', None)
        if encoder is None:
            encoder = dnnlib.util.construct_class_by_name(class_name='training.encoders.StandardRGBEncoder')
        n_params = sum(p.numel() for p in net_pos.parameters())
        dist.print0(f'Loaded positive network from "{pos_pkl}" with {n_params:,.0f} parameters.')
    else:
        net_pos = None

    if neg_pkl is not None:
        data = open_data(neg_pkl)
        net_neg = data['ema'].to(device)
        n_params = sum(p.numel() for p in net_neg.parameters())
        dist.print0(f'Loaded negative network from "{neg_pkl}" with {n_params:,.0f} parameters.')
    else:
        net_neg = None

    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier(device_ids=[dist.get_rank()])

    return net_pos, encoder, net_neg


def load_snaps(which_snaps, dataset, model_type, res, model_size=None, ema='0.100'):
    """Prepare model paths"""
    assert type(which_snaps) == int or which_snaps == 'all', "Invalid which_snaps."

    if model_type == 'edm2':
        network_folder = f"/home/shared/generative_models/EDM2/IN{res}/models/edm2-img{res}-{model_size}/"
        # network_folder = f"https://nvlabs-fi-cdn.nvidia.com/edm2/raw-snapshots/edm2-img{res}-{model_size}/"   # Online from NVIDIA
        all_snaps = sorted([os.path.join(network_folder, s) for s in os.listdir(network_folder) if ('edm2-' in s and ema in s)])
    elif 'edm' in model_type or ('P_' in model_type) or 'theory' in model_type or 'ours' in model_type:
        if 'ours' in model_type:
            run_folder = f"00{model_type.replace('ours-', '')}-{dataset}-32x32-cond-ddpmpp-ours-gpu0-batch1024-fp16"
            network_folder = f"/home/shared/generative_models/training/edm/{dataset}/sigma_eps_tuning/gamma_6/{run_folder}"
        elif 'kmeans' in model_type:
            n_clusters = model_type.split('-')[-1]
            network_folder = f"/home/shared/generative_models/training/edm/{dataset}/self-conditioning/kmeans-{n_clusters}/"
        elif model_type == 'edm-uncond':
            network_folder = f"/home/shared/generative_models/inductive_bias/edm/{dataset}/training/uncond-default/"
        else:
            if 'theory' in model_type:
                run_folder = f"cond-theory-loss"
            else:
                run_folder = f"cond-default{model_type.replace('edm', '')}"
            network_folder = f"/home/shared/generative_models/inductive_bias/edm/{dataset}/training/{run_folder}/"
        all_snaps = sorted([os.path.join(network_folder, s) for s in os.listdir(network_folder) if ('snapshot' in s)])[
                    1:]  # Exclude 0M
    else:
        raise NotImplementedError(f"Unsupported model type: {model_type}")

    if which_snaps != 'all':
        snaps = [all_snaps[which_snaps]]
    else:
        snaps = all_snaps

    dist.print0(f"Loading {len(snaps)} snap(s).")
    return snaps
