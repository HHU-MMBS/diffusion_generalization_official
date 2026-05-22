import os
import torch
import numpy as np
import time

from generalization_gap_exps import load_data, load_snaps
from frechet_utils import get_detector
from torch_utils import distributed as dist
from tqdm import tqdm


def get_sample_path(dataset: str, model_size: str, snap: int, probe_type: str, sigma: str) -> str:
    """
    Return a list of image paths (or npz/npy batch files) for the given
    probe_type ∈ {'xr_train', 'xr_val', 'xt'}.
    """
    if probe_type == 'xr_train':
        return f"/home/shared/generative_models/diffusion_overfit/data/loss_analysis/{dataset}/edm2-{model_size}/train_data/samples-{snap}-0.100/{sigma}/"
    elif probe_type == 'xr_val':
        return f"/home/shared/generative_models/diffusion_overfit/data/loss_analysis/{dataset}/edm2-{model_size}/val_data/samples-{snap}-0.100/{sigma}/"
    elif 'xt' in probe_type:
        if not 'stop' in probe_type:
            return f"/home/shared/generative_models/diffusion_overfit/fd_analysis/{dataset}/gen/uniform/edm2/{model_size}/{snap}/none/samples/"
        else:
            return f"/home/shared/generative_models/diffusion_overfit/fd_analysis/{dataset}/gen/uniform/edm2/{model_size}/{snap}/none/{probe_type.split('_')[-1]}/samples/"
    else:
        raise ValueError(f"Invalid probe_type: {probe_type}")


def get_reference_path(dataset: str, split: str) -> str:
    """
    Return a path for the reference set.
    split ∈ {'train', 'val'}
    """
    return f"/home/shared/generative_models/diffusion_overfit/fd_analysis/{dataset}/{split}/uniform/subsamples_50000_01/samples/"


def get_ref_features_cache_path(dataset: str, split: str) -> str:
    """Return a .pt path for caching reference features."""
    return f"/home/shared/generative_models/diffusion_overfit/fd_analysis/{dataset}/{split}/uniform/subsamples_50000_01/feats.pt"


def get_output_path(dataset: str, model_size: str, snap: int, probe_type: str, sigma: str, f_extractor: str) -> str:
    """
    Return a .npy path for saving per-sample 1-NN cosine similarities.
    e.g. results/{dataset}/{model_size}/snap{snap:06d}_{probe_type}_nn_cossim.npy
    """
    # if f_extractor == 'fd_dinov2':
    #     f_str = 'dino'
    # elif f_extractor == 'fid':
    #     f_str = 'incp'
    f_str = f_extractor  # todo delete and rerun
    return f"/home/shared/generative_models/diffusion_overfit/data/fnn_analysis/{dataset}/edm2-{model_size}/{snap}-0.100/{sigma}/{probe_type}/cos-sims-{f_str   }.npy"

# ── internals ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_features(
        f_extractors: list[torch.nn.Module],
        dataset_obj: torch.utils.data.Dataset,
        batch_size: int = 256,
        device: torch.device = torch.device('cuda'),
) -> list[torch.Tensor]:
    """Extract features for a list of image paths. Returns (N, D)."""
    # Arange batches of image indices for each GPU
    num_samples = len(dataset_obj)
    idx = torch.arange(num_samples, device=device)
    num_batches = max((num_samples - 1) // (batch_size * world_size) + 1, 1) * world_size
    img_batches = idx.tensor_split(num_batches)[rank::world_size]
    dataset_loader =torch.utils.data.DataLoader(dataset=dataset_obj, batch_sampler=img_batches, pin_memory=True, num_workers=0)

    feats = [torch.zeros(num_samples, f.feature_dim, dtype=torch.float64, device=device) for f in f_extractors]
    for i, (tensors, raw_idx) in enumerate(zip(dataset_loader, img_batches)):
        images, _ = tensors
        clean = images.to(device)

        for k, f in enumerate(f_extractors):
            feats[k][raw_idx] = f(clean).to(torch.float64)

    # reduce across GPUs
    for k in range(len(f_extractors)):
        torch.distributed.all_reduce(feats[k], op=torch.distributed.ReduceOp.SUM)
        feats[k] = feats[k].cpu()


    torch.distributed.barrier(device_ids=[rank])
    return feats  # (N, D)


def load_or_compute_ref_features(
        dataset: str,
        split: str,
        model_type: str,
        f_extractors: list[torch.nn.Module],
        batch_size: int,
        device: torch.device,
) -> list[torch.Tensor]:
    """Load cached reference features if present, else compute and cache."""
    cache_path = get_ref_features_cache_path(dataset, split)
    if not os.path.exists(cache_path):
        dist.print0(f"Computing {split} reference features …")
        ref_path = get_reference_path(dataset, split)
        dataset_obj, _, _ = load_data(dataset=ref_path, model_type=model_type, split=split, get_loader=False, get_iter=False)  # No labels needed
        feats = extract_features(f_extractors, dataset_obj, batch_size, device)
        if rank == 0:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            torch.save(feats, cache_path)
            dist.print0(f"Saved to {cache_path}")
        torch.distributed.barrier(device_ids=[rank])
        
    # Reload clean copy on all GPUs
    feats = torch.load(cache_path, weights_only=True)
    for k in range(len(f_extractors)):
        feats[k] = feats[k].to(device)
    torch.distributed.barrier(device_ids=[rank])
    return feats


@torch.no_grad()
def nearest_neighbor_cossim(
    query_feats: torch.Tensor,      # (N, D), L2-normalised, full tensor on every GPU
    ref_feats: torch.Tensor,        # (M, D), L2-normalised, full tensor on every GPU
    chunk_size: int = 1024,
):
    """
    Multi-GPU 1-NN cosine similarity.
    Each GPU fills its strided slice [rank :: world_size] of a shared result tensor.
    all_reduce (sum) collects all slices. Rank 0 returns the full (N,) array.
    """
    device = torch.device("cuda", rank)
    N = len(query_feats)

    ref_feats = ref_feats.to(device)
    ref_feats = torch.nn.functional.normalize(ref_feats, dim=-1)  # L2-normalization

    # Each GPU processes its own strided slice of the queries
    local_indices = range(rank, N, world_size)
    local_query = query_feats[local_indices].to(device)     # (n_local, D)
    local_query = torch.nn.functional.normalize(local_query, dim=-1)  # L2-normalization

    # Compute 1-NN cos-sim for this GPU's queries
    local_sims = []
    for i in range(0, len(local_query), chunk_size):
        q = local_query[i : i + chunk_size]                 # (B, D)
        sims = q @ ref_feats.T                              # (B, M)
        local_sims.append(sims.max(dim=-1).values)
    local_sims = torch.cat(local_sims)                      # (n_local,)

    # Write into the correct positions of a full-size zero tensor
    result = torch.zeros(N, dtype=torch.float32, device=device)
    result[list(local_indices)] = local_sims.to(torch.float32)

    # Sum across GPUs — each position is nonzero on exactly one GPU
    torch.distributed.all_reduce(result, op=torch.distributed.ReduceOp.SUM)

    if rank != 0:
        return None
    return result.cpu().numpy()


# ── main entry point ──────────────────────────────────────────────────────────

def compute_nn_cossim(
        dataset: str,
        model_size: str,
        snaps: list[int],
        all_sigmas: list[str],
        num_samples: int,
        batch_size: int = 256,
        nn_chunk_size: int = 1024,
        overwrite: bool = False,
        device: torch.device = torch.device('cuda'),
) -> None:
    """
    For each snap, compute per-sample 1-NN cosine similarity for:
      - xr_train  →  query against train reference
      - xr_val    →  query against val   reference
      - xt        →  query against train reference
      - xt        →  query against val   reference

    Results saved as .npy arrays of shape (N,), one file per (snap, probe_type, ref_split).
    """
    model_type = 'edm2'  # not implemented for 'edm' for now

    # Load feature extractors
    f_names = ['fd_dinov2', 'fid']
    f_extractors = [get_detector(metric, verbose=True) for metric in f_names]
    for f in f_extractors:
        f.model.eval()
        for p in f.model.parameters():
            p.requires_grad_(False)

    # Pre-load both reference sets (with caching)
    ref_feats = {
        "train": load_or_compute_ref_features(dataset, "train", model_type, f_extractors, batch_size, device),
        "val": load_or_compute_ref_features(dataset, "val", model_type, f_extractors, batch_size, device),
    }

    # probe_type → which reference split(s) to query against
    probe_ref_map = {
        "xr_train": ["train", "val"],
        "xr_val": ["train", "val"],
        "xt": ["train", "val"],
        "xt_stop18": ["train", "val"]
    }

    pbar = tqdm(snaps, desc="", disable=rank != 0)

    for snap in pbar:
        for sigma in all_sigmas:
            pbar.set_description(f"snap {snap}, sigma: {sigma}")
            for probe_type, ref_splits in probe_ref_map.items():
                sample_path = get_sample_path(dataset, model_size, snap, probe_type, sigma)

                # Check existence of results, skip if results against both ref_splits exist
                skip = 0
                counter = 0
                for ref_split in ref_splits:
                    for k in range(len(f_extractors)):
                        out_path = get_output_path(dataset, model_size, snap, f"{probe_type}_vs_{ref_split}", sigma, f_names[k])
                        if os.path.exists(out_path) and not overwrite:
                            skip += 1
                        counter += 1

                if skip == counter:  # If all exist
                    pbar.set_description(f"snap {snap}, sigma: {sigma}: Skipping {probe_type}, because results already exist")
                    time.sleep(0.5)
                    continue

                # Check if query samples exist, otherwise skip
                pbar.set_description(f"snap {snap}, sigma: {sigma}: Extracting features for {probe_type}")
                if not os.path.exists(sample_path):
                    pbar.set_description(f"snap {snap}, sigma: {sigma}: Skipping, because {sample_path} doesn't exist")
                    time.sleep(3)
                    continue

                # Extract query features once per probe type, reuse for both ref splits
                dataset_obj, _, _ = load_data(dataset=sample_path, model_type=model_type, split=None, get_loader=False, get_iter=False, max_size=num_samples, verbose=False)  # No labels needed
                query_feats = extract_features(f_extractors, dataset_obj, batch_size, device)

                for ref_split in ref_splits:
                    for k in range(len(f_extractors)):
                        out_path = get_output_path(dataset, model_size, snap, f"{probe_type}_vs_{ref_split}", sigma, f_names[k])
                        if os.path.exists(out_path) and not overwrite:
                            continue

                        pbar.set_description(f"snap {snap}, sigma: {sigma}: Computing 1-NN cos-sim: {probe_type} vs {ref_split} …")
                        nn_sims = nearest_neighbor_cossim(query_feats[k], ref_feats[ref_split][k], nn_chunk_size)

                        if rank == 0:
                            os.makedirs(os.path.dirname(out_path), exist_ok=True)
                            np.save(out_path, nn_sims)
                        torch.distributed.barrier(device_ids=[rank])


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    dist.init()
    rank, world_size = dist.get_rank(), dist.get_world_size()
    model_type = 'edm2'
    ema = '0.100'

    for dataset in ['in64']:
        res = int(dataset.replace('in', '')) if 'in' in dataset else 32
        for model_size in ['s']:
            snaps = load_snaps(which_snaps='all',  # which_snap is an index or 'all'
                               model_type=model_type,
                               res=res,
                               model_size=model_size,
                               ema=ema)[::-1]

            if len(snaps) >= 32:
                snaps = snaps[::4]
            elif len(snaps) >= 16:
                snaps = snaps[::2]

            snaps = [snap.split('-')[-2] for snap in snaps]  # only keep snap numbers
            compute_nn_cossim(dataset, 
                              model_size, 
                              snaps, 
                              num_samples=8192,  # How many samples to use for query features
                              all_sigmas=['1.19e+00'],  # '1.01e+00', '1.19e+00', '1.41e+00', '1.67e+00'
                              )
