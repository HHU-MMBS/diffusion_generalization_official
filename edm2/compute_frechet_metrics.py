import wandb
import os
import random
import PIL.Image
import torch
import numpy as np
import time

from tqdm import tqdm
from torch_utils import distributed as dist
from frechet_utils import calculate_stats_for_files, calculate_metrics_from_stats, save_stats, load_stats
from exp_utils import load_data



def wait_for_available_memory(required_memory_gb, gpu_id=0, check_interval=10):
    """
    Waits until the specified GPU has at least `required_memory_gb` of free memory.

    Args:
        required_memory_gb (float): Minimum required free memory in gigabytes.
        gpu_id (int): ID of the GPU to check.
        check_interval (int): Time to wait (in minutes) before checking again.
    """
    required_memory_mb = required_memory_gb * 1024  # Convert GB to MB
    while True:
        # Clear unused memory
        torch.cuda.empty_cache()

        # Get memory stats
        total_memory = torch.cuda.get_device_properties(gpu_id).total_memory  # Total memory in bytes
        reserved_memory = torch.cuda.memory_reserved(gpu_id)  # Memory reserved by CUDA in bytes
        allocated_memory = torch.cuda.memory_allocated(gpu_id)  # Allocated memory in bytes

        # Free memory is the difference between total memory and reserved memory
        free_memory_bytes = total_memory - reserved_memory + (reserved_memory - allocated_memory)
        free_memory_mb = free_memory_bytes / (1024 ** 2)  # Convert bytes to MB

        print(f"Currently available memory on GPU {gpu_id}: {free_memory_mb:.2f} MB")

        if free_memory_mb >= required_memory_mb:
            print(f"Sufficient memory available! ({free_memory_mb:.2f} MB, required: {required_memory_mb} MB)")
            break

        print(f"Not enough memory! Waiting for {check_interval} minutes before checking again...")
        time.sleep(check_interval * 60)


def hamilton_allocation_torch(freqs: torch.Tensor, n: int) -> torch.Tensor:
    """
    Given a 1D torch tensor of non-negative class frequencies (counts or probabilities),
    return a 1D tensor of integer class counts summing to n using the
    Hamilton (largest-remainder) method.

    freqs: torch.Tensor, shape (K,)
        Non-negative frequencies (need not sum to 1). Must be float or convertible.
    n: int
        Number of samples to allocate.
    """
    if freqs.ndim != 1:
        raise ValueError("freqs must be a 1D tensor.")
    if (freqs < 0).any():
        raise ValueError("freqs must be non-negative.")

    freqs = freqs.to(dtype=torch.float)

    total = freqs.sum()
    if total <= 0:
        raise ValueError("Sum of frequencies must be positive.")

    # Normalize to probabilities
    p = freqs / total

    # Ideal fractional allocations
    fractional = p * n

    # Integer part
    base = torch.floor(fractional).to(torch.int64)

    # How many samples remain unassigned?
    remainder = int(n - base.sum())

    if remainder > 0:
        # fractional remainders
        frac_parts = fractional - base

        # indices of largest fractional remainders
        _, idx = torch.topk(frac_parts, remainder, largest=True, sorted=False)

        # allocate one extra sample to each selected class
        base[idx] += 1

    return base


def get_label_frequencies(dataset_obj):
    raw_labels = torch.from_numpy(dataset_obj._get_raw_labels())  # (n_images,)
    return torch.unique(raw_labels, return_counts=True)[1]  # (label_dim,)


def calculate_ref(image_dir, metrics):
    """Calculate a reference file with the Frechet Distance statistics for images from the given path"""
    torch.distributed.barrier(device_ids=[rank])
    stats_iter = calculate_stats_for_files(metrics=metrics,
                                           image_path=image_dir,  # Path to a directory or ZIP file containing the images.
                                           num_images=None,  # Number of images to use. None = all available images.
                                           max_batch_size=512)  # Maximum batch size.
    for r in tqdm(stats_iter, unit='batch', disable=(rank != 0)):
        pass

    if rank == 0:
        save_stats(r.stats, os.path.join(image_dir, 'fd_refs.pkl'))


def subsample_dataset(n, res, dataset_obj, subsample_dir, freqs, subdirs=True, skip=False):
    """
    Subsamples n images from a dataset. Supports either random sampling or a fixed
    number of samples per class specified by the `freqs` tensor.

    Args:
        n (int): Number of images to sample.
        res (int): Expected resolution of the images.
        dataset_obj: Dataset object with methods to access data and labels.
        subsample_dir (str): Base directory where subsamples are stored.
        subdirs (bool): Whether or not to create subdirectories for subsampling.
        freqs (torch.Tensor, optional): Tensor specifying the number of samples to take
                                         per class. If provided, the total sum should equal `n`,
                                         and its shape should match `dataset_obj.get_label_dim`.
        skip: Just return outdir without subsampling
    """

    # Validate the `freqs` tensor
    if freqs.sum().item() != n:
        raise ValueError(f"The sum of `freqs` ({freqs.sum().item()}) must equal `n` ({n}).")
    if freqs.shape[0] != dataset_obj.get_label_dim():
        raise ValueError(f"The shape of `freqs` ({freqs.shape[0]}) must match the label dimension "
                         f"({dataset_obj.get_label_dim()}).")

    # Generate the output directory path with incrementing number suffix
    counter = 1
    while True:
        outdir = os.path.join(subsample_dir, f"subsamples_{n}_{counter:02d}")
        if not os.path.exists(outdir):
            break
        counter += 1
    torch.distributed.barrier(device_ids=[rank])  # Don't create outdir before others have finished scanning
    if skip:
        return outdir, None, None

    os.makedirs(outdir)

    # Set random seed for reproducibility
    random.seed(counter)
    torch.manual_seed(counter)

    # Class prior
    indices = []
    all_labels = dataset_obj._get_raw_labels()
    for class_idx, count in enumerate(tqdm(freqs.tolist(), desc=f"Sampling {n} class labels")):
        # Get all indices for the current class
        class_indices = (all_labels == class_idx).nonzero()[0]  # (n_class,)

        # Ensure there are enough samples for the class
        if len(class_indices) < count:
            raise ValueError(f"Not enough samples for class {class_idx}. Requested {count}, but only {len(class_indices)} available.")

        # Sample `count` indices without replacement
        sampled_indices = torch.multinomial(torch.ones(len(class_indices)), count, replacement=False)  # (count,)
        indices.extend(class_indices[sampled_indices].tolist())  # Add count random indices to the list

    # Subsample and save images
    labels = torch.zeros(n, dtype=torch.int)
    for seed, idx in enumerate(tqdm(indices, desc=f"Subsampling {n} images...")):
        image, label = dataset_obj[idx]  # Assuming dataset[idx] returns (image, label)
        labels[seed] = np.argmax(label).item()  # One-hot -> int
        image = image.transpose(1, 2, 0)
        assert image.shape[0] == res, f"Expected image resolution {res}, got {image.shape[0]}"
        image_dir = os.path.join(outdir, f"samples/{seed // 1000 * 1000:07d}") if subdirs else outdir
        os.makedirs(image_dir, exist_ok=True)
        PIL.Image.fromarray(image, "RGB").save(os.path.join(image_dir, f"{seed:07d}.png"))

    assert torch.equal(torch.unique(labels, return_counts=True)[1], freqs), f"Sampled labels do not match the provided frequencies."
    dist.print0(f"Subsampled {n} images to {outdir}")
    return outdir, indices, labels


def create_subsets(dataset, n_samples, val_settings, n_subsets):
    if 'cifar' in dataset:
        model_type = 'edm'
    elif 'in' in dataset:
        model_type = 'edm2'
    else:
        raise ValueError(f"Invalid dataset: {dataset}")

    res = int(dataset.replace('in', '')) if 'in' in dataset else 32  # 32 for cifar
    for val in val_settings:
        dataset_obj = load_data(dataset, res, model_type, split='train' if not val else 'val', get_loader=False, get_iter=False, batch_gpu=64)[0]
        freqs = torch.ones(dataset_obj.get_label_dim(), dtype=torch.int) * (n_samples // dataset_obj.get_label_dim())  # 50 images per class, 1000 classes = 50k images

        for i in range(n_subsets):
            subsample_dir = f"{proj_folder}/fd_analysis/{dataset}/{'val' if val else 'train'}/uniform/"
            subsample_path = os.path.join(subsample_dir, f"subsamples_{n_samples}_{i+1:02d}")
            if os.path.exists(subsample_path):
                dist.print0(f"Subsample directory {subsample_path} already exists. Skipping...")
                continue

            outdir, indices, labels = subsample_dataset(n=n_samples,
                                                        res=res,
                                                        dataset_obj=dataset_obj,
                                                        subsample_dir=subsample_dir,
                                                        freqs=freqs,
                                                        skip=rank != 0)

            if rank == 0:
                torch.save(indices, os.path.join(outdir, f"indices.pt"))
                torch.save(labels, os.path.join(outdir, f"labels.pt"))


def compute_metrics(paths1, paths2, metrics=['fid', 'fd_dinov2'], wandb_log=None, use_wandb=False):
    """
    dataset:
    metrics:
    n_trainsets:
    use_wandb:
    """

    for path1 in paths1:
        for path2 in paths2:
            # Calculate Frecheet references if needed
            ref_path1 = os.path.join(path1, "fd_refs.pkl")
            if not os.path.exists(ref_path1):
                calculate_ref(path1, metrics)

            ref_path2 = os.path.join(path2, "fd_refs.pkl")
            if not os.path.exists(ref_path2):
                calculate_ref(path2, metrics)

            # Make sure the class priors are identical
            if not ("uncond" in path1):
                labels1 = torch.load(os.path.join(path1, "labels.pt"), weights_only=True)
                labels1_unique, prior1 = torch.unique(labels1, return_counts=True)
                prior1 = prior1.cpu()  # (label_dim,), How many of each class
                labels2 = torch.load(os.path.join(path2, "labels.pt"), weights_only=True)
                labels2_unique, prior2 = torch.unique(labels2, return_counts=True)
                prior2 = prior2.cpu()  # (label_dim,), How many of each class
                if len(labels1_unique) != len(labels2_unique):
                    dist.print0(f"Warning: References have different number of classes.")
                elif not torch.equal(prior1, prior2):
                    dist.print0(f"Warning: Class priors don't match excatly between {path1} and {path2}.")
                    ratio = prior1 / prior2  # assert that each prior has the same class distribution, but not necessarily the same number of samples
                    if (ratio.max() - ratio.min()).abs() > 1e-6:
                        dist.print0(f"Warning: The class priors differ in more than sample count between {path1} and {path2}.")

            # Customize WandB logging
            if use_wandb:
                wandb_groupname = None  # What kind of experiment?
                wandb_runname = None  # What concrete run?
            else:
                wandb_groupname = None
                wandb_runname = None
                wandb_log = None

            # Compute Frechet Distances
            if rank == 0:
                if use_wandb:
                    # todo wandb setup
                    if os.access("/home/tikai103", os.R_OK):
                        os.environ["WANDB_API_KEY"] = "a9efb3b6cddc090dbf125d4c5d0dff12b178eb36"
                    wandb.init(project="diffusion_overfit", entity="hhu-mmbs", group=wandb_groupname)
                    wandb.run.name = wandb_runname
                    wandb.config.update(wandb_log)

                stats1 = load_stats(ref_path1)
                stats2 = load_stats(ref_path2)
                results = calculate_metrics_from_stats(stats=stats1, ref=stats2, metrics=metrics, verbose=True)

                if use_wandb:
                    if 'stop' in ref_path1:
                        early_stop = int(path1.split('stop')[-1].split('/')[0])
                    else:
                        early_stop =-1

                    log = {'ref_path1': ref_path1,
                           'ref_path2': ref_path2,
                           'early_stop': early_stop}

                    for metric in metrics:
                        name = "FID" if metric == 'fid' else 'FDD'
                        log[name] = results[metric]

                    wandb.config.update(log)
                    wandb.finish()
            torch.distributed.barrier(device_ids=[rank])


# Run with torchrun --nproc_per_node=4 compute_frechet_metrics.py if you need GPUs for FD-reference computation
# Run with python compute_frechet_metrics.py if all FD-references are already computed
if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    dist.init()
    rank, world_size = dist.get_rank(), dist.get_world_size()
    proj_folder = '/home/shared/generative_models/diffusion_overfit_official'  # todo: specify
    metrics = ['fid', 'fd_dinov2']

    # Measure FDs from references
    use_wandb = True
    datasets = ['in64']  # 'in64', 'in512', 'cifar10', 'cifar100'
    model_sizes = ['xs-uncond', 'xxs', 'xs', 's', 'm', 'l', 'xl', 'xxl']  # 'xs-uncond', 'xxs', 'xs', 's', 'm', 'l', 'xl', 'xxl' and [None] for edm
    ema = '0.100'
    guidance_mode = 'none'  # 'none' for no guidance, 'auto' for Autoguidance, 'regular' for CFG
    snap_idx = -1  # 'all' or specific index, -1 is allowed
    pseudo_labels = False  # kmeans pseudo labels
    for early_stop in [-1]:  # '-1' for no early_stopping, otherwise indices <= num_steps
        for mode in ['gen-v-train', 'gen-v-val']:  # 'gen-v-train', 'gen-v-val', for IN-distribution-mismatch: 'train-v-train', 'train-v-val'
            for dataset in datasets:  # 'cifar10', 'cifar100', 'in64', 'in512'
                n_trainsets = 15 if 'in' in dataset else 10  # how many train subsets to use

                res = int(dataset.replace('in', '')) if 'in' in dataset else 32
                model_type = 'edm2' if 'in' in dataset else 'edm'

                # Set train and val paths
                n_valsamples = 50000 if 'in' in dataset else 10000  # 10k for cifar
                val_paths = [f"{proj_folder}/fd_analysis/{dataset}/val/uniform/subsamples_{n_valsamples}_01/"]  # Just one
                train_folder = f"{proj_folder}/fd_analysis/{dataset}/train/uniform/"
                n_trainsamples = 50000 if 'in' in dataset else 10000  # 10k for cifar
                train_paths_arg2 = [os.path.join(train_folder, f"subsamples_{n_trainsamples}_{i+1:02d}/") for i in range(n_trainsets)]
                if 'cifar' in dataset:
                    # For CIFAR, all experiments have the form "XX-50k vs XX-10k" for comparibility
                    train_paths_arg1 = [os.path.join(train_folder, f"subsamples_50000_{i + 1:02d}/") for i in range(n_trainsets)]
                else:
                    train_paths_arg1 = train_paths_arg2

                # Set generative paths
                if 'gen' in mode:
                    gen_paths = []
                    gen_folder = f"{proj_folder}/fd_analysis/{dataset}/gen/uniform/{model_type}"
                    if 'in' in dataset:
                        for model_size in model_sizes:
                            snap_folder = os.path.join(gen_folder, model_size)
                            all_snaps = sorted(os.listdir(snap_folder))

                            if early_stop != -1:  # Search the snap folders for those that contain "stop.." folders
                                matches = []
                                for entry in os.listdir(snap_folder):  # List all items in the top directory
                                    snap_path = os.path.join(snap_folder, entry)
                                    target_structure = os.path.join(snap_path, 'none', f'stop{early_stop}')
                                    if os.path.isdir(target_structure):
                                        matches.append(snap_path.split('/')[-1])
                                all_snaps = sorted(matches)

                            if snap_idx != 'all':
                                all_snaps = [all_snaps[snap_idx]]

                            for snap in all_snaps:
                                if guidance_mode == 'none':
                                    if early_stop == -1:
                                        gen_paths.append(os.path.join(gen_folder, f"{model_size}/{snap}/{guidance_mode}/"))
                                    else:
                                        gen_paths.append(os.path.join(gen_folder, f"{model_size}/{snap}/{guidance_mode}/stop{early_stop}/"))
                                else:
                                    parent_dir = os.path.join(gen_folder, f"{model_size}/{snap}/{guidance_mode}/")
                                    for g_weight in sorted(os.listdir(parent_dir)):
                                        if 'stop' in g_weight and early_stop == -1:
                                            continue
                                        if 'stop' not in g_weight and early_stop != -1:
                                            continue
                                        gen_paths.append(os.path.join(parent_dir, f"{g_weight}/"))
                    else:
                        all_snaps = sorted(os.listdir(gen_folder))

                        if snap_idx != 'all':
                            all_snaps = [all_snaps[snap_idx]]

                        for snap in all_snaps:
                            if pseudo_labels:
                                for k in [2, 5, 10, 20, 50, 100, 200, 300, 400, 500]:
                                    snap_no = int(snap.split('/')[-1])
                                    gen_paths.append(f"{gen_folder}-kmeans{k}/{int(round(snap_no//100)*100)}/{guidance_mode}/")
                            else:
                                gen_paths.append(os.path.join(gen_folder, f"{snap}/none/"))

                # Create train subsets and fd-references if they don't exist yet
                create_subsets(dataset=dataset,  # 'cifar-10', 'cifar-100', 'in64', 'in512'
                               n_samples=n_trainsamples,  # Number of samples to generate per subset
                               val_settings=[False, True],  # train and val subsets
                               n_subsets=n_trainsets)

                if mode == 'gen-v-train':  # Measure fit to training distribution
                    paths1 = gen_paths
                    paths2 = train_paths_arg2
                elif mode == 'gen-v-val':  # Measure fit to validation distribution
                    paths1 = gen_paths
                    paths2 = val_paths
                elif mode == 'train-v-train':  # Isolate sampling noise, first vs everyone else
                    assert len(train_paths_arg2) > 1, f"train-v-train doesn't make sense if there is only one train subset."
                    paths1 = [train_paths_arg1[0]]
                    paths2 = train_paths_arg2[1:]
                elif mode == 'train-v-val':  # Isolate distribution mismatch
                    paths1 = train_paths_arg1
                    paths2 = val_paths

                if use_wandb:
                    wandb_log = {'dataset': dataset,
                                 'model_type': model_type,
                                 'mode': mode,
                                 'guidance_mode': guidance_mode,
                                 'pseudo_labels': pseudo_labels,
                                 }

                compute_metrics(paths1=paths1,
                                paths2=paths2,
                                metrics=metrics,
                                wandb_log=wandb_log,
                                use_wandb=use_wandb)
