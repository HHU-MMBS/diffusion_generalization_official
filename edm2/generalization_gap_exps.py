import os
import torch
import numpy as np
import PIL.Image
import torch.nn.functional as F

from tqdm import tqdm
from generate_images import StackedRandomGenerator
from frechet_utils import get_detector
from exp_utils import load_data, load_snaps, load_models
from torch_utils import distributed as dist


@torch.no_grad()
def SWG(net_forward, net, x, t, labels, noise, swg_size=40, swg_steps=2,
        mode='average', dtype=torch.float32):
    """
    modes:
        - 'average': Average overlapping pixels
        - 'cut': Every pixel is assinged to the window with the closest center
    """

    def edges(swg_size, stride, i, j):
        le = stride * i
        te = stride * j
        re = stride * i + swg_size
        be = stride * j + swg_size
        return le, re, te, be

    img_size = x.shape[-1]
    y_neg = torch.zeros_like(x)
    if mode == 'cut':
        center_tracker = torch.full_like(x, fill_value=x.shape[-1], dtype=torch.long)  # Maximum distance is feature-dim
        # center_indices = torch.zeros_like(x, dtype=torch.int32)
    elif mode == 'average':
        w = torch.zeros_like(x)
    elif mode == 'center':
        pass
    else:
        raise ValueError(f"Invalid mode: {mode}")

    assert swg_steps in [1, 2, 3, 5]
    stride = (img_size - swg_size) // (swg_steps - 1) if swg_steps > 1 else 0

    logvar = None
    if mode == 'center':
        gap = (x.shape[-1] - swg_size) // 2  # gap on either seide of the window
        xc = x[:, :, gap:-gap, gap:-gap]
        y_posc, logvar = net_forward(net, xc, t, labels, noise)
        y_posc = y_posc.to(dtype)
        y_neg[:, :, gap:-gap, gap:-gap] = y_posc
    elif mode == 'average' or mode == 'cut':
        for i in range(swg_steps):
            for j in range(swg_steps):
                # Calculate window edges
                le, re, te, be = edges(swg_size, stride, i, j)

                # Extract the local window and run through net_forward
                xc = x[:, :, te:be, le:re]
                noise_c = noise[:, :, te:be, le:re]
                y_posc, logvar = net_forward(net, xc, t, labels, noise_c)
                y_posc = y_posc.to(dtype)

                if mode == 'average':
                    # Accumulate local results in y_neg and count pixel overlap
                    y_neg[:, :, te:be, le:re] += y_posc
                    w[:, :, te:be, le:re] += torch.ones_like(y_posc)
                elif mode == 'cut':
                    # Compute the absolute (L1) distance from the window center to the current pixels
                    window_center_x = (te + be) // 2
                    window_center_y = (le + re) // 2

                    distance = torch.abs(torch.arange(te, be, device=x.device)[None, None, :, None] - window_center_x) + \
                               torch.abs(torch.arange(le, re, device=x.device)[None, None, None,
                                         :] - window_center_y)  # (swg_size, swg_size)
                    distance = distance.repeat_interleave(x.shape[0], dim=0)  # broadcast batch dimension
                    distance = distance.repeat_interleave(x.shape[1], dim=1)  # broadcast channel dimension

                    # Only update where the distance to this center is smaller
                    mask = distance < center_tracker[:, :, te:be, le:re]
                    center_tracker[:, :, te:be, le:re][mask] = distance[mask]  # Update center tracker
                    # center_indices[:, :, te:be, le:re][mask] = i * swg_steps + j
                    y_neg[:, :, te:be, le:re][mask] = y_posc[mask]  # Update y_neg

    if mode == 'average':
        y_neg = y_neg / w

    return y_neg, logvar


@torch.no_grad()
def generalization_gap(dataset, snaps, res, model_type, model_size, split='train',
                       num_steps=64, num_samples=8192, max_batch_size=256, encoder_batch_size=4,
                       neg_snap=None, guidance_weights=[0.0], swg_sizes=[None], save_preds=False,
                       custom_img_dir=None, pseudo_label_path=None, sigma_ids=None, device='cuda',
                       ):
    """
    num_steps: number of noise levels
    num_samples: number of samples for each noise level
    neg_snap: index of snap for negative model, default is last, i.e., longest training
    """
    if custom_img_dir is None:
        dataset_str = dataset
    else:
        dataset_str = custom_img_dir
        print(f"Warning: Custom folders are expected to contain raw images.")
    dataset_obj, _, _ = load_data(dataset=dataset_str, model_type=model_type, split=split, get_loader=False, get_iter=False, pseudo_label_path=pseudo_label_path)
    if res == 512:  # For the reference stats, we need the clean data, before encoding
        ref_dataset_obj, _, _ = load_data(dataset=f"{dataset}-raw", model_type=model_type, split=split, get_loader=False, get_iter=False)  # Has the exact same image ordering as the encoded dataset
    else:
        ref_dataset_obj = dataset_obj

    # Load feature extractors
    f_names = ['fd_dinov2', 'fid']
    f_extractors = [get_detector(metric, verbose=True) for metric in f_names]
    for f in f_extractors:
        f.model.eval()
        for p in f.model.parameters():
            p.requires_grad_(False)

    feat_dims = [f.feature_dim for f in f_extractors]
    num_f_extractors = len(f_extractors)

    # Draw the same random images every time, for all snaps and sigmas, and broadcast to all GPUs
    np.random.seed(0)
    idx = torch.from_numpy(np.random.choice(np.arange(len(dataset_obj)), size=(num_samples,), replace=False)).to(device)  # e.g. 7, 4, 6, 2, ...
    torch.distributed.broadcast(idx, src=0)
    assert idx.max() < len(dataset_obj), f'Index must be smaller than dataset. {idx.max()}, {len(dataset_obj)}'

    # Arange batches of image indices for each GPU
    num_batches = max((num_samples - 1) // (max_batch_size * world_size) + 1, 1) * world_size
    img_batches = idx.tensor_split(num_batches)[rank::world_size]
    raw_batches = torch.arange(num_samples, device=device).tensor_split(num_batches)[rank::world_size]

    # Sigma steps, evenly spaced on a log scale
    start = float(torch.log(torch.tensor([sigma_min])) / torch.log(torch.tensor([10])))  # log(sigma_min) base 10
    end = float(torch.log(torch.tensor([sigma_max])) / torch.log(torch.tensor([10])))  # log(sigma_max) base 10
    t_steps = torch.logspace(start, end, num_steps, device=device)  # [num_steps]

    # Precompute noise for each image, same across snaps and sigmas
    latent_shape = dataset_obj.image_shape  # CHW
    if res == 512:
        latent_shape[0] = 4  # latents in dataset are 8 (mu, std for 4 channels) and get reduced in the loop to 4
    rng = StackedRandomGenerator(device, idx)
    all_noise = rng.randn((num_samples, *latent_shape), device=device)

    # Use the img_batches to tell the dataloader which batches to sample, different for every GPU
    def get_dataset_loader(dataset_obj, img_batches):
        return torch.utils.data.DataLoader(dataset=dataset_obj, batch_sampler=img_batches,
                                           pin_memory=True, num_workers=0)

    def net_forward(net, y, sigma, labels, noise):
        if model_type == 'edm2':
            return net(y + noise * sigma, sigma, labels, return_logvar=True)  # ([bs, C, H, W], logvar)
        else:
            return net(y + noise * sigma, sigma, labels), torch.zeros(1, device=device)  # ([bs, C, H, W], zeros)

    def frechet_distance_precomp_np(mu, sigma, mu_ref, A, trace_sigma_ref):

        m = np.dot(mu - mu_ref, mu - mu_ref)

        M = A @ sigma @ A

        evals = np.linalg.eigvalsh(M)
        evals = np.clip(evals, 0, None)

        return m + np.trace(sigma) + trace_sigma_ref - 2.0 * np.sum(np.sqrt(evals))

    # 1. Enable guidance and use the guidance loop 2. Enable swg and use the swg loop 3. Use the snaps loop
    for snap in snaps:
        for g_weight in guidance_weights:
            for swg_size in swg_sizes:
                torch.distributed.barrier(device_ids=[rank])

                # Create save_file str
                assert not (g_weight != 0 and swg_size is not None), f"Can use either guidance or swg"

                if swg_size is not None:
                    suffix = f'-swg{swg_size}'
                elif g_weight != 0:
                    suffix = f'-w{g_weight:.1f}'
                else:
                    suffix = ''

                if model_type == 'edm2':
                    snap_nr = snap.split('/')[-1].split('-')[-2]
                    save_path = f"{proj_folder}/data/rec_based/in{res}/{model_type}-{model_size}{suffix}/{split}_data"
                elif 'edm' in model_type or ('P_' in model_type) or 'theory' in model_type or 'ours' in model_type:
                    snap_nr = snap.split('/')[-1].split('-')[-1].split('.')[0]
                    save_path = f"{proj_folder}/data/rec_based/{dataset}/{model_type}{suffix}/{split}_data"
                else:
                    raise NotImplementedError(f"Unsupported model type: {model_type}")

                os.makedirs(save_path, exist_ok=True)
                save_file = f"{save_path}/data-{snap_nr}-{ema}.pt"
                dist.print0(f"Saving results to {save_file}")

                if save_preds:
                    outdir = f"{save_path}/samples-{snap_nr}-{ema}/"
                    dist.print0(f"Saving predictions to {outdir}")

                # Check if file exists on rank 0 and broadcast decision -> Avoid asynchronous runs
                file_exists = int(os.path.exists(save_file)) if rank == 0 else 0
                file_exists = torch.tensor(file_exists, device=device)
                torch.distributed.broadcast(file_exists, src=0)

                # if file_exists.item():
                #     dist.print0(f"File {save_file} already exists, continue...")
                #     continue

                if g_weight != 0:
                    assert snap != neg_snap, f"Negative snap needs to differ from positive snap"
                    net_pos, encoder, net_neg = load_models(pos_pkl=snap, neg_pkl=neg_snap)
                else:
                    net_pos, encoder, net_neg = load_models(pos_pkl=snap)

                encoder.batch_size = encoder_batch_size

                # Initialize results tensors, storing running mean and std for each noise level
                all_raw_l2 = torch.zeros(num_steps, 2, dtype=torch.float32, device=device)
                all_feat_l2 = [torch.zeros(num_steps, 2, dtype=torch.float32, device=device) for _ in range(num_f_extractors)]  # Per-extractor feature reconstruction error

                # To compute references of train/val/gen data for FD computations
                ref_path = f"{proj_folder}/data/rec_based/{dataset}/{split}_ref_{num_samples}.pt"
                compute_ref = (rank == 0 and not os.path.isfile(ref_path))
                compute_ref = torch.tensor(int(compute_ref), device=device)
                torch.distributed.broadcast(compute_ref, src=0)
                compute_ref = bool(compute_ref.item())

                # Before the batch loop, load precomputed decoded images if they exist
                feats_path = f"{proj_folder}/data/rec_based/{dataset}/feats_{split}_{num_samples}.pt"

                if not os.path.isfile(feats_path):
                    feats_gt = [torch.zeros(num_samples, f.feature_dim, dtype=torch.float64, device=device) for f in f_extractors]
                    # feats_dec = [torch.zeros(num_samples, f.feature_dim, dtype=torch.float64, device=device) for f in f_extractors]
                    gt_loader = get_dataset_loader(ref_dataset_obj, img_batches)
                    for i, (tensors, raw_idx) in enumerate(tqdm(zip(gt_loader, raw_batches), total=len(raw_batches), disable=rank != 0,
                             ascii=True, desc="Pre-decoding ground truth images and pre-computing ground truth features.")):
                        images, _ = tensors
                        clean = images.to(device)
                        # decoded = encoder.decode(encoder.encode_latents(encoder.encode_pixels(clean.to(device))))  # encode + decode, but this is stochastic on IN512

                        for k, f in enumerate(f_extractors):
                            feats_gt[k][raw_idx] = f(clean).to(torch.float64)  # insert 'decoded' here if you enable the line above
                            # feats_dec[k][raw_idx] = f(decoded).to(torch.float64)

                    # reduce across GPUs
                    for k in range(num_f_extractors):
                        torch.distributed.all_reduce(feats_gt[k], op=torch.distributed.ReduceOp.SUM)
                        # torch.distributed.all_reduce(feats_dec[k], op=torch.distributed.ReduceOp.SUM)

                    if rank == 0:
                        torch.save([f.cpu() for f in feats_gt], feats_path)
                        dist.print0(f"Saved GT features to {feats_path}")
                    torch.distributed.barrier(device_ids=[rank])

                # reload clean copy
                feats_gt = torch.load(feats_path, weights_only=True)
                for k in range(num_f_extractors):
                    feats_gt[k] = feats_gt[k].to(device)

                if rank == 0:
                    fd_scores = [np.zeros((num_steps,), dtype=np.float64) for _ in range(num_f_extractors)]

                # Reference statistics computation
                if compute_ref:
                    # Finalize reference stats (rank 0)
                    if rank == 0:
                        ref_stats = {}
                        for k in range(num_f_extractors):
                            feats = feats_gt[k].cpu().numpy()
                            mu = feats.mean(axis=0)
                            sigma = np.cov(feats, rowvar=False)

                            evals, evecs = np.linalg.eigh(sigma)
                            evals = np.clip(evals, 1e-6, None)

                            A = (evecs * np.sqrt(evals)) @ evecs.T

                            ref_stats[f_names[k]] = dict(
                                mu=mu,
                                sigma=sigma,
                                A=A,
                                trace_sigma=float(evals.sum()),
                            )

                        os.makedirs(os.path.dirname(ref_path), exist_ok=True)
                        torch.save(ref_stats, ref_path)
                        dist.print0(f"Saved FD-ref to {ref_path}")
                else:
                    if rank == 0:
                        ref_stats = torch.load(ref_path, weights_only=False)

                # Ensure everyone waits until ref_stats is ready
                torch.distributed.barrier(device_ids=[rank])

                # Main calculation loop
                if sigma_ids is not None:
                    all_sigma_ids = sigma_ids
                    dist.print0(f"Restricting sigmas to:", [f"{t_steps[s]:.1e}" for s in all_sigma_ids])
                else:
                    all_sigma_ids = range(num_steps)

                pbar = tqdm(all_sigma_ids,
                    total = len(all_sigma_ids),
                    disable = rank != 0,
                    ascii = True,
                    desc = "",
                    )

                for sigma_idx in pbar:
                    sigma = t_steps[sigma_idx].to(device)  # [1,]

                    # Reset per-sigma accumulators (EXACTLY what used to live at index s)
                    raw_sum = torch.zeros([], dtype=torch.float64, device=device)
                    raw_sq_sum = torch.zeros([], dtype=torch.float64, device=device)

                    feat_sum = [torch.zeros([], dtype=torch.float64, device=device) for _ in range(num_f_extractors)]
                    feat_sq_sum = [torch.zeros([], dtype=torch.float64, device=device) for _ in range(num_f_extractors)]

                    cum_mu = [torch.zeros(D, dtype=torch.float64, device=device) for D in feat_dims]
                    cum_sigma = [torch.zeros(D, D, dtype=torch.float64, device=device) for D in feat_dims]

                    dataset_loader = get_dataset_loader(dataset_obj, img_batches)

                    # Batch loop (identical computation, fixed sigma)
                    for (tensors, raw_idx) in zip(dataset_loader, raw_batches):
                        if len(raw_idx) == 0:
                            continue

                        pbar.set_description(f'Sigma: {sigma:.2e}. Encoding images')
                        images, labels = tensors
                        if custom_img_dir:  # custom folders are expected to contain raw images
                            images = encoder.encode_pixels(images.to(device))
                        else:
                            images = images.to(device)
                        y = encoder.encode_latents(images)  # raw latents -> final latents
                        labels = labels.to(device)

                        # Compute reconstruction error
                        pbar.set_description(f'Sigma: {sigma:.2e}. Computing predictions')
                        if swg_size is not None:
                            preds, logvar = SWG(net_forward, net_pos, y, sigma, labels, all_noise[raw_idx], swg_size=swg_size)
                        else:
                            preds = net_forward(net_pos, y, sigma, labels, all_noise[raw_idx])[0]  # no guidance
                            if g_weight != 0:
                                preds_neg = net_forward(net_neg, y, sigma, labels, all_noise[raw_idx])[0]
                                preds = preds + g_weight * (preds - preds_neg)  # Short-checkpoint guidance

                        pbar.set_description(f'Sigma: {sigma:.2e}. Decoding predictions')
                        preds_decoded = encoder.decode(preds)  # reconstructed

                        # Saving predictions
                        if save_preds:
                            pbar.set_description(f'Sigma: {sigma:.2e}. Saving predictions')
                            outdir_sigma = os.path.join(outdir, f"{sigma:.2e}")
                            os.makedirs(outdir_sigma, exist_ok=True)
                            for seed, image in zip(raw_idx, preds_decoded.permute(0, 2, 3, 1).cpu().numpy()):
                                image_dir = os.path.join(outdir_sigma, f'{seed // 1000 * 1000:07d}')
                                os.makedirs(image_dir, exist_ok=True)
                                PIL.Image.fromarray(image, 'RGB').save(os.path.join(image_dir, f'{seed:07d}.png'))
                            torch.distributed.barrier(device_ids=[rank])

                        # Compute the avg. L2 reconstruction error in latent space
                        pbar.set_description(f'Sigma: {sigma:.2e}. Computing L2 error')
                        eps = ((y - preds) ** 2).mean(dim=(1, 2, 3))  # per-sample MSE
                        raw_sum += eps.sum()
                        raw_sq_sum += (eps ** 2).sum()

                        # Compute the average reconstruction error in feature space
                        pbar.set_description(f'Sigma: {sigma:.2e}. Computing feature error')
                        chunk_size = max_batch_size
                        feats_pred = [f(preds_decoded).to(torch.float64) for f in f_extractors]
                        for k in range(num_f_extractors):
                            feat_eps = ((feats_pred[k] - feats_gt[k][raw_idx]) ** 2).mean(dim=1)

                            # Feature L2 error
                            feat_sum[k] += feat_eps.sum()
                            feat_sq_sum[k] += (feat_eps ** 2).sum()

                            # rFD stats
                            cum_mu[k] += feats_pred[k].sum(dim=0)
                            cum_sigma[k] += feats_pred[k].T @ feats_pred[k]

                        pbar.set_description(f'Sigma: {sigma:.2e}. Computing feature-nearest-neighbor')
                        for k in range(num_f_extractors):
                            # Normalize to unit length → cosine similarity = dot product
                            pred = F.normalize(feats_pred[k], dim=1)
                            train = F.normalize(feats_gt[k][raw_idx], dim=1)

                            bs = pred.shape[0]

                            best_sim = torch.full((bs,), -float("inf"), device=device)

                            for start in range(0, train.size(0), chunk_size):
                                end = start + chunk_size
                                train_chunk = train[start:end]  # (chunk, D)

                                sims = pred @ train_chunk.T  # (bs, chunk)
                                best_sim = torch.maximum(best_sim, sims.max(dim=1).values)

                            avg_sim = best_sim.mean()

                    # Reduce across GPUs
                    torch.distributed.reduce(raw_sum, dst=0)
                    torch.distributed.reduce(raw_sq_sum, dst=0)

                    for k in range(num_f_extractors):
                        torch.distributed.reduce(feat_sum[k], dst=0)
                        torch.distributed.reduce(feat_sq_sum[k], dst=0)
                        torch.distributed.reduce(cum_mu[k], dst=0)
                        torch.distributed.reduce(cum_sigma[k], dst=0)

                    if rank == 0:
                        mean = raw_sum / num_samples
                        var = raw_sq_sum / num_samples - mean ** 2
                        std = torch.sqrt(var.clamp(min=0))

                        all_raw_l2[sigma_idx, 0] = mean
                        all_raw_l2[sigma_idx, 1] = std

                        for k in range(num_f_extractors):
                            mean = feat_sum[k] / num_samples
                            var = feat_sq_sum[k] / num_samples - mean ** 2
                            std = torch.sqrt(var.clamp(min=0))

                            all_feat_l2[k][sigma_idx, 0] = mean
                            all_feat_l2[k][sigma_idx, 1] = std

                            # Empirical mean
                            mu = cum_mu[k].cpu() / num_samples  # [D], torch CPU
                            mu_np = mu.numpy()

                            # Empirical covariance
                            sigma = (cum_sigma[k].cpu() - num_samples * torch.outer(mu, mu)) / (num_samples - 1)  # [D, D], torch CPU
                            sigma_np = sigma.numpy()

                            # Reference stats
                            mu_ref = ref_stats[f_names[k]]['mu']
                            A = ref_stats[f_names[k]]['A']
                            trace_ref = ref_stats[f_names[k]]['trace_sigma']

                            fd = frechet_distance_precomp_np(mu_np, sigma_np, mu_ref, A, trace_ref)
                            if not np.isfinite(fd):
                                print("FD computation contained singularities, skipping...")
                            else:
                                fd_scores[k][sigma_idx] = fd # max(fd, 0.0)  # Clamp negative values, if covariances were singular

                # Save results
                done = torch.tensor(0, device=device)
                if rank == 0:
                    # Prepare for saving
                    all_raw_l2 = all_raw_l2.cpu().numpy()
                    all_feat_l2 = [eps.cpu().numpy() for eps in all_feat_l2]

                    # Save data, all numpy arrays
                    data = {'num_steps': num_steps, 'num_samples': num_samples, 't_steps': t_steps.cpu().numpy(),
                            'l2_eps_mean': all_raw_l2[:,0],
                            'l2_eps_std': all_raw_l2[:,1],
                            'dino_eps_mean': all_feat_l2[0][:,0],
                            'dino_eps_std': all_feat_l2[0][:,1],
                            'incp_eps_mean': all_feat_l2[1][:,0],
                            'incp_eps_std': all_feat_l2[1][:,1],
                            'rfdd': fd_scores[0],
                            'rfid': fd_scores[1],
                            'pos_snap': snap,
                            'neg_snap': neg_snap,
                            }

                    torch.save(data, save_file)
                    dist.print0(f'Saved results to {save_file}')
                    done.fill_(1)
                del encoder, net_pos, net_neg
                torch.distributed.broadcast(done, src=0)  # this is crucial, it avoids NCCL timeout because rank 0 does work > 600s
    del dataset_obj


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    dist.init()
    rank, world_size = dist.get_rank(), dist.get_world_size()
    proj_folder = '/home/shared/generative_models/diffusion_overfit_official'  # todo: specify

    # EDM sampler setup
    num_steps, rho = 32, 7
    sigma_min, sigma_max = 2e-3, 80

    datasets = ['in64']  # 'in64', 'in512', 'cifar10', 'cifar100'
    model_sizes = ['xs-uncond', 'xxs', 'xs', 's', 'm', 'l', 'xl', 'xxl']  # 'xs-uncond', 'xxs', 'xs', 's', 'm', 'l', 'xl', 'xxl' and [None] for edm
    guidance_mode = 'none'  # 'none' or 'cfg' or 'autoguidance'

    # snap = "1073741" if res == 64 else "2147483"
    # for dataset in ['cifar10']:
    #     for model_type in [
    #                         # 'edm-uncond',
    #                         # 'edm-kmeans-2',
    #                         # 'edm-kmeans-5',
    #                         # 'edm-kmeans-10',
    #                         # 'edm-kmeans-20',
    #                         # 'edm-kmeans-50',
    #                         # 'edm-kmeans-100',
    #                         # 'edm-kmeans-200',
    #                         # 'edm-kmeans-300',
    #                         # 'edm-kmeans-400',
    #                         'edm-kmeans-500',
    #                       ]:  # edm or edm-kmeans-XX edm2 or edm-P_std_X.X or edm-P_mean_X.X or edm-theory-loss or our-XXX
    for dataset in datasets:  # 'in64', 'in512', 'cifar10', 'cifar100'
        model_type = 'edm2' if 'in' in dataset else 'edm'
        res = int(dataset.replace('in', '')) if 'in' in dataset else 32
        for model_size in model_sizes:
            for val in [False, True]:
                for ema in ['0.100']:  # ['0.020', '0.040', '0.060', '0.080', '0.100', '0.120', '0.140', '0.160', '0.180', '0.200']
                    snaps = load_snaps(which_snaps='all',   # which_snap is an index or 'all'
                                       dataset=dataset,
                                       model_type=model_type,
                                       res=res,
                                       model_size=model_size,
                                       ema=ema)[::-1]  # start with the highest snap

                    if len(snaps) > 32:
                        snaps = snaps[::4]
                    elif len(snaps) > 16:
                        snaps = snaps[::2]

                    if guidance_mode == 'cfg':
                        neg_snap = load_snaps(which_snaps=31,   # which_snap is an index or 'all'
                                              dataset = dataset,
                                              model_type=model_type,
                                              res=res,
                                              model_size='xs-uncond',
                                              ema=ema)[0]
                        g_weights = np.arange(0.2, 2.1, 0.2).tolist()
                    elif guidance_mode == 'autoguidance':
                        neg_snap = load_snaps(which_snaps=31,  # which_snap is an index or 'all'
                                              dataset=dataset,
                                              model_type=model_type,
                                              res=res,
                                              model_size=model_size,
                                              ema=ema)[0]
                        g_weights = np.arange(0.2, 2.1, 0.2).tolist()
                    else:
                        neg_snap = None
                        g_weights = [0.0]

                    generalization_gap(dataset=dataset,
                                       snaps=snaps,
                                       neg_snap=neg_snap,
                                       guidance_weights=g_weights,
                                       swg_sizes=[None],  # [32, 40, 48, 56]
                                       res=res,
                                       model_type=model_type,
                                       model_size=model_size,
                                       split='val' if val else 'train',
                                       num_steps=64,  # Paper: 64
                                       num_samples=50000,  # Paper: 8192
                                       max_batch_size=512 if res != 512 else 256,
                                       encoder_batch_size=256 if res != 512 else 16,
                                       # custom_img_dir=f"{proj_folder}/fd_analysis/{dataset}/{'val' if val else 'train'}/uniform/subsamples_50000_01/samples",  # custom dataset
                                       # pseudo_label_path=f"{proj_folder}/fd_analysis/{dataset}/{'val' if val else 'train'}/uniform/subsamples_50000_01/labels.pt",  # pseudo labels
                                       save_preds=True,
                                       # sigma_ids=[37, 38, 39, 40],  # Restrict to specific sigma_ids
                                       )
                    torch.cuda.empty_cache()
