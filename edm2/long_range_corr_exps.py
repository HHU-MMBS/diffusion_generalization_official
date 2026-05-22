import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch.func import jacrev, vmap
from torch_utils import distributed as dist
from generate_images import StackedRandomGenerator, parse_int_list
from exp_utils import load_data, load_models, load_snaps



@torch.no_grad()
def relative_error(net_pos, net_neg, x, sigma, labels):
    epsilon = 1e-8
    alpha = 1.0

    D_b = net_pos(x, sigma, labels).detach()  # base model
    D_w = net_neg(x, sigma, labels)  # weak model
    range_weight = (D_b - D_w + epsilon).abs() ** alpha
    # range_norm, range_std = range_weight.mean(dim=(1, 2, 3), keepdim=True), range_weight.std(dim=(1, 2, 3))  # Before normalization
    # range_weight /= range_norm  # Normalize
    return range_weight


@torch.no_grad()
def long_range_corr(dataset, res, model_type, device='cuda'):
    snaps = load_snaps(which_snaps='all', dataset=dataset, model_type=model_type, res=res, model_size='s')
    net, encoder, net_neg = load_models(pos_pkl=snaps[-1], neg_pkl=snaps[0])
    dataset_obj, dataset_loader, dataset_iter = load_data(dataset, model_type)

    n_seeds = 128

    images, labels = next(dataset_iter)
    y = encoder.encode_latents(images.to(device))
    labels = labels.to(device)

    # Repeat images and labels n_seeds times
    y = y.repeat_interleave(n_seeds, dim=0)
    labels = labels.repeat_interleave(n_seeds, dim=0)

    for sigma_int in [1e-2, 2.5e-2, 5e-2, 1e-1, 2.5e-1, 5e-1, 1e0, 2.5e0, 5e0, 1e1, 2.5e1, 5e1, 1e2]:
        sigma = sigma_int * torch.ones(y.shape[0], 1, 1, 1, device=device)
        x = y + sigma * torch.randn_like(y)  # [bs, C, H, W]

        range_weight = relative_error(net, net_neg, x, sigma, labels)

        plt.figure(figsize=(8, 4))
        # Average over batch
        weight = range_weight.mean(dim=0)
        img, plot = y.mean(dim=0), weight.permute(1, 2, 0).cpu().numpy()
        plot = plot.mean(axis=-1)  # Average over channels
        plot_min, plot_max, plot_mean = plot.min(), plot.max(), plot.mean()
        plot = (plot - plot_min) / (plot_max - plot_min)  # normalize
        # norm, std = range_norm.mean(dim=0), range_std.mean(dim=0)  # Stats before normalization
        # plt.suptitle(f"Sigma: {sigma[0].item():.1e}. Mean: {norm.item():.1e} , Std: {std.item():.1e}")
        plt.suptitle(f"Sigma: {sigma[0].item():.1e}. Intensity range: [{plot_min.item():.1e}, {plot_mean.item():.1e}]")
        dist.print0(f"Sigma: {sigma[0].item():.1e}. Max: {plot_max.item():.1e}, Mean: {plot_min.item():.1e}")
        plt.subplot(1, 2, 1)
        plt.imshow(img.permute(1, 2, 0).cpu().numpy() / 2 + 0.5)

        plt.subplot(1, 2, 2)
        plt.imshow(plot)
        cbar = plt.colorbar(ticks=np.linspace(0, 1, 5), fraction=0.046, pad=0.04)
        cbar.ax.set_yticklabels(np.round(np.linspace(plot_min, plot_max, 5), 4))  # vertically oriented colorbar

        path = f'/home/tikai103/inductive_bias_diffusion/edm2-long_range_corr_avg-{sigma_int}.png'
        plt.savefig(path, dpi=200, bbox_inches='tight')
        # dist.print0(f'Saved to {path}.')
        plt.close()

        # eps = (y - D_x) ** 2  # [bs, C, H, W], eq. (10)
        # weight = sigma ** -2  # [bs, 1, 1, 1]


def long_range_gradients(split='train', device='cuda'):
    model_size = 's'

    snaps = load_snaps(which_snaps=-1, dataset=dataset, model_type=model_type, res=res, model_size=model_size)
    net, encoder, net_neg = load_models(pos_pkl=snaps[0])
    dataset_obj, dataset_loader, dataset_iter = load_data(dataset, model_type, split=split, batch_gpu=1024)

    # Enable gradients
    net.requires_grad_(True)

    images, labels = next(dataset_iter)
    latents = encoder.encode_latents(images.to(device))
    labels = labels.to(device)

    # Select a few images
    img_idx_str = '0-31'
    img_idx = parse_int_list(img_idx_str)
    images = images[img_idx]
    latents = latents[img_idx]
    labels = labels[img_idx]

    latents.requires_grad = True
    bs, C, H, W = latents.shape[0], latents.shape[1], latents.shape[2], latents.shape[3]

    # # Sigma steps, evenly spaced on a log scale
    # num_steps = 32
    # start = float(torch.log(torch.tensor([sigma_min])) / torch.log(torch.tensor([10])))  # log(sigma_min) base 10
    # end = float(torch.log(torch.tensor([sigma_max])) / torch.log(torch.tensor([10])))  # log(sigma_max) base 10
    # sigmas = torch.logspace(start, end, num_steps, device=device)
    # all_sigmas = torch.ones(1, device=device) * 1
    all_sigmas = torch.logspace(start=torch.log10(torch.tensor(2e-3)), end=torch.log10(torch.tensor(80.)), steps=16,
                                device=device)
    print("sigmas: ", all_sigmas)
    rnd = StackedRandomGenerator(device, torch.arange(len(latents)))  # The same noise for the same images

    # corr_score = np.zeros((len(sigmas), images.shape[0]))  # [num_steps, bs]
    # rand_pixels = [(np.random.randint(H), np.random.randint(H)) for _ in range(bs)]  # Random pixel for each image, same for all sigmas
    # rand_pixels = [(H // 2, H // 2) for _ in range(bs)]  # center pixels
    for i, sigmas in tqdm(enumerate(all_sigmas), total=len(all_sigmas), disable=False):
        sigmas = sigmas * torch.ones(bs, 1, 1, 1, device=device)
        noises = rnd.randn_like(latents)
        # D_x = net(latents + sigmas * noise, sigmas, labels)  # [bs, C, H, W]
        for b in tqdm(range(bs), disable=True):
            # corr_score = np.zeros((C, H, H))
            # nx, ny = np.meshgrid(np.arange(H), np.arange(H), indexing='ij')
            #
            # # >4 mins per image
            # # Compute gradients w.r.t input pixels
            # # pixels = [(H // 2, H // 2)]  # Only center pixel
            # # pixels = [(H // 3, H // 3), (H // 3, 2 * H // 3), (2 * H // 3, H // 3), (2 * H // 3, 2 * H // 3), (H // 2, H // 2)]  # Square and center
            # # D_x[j, 0, rand_pixels[j][0], rand_pixels[j][1]].backward(retain_graph=True, inputs=latents)  # 0-th channel, random pixel
            # pixels = [(x, y) for x in range(H) for y in range(H)]  # all pixels
            # for pixel in tqdm(pixels):
            #     distance_matrix = np.sqrt((nx - pixel[0]) ** 2 + (ny - pixel[1]) ** 2) # [H, H], matrix that encodes the L2 distance from current pixel
            #     for c in range(C):
            #         latents.grad = None
            #         D_x[b, c, pixel[0], pixel[1]].backward(retain_graph=True, inputs=latents)  # 0-th channel, gradients w.r.t pixel [H, H]
            #         gradient = latents.grad[b].abs().cpu().numpy()  # j-th image, 0-th channel, [C, H, H]
            #         corr_score[:, pixel[0], pixel[1]] = (distance_matrix * gradient).mean(axis=(1,2)) / gradient.mean(axis=(1,2))  # [C, H, H] * [H, H] -> [C]

            # # >2 mins per image
            # for (i, j) in tqdm([(x, y) for x in range(H) for y in range(H)]):
            #     grad_outputs = torch.zeros_like(D_x)
            #     grad_outputs[:, :, i, j] = 1  # Select one pixel
            #     gradient = torch.autograd.grad(outputs=D_x, inputs=latents, grad_outputs=grad_outputs, retain_graph=True)[0].squeeze(0)  # [C, H, W]
            #     gradient = gradient.abs().cpu().numpy()
            #     distance_matrix = np.sqrt((nx - i) ** 2 + (ny - j) ** 2)  # [H, W], matrix that encodes the L2 distance from current pixel
            #     corr_score[:, i, j] = (distance_matrix * gradient).mean(axis=(1, 2)) / gradient.mean(axis=(1, 2))  # [C, H, W] -> [bs, C]

            # VMAP approach, < 30s per image
            def compute_corr_score(latent, sigma, label, noise):
                nx, ny = torch.meshgrid(torch.arange(H), torch.arange(H), indexing="ij")
                nx, ny = nx.to(latent.device), ny.to(latent.device)

                def single_output_pixel(latent, sigma, label, noise, h_idx, w_idx):
                    latent = latent.unsqueeze(0)
                    sigma = sigma.unsqueeze(0)
                    label = label.unsqueeze(0)
                    D_x = net(latent + sigma * noise, sigma, label).squeeze(0)  # [C, H, W]
                    return D_x[:, h_idx.unsqueeze(-1), w_idx.unsqueeze(-1)].squeeze(-1)  # [C]

                def per_pixel_jac(h_idx, w_idx):
                    grad = jacrev(single_output_pixel, argnums=0)(latent, sigma, label, noise, h_idx,
                                                                  w_idx).abs()  # [C, C, H, W]
                    grad = torch.diagonal(grad, dim1=0, dim2=1).permute(2, 0,
                                                                        1)  # [C, H, W], Consider the channels separately

                    with torch.no_grad():
                        distance_matrix = torch.sqrt((nx - h_idx) ** 2 + (ny - w_idx) ** 2).to(grad.device)  # [H, H]

                    return (distance_matrix * grad).mean(dim=(1, 2)) / grad.mean(dim=(1, 2))  # [C, C]

                # Vectorizing over all pixels to avoid looping. Organize into chunks to trade memory for speed
                pixel_indices = torch.arange(H, device=latent.device)
                chunk_size = 16
                corr_score_chunks = []
                for i in range(0, H, chunk_size):
                    for j in range(0, H, chunk_size):
                        h_chunk = pixel_indices[i: i + chunk_size]
                        w_chunk = pixel_indices[j: j + chunk_size]

                        with torch.no_grad():  # Save some memory
                            corr_score_chunk = vmap(vmap(per_pixel_jac, in_dims=(None, 0)), in_dims=(0, None))(h_chunk, w_chunk)  # [chunk_size, chunk_size, C]

                        corr_score_chunks.append(corr_score_chunk)

                return torch.stack(corr_score_chunks)  # [num_chunks, chunk_size, chunk_size, C]

            # Compute vectorized
            corr_score_chunks = vmap(compute_corr_score)(latents[b:b + 1],
                                                         sigmas[b:b + 1],
                                                         labels[b:b + 1],
                                                         noises[b:b + 1]).cpu()  # [bs, num_chunks, chunk_size, chunk_size, C]
            corr_score = torch.zeros(C, H, W)

            # Rearrange the chunks back together
            chunk_size = corr_score_chunks[0].shape[1]
            counter = 0
            for i in range(0, H, chunk_size):
                for j in range(0, H, chunk_size):
                    chunk = corr_score_chunks[0, counter]
                    corr_score[:, i: i + chunk.shape[0], j: j + chunk.shape[1]] = chunk.permute(2, 0, 1)
                    counter += 1

            corr_score = corr_score.numpy()

            # # Plot gradients
            # # grad = y.grad[0] / (len(pixels) * y.shape[1]) # Average over pixels and channels
            # grad = y.grad[0,0]
            # plt.figure(figsize=(8, 4))
            # plt.suptitle(f"Sigma: {sigma[0].item():.1e}")
            # plt.subplot(1, 2, 1)
            # plt.imshow(images[img_idx].permute(1, 2, 0).cpu().numpy())
            # plt.subplot(1, 2, 2)
            # plt.imshow(grad.abs().cpu().numpy())
            # plt.colorbar(fraction=0.046, pad=0.04)
            # plt.savefig(f'/home/tikai103/diffusion_overfit/plotsgradients/edm2-long_range_gradients_{sigma[0].item():.1e}.png', dpi=200, bbox_inches='tight')
            # plt.close()

            # Save results
            assert corr_score.shape == (C, H, W), f"Shape mismatch. {corr_score.shape}, must be {(C, H, W)}"
            save_path = (f"{proj_folder}/data/gradients/in{res}/edm2-{model_size}/"
                         f"corr_scores_{split}_{img_idx[b]}_s{sigmas[b].item():.1e}.pt")
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            torch.save((corr_score, images[b].cpu().numpy(), sigmas[b].cpu().numpy()), save_path)
            # dist.print0(f'Saved to {save_path}')


@torch.no_grad()
def get_attention_map(dataset, res, model_type, model_size, device='cuda'):
    assert world_size == 1, "Only single GPU supported for now."
    from training.networks_edm2 import mp_sum, mp_silu, mp_cat, resample, normalize

    snaps = load_snaps(which_snaps='last', dataset=dataset, model_type=model_type, res=res, model_size=model_size)
    net, encoder, _ = load_models(pos_pkl=snaps[-1])
    dataset_obj, dataset_loader, dataset_iter = load_data(dataset, model_type)

    def conv_forward(conv, x, gain=1):
        w = conv.weight.to(torch.float32)
        if conv.training:
            with torch.no_grad():
                conv.weight.copy_(normalize(w))  # forced weight normalization
        w = normalize(w)  # traditional weight normalization
        w = w * (gain / np.sqrt(w[0].numel()))  # magnitude-preserving scaling
        w = w.to(x.dtype)
        if w.ndim == 2:
            return x @ w.t()
        assert w.ndim == 4
        return torch.nn.functional.conv2d(x, w, padding=(w.shape[-1] // 2,))

    def block_forward(block, x, emb):
        """Copy of forward of Block class in networks_edm2.py"""
        # Main branch.
        x = resample(x, f=block.resample_filter, mode=block.resample_mode)
        if block.flavor == 'enc':
            if block.conv_skip is not None:
                x = block.conv_skip(x)
            x = normalize(x, dim=1)  # pixel norm

        # Residual branch.
        y = block.conv_res0(mp_silu(x))
        c = block.emb_linear(emb, gain=block.emb_gain) + 1
        y = mp_silu(y * c.unsqueeze(2).unsqueeze(3).to(y.dtype))
        if block.training and block.dropout != 0:
            y = torch.nn.functional.dropout(y, p=block.dropout)
        y = block.conv_res1(y)

        # Connect the branches.
        if block.flavor == 'dec' and block.conv_skip is not None:
            x = block.conv_skip(x)
        x = mp_sum(x, y, t=block.res_balance)

        # Self-attention.
        if block.num_heads != 0:
            y = block.attn_qkv(x)
            y = y.reshape(y.shape[0], block.num_heads, -1, 3, y.shape[2] * y.shape[3])
            q, k, v = normalize(y, dim=2).unbind(3)  # pixel norm & split
            w = torch.einsum('nhcq,nhck->nhqk', q, k / np.sqrt(q.shape[2])).softmax(dim=3)  # qk map
            y = torch.einsum('nhqk,nhck->nhcq', w, v)
            y = block.attn_proj(y.reshape(*x.shape))
            x = mp_sum(x, y, t=block.attn_balance)
        else:
            w = None

        # Clip activations.
        if block.clip_act is not None:
            x = x.clip_(-block.clip_act, block.clip_act)
        return x, w

    def unet_foward(unet, x, sigma, class_labels):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        class_labels = None if net.label_dim == 0 else torch.zeros([1, net.label_dim],
                                                                   device=x.device) if class_labels is None else class_labels.to(
            torch.float32).reshape(-1, net.label_dim)
        dtype = torch.float32

        noise_labels = sigma.flatten().log() / 4
        c_in = 1 / (sigma_data ** 2 + sigma ** 2).sqrt()
        x = (c_in * x).to(dtype)

        # Embedding.
        emb = unet.emb_noise(unet.emb_fourier(noise_labels))
        if unet.emb_label is not None:
            emb = mp_sum(emb, unet.emb_label(class_labels * np.sqrt(class_labels.shape[1])), t=unet.label_balance)
        emb = mp_silu(emb)

        # Encoder.
        x = torch.cat([x, torch.ones_like(x[:, :1])], dim=1)
        skips = []
        qk_maps = []
        for name, block in unet.enc.items():
            x, qk_map = (conv_forward(block, x), None) if 'conv' in name else block_forward(block, x, emb)
            skips.append(x)
            if qk_map is not None:
                qk_maps.append((f"enc-{name.replace('block', '')}", qk_map.cpu()))

        # Decoder.
        for name, block in unet.dec.items():
            if 'block' in name:
                x = mp_cat(x, skips.pop(), t=unet.concat_balance)
            x, qk_map = block_forward(block, x, emb)
            if qk_map is not None:
                qk_maps.append((f"dec-{name.replace('block', '')}", qk_map.cpu()))
        x = unet.out_conv(x, gain=unet.out_gain)

        return x, qk_maps

    images, labels = next(dataset_iter)
    y = encoder.encode_latents(images.to(device))
    labels = labels.to(device)

    # Sigma steps, evenly spaced on a log scale
    num_steps = 10
    start = float(torch.log(torch.tensor([sigma_min])) / torch.log(torch.tensor([10])))  # log(sigma_min) base 10
    end = float(torch.log(torch.tensor([sigma_max])) / torch.log(torch.tensor([10])))  # log(sigma_max) base 10
    t_steps = torch.logspace(start, end, num_steps, device=device)  # [num_steps]

    rnd = StackedRandomGenerator(device, torch.arange(len(images)))
    all_qk_maps = {}
    for sigma in tqdm(t_steps):
        sigma = sigma[None, None, None, None].to(device)
        x = y + rnd.randn_like(y) * sigma
        _, qk_maps = unet_foward(net.unet, x, sigma, labels)
        all_qk_maps[sigma[0].item()] = qk_maps

    save_path = f"{proj_folder}/data/loss_analysis/in{res}/edm2-{model_size}/qk-maps.pt"
    torch.save(all_qk_maps, save_path)
    dist.print0(f'Saved to {save_path}')



if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    dist.init()
    rank, world_size = dist.get_rank(), dist.get_world_size()
    sigma_data = 0.5
    proj_folder = '/home/shared/generative_models/diffusion_overfit'

    # EDM sampler setup
    num_steps, rho = 32, 7
    sigma_min, sigma_max = 2e-3, 80

    model_type = 'edm2'  # edm or edm2 or edm-P_std_X.X or edm-P_mean_X.X or edm-theory-loss or our-XXX
    dataset = 'in512'  # cifar10 or cifar100 or in64 or in512
    res = int(dataset.replace('in', '')) if 'in' in dataset else 32

    long_range_corr(dataset, res, model_type)

    # for val in [True, False]:
    #     long_range_gradients(val)

    # for res in [64, 512]:
    #     model_sizes = ['s', 'xl'] if res == 64 else ['xs', 'xxl']
    #     for model_size in model_sizes:
    #         get_attention_map(dataset, res, model_type, model_size=model_size)
