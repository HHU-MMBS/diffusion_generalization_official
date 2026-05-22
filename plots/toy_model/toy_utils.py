import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from scipy.linalg import sqrtm
from tqdm import tqdm
from generalization_gap_utils import multiline, plot_layout
from plots.toy_model.toy_math import ToyModel


def plot_traject(trajec_list, t_steps, alpha=None, scale=1, width=0.005):
    n = len(t_steps) - 2

    # Linear steps from a threshold onwards, from 1 to 0
    color_steps = np.clip(np.arange(n, 0, -1), 0, n // 2)
    c_min, c_max = color_steps.min(), color_steps.max()
    color_steps = (color_steps - c_min) / (c_max - c_min)

    # Linear steps from a threshold onwards, from 0 to 1
    alpha_steps = np.clip(np.arange(n), n // 2, n)
    c_min, c_max = alpha_steps.min(), alpha_steps.max()
    alpha_steps = (alpha_steps - c_min) / (c_max - c_min)
    for c, trajec in enumerate(trajec_list):
        plt.quiver(trajec[:-1, 0], trajec[:-1, 1],
                   trajec[1:, 0] - trajec[:-1, 0],
                   trajec[1:, 1] - trajec[:-1, 1],
                   color_steps,
                   cmap='viridis',
                   scale_units='xy',
                   angles='xy',
                   scale=scale,
                   width=width,
                   alpha=alpha_steps if alpha is None else alpha,
                   headlength=0,
                   headaxislength=0)


def plot_endpoints(data, trajects):
    for c, trajec in enumerate(trajects):
        dist = np.power(trajec[-1] - data, 2).sum(axis=1)
        plt.scatter(trajec[-1, 0], trajec[-1, 1], s=5, c=dist.min() * 10)


def setup_plot(window_size, data, data_split, sigma=None, s=10, grid=True):
    """Limit the plot to a square window of size [-window_size, window_size]"""
    for split, color, lbl, marker in zip([data[data_split], data[~data_split]], ['orangered', 'firebrick'], ['training point', 'val'], ['+', 'x']):  # firebrick
        if len(split) == 0:
            continue
        plt.scatter(split[:, 0], [split[:, 1]], label=lbl, marker=marker, s=s, color=color, zorder=3)

        # if lbl == 'val':
        #     # Plot circles for each data point
        #     for point in split:
        #         circle = plt.Circle(
        #             (point[0], point[1]),  # Center at the data point
        #             radius=sigma,  # Define the radius of the circle
        #             color=color,  # Same color as the scatter point
        #             alpha=0.5,  # Set transparency
        #             fill=False,  # Make it an outline
        #             # linestyle=(0, (5, 5))  # Make the circle dashed (dash pattern style)
        #         )
        #         # Add the circle to the current axes via plt.gca
        #         plt.gca().add_artist(circle)

    plt.xlim(-window_size, window_size)
    plt.ylim([-window_size, window_size])
    if grid:
        plt.grid(alpha=0.2)
    plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)


def plot_row(n, m, i, trajects, preds, ylabel, titles, window_size, data, data_split, t_steps):
    """
    Args:
        n: Number of trajectories
        m: Number of trajectories to plot
        i: Row index
        trajects: Trajectories x(t), [n, n_steps, 2]
        preds: Predictions y(t), [n, n_steps, 2]
        t_steps: Noise levels
    """
    rows = 4
    columns = 3

    # Trajectories
    plt.subplot(rows, columns, columns * i + 1)
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=14)
    if titles:
        plt.title('$x(t)$')

    plot_traject(trajects[::n // m], t_steps)
    setup_plot(window_size, data, data_split, s=window_size * 4)

    # Predictions
    plt.subplot(rows, columns, columns * i + 2)
    plot_traject(preds[::n // m], t_steps)
    setup_plot(window_size, data, data_split, s=window_size * 4)
    if titles:
        plt.title('$y(t)$')

    # Endpoints
    plt.subplot(rows, columns, columns * i + 3)
    plot_endpoints(data, trajects)
    setup_plot(window_size, data, data_split, s=window_size * 4)
    if titles:
        plt.title('$x(0)$')

    total_error = (((trajects[None, :, -1] - data[:, None]) ** 2).min(axis=0).sum(axis=-1) ** 0.5).mean()
    plt.xlabel(f"{total_error:.1e}")

    return total_error


def simulate_trajectories(model, x_labels, n, m, init, delta, data, data_split, window_size, full_row=False, save_as=None):
    """
    Args:
        model: ToyModel instance to use for sampling
        x_labels: Array of labels for each trajectory, shape (n,)
        n: Number of trajectories
        m: Number of trajectories to plot
        interval: Interval where the guidance weight is used
        save_as: Path to save the plot in
    """

    total_error = []

    if full_row:
        plt.figure(figsize=(9, 12), facecolor='white')
        plt.subplots_adjust(wspace=0, hspace=0)
    else:
        plt.figure(figsize=(4,4), facecolor='white')

    model_kwargs = {'x_labels': x_labels, 'n': n, 'init': init}
    trajects, preds = model.run(**model_kwargs)

    if full_row:
        counter = 0  # Row counter
        plot_labels = [f'$\\epsilon_{{{delta}}}(x, t)$',]
        for plot_label in plot_labels:
            total_error.append(plot_row(n=n,              # Number of trajectories
                                        m=m,              # Number of trajectories to plot
                                        i=counter,        # Row index
                                        trajects=trajects,
                                        preds=preds,
                                        ylabel=plot_label,          # ylabels
                                        titles=True,
                                        window_size=window_size,    # Window size
                                        data=data,     # Data
                                        data_split=data_split,
                                        t_steps=model.t_steps)) # Noise levels
            counter += 1
    else:
        # Trajectories
        plot_traject(trajects[::n // m], model.t_steps)
        setup_plot(window_size, data, data_split, s=window_size * 20)
        plt.legend(loc=1)

    if save_as is not None:
        plt.savefig(save_as, dpi=150, bbox_inches='tight')
    plt.show()


def l2_metric(model, delta, data, data_split, t_steps, guid_weight=0.0, delta_neg=None, samples_per_point=100, x_labels=None, cond=False, window_size=2.5,
              plot=True, plot_points=100, plot_samples_per_point=50, plot_sigmas=None, plot_counter=1, n_rows=1,
              mode=None, contours=False, threshold=True, scale=1, save_folder=None):

    relative_error = np.zeros(len(t_steps))
    np.random.seed(0)
    noise = np.random.randn(samples_per_point, *data.shape)  # same noise across sigmas, averaged over multiple samples per data point
    first_plot = True  # first plot get a ylabel
    for i, sigma in enumerate(tqdm(t_steps)):
        assert sigma != 0, "Sigma cannot be zero"
        # Noise data with
        noised = data + noise * sigma  # (N,D) + (M,N,D) * (1,) = (M,N,D)
        if plot:
            plot_noised = noised[:plot_samples_per_point, :plot_points].reshape(-1, data.shape[1])
        noised = noised.reshape(-1, data.shape[1])  # (N*M, D)

        # Denoise with model
        if x_labels is not None:
            x_labels = np.array(list(x_labels) * samples_per_point)  #
        denoised = model.y_delta(x=noised,
                                 t=sigma,
                                 x_labels=x_labels,  # (N*M)
                                 delta=delta,
                                 cond=cond)  # (N*M,D)

        if guid_weight:
            denoised_neg = model.y_delta(x=noised,
                                         t=sigma,
                                         x_labels=x_labels,  # (N*M)
                                         delta=delta_neg,
                                         cond=cond)  # (N*M,D)
            denoised = denoised + guid_weight * (denoised - denoised_neg)

        denoised = denoised.reshape(samples_per_point, -1, data.shape[1])  # (M,N,D)
        if (~data_split).sum() > 0:
            pred_err = np.sum((denoised - data)**2, axis=2) ** 0.5  # (M,N,)
            pred_err = pred_err.mean(axis=0)  # (N,)
            train_err = pred_err[data_split].mean()  # 1: train
            val_err = pred_err[~data_split].mean()  # 0: val
            relative_error[i] = (val_err - train_err) / train_err
        else:
            relative_error = None

        if plot and sigma in plot_sigmas:
            plot_denoised = denoised.reshape(samples_per_point, -1, data.shape[1])[:plot_samples_per_point, :plot_points].reshape(-1, data.shape[1])

            if mode == 'error':
                draw_error_field(model, sigma, delta, scale=scale)
            else:
                if save_folder is not None:
                    plt.figure(figsize=(4, 4))
                else:
                    plt.subplot(n_rows, len(plot_sigmas), plot_counter)
                    if first_plot:
                        plt.legend(loc=1)
                        plt.xlabel(f"$\sigma = {sigma:.1e}$")
                        plt.ylabel(f"$\delta$ = {delta:.1f}")
                        first_plot = False
                    else:
                        plt.xlabel(f"$\sigma = {sigma:.1e}$")

            if mode == 'displacement':
                setup_plot(window_size=window_size, data=data, data_split=data_split, sigma=sigma, s=30, grid=False)
                draw_displacement_region(model, sigma, delta, window_size, contours=contours, threshold=threshold)

            if mode == 'flow':
                setup_plot(window_size=window_size, data=data, data_split=data_split, sigma=sigma, s=50, grid=False)
                draw_flow_field(model, sigma, delta, window_size=window_size)

            if plot_points > 0 and plot_samples_per_point > 0 and mode != 'error':
                # for arr, lbl, color in zip([plot_noised, plot_denoised], ['noised', 'denoised'], ['green', 'blue']):
                for arr, lbl, color in zip([plot_denoised], ['target pred.'], ['blue']):
                    plt.scatter(arr[:, 0], [arr[:, 1]], label=lbl, marker='o', s=2, color=color, alpha=0.5, zorder=0)

            if save_folder is not None:
                plt.legend(loc=1)
                os.makedirs(save_folder, exist_ok=True)
                plt.savefig(os.path.join(save_folder, f"{mode}_d{delta:.1e}_s{sigma:.1e}.png"), dpi=200, bbox_inches='tight')
                plt.close()

            plot_counter += 1

    return relative_error, plot_counter


def draw_flow_field(model, sigma, delta, window_size=2.5, grid_points=50,
                    arrow_length_scale=1.5, arrow_alpha_low=0.6, arrow_alpha_high=1.0,
                    show_bg=True, cmap='viridis_r'):
    """
    Draw the flow field of a model's predictions, optimized for visibility of direction
    even at low magnitudes.

    Args:
        model: The model to evaluate (must have a `y_delta` method).
        sigma: Noise level.
        delta: Parameter for the model's `y_delta`.
        window_size: Square window for the plot. Grid from -window_size to +window_size.
        grid_points: Number of points in each axis.
        arrow_length_scale: Multiplier for arrow length (magnitude-dependent).
        arrow_alpha_low: Minimum alpha for arrows (low-magnitude visibility).
        arrow_alpha_high: Maximum alpha for arrows (high-magnitude visibility).
        show_bg: If True, overlay a background magnitude heatmap.
        cmap: Colormap for arrows.
    """

    # Grid
    xs = np.linspace(-window_size, window_size, grid_points)
    ys = np.linspace(-window_size, window_size, grid_points)
    X, Y = np.meshgrid(xs, ys)
    clean_grid = np.stack([X, Y], axis=-1)
    clean_flat = clean_grid.reshape(-1, 2)  # (grid_points^2, 2)

    # Model prediction
    pred_flat = model.y_delta(clean_flat, sigma, x_labels=None, delta=delta, cond=False)  # (grid_points^2, 2)
    dists_to_y = np.sum((pred_flat[:, None] - model.data[None])**2, axis=2)  # (grid_points^2, N)
    pred_error = np.min(dists_to_y, axis=1).reshape(grid_points, grid_points)  # (grid_points^2,)
    displacements = pred_flat - clean_flat
    magnitudes = np.linalg.norm(displacements, axis=1)

    # Reshape for plotting
    M = magnitudes.reshape(grid_points, grid_points)

    # 1. Arrow length scaling (keep some proportionality to magnitude)
    norm_disp = displacements / (magnitudes[:, None] + 1e-8)  # Normalize but keep a small magnitude-dependent factor
    scaled_disp = norm_disp * (arrow_length_scale * np.sqrt(magnitudes)[:, None])  # sqrt makes low-mags more visible
    # max_length = 0.8
    # scaled_disp = np.clip(scaled_disp, -max_length, max_length)

    # # --- Arrow length soft-capping ---
    # m_dead = np.percentile(magnitudes, 0)  # defines white region size
    # m0 = np.percentile(magnitudes, 100)  # saturation onset
    # Lmax = arrow_length_scale
    # lengths = np.zeros_like(magnitudes)
    #
    # mask = magnitudes > m_dead
    # m_eff = magnitudes[mask] - m_dead
    #
    # lengths[mask] = Lmax * m_eff / (m_eff + m0)
    #
    # norm_disp = displacements / (magnitudes[:, None] + 1e-8)
    # scaled_disp = norm_disp * lengths[:, None]

    U = scaled_disp[:, 0].reshape(grid_points, grid_points)
    V = scaled_disp[:, 1].reshape(grid_points, grid_points)

    # 2. Alpha by magnitude (low-magnitude arrows stand out more)
    alpha = arrow_alpha_low + (arrow_alpha_high - arrow_alpha_low) * (M - M.min()) / (M.max() - M.min() + 1e-8)

    # 3. Optional magnitude background
    if show_bg:
        plt.imshow(M, extent=(-window_size, window_size, -window_size, window_size),
                   origin='lower', cmap=cmap, alpha=0.2, zorder=0)  # cmap='Greys'
        # plt.imshow(pred_error, extent=(-window_size, window_size, -window_size, window_size),
        #            origin='lower', cmap=cmap, alpha=0.2, zorder=0)  # cmap='Greys'

    # Quiver plot
    # width = 0.003 + 0.006 * (M / (M.max() + 1e-8))
    # q = plt.quiver(X, Y, U, V, M,
    #                cmap=cmap,
    #                pivot='middle',
    #                alpha=alpha,
    #                scale_units='xy',
    #                scale=2,          # Keep length scaling explicit
    #                width=0.003,        # Thicker arrows
    #                # width=width[::step, ::step].reshape(-1),
    #                headwidth=4,
    #                headlength=6,
    #                headaxislength=4,
    #                )

    # High-density streamplot
    strm = plt.streamplot(
        X, Y, U, V,
        density=1.2,
        color='black',  # 'M' for magnitude as color
        linewidth=0.5,  # VERY thin
        arrowsize=0.75,  # tiny arrowheads
        arrowstyle='->',  # sharp, minimal
        minlength=0.05,
        maxlength=1.0,
        zorder=2,
        cmap=cmap,
    )

    # CURRENT PAPER VERSION:
    # # High-density streamplot
    # strm = plt.streamplot(
    #     X, Y, U, V,
    #     density=1.2,
    #     color=M,  # 'M' for magnitude as color
    #     linewidth=0.5,  # VERY thin
    #     arrowsize=0.75,  # tiny arrowheads
    #     arrowstyle='->',  # sharp, minimal
    #     minlength=0.05,
    #     maxlength=1.0,
    #     zorder=2,
    #     cmap=cmap,
    # )

    # Colorbar
    # plt.colorbar(q, label='Magnitude of change')
    # plt.axis('equal')


def draw_error_field(model, sigma, delta, scale, manifold_points=200, n_points=500):
    """

    """

    # Set of points along the manifold
    u = np.linspace(0, 1, manifold_points, endpoint=False)  # uniform in [0, 1]
    theta = np.pi * u  # angle
    clean_points = np.stack([np.cos(theta), np.sin(theta)], axis=1) * scale  # (M, 2)

    def expected_error(clean_points):
        """Noise each sample n_points times and compute average error."""
        # Shared noise samples (same for all grid points)
        np.random.seed(1)
        eps = np.random.randn(n_points, 2)  # (K, 2)

        # Construct noisy inputs
        noisy = clean_points[:, None, :] + sigma * eps[None, :, :]
        noisy_flat = noisy.reshape(-1, 2)  # shape: (G, K, 2) -> (G*K, 2)

        # Denoise in one forward pass
        pred_flat = model.y_delta(
            noisy_flat,
            sigma,
            x_labels=None,
            delta=delta,
            cond=False,
        )

        # Compute squared L2 error to clean points
        pred = pred_flat.reshape(clean_points.shape[0], n_points, 2)
        clean = clean_points[:, None, :]  # (G, 1, 2)
        sq_error = np.sum((pred - clean) ** 2, axis=-1) ** 0.5  # (G, K)
        return sq_error.mean(axis=1)  # (G,)

    test_error = expected_error(clean_points)
    train_error = expected_error(model.data).mean()  # all training points
    rel_error = (test_error - train_error) / train_error

    # Plot relative gen gap vs angle
    plt.plot(theta, rel_error, label=f"$\\delta = {delta}$")
    plt.xlabel("Angle on manifold")
    plt.xticks(np.linspace(0, 2 * np.pi, 5, endpoint=True),
               ['0', '$1/2\pi$', '$\pi$', '$3/2\pi$', '$2\pi$'])
    plt.xlim([0, np.pi])
    plt.grid(alpha=0.4)


def draw_displacement_region(model, sigma, delta, window_size=2.5, n_points=200, contours=True, threshold=False):
    x = np.linspace(-window_size, window_size, n_points)
    y = np.linspace(-window_size, window_size, n_points)
    X, Y = np.meshgrid(x, y)
    grid = np.stack([X.ravel(), Y.ravel()], axis=1)

    y_delta = model.y_delta(grid, sigma, x_labels=None, delta=delta, cond=False)
    # y_star = y_delta * (sigma**2 + delta**2) / sigma**2
    Z = np.linalg.norm(y_delta - grid, axis=1).reshape(n_points, n_points)
    threshold = sigma

    if threshold:
        # Shade "identity-like" region
        plt.contourf(
            X, Y, Z,
            levels=[0, threshold],
            colors=['#66c2a5'],
            alpha=0.35
        )

        # Draw boundary explicitly
        plt.contour(
            X, Y, Z,
            levels=[threshold],
            colors='k',
            linewidths=1.5
        )

    if contours:
        plt.contourf(
            X, Y, Z,
            levels=30,
            cmap='viridis',
            alpha=0.15
        )


def plot_generalization_gap(t_steps, all_errors, color_range, idx=None, cbar_ticks=None, x_formatter=None, ylim=[-0.05, 0.16], cmap='viridis_r'):
    """
    Assume the figure is setup outside this function.

    Args:
        t_steps: Noise levels
        all_errors: Array of relative errors, shape (N_models, len(t_steps))
        idx: Select a subset of data to plot
    """

    line_collection = multiline(
        xs=np.array([t_steps] * len(all_errors)),
        ys=all_errors,
        c=color_range,
        cmap=cmap,
        alpha=1,
        idx=idx
    )

    # Plot layout and configuration
    plot_layout(t_steps[0], t_steps[-1], scale_y=False, legend=False)
    plt.xlabel('$\sigma$')
    if x_formatter is not None:
        color_bar = plt.colorbar(line_collection, format=ticker.FuncFormatter(x_formatter))
    else:
        color_bar = plt.colorbar(line_collection)

    # Customize color bar ticks if provided
    if cbar_ticks is not None:
        color_bar.set_ticks(cbar_ticks)

    plt.ylim(ylim)


def frechet_stats(array):
    """Compute mean and covariance matrix of array."""
    mu = np.mean(array, axis=0)
    cov = np.cov(array, rowvar=False)
    return mu, cov


def frechet_distance(mu_gen, cov_gen, mu_ref, cov_ref, eps=1e-6):
    """
    Compute Fréchet Distance between two point clouds.

    Args:
        X_gen: np.ndarray of shape [N, D]  (generated samples)
        X_ref: np.ndarray of shape [M, D]  (reference samples, train or val)
        eps: small jitter for numerical stability

    Returns:
        fd: scalar Fréchet distance
    """
    # Stabilize
    cov_gen += eps * np.eye(cov_gen.shape[0])
    cov_ref += eps * np.eye(cov_ref.shape[0])

    # Mean term
    mean_diff = mu_gen - mu_ref
    mean_term = mean_diff @ mean_diff

    # Covariance term
    covmean = sqrtm(cov_gen @ cov_ref)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    cov_term = np.trace(cov_gen + cov_ref - 2 * covmean)

    return mean_term, cov_term


# Helper functions just for the toy_model.ipynb notebook
# ----------------------------------------

def prepare_data(random_data, N=None, seed=0, split=True, offset=1e-2):
    """
    Prepare 2D-data on a circle manifold.
    Args:
        random_data: If True, generate random data. Otherwise, use equidistant angles.
        N: Total number of points
        split: If True, split data into train and validation sets.
        offset: Rotate all angles by this amount, to avoid trajectories "stuck" between points
    """
    data_str = 'random' if random_data else 'symmetric'
    if not random_data: # Symmetric on circle
        N = 20 if N is None else N
        u = np.linspace(0, 1, N, endpoint=False) + offset  # uniform in [0, 1]
        theta = 2 * np.pi * u  # angle
        data = np.stack([np.cos(theta), np.sin(theta)], axis=1)
        data_labels = np.arange(N)  # One label per datapoint for now
        data_split = np.ones(N, dtype=bool)
        if split:
            data_split[1::2] = 0  # 1 = train, 0 = validation
    else: # Random on Circle
        np.random.seed(seed)
        N = 40 if N is None else N
        u = np.random.rand(N)  # uniform in [0, 1]
        theta = 2 * np.pi * u  # angle
        data = np.stack([np.cos(theta), np.sin(theta)], axis=1)
        data_labels = np.arange(N)  # One label per datapoint for now
        data_split = np.ones(N, dtype=bool)
        if split:
            data_split = np.array([0] * (N // 2) + [1] * (N // 2), dtype=bool)  # 1 = train, 0 = validation
    return data_str, N, data, data_labels, data_split


def model_and_metric(data, data_split, data_labels, num_steps, sigma_min, sigma_max, rho, delta, window_size,
                     guid_weight=0.0, delta_neg=None,
                     samples_per_point=0, sigma_idx=None,
                     plot_counter=1,
                     plot=False, contours=False, threshold=False, plot_points=0, plot_samples_per_point=0, n_rows=1,
                     plot_sigma_idx=None,
                     mode=None, scale=1, save_folder=None):
    # Toy Model settings
    model = ToyModel(data=data[data_split],
                     data_labels=data_labels[data_split],
                     num_steps=num_steps,
                     sigma_min=sigma_min,
                     sigma_max=sigma_max,
                     rho=rho,
                     model_kwargs={'delta': delta, 'cond': False},  # positive network
                     precond=False,
                     heun=False)

    # Compute L2 denoising error
    t_steps = model.t_schedule()[:-1][::-1]
    relative_error, plot_counter = l2_metric(model,
                                             model.model_kwargs['delta'],
                                             data,
                                             data_split,
                                             t_steps[sigma_idx],
                                             guid_weight=guid_weight,
                                             delta_neg=delta_neg,
                                             samples_per_point=samples_per_point,
                                             # How many samples to run per datapoint
                                             x_labels=None,
                                             cond=False,
                                             window_size=window_size,
                                             plot=plot,
                                             mode=mode,
                                             contours=contours,
                                             threshold=threshold,
                                             plot_points=plot_points,  # How many datapoints to include in the plot
                                             plot_samples_per_point=plot_samples_per_point, # How many samples per datapoint to include in the plot
                                             plot_sigmas=t_steps[plot_sigma_idx],
                                             plot_counter=plot_counter,
                                             n_rows=n_rows,  # How many samples per datapoint to show in the plot
                                             scale=scale,
                                             save_folder=save_folder,
                                             )
    return relative_error, plot_counter, t_steps


def gen_gap_plot(t_steps, all_errors, color_range, cbar_ticks, ylim, fmt=None, cmap='viridis_r', save_as=None):
    # Make generalization gap plot
    plt.figure(figsize=(6, 5))
    plot_generalization_gap(t_steps,
                            all_errors,
                            color_range=color_range,
                            idx=None,
                            cbar_ticks=cbar_ticks,
                            x_formatter=fmt,  # fmt here for custom formatter
                            ylim=ylim,
                            cmap=cmap,
                            )

    # for sigma in t_steps[sigma_idx]:
    #     plt.axvline(sigma, color='black', linestyle=':', alpha=0.5)

    # for delta in all_deltas:
    #     plt.plot(t_steps, t_steps**2 / (t_steps**2 + delta**2))

    if save_as:
        plt.savefig(save_as, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_cov_ellipse(pos, cov, nstd=1, ax=None, **kwargs):
    """Plots an ellipse representing the covariance matrix (cov) at position (pos)."""
    if ax is None:
        ax = plt.gca()

    # Eigenvalues and eigenvectors
    vals, vecs = np.linalg.eigh(cov)
    # Sort eigenvalues to get major axis
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]

    # Calculate angle and dimensions
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    # Width and height are 2 * nstd * sqrt(eigenvalues)
    width, height = 2 * nstd * np.sqrt(vals)

    ell = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)
    ell.set_facecolor('none')  # Transparent fill










