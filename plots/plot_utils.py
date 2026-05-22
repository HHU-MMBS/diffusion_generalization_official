import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker

from matplotlib.collections import LineCollection


def mean_std_plot(t_steps, mean, std, label, color, scaling=1.):
    """Plots the mean with +/- std and normalized to the range [0,scaling] """
    min, max = (mean - std).min(), (mean + std).max()
    def normalized(x):
        return (x - min) / (max - min) * scaling
    plt.plot(t_steps, normalized(mean), label=label, color=color)
    plt.fill_between(t_steps, normalized(mean + std), normalized(mean - std), alpha=.25, color=color)


def plot_layout(sigma_min, sigma_max, scale_y=True, legend=True):
    plt.grid(alpha=0.4)
    plt.xscale('log')
    plt.xlim(left=sigma_min, right=sigma_max)
    if legend:
        plt.legend(loc=1)
    if scale_y:
        plt.ylim(bottom=0, top=1.1)
        plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        plt.tick_params(left=False)


def load_data(path):
    data = torch.load(path, weights_only=False)
    eps_mean, eps_std = data['eps_mean'], data['eps_std']
    loss_mean, loss_std = data['loss_mean'], data['loss_std']
    sigma_pdf = data['sigma_pdf']
    eff_loss_mean, eff_loss_std = data['eff_loss_mean'], data['eff_loss_std']
    t_steps = data['t_steps']
    varweight_mean, varweight_std = data['varweight_mean'], data['varweight_std']
    return (eps_mean, eps_std, loss_mean, loss_std, sigma_pdf, eff_loss_mean, eff_loss_std,
            t_steps, varweight_mean, varweight_std)


def extract_snapshot_from_path(path):
    """Extract snapshot number from file path"""
    snap = path.replace('-0.100', '').split('data-')[-1].split('.')[0]
    return "200000" if snap == "200090" else snap


def log_normal(x, mu, sigma):
    return np.exp(((np.log(x) - mu) / sigma) ** 2 / -2) / (x * sigma * (2 * np.pi) ** 0.5)


def log_area(x, y):
    """Computes the area under the curve in log-space, given points (x,y) in real space"""
    log_x = np.log(x)
    return ((y[1:] + y[:-1]) / 2 * (log_x[1:] - log_x[:-1])).sum()


def multiline(xs, ys, c, idx=None, ax=None, **kwargs):
    """Plot lines with different colorings

    Parameters
    ----------
    xs : iterable container of x coordinates
    ys : iterable container of y coordinates
    c : iterable container of numbers mapped to colormap
    idx: optional, select a subset of data to plot
    ax (optional): Axes to plot on.
    kwargs (optional): passed to LineCollection

    Notes:
        len(xs) == len(ys) == len(c) is the number of line segments
        len(xs[i]) == len(ys[i]) is the number of points for each line (indexed by i)

    Returns
    -------
    lc : LineCollection instance.
    """

    if idx is not None:
        xs = xs[idx]
        ys = ys[idx]
        c = c[idx]

    # find axes
    ax = plt.gca() if ax is None else ax

    # create LineCollection
    segments = [np.column_stack([x, y]) for x, y in zip(xs, ys)]
    lc = LineCollection(segments, **kwargs)

    # set coloring of line segments
    #    Note: I get an error if I pass c as a list here... not sure why.
    lc.set_array(np.asarray(c))

    # add lines to axes and rescale
    #    Note: adding a collection doesn't autoscalee xlim/ylim
    ax.add_collection(lc)
    ax.autoscale()
    return lc


def load_residuals(paths, metric):
    sigma_bins = None
    assert len(paths) > 0, "No data files provided."
    for i, path in enumerate(paths):
        data = torch.load(path, weights_only=False)

        if i == 0:  # Save sigma bins only for the first file
            sigma_bins = data['t_steps']
            if 'num_steps' and 'num_samples' in data.keys():
                num_steps, num_samples = data['num_steps'], data['num_samples']
                # print(f"The data contains {num_steps} noise levels, each averaged over {num_samples} samples.")
            else:
                num_steps = data['t_steps'].shape[0]
            residuals = np.zeros((len(paths), num_steps))
        if metric == 'pl2':
            try:
                residuals[i] = data['l2_eps_mean']
            except:
                residuals[i] = data['eps_mean']
        elif metric == 'incp-l2':
            residuals[i] = data['dino_eps_mean']  #
        elif metric == 'dino-l2':
            residuals[i] = data['incp_eps_mean']  #
        elif metric == 'rfid':
            residuals[i] = data['rfid']  # rfid
        elif metric == 'rfdd':
            residuals[i] = data['rfdd']  #
        else:
            raise ValueError(f"Invalid metric '{metric}'.")
    return residuals, sigma_bins


def get_residuals(dataset, models, ema=False):
    """
    Load training and validation residuals for a given dataset and list of models.

    Args:
        dataset (str): Name of the dataset (e.g., "cifar10").
        models (list of str): List of models to process.
        ema (bool, optional): Flag to include or exclude ema data for older data. Defaults to False.

    Returns:
        tuple:
            - train_paths (dict): Dictionary of training data paths for each model.
            - val_paths (dict): Dictionary of validation data paths for each model.
    """
    # Initialize dictionaries to store file paths and residual data
    train_paths = {}
    val_paths = {}
    train_residuals = {}
    val_residuals = {}

    # Step 1: Iterate over each model
    for model in models:
        train_paths[model], val_paths[model] = [], []

        # Step 2: Locate training and validation file paths
        for mode, path_list in [('train', train_paths[model]), ('val', val_paths[model])]:
            data_folder = f'/home/shared/generative_models/diffusion_overfit/data/loss_analysis_l2_only/{dataset}/{model}/{mode}_data'  # todo: change back to regular loss_analysis folder
            if os.path.exists(data_folder):  # Ensure the data folder exists
                for file_path in sorted([os.path.join(data_folder, f) for f in os.listdir(data_folder) if 'data' in f]):
                    if not ema and '-0.' in file_path:  # Exclude EMA files if `ema` set to False
                        continue
                    path_list.append(file_path)
            else:
                print(f"Warning: Missing data folder for {model} in mode '{mode}'.")

        # Step 3: Load residuals from the paths and save into residual dictionaries
        train_residuals[model], val_residuals[model] = {}, {}
        for residual_dict, paths in [(train_residuals[model], train_paths[model]),
                                     (val_residuals[model], val_paths[model])]:
            for path in paths:
                snapshot = extract_snapshot_from_path(path)
                data = torch.load(path, weights_only=False)
                residual_dict[snapshot] = data['eps_mean']
                sigma_bins = data['t_steps']

        # Step 4: Validation checks
        train_values = np.array(list(train_residuals[model].values()))
        val_values = np.array(list(val_residuals[model].values()))
        assert train_values.min() > 0, f"Training residuals for model {model} contain invalid values."
        assert val_values.min() > 0, f"Validation residuals for model {model} contain invalid values."

    return train_paths, val_paths, sigma_bins, train_residuals, val_residuals


def setup_plot(xlabel="Images seen", snaps_on_x=True, fontsize=None):
    if snaps_on_x:
        def fmt(x, pos):
            return f'{int(x // 1000)}M'

        plt.gca().xaxis.set(major_formatter=ticker.FuncFormatter(fmt))
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.grid(alpha=0.4)
    # plt.legend()


def vs_sigma_plot(rel_error, sigma_bins, color_range, alpha=1):
    """ema_weight=0 for no smoothing"""
    lc = multiline(xs=np.array([sigma_bins]*len(rel_error)),
                   ys=rel_error,
                   c=color_range,
                   cmap='viridis',
                   alpha=alpha)
    plot_layout(sigma_bins[0], sigma_bins[-1], scale_y=False, legend=False)
    plt.xlabel('$\sigma$')
    def fmt(x, pos):
        return f'{int(x//1000)}M'
    plt.colorbar(lc, format=ticker.FuncFormatter(fmt))
