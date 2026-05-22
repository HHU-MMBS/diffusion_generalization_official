import matplotlib.ticker as ticker

from plot_utils import *

# dict with sigma_idx for all combinations of models and metrics
sigma_ids = {
    'in64': {
        'pl2': {
            'xs-uncond': 38,
            'xs': 40,
            's': 40,
            'm': 41,
            'l': 40,
            'xl': 40,
            'avg': 40,
            },
        'incp-l2': {
            'xs-uncond': 34,
            'xs': 37,
            's': 37,
            'm': 37,
            'l': 38,
            'xl': 37,
            'avg': 37,
        },
        'dino-l2': {
            'xs-uncond': 37,
            'xs': 39,
            's': 40,
            'm': 40,
            'l': 40,
            'xl': 40,
            'avg': 39,
        },
        'rfid': {
            'xs-uncond': 38,
            'xs': 41,
            's': 41,
            'm': 42,
            'l': 43,
            'xl': 43,
            'avg': 41,
        },
        'rfdd': {
            'xs-uncond': 35,
            'xs': 39,
            's': 38,
            'm': 40,
            'l': 40,
            'xl': 40,
            'avg': 39,
        },
    },
    'in512': {
        'pl2': {
            'xs-uncond': 38,
            'xxs': 39,
            'xs': 39,
            's': 39,
            'm': 40,
            'l': 40,
            'xl': 40,
            'xxl': 39,
            'avg': 39,
            },
        'incp-l2': {
            'xs-uncond': None,
            'xxs': None,
            'xs': None,
            's': 38,
            'm': None,
            'l': None,
            'xl': None,
            'xxl': None,
            'avg': 38,
            },
        'dino-l2': {
            'xs-uncond': None,
            'xxs': None,
            'xs': None,
            's': 37,
            'm': None,
            'l': None,
            'xl': None,
            'xxl': None,
            'avg': 37,
            },
        'rfid': {
            'xs-uncond': None,
            'xxs': None,
            'xs': None,
            's': 40,
            'm': None,
            'l': None,
            'xl': None,
            'xxl': None,
            'avg': 40,
            },
        'rfdd': {
            'xs-uncond': None,
            'xxs': None,
            'xs': None,
            's': 39,
            'm': None,
            'l': None,
            'xl': None,
            'xxl': None,
            'avg': 39,
            },
        },
    'cifar10': {
        'pl2': 35,
        'dino-l2': 33,
        'incp-l2': 35,
        'rfdd': 39,
        'rfid': 36,
    },
    'cifar100': {
        'pl2': 38,
        'dino-l2': 33,
        'incp-l2': 38,
        'rfdd': 40,
        'rfid': 39,
    },
}


def gen_gap_vs_sigma(
        train_paths, val_paths, color_range,
        x_formatter=False, ylim=[-0.05, 0.16],
        idx=None, cbar_ticks=None,
        metric='pl2',  # 'pl2' or 'incp-l2' 'dino-l2' or 'rfid' or 'rfdd'
):
    """
    Generate a generalization gap plot showing relative error between training and validation fits.

    Args:
        train_paths (list of str): Paths to training data files.
        val_paths (list of str): Paths to validation data files.
        color_range (list): Range of values to use for coloring lines on the plot.
        x_formatter (bool, optional): Format color bar ticks as multiples. Defaults to False.
        ylim (list, optional): Y-axis limits. Defaults to [-0.05, 0.16].
        idx (int, optional): Select a subset of the data to plot. Defaults to None.
        cbar_ticks (list, optional): Custom tick labels for the color bar. Defaults to None.

    Returns:
        tuple: Training residuals, validation residuals, and sigma_bins.
    """
    # Step 1: Prepare residual arrays and variables
    num_paths = len(train_paths)
    assert num_paths == len(val_paths), f"Mismatch between number of train ({len(train_paths)}) and validation paths ({len(val_paths)})."

    # Load data for train and validation
    train_residuals, sigma_bins = load_residuals(train_paths, metric=metric)
    val_residuals, _ = load_residuals(val_paths, metric=metric)

    # Validation checks
    assert train_residuals.min() > 0, "Not all values in train residuals were filled"
    assert val_residuals.min() > 0, "Not all values in val residuals were filled"

    # Step 3: Calculate relative error
    relative_error = (val_residuals - train_residuals) / train_residuals

    # Step 4: Multiline plot for relative error
    line_collection = multiline(
        xs=np.array([sigma_bins] * len(relative_error)),
        ys=relative_error,
        c=color_range,
        cmap='viridis',
        alpha=1,
        idx=idx
    )

    # Step 5: Plot layout and configuration
    plot_layout(sigma_bins[0], sigma_bins[-1], scale_y=False, legend=False)
    plt.xlabel('$\sigma$')
    if x_formatter:
        def fmt(x, pos):
            return f'{int(x // 1000)}M'  # Format tick labels

        color_bar = plt.colorbar(line_collection, format=ticker.FuncFormatter(fmt))
    else:
        color_bar = plt.colorbar(line_collection)

    # Customize color bar ticks if provided
    if cbar_ticks is not None:
        color_bar.set_ticks(cbar_ticks)

    plt.ylim(ylim)

    # Step 6: Return results
    return relative_error, train_residuals, val_residuals, sigma_bins


def gen_gap_vs_model_error(
        train_paths, val_paths, x_axis, sigma_idx, metric,
        model_name, color, mode='overfit', label=None, **plot_kwargs,
    ):
    """
    """
    # Prepare residual arrays and variables
    num_paths = len(train_paths)
    if num_paths != len(val_paths):
        print(f"Mismatch between number of train ({len(train_paths)}) and validation paths ({len(val_paths)}) for {model_name}, skipping...")
        return

    # Load data for train and validation
    train_residuals, sigma_bins = load_residuals(train_paths, metric=metric)
    val_residuals, _ = load_residuals(val_paths, metric=metric)

    # Validation checks
    assert train_residuals.min() > 0, "Not all values in train residuals were filled"
    assert val_residuals.min() > 0, "Not all values in val residuals were filled"

    # Select one sigma to plot
    y_tr = train_residuals[:, sigma_idx]
    y_val = val_residuals[:, sigma_idx]
    y = (y_val - y_tr) / y_tr
    print(f"Selecting sigma {sigma_bins[sigma_idx]:.2f} for model {model_name}.")
    # print(relative_error.argmax(axis=1))

    sort_idx = np.argsort(x_axis)
    x_axis = x_axis[sort_idx]
    y = y[sort_idx]
    before = len(x_axis)
    if before > 32:  # more than 32 snaps
        step = 4
    elif before > 16:
        step = 2
    else:
        step = 1
    x_axis = x_axis[::step]
    y = y[::step]
    y_tr = y_tr[::step]
    y_val = y_val[::step]
    if step > 1:
        print(f"Reducing number of points from {before} to {len(x_axis)} for model {model_name}.")

    label = model_name if label is None else label
    if mode == 'overfit':
        plt.plot(x_axis, y, color=color, label=label, **plot_kwargs)
        plt.scatter(x_axis, y, color=color, marker='o', s=10)
    elif mode == 'train-val':
        for i, y in enumerate((y_tr, y_val)):
            plt.plot(x_axis, y, color=color, label=label if i == 0 else None, linestyle=':' if i == 1 else '-')
            plt.scatter(x_axis, y, color=color, marker='o', s=10)
    else:
        raise ValueError(f"Invalid mode '{mode}'.")


def loss_decomposition(model, dataset, snaps, title, labels, dist=True, res=True, unc=True, lam=True, loss=True, eff_loss=True):
    for snap in snaps:
        # Load data
        path = f"/diffusion_overfit/data/loss_analysis/{dataset}/{model}/train_data/data-{snap}.pt"
        (eps_mean, eps_std, loss_mean, loss_std, sigma_pdf, eff_loss_mean, eff_loss_std,
         t_steps, varweight_mean, varweight_std) = load_data(path)

        # Plot figure
        plt.figure(figsize=(12, 5))
        plt.title(title)
        # plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.08)

        if dist:
            plt.plot(t_steps, sigma_pdf / sigma_pdf.max(),     label=labels[0], linestyle=':', color='C3')
        if res:
            mean_std_plot(t_steps, eps_mean, eps_std,               label=labels[1], color='C1')
        if unc and varweight_mean is not None:
            logvar = np.log(varweight_mean ** -1)
            mean_std_plot(t_steps, varweight_mean, varweight_std,   label=labels[2], color='C2')
            mean_std_plot(t_steps, logvar, 0,                   label=labels[3], color='C6')
        if lam:
            sigma_data = 0.5
            edm_weight = (t_steps ** 2 + sigma_data ** 2) / (t_steps * sigma_data) ** 2
            plt.plot(t_steps, edm_weight / edm_weight.max(),   label=labels[4], color='C5')
        if loss:
            mean_std_plot(t_steps, loss_mean, loss_std,             label=labels[5], color='C0')  # loss
        if eff_loss:
            mean_std_plot(t_steps, eff_loss_mean, eff_loss_std,     label=labels[6], color='C4')

        plot_layout(t_steps[0], t_steps[-1])
        # plt.tick_params(left=True)
        # plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1], [0, 0.2, 0.4, 0.6, 0.8, 1])
        ticks = [0.002, 0.005, 0.02, 0.1, 0.5, 1, 2, 5, 10, 20, 50]
        plt.xticks(ticks, [f'$\sigma=${0.002}   ', '', 0.02, 0.1, 0.5, 1, 2, 5, 10, 20, 50])
        
        # plt.savefig('/home/tikai103/plots/inductive_bias_diffusion/figs/loss_analysis.png', dpi=300)
        # plt.show()
        # plt.close()



def gg_vs_snaps(dataset, model, overfit=True, fontsize=18):
    # Get generalization-gap data
    tr_paths, val_paths, sigma_bins, tr_eps, val_eps = get_residuals(dataset, [model], ema=True)
    pop_idx = ['-' not in key for key in tr_eps[model].keys()]  # remove snaps with ema suffix
    tr_eps = np.array(list(tr_eps[model].values()))[pop_idx]  # [len(tr_paths), 64]
    val_eps = np.array(list(val_eps[model].values()))[pop_idx]  # [len(tr_paths), 64]

    color_range = sigma_bins
    x_axis = [int(tr_paths[model][i].split('.')[0].split('-')[-2]) for i in range(len(tr_paths[model]))]
    print(f"Found {len(tr_paths[model])} checkpoints: {x_axis}")

    val_eps = val_eps.transpose(1, 0)
    tr_eps = tr_eps.transpose(1, 0)

    if dataset == 'in512':
        s_idx = 40
    elif dataset == 'in64':
        s_idx = 41
    else:
        s_idx = 36  # in512: 40 (1.67), 33 (0.51), 24 (0.11), cifar-10: 36
    print(f"Plotting for $\sigma = $ {sigma_bins[s_idx]:.1f}")
    plt.plot(x_axis, tr_eps[s_idx], label='Training')
    plt.plot(x_axis, val_eps[s_idx], label='Validation')
    plt.ylabel('Training objective', fontsize=fontsize)

    # Overfit annotation
    overfit_th = x_axis[val_eps[s_idx].argmin()]
    if 'in' in dataset:
        if s_idx == 40:
            anno_x = overfit_th + 20000
            anno_y = val_eps[s_idx].min() - 0.0042
            arrow_start = (anno_x + 100000, anno_y + 0.0001)
            arrow_end = (anno_x + 200000, anno_y + 0.0001)
        elif s_idx == 33:
            anno_x = overfit_th + 10000
            anno_y = val_eps[s_idx].min() + 0.0005
            arrow_start = (anno_x + 130000, anno_y + 0.00005)
            arrow_end = (anno_x + 220000, anno_y + 0.00005)
    else:
        anno_x = overfit_th + 10000
        anno_y = val_eps[s_idx].min() + 0.001
        arrow_start = (anno_x + 70000, anno_y + 0.00011)
        arrow_end = (anno_x + 120000, anno_y + 0.00011)

    if not (s_idx == 24 and 'in' in dataset) and overfit:
        plt.axvline(overfit_th, color='black', linestyle=':')
        plt.text(anno_x, anno_y, f'Overfit', fontsize=fontsize)
        plt.annotate(text='', xy=arrow_start, xytext=arrow_end, arrowprops=dict(arrowstyle='<-'))

    # Generalization gap annotation
    gg_color = 'C5'
    plt.fill_between(x_axis, tr_eps[s_idx], val_eps[s_idx], alpha=0.1, color=gg_color)
    if dataset == 'in512':
        if s_idx != 24:
            gg_idx = -8  # where on the x_axis, only modify this!
            anno_x = x_axis[gg_idx]
            anno_bot, anno_top = tr_eps[s_idx][gg_idx], val_eps[s_idx][gg_idx]
            anno_x_text, anno_y_text = anno_x + 10000, (anno_top + anno_bot) / 2 + 0.0004
        else:
            gg_idx = -1
            anno_x = x_axis[gg_idx]
            anno_bot, anno_top = tr_eps[s_idx][gg_idx], val_eps[s_idx][gg_idx]
            anno_x_text, anno_y_text = anno_x - 300000, anno_top + 0.000015
    elif dataset == 'in64':
        gg_idx = -18  # where on the x_axis, only modify this!
        anno_x = x_axis[gg_idx]
        anno_bot, anno_top = tr_eps[s_idx][gg_idx], val_eps[s_idx][gg_idx]
        anno_x_text, anno_y_text = anno_x + 30000, (anno_top + anno_bot) / 2 + 0.0003
    else:
        gg_idx = -12
        anno_x = x_axis[gg_idx]
        anno_bot, anno_top = tr_eps[s_idx][gg_idx], val_eps[s_idx][gg_idx]
        anno_x_text, anno_y_text = anno_x + 10000, (anno_top + anno_bot) / 2

    print(anno_x, anno_bot)
    plt.annotate(text='', xy=(anno_x, anno_bot), xytext=(anno_x, anno_top),
                 arrowprops=dict(arrowstyle='<->', color=gg_color))
    plt.text(anno_x_text, anno_y_text, 'Generalization gap', fontsize=fontsize-1, color=gg_color)
    if dataset == 'in512':
        plt.xticks([200000, 400000, 600000, 800000], ['200M', '400M', '600M', '800M'], fontsize=fontsize - 6)
    elif dataset == 'in64':
        plt.xticks([500000, 1000000, 1500000, 2000000], ['500M', '1000M', '1500M', '2000M'], fontsize=fontsize - 6)
        plt.ylim([0.04, 0.05])
        # plt.yticks([0.042, 0.044, 0.046, 0.048, 0.050], [0.042, 0.044, 0.046, 0.048, 0.050])
    # plt.title(f"{dataset}, {model}")
    plt.xlim([x_axis[0], x_axis[-1]])
    plt.yticks(fontsize=fontsize - 6)
    setup_plot(fontsize=fontsize)
    return s_idx, sigma_bins
