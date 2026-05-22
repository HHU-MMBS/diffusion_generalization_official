import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as ticker

from plot_utils import *


def get_fds(dataset, n_trainruns=15, early_stop=-1, guidance='none'):
    """
    Load FD results into an nested dictionary with keys:
    1st level: ['xxs', 'xs', 's', 'm', 'l', 'xl', 'xxl', 'xxs-uncond', 'xs-uncond']
    2nd level: ['train', 'val']
    3rd level: ['FID', 'FDD', 'snaps']

    n_trainruns is the expected number of runs for gen-v-train.
    """
    # Get metrics and their snaps
    # data = pd.read_csv(f"/home/tikai103/plots/inductive_bias_diffusion/data/{model}-{dataset}-none.csv")

    fd_data = {}
    if 'in' in dataset:
        model_sizes = ['xxs-uncond', 'xxs', 'xs-uncond', 'xs', 's', 'm', 'l', 'xl', 'xxl']
    elif 'cifar' in dataset:
        model_sizes = [None]
    for model_size in model_sizes:
        level_2 = {'train': None, 'val': None}
        for mode in ['train', 'val']:
            data = pd.read_csv(f"/home/shared/generative_models/diffusion_overfit/data/fd_analysis/gen-v-{mode}/{dataset}.csv")

            if 'in' in dataset:
                model_data_idx = [f"/{model_size}/" in s for s in data['ref_path1']]
                if sum(model_data_idx) == 0:
                    break  # no data for this model size
                arrs = data[model_data_idx]
            elif 'cifar' in dataset:
                arrs = data  # use all data

            if 'auto' in dataset or 'regular' in dataset:  # x-axis is training time (snaps)
                if early_stop == -1:
                    x_all = np.array([float(s.split('/')[-2].split('w')[-1]) for s in arrs['ref_path1']])  # different weights
                else:
                    x_all = np.array([float(s.split('/')[-2].split('w')[-1].split('_')[0]) for s in arrs['ref_path1']])  # different weights
                x_unique = np.unique(x_all)
                key = 'weights'
            elif 'kmeans' in dataset:
                x_all = np.array([int(s.split('/')[9].split('kmeans')[-1]) for s in arrs['ref_path1']])  # different number of classes
                x_unique = np.unique(x_all)
                key = 'kmeans'
            else:
                if early_stop == -1:
                    x_all = np.array([int(s.split('/')[-3]) for s in arrs['ref_path1']])  # different snap numbers
                else:
                    x_all = np.array([int(s.split('/')[-4]) for s in arrs['ref_path1']])  # different snap numbers
                x_unique = np.unique(x_all)
                key = 'snaps'

            level_3 = {}
            level_3[key] = np.sort(x_unique)
            level_3['FID'] = np.zeros(len(x_unique))  # One FD value for each snap/weight
            level_3['FDD'] = np.zeros(len(x_unique))  # One FD value for each snap/weight
            fid_all = arrs['FID'].to_numpy()  # (n_snaps * n_trainruns,), different train-50k sets for each snap/weight
            fdd_all = arrs['FDD'].to_numpy()  # (n_snaps * n_trainruns,), different train-50k sets for each snap/weight
            for s, x in enumerate(x_unique):
                x_mask = x == x_all
                # if x_mask.sum() != n_trainruns and mode == 'train':
                #     print(f"All snaps/weights need to have the same number of re-runs (found {x_mask.sum()} instead of {n_trainruns}. Model size: {model_size}, mode: {mode}, {key}: {x}")
                level_3['FID'][s] = fid_all[x_mask].mean()  # Average all trainruns
                level_3['FDD'][s] = fdd_all[x_mask].mean()  # Average all trainruns
                
            level_2[mode] = level_3
        if 'in' in dataset:
            fd_data[model_size] = level_2
        elif 'cifar' in dataset:
            fd_data = level_2

        # Add baseline data
        # ds_baseline = dist_shift_baseline(dataset, verbose=False)
        # s_baseline = sampling_baseline(dataset, verbose=False)
        # for metric in ['FID', 'FDD']:
        #     # Load baselines to account for distribution shift
        #     ds_mean = ds_baseline[0] if metric == 'FDD' else ds_baseline[2]
        #     s_mean = s_baseline[0] if metric == 'FDD' else s_baseline[2]
        #     fd_shift = ds_mean - s_mean  # How much does the FD increase because of distribution shift between train and val data
        #     fd_data[f"{metric}_val_clean"] = fd_data[f"{metric}_val"] - fd_shift
        # fd_data[f"snaps_val_clean"] = fd_data[f"snaps_val"]
    return fd_data


def fd_vs_snaps(fd_data, model_sizes):
    for metric in ['FDD', 'FID']:
        plt.figure(figsize=(6, 5))
        for model_size in model_sizes:
            model_data = fd_data[model_size]
            # plot_lines = []  # to draw "fill between" later
            for j, mode in enumerate(['train', 'val']):
                fd_snaps = model_data[mode][f'snaps']
                fd_values = model_data[mode][metric]

                plt.plot(fd_snaps, fd_values, color=f"C{j}", label=f'{mode}')
                # plot_lines.append(fd_values)
                # plt.scatter(x_axis, m, color=f"C{j}", marker='o', s=20)

            plt.ylabel(metric)
            # if 'edm2' in model:
            #     plt.ylim([0, 5.2]) if metric == 'FID' else plt.ylim([0, 100])  # IN512
            # else:
            #     plt.ylim([1.7, 5]) if metric == 'FID' else plt.ylim([135, 300])  # CIFAR

            setup_plot()

            # # Generalization gap annotation, only setup for EDM2-XXL IN512
            # gg_color = 'C5'
            # plt.fill_between(x_axis, plot_lines[0], plot_lines[1], alpha=0.1, color=gg_color)
            # gg_idx = -6
            # anno_x = x_axis[gg_idx]
            # anno_bot, anno_top = plot_lines[0][gg_idx], plot_lines[1][gg_idx]
            # anno_x_text, anno_y_text = anno_x + 10000, (anno_top + anno_bot) / 2.05

            # plt.annotate(text='', xy=(anno_x, anno_bot), xytext=(anno_x, anno_top), arrowprops=dict(arrowstyle='<->', color=gg_color))
            # plt.text(anno_x_text, anno_y_text, 'Generalization gap', fontsize=12, color=gg_color)

        #     # plt.savefig(f'/home/tikai103/diffusion_overfit/plots/figs/icml2025/{metric}-vs-Mimg_{dataset}-{model}.png', dpi=200, bbox_inches='tight')
        #     plt.show()


def overfit_vs_snaps(models_data, metric, mode, x_axis='snaps', x_key='snaps', label=None, c=None,
                     stop_in_label=False, **plot_kwargs):
    """
    models_data: dict with strings for the legend as keys and model_data dicts as values
    """
    for m, model_name in enumerate(models_data.keys()):
        model_data = models_data[model_name]  # Model size for ImageNet, dataset for CIFAR10/100

        if model_data['train'] is None:
            print(f"Didn't find data for {model_name}, skipping...")
            continue  # no data for this model size

        x = model_data['train'][x_key]
        assert np.all(x == model_data['val'][x_key]), "Training and validation snaps don't match"
        fd_train, fd_val = model_data['train'][metric], model_data['val'][metric]

        if x_axis == 'fd':
            x = fd_train

        if mode == 'train':
            y = fd_train
            print(f"model_name: {y.min()}")
        elif mode == 'val':
            y = fd_val
        elif mode == 'train-val':
            y = fd_train
            y2 = fd_val
        elif mode == 'overfit':
            y = (fd_val - fd_train) / fd_train  # relative overfit, same as L2 metric
        else:
            raise ValueError(f"Unknown mode: {mode}")

        if x_axis == 'snaps':
            cur_c = plt.cm.viridis(1 - m / len(models_data.keys())) if c is None else c
            cur_label = model_name if label is None else label
            if "stop" in cur_label and not stop_in_label:
                cur_label = cur_label.split('-stop')[0]
        else:
            cur_c = 'C3' if c is None else c
            cur_label = f"FDD" if label is None else label

        if 'cifar' in model_name:
            c = f'C{m}'

        plt.plot(x, y, color=cur_c, label=cur_label, **plot_kwargs)
        plt.scatter(x, y, color=cur_c, marker='o', s=10)

        if mode == 'train-val':
            val_alpha = 0.6
            plt.plot(x, y2, color=cur_c, linestyle='--', alpha=val_alpha)
            plt.scatter(x, y2, color=cur_c, marker='x', s=10, alpha=val_alpha)

        if 'train' in models_data.keys():  # cifar10/100
            break

    if x_axis == 'snaps':
        setup_plot()
    elif x_axis == 'fd':
        plt.grid(alpha=0.4)
        plt.legend()

    # Finetune for final plots
    if mode == 'overfit':
        plt.ylim([-0.1, 0.4])

    # plt.ylabel(metric)
