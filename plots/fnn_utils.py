# fnn_utils.py
from pathlib import Path
import numpy as np


PROBE_KEYS = ["xr_train_vs_train", "xr_val_vs_train", "xt_vs_train", "xr_train_vs_val", "xt_stop18_vs_train", "xr_val_vs_val", "xt_vs_val", "xt_stop18_vs_val"]


def parse_snap(folder_name: str) -> int:
    """Extract snap integer from folder name like '1073741-0.100'."""
    return int(folder_name.split("-")[0])


def load_snap_data(results_root: Path, dataset: str, model_size: str, sigma: str, f_extractor: str, ema: str) -> dict:
    """
    Load all cos-sim arrays for every snap found under
    results_root / dataset / model_size / {snap}-{ema} / {probe_key} / cos-sims.npy

    Returns:
        {snap_int: {probe_key: np.ndarray}}
    """
    base = results_root / dataset / model_size
    data = {}
    for snap_dir in sorted(base.iterdir()):
        if not snap_dir.is_dir():
            continue
        snap_str, snap_ema = snap_dir.name.split("-", 1)
        if snap_ema != ema:
            continue
        snap = int(snap_str)
        data[snap] = {}
        for key in PROBE_KEYS:
            p = snap_dir / sigma / key / f"cos-sims-{f_extractor}.npy"
            if p.exists():
                data[snap][key] = np.load(p)
    return data


def compute_mean_series(snap_data: dict, probe_key: str) -> tuple[np.ndarray, np.ndarray]:
    """
    For a given probe_key, return (snaps, means) sorted by snap.
    Only includes snaps where the key is present.
    """
    records = [
        (snap, vals[probe_key].mean())
        for snap, vals in snap_data.items()
        if probe_key in vals
    ]
    records.sort(key=lambda x: x[0])
    snaps = np.array([r[0] for r in records])
    means = np.array([r[1] for r in records])
    return snaps, means