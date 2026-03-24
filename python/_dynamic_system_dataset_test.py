import numpy as np
import sys
from pathlib import Path


from reservoirpy.datasets import (
    doublescroll,
    henon_map,
    kuramoto_sivashinsky,
    logistic_map,
    lorenz,
    lorenz96,
    mackey_glass,
    multiscroll,
    narma,
    rabinovich_fabrikant,
    rossler,
)


SEQ_LEN = 96
LABEL_LEN = 48
PRED_LEN = 96


def _split_borders(total_rows: int):
    num_train = int(total_rows * 0.7)
    num_test = int(total_rows * 0.2)
    num_val = max(0, total_rows - num_train - num_test)

    raw_border1s = (
        0,
        max(0, num_train - SEQ_LEN),
        max(0, total_rows - (num_test + SEQ_LEN)),
    )
    raw_border2s = (num_train, num_train + num_val, total_rows)

    border1s = tuple(min(total_rows, idx) for idx in raw_border1s)
    border2s = tuple(min(total_rows, idx) for idx in raw_border2s)
    return border1s, border2s


def _from_series(series: np.ndarray):
    total_rows = series.shape[0]
    border1s, border2s = _split_borders(total_rows)

    start_idx, end_idx = border1s[2], border2s[2]

    train_slice = series[border1s[0] : border2s[0], :]
    mean = train_slice.mean(axis=0)
    scale = train_slice.std(axis=0, ddof=0)
    scale = np.where(scale == 0.0, 1.0, scale)

    scaled_data = (series - mean) / scale
    data_x = scaled_data[start_idx:end_idx, :]
    data_y = data_x.copy()
    data_stamp = np.zeros((max(0, end_idx - start_idx), 1), dtype=np.float64)
    return data_x, data_stamp, data_y


def _as_2d(series: np.ndarray) -> np.ndarray:
    arr = np.asarray(series, dtype=np.float64)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr


def _series_for_system(system_name: str):
    n_timesteps = 400

    if system_name == "logistic_map":
        return _as_2d(logistic_map(n_timesteps, r=3.9, x0=0.1))
    if system_name == "henon_map":
        return _as_2d(henon_map(n_timesteps, a=1.4, b=0.3, x0=[0.0, 0.0]))
    if system_name == "lorenz":
        return _as_2d(
            lorenz(
                n_timesteps,
                rho=28.0,
                sigma=10.0,
                beta=2.6666666666666665,
                x0=[1.0, 1.0, 1.0],
                h=0.01,
            )
        )
    if system_name == "lorenz96":
        return _as_2d(
            lorenz96(n_timesteps, warmup=0, N=8, F=8.0, dF=0.01, h=0.01, x0=None)
        )
    if system_name == "rossler":
        return _as_2d(
            rossler(n_timesteps, a=0.2, b=0.2, c=5.7, x0=[1.0, 1.0, 1.0], h=0.01)
        )
    if system_name == "doublescroll":
        return _as_2d(
            doublescroll(
                n_timesteps,
                r1=1.2,
                r2=3.44,
                r4=0.193,
                ir=2.25,
                beta=11.6,
                x0=[0.1, 0.0, 0.0],
                h=0.01,
            )
        )
    if system_name == "multiscroll":
        return _as_2d(
            multiscroll(n_timesteps, a=36.0, b=3.0, c=20.0, x0=[0.1, 0.0, 0.0], h=0.01)
        )
    if system_name == "rabinovich_fabrikant":
        return _as_2d(
            rabinovich_fabrikant(
                n_timesteps,
                alpha=0.14,
                gamma=0.1,
                x0=[0.1, 0.1, 0.1],
                h=0.005,
            )
        )
    if system_name == "mackey_glass":
        return _as_2d(
            mackey_glass(
                n_timesteps,
                tau=0,
                a=0.2,
                b=0.1,
                n=10,
                x0=1.2,
                h=0.1,
                seed=None,
                history=None,
            )
        )
    if system_name == "narma":
        order = 10
        u = np.array(
            [(i % 7) * 0.05 for i in range(n_timesteps + order)], dtype=np.float64
        )
        x0 = np.zeros(order, dtype=np.float64)
        _u, y = narma(
            n_timesteps,
            order=order,
            a1=0.3,
            a2=0.05,
            b=1.5,
            c=0.1,
            x0=x0,
            u=u,
            seed=None,
        )
        return _as_2d(y)
    if system_name == "kuramoto_sivashinsky":
        return _as_2d(kuramoto_sivashinsky(120, warmup=0, N=16, M=8.0, x0=None, h=0.25))

    raise ValueError(f"Unknown system_name: {system_name}")


def dynamic_system_dataset_test(system_name: str):
    series = _series_for_system(system_name)
    return _from_series(series)
