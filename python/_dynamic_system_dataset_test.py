import numpy as np


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


def _logistic_map(n_timesteps: int, r: float, x0: float):
    states = np.zeros((n_timesteps, 1), dtype=np.float64)
    states[0, 0] = x0
    for i in range(1, n_timesteps):
        states[i, 0] = r * states[i - 1, 0] * (1.0 - states[i - 1, 0])
    return states


def _henon_map(n_timesteps: int, a: float, b: float, x0):
    states = np.zeros((n_timesteps, 2), dtype=np.float64)
    states[0, :] = x0
    for i in range(1, n_timesteps):
        states[i, 0] = 1.0 - a * states[i - 1, 0] * states[i - 1, 0] + states[i - 1, 1]
        states[i, 1] = b * states[i - 1, 0]
    return states


def _rk4_system(diff, state: np.ndarray, dt: float, *args):
    k1 = diff(state, *args)
    k2 = diff(state + 0.5 * dt * k1, *args)
    k3 = diff(state + 0.5 * dt * k2, *args)
    k4 = diff(state + dt * k3, *args)
    return state + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def _lorenz_diff(state: np.ndarray, rho: float, sigma: float, beta: float):
    x, y, z = state
    return np.array(
        [sigma * (y - x), x * (rho - z) - y, x * y - beta * z], dtype=np.float64
    )


def _lorenz(n_timesteps: int, rho: float, sigma: float, beta: float, x0, h: float):
    out = np.zeros((n_timesteps, 3), dtype=np.float64)
    out[0, :] = x0
    state = np.array(x0, dtype=np.float64)
    for i in range(1, n_timesteps):
        state = _rk4_system(_lorenz_diff, state, h, rho, sigma, beta)
        out[i, :] = state
    return out


def _lorenz96_diff(state: np.ndarray, f: float):
    n = state.shape[0]
    ds = np.zeros(n, dtype=np.float64)
    for i in range(n):
        ip1 = (i + 1) % n
        im1 = (i + n - 1) % n
        im2 = (i + n - 2) % n
        ds[i] = (state[ip1] - state[im2]) * state[im1] - state[i] + f
    return ds


def _lorenz96(n_timesteps: int, n: int, f: float, df: float, h: float):
    out = np.zeros((n_timesteps, n), dtype=np.float64)
    state = np.full(n, f, dtype=np.float64)
    state[0] = f + df
    out[0, :] = state
    for i in range(1, n_timesteps):
        state = _rk4_system(_lorenz96_diff, state, h, f)
        out[i, :] = state
    return out


def _rossler_diff(state: np.ndarray, a: float, b: float, c: float):
    x, y, z = state
    return np.array([-y - z, x + a * y, b + z * (x - c)], dtype=np.float64)


def _rossler(n_timesteps: int, a: float, b: float, c: float, x0, h: float):
    out = np.zeros((n_timesteps, 3), dtype=np.float64)
    out[0, :] = x0
    state = np.array(x0, dtype=np.float64)
    for i in range(1, n_timesteps):
        state = _rk4_system(_rossler_diff, state, h, a, b, c)
        out[i, :] = state
    return out


def _doublescroll_diff(
    state: np.ndarray, r1: float, r2: float, r4: float, ir: float, beta: float
):
    v1, v2, current = state
    dv = v1 - v2
    factor = (dv / r2) + ir * np.sinh(beta * dv)
    dv1 = (v1 / r1) - factor
    dv2 = factor - current
    di = v2 - r4 * current
    return np.array([dv1, dv2, di], dtype=np.float64)


def _doublescroll(
    n_timesteps: int,
    r1: float,
    r2: float,
    r4: float,
    ir: float,
    beta: float,
    x0,
    h: float,
):
    out = np.zeros((n_timesteps, 3), dtype=np.float64)
    out[0, :] = x0
    state = np.array(x0, dtype=np.float64)
    for i in range(1, n_timesteps):
        state = _rk4_system(_doublescroll_diff, state, h, r1, r2, r4, ir, beta)
        out[i, :] = state
    return out


def _multiscroll_diff(state: np.ndarray, a: float, b: float, c: float):
    x, y, z = state
    return np.array(
        [a * (y - x), (c - a) * x - x * z + c * y, x * y - b * z], dtype=np.float64
    )


def _multiscroll(n_timesteps: int, a: float, b: float, c: float, x0, h: float):
    out = np.zeros((n_timesteps, 3), dtype=np.float64)
    out[0, :] = x0
    state = np.array(x0, dtype=np.float64)
    for i in range(1, n_timesteps):
        state = _rk4_system(_multiscroll_diff, state, h, a, b, c)
        out[i, :] = state
    return out


def _rf_diff(state: np.ndarray, alpha: float, gamma: float):
    x, y, z = state
    return np.array(
        [
            y * (z - 1.0 + x * x) + gamma * x,
            x * (3.0 * z + 1.0 - x * x) + gamma * y,
            -2.0 * z * (alpha + x * y),
        ],
        dtype=np.float64,
    )


def _rabinovich_fabrikant(n_timesteps: int, alpha: float, gamma: float, x0, h: float):
    out = np.zeros((n_timesteps, 3), dtype=np.float64)
    out[0, :] = x0
    state = np.array(x0, dtype=np.float64)
    for i in range(1, n_timesteps):
        state = _rk4_system(_rf_diff, state, h, alpha, gamma)
        out[i, :] = state
    return out


def _mg_rk4(xt: float, xtau: float, a: float, b: float, n: float, h: float):
    bh = -b * h
    k1 = bh * xt + a * xtau / (1.0 + xtau**n)
    k2 = 2.0 * k1 + bh * k1
    k3 = 2.0 * k1 + bh * k2
    k4 = k1 + bh * k3
    return xt + (k1 + k2 + k3 + k4) / 6.0


def _mackey_glass(
    n_timesteps: int, tau: int, a: float, b: float, n: int, x0: float, h: float
):
    history_length = int(np.floor(float(tau) / h))
    xt = x0
    x = np.zeros(history_length + n_timesteps, dtype=np.float64)

    for i in range(history_length, history_length + n_timesteps):
        x[i] = xt
        xtau = x[i - history_length] if (tau > 0 and history_length > 0) else 0.0
        xt = _mg_rk4(xt, xtau, a, b, float(n), h)

    return x[history_length:].reshape(-1, 1)


def _narma(
    n_timesteps: int,
    order: int,
    a1: float,
    a2: float,
    b: float,
    c: float,
    x0,
    u,
):
    y = np.zeros(n_timesteps + order, dtype=np.float64)
    for i, v in enumerate(x0[: y.shape[0]]):
        y[i] = v

    for t in range(order, n_timesteps + order - 1):
        sum_hist = np.sum(y[t - order : t])
        y[t + 1] = a1 * y[t] + a2 * y[t] * sum_hist + b * u[t - order] * u[t] + c

    return y[order:].reshape(-1, 1)


def _ks_etdrk4(
    v: np.ndarray,
    g: np.ndarray,
    e: np.ndarray,
    q: np.ndarray,
    f1: np.ndarray,
    f2: np.ndarray,
    f3: np.ndarray,
):
    n = v.shape[0]
    out = np.zeros(n, dtype=np.float64)
    for i in range(n):
        ip1 = (i + 1) % n
        im1 = (i + n - 1) % n
        im2 = (i + n - 2) % n

        ux = 0.5 * (v[ip1] - v[im1])
        uxx = v[ip1] - 2.0 * v[i] + v[im1]
        uxxxx = v[ip1] - 4.0 * v[i] + 6.0 * v[im1] - 4.0 * v[im2] + v[(i + 2) % n]

        nonlinear = -v[i] * ux
        stiff = -uxx - uxxxx
        nv = g[i] * nonlinear

        out[i] = e[i] * v[i] + q[i] * nv + f1[i] * nv + f2[i] * stiff + f3[i] * stiff
    return out


def _kuramoto_sivashinsky(n_timesteps: int, n: int, m: float, h: float):
    x0 = np.array(
        [
            np.cos((2.0 * m * np.pi * (idx + 1) / n) / m)
            * (1.0 + np.sin((2.0 * m * np.pi * (idx + 1) / n) / m))
            for idx in range(n)
        ],
        dtype=np.float64,
    )

    k = (
        np.array(
            [float(i) if i <= n // 2 else float(i) - float(n) for i in range(n)],
            dtype=np.float64,
        )
        / m
    )
    l = k * k - np.power(k, 4)
    e = np.exp(h * l)
    q = np.full(n, h, dtype=np.float64)
    f1 = np.full(n, h, dtype=np.float64)
    f2 = np.full(n, 0.5 * h, dtype=np.float64)
    f3 = np.full(n, 0.5 * h, dtype=np.float64)
    g = -0.5 * k

    out = np.zeros((n_timesteps, n), dtype=np.float64)
    out[0, :] = x0
    v = x0
    with np.errstate(over="ignore", invalid="ignore"):
        for i in range(1, n_timesteps):
            v = _ks_etdrk4(v, g, e, q, f1, f2, f3)
            out[i, :] = v
    return out


def _series_for_system(system_name: str):
    n_timesteps = 400

    if system_name == "logistic_map":
        return _logistic_map(n_timesteps, 3.9, 0.1)
    if system_name == "henon_map":
        return _henon_map(n_timesteps, 1.4, 0.3, [0.0, 0.0])
    if system_name == "lorenz":
        return _lorenz(
            n_timesteps, 28.0, 10.0, 2.6666666666666665, [1.0, 1.0, 1.0], 0.01
        )
    if system_name == "lorenz96":
        return _lorenz96(n_timesteps, 8, 8.0, 0.01, 0.01)
    if system_name == "rossler":
        return _rossler(n_timesteps, 0.2, 0.2, 5.7, [1.0, 1.0, 1.0], 0.01)
    if system_name == "doublescroll":
        return _doublescroll(
            n_timesteps, 1.2, 3.44, 0.193, 2.25, 11.6, [0.1, 0.0, 0.0], 0.01
        )
    if system_name == "multiscroll":
        return _multiscroll(n_timesteps, 36.0, 3.0, 20.0, [0.1, 0.0, 0.0], 0.01)
    if system_name == "rabinovich_fabrikant":
        return _rabinovich_fabrikant(n_timesteps, 0.14, 0.1, [0.1, 0.1, 0.1], 0.005)
    if system_name == "mackey_glass":
        return _mackey_glass(n_timesteps, 0, 0.2, 0.1, 10, 1.2, 0.1)
    if system_name == "narma":
        order = 10
        u = np.array(
            [(i % 7) * 0.05 for i in range(n_timesteps + order)], dtype=np.float64
        )
        x0 = np.zeros(order, dtype=np.float64)
        return _narma(n_timesteps, order, 0.3, 0.05, 1.5, 0.1, x0, u)
    if system_name == "kuramoto_sivashinsky":
        return _kuramoto_sivashinsky(120, 16, 8.0, 0.25)

    raise ValueError(f"Unknown system_name: {system_name}")


def dynamic_system_dataset_test(system_name: str):
    series = _series_for_system(system_name)
    return _from_series(series)
