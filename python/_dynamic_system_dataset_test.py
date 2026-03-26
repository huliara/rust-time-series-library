import numpy as np

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


def _series_for_system(system_name: str):
    n_timesteps = 400

    if system_name == "logistic_map":
        return logistic_map(n_timesteps, r=3.9, x0=0.1)
    if system_name == "henon_map":
        return henon_map(n_timesteps, a=1.4, b=0.3, x0=[0.0, 0.0])
    if system_name == "lorenz":
        return lorenz(
            n_timesteps,
            rho=28.0,
            sigma=10.0,
            beta=2.6666666666666665,
            x0=[1.0, 1.0, 1.0],
            h=0.01,
        )

    if system_name == "lorenz96":
        return lorenz96(n_timesteps, warmup=0, N=8, F=8.0, dF=0.01, h=0.01, x0=None)

    if system_name == "rossler":
        return rossler(n_timesteps, a=0.2, b=0.2, c=5.7, x0=[1.0, 1.0, 1.0], h=0.01)

    if system_name == "doublescroll":
        return doublescroll(
            n_timesteps,
            r1=1.2,
            r2=3.44,
            r4=0.193,
            ir=2.25,
            beta=11.6,
            x0=[0.1, 0.0, 0.0],
            h=0.01,
        )

    if system_name == "multiscroll":
        return multiscroll(
            n_timesteps, a=36.0, b=3.0, c=20.0, x0=[0.1, 0.0, 0.0], h=0.01
        )

    if system_name == "rabinovich_fabrikant":
        return rabinovich_fabrikant(
            n_timesteps,
            alpha=0.14,
            gamma=0.1,
            x0=[0.1, 0.1, 0.1],
            h=0.005,
        )

    if system_name == "mackey_glass":
        return mackey_glass(
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
        return y
    if system_name == "kuramoto_sivashinsky":
        return kuramoto_sivashinsky(120, warmup=0, N=16, M=8.0, x0=None, h=0.25)

    raise ValueError(f"Unknown system_name: {system_name}")


def dynamic_system_dataset_test(system_name: str):
    return _series_for_system(system_name).flatten().tolist()


if __name__ == "__main__":
    system_name = "lorenz"
    series = dynamic_system_dataset_test(system_name)
    print(f"{system_name} series length: {len(series)}")
