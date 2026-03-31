"""
========
model.py
========

Non-linear Vector Autoregressive (NVAR) model from paper:

    Gauthier, D. J., Bollt, E., Griffith, A., & Barbosa, W. A. S. (2021).
    Next generation reservoir computing.
    Nature Communications, 12(1), 5564.
    https://doi.org/10.1038/s41467-021-25801-2

Author: Nathan Trouvain <nathan.trouvain@inria.fr>
Licence: GNU GENERAL PUBLIC LICENSE v3
Copyright (c) 2022 Nathan Trouvain
"""

import itertools as it

import numpy as np
import scipy.linalg
from mip import Model, minimize, xsum


class NGRC:
    def __init__(self, config):
        self.k = config["model"]["params"]["delay"]
        self.s = config["model"]["params"]["stride"]
        self.p = config["model"]["params"]["poly_order"]
        self.alpha = config["model"]["params"]["ridge_param"]
        self.transients = config["model"]["params"]["transients"]
        self.bias = config["model"]["params"]["bias"]
        self.loss = config["model"]["params"]["loss"]
        self.Wout = None
        self.dim = None

    def fit(self, train_data):
        lin_features, nlin_features, _ = self.__nvar(train_data[:-1])
        dTrain = train_data[1:] - train_data[:-1]
        if self.loss == "mse":
            self.Wout = self.__tikhonov_regression(
                lin_features,
                nlin_features,
                dTrain,
            )
        elif self.loss == "mae":
            self.Wout = self.__lp_solve(
                lin_features,
                nlin_features,
                dTrain,
            )
        else:
            raise ValueError(f"Loss {self.loss} not recognized.")

    def predict(self, lin_features, nlin_features):
        """Use the NVAR features and the learned readout matrix for inference.

        Parameters
        ----------
        Wout : numpy.ndarray
            Readout matrix of shape
            (target_dimension, linear_dimension + non_linear_dimension + bias)
        lin_features : numpy.ndarray
            NVAR linear features of shape (timesteps, linear_dimension)
        nlin_features : numpy.ndarray
            NVAR non-linear features of shape (timesteps, nonlinear_dimension)

        Returns
        -------
            numpy.ndarray
                A predicited signal of shape (timesteps, target_dimensions)
        """

        tot_features = np.c_[lin_features, nlin_features]

        # If bias:
        if self.Wout.shape[1] == tot_features.shape[1] + 1:
            W, bias = self.Wout[:, 1:], self.Wout[:, :1]
        else:
            W, bias = self.Wout, np.zeros((self.Wout.shape[0], 1))

        # Transpose to keep time in first axis.
        return (np.dot(W, tot_features.T) + bias).T

    def forecast(self, context, steps):
        _, _, window = self.__nvar(context[-1 * (self.k) - 1 : -1])
        u = np.atleast_2d(context[-1])
        Y = np.zeros((steps, u.shape[1]))
        for i in range(steps):
            lin_features, nlin_features, window = self.__nvar(u, window=window)
            u = u + self.predict(lin_features, nlin_features)
            Y[i, :] = u
        return Y

    def __nvar(self, X, window=None):
        """Apply Non-linear Vecror Autoregressive model (NVAR) to timeseries.

        Parameters
        ----------
        X : numpy.ndarray
            Multivariate timeseries of shape (timesteps, n_dimensions)
        k : int
            Delay of the NVAR
        s : int
            Strides of the NVAR
        p : int
            Order of the non-linear features (i.e. monomials order)

        Returns
        -------
            numpy.ndarray, numpy.ndarray, numpy.ndarray
                Linear features vector, non-linear features vector,
                last sliding window over the signal.
        """
        if self.k < 1:
            raise ValueError("k should be >= 1.")
        if self.s < 1:
            raise ValueError("s should be >= 1.")
        if self.p < 1:
            raise ValueError("p should be > 0.")
        if X.ndim < 2:
            X = X.reshape(-1, 1)

        # Inputs must be of shape (timesteps, dimension)
        n_steps, n_dim = X.shape

        lin_dim = n_dim * self.k  # Linear features dimension

        # Finding all monomials of order p in lin_features requires finding all
        # unique combinations of p lin_features elements, with replacement.
        lin_idx = np.arange(lin_dim)
        monom_idx = np.array(list(it.combinations_with_replacement(lin_idx, self.p)))

        nlin_dim = monom_idx.shape[0]

        # A sliding window to store all lagged inputs, including discarded ones.
        # By default, the window is initialized with zeros,
        # transient features will have unexpected zeros.
        win_dim = (self.k - 1) * self.s + 1  # lagged window dimension
        if window is None:
            window = np.zeros((win_dim, n_dim))
        else:
            if window.shape != (win_dim, n_dim):
                raise ValueError(
                    f"window must be of shape ({win_dim}, {n_dim}) "
                    f"but is of shape {window.shape}."
                )

        # Linear features and non-linear features vectors.
        lin_features = np.zeros((n_steps, lin_dim))
        nlin_features = np.zeros((n_steps, nlin_dim))

        for i in range(n_steps):
            window = np.roll(window, -1, axis=0)
            window[-1, :] = X[i]

            lin_feat = window[:: self.s, :].flatten()
            nlin_feat = np.prod(lin_feat[monom_idx], axis=1)

            lin_features[i, :] = lin_feat
            nlin_features[i, :] = nlin_feat

        return lin_features, nlin_features, window

    def __lp_solve(self, lin_features, nlin_features, target):
        Y = target[self.transients :]
        tot_features = self.__total_feature(lin_features, nlin_features)
        m = Model()
        Wout_var = m.add_var_tensor(
            (Y.shape[1], tot_features.shape[1]), lb=-5.0, ub=5.0
        )

        z = m.add_var_tensor(Y.shape, lb=0.0)
        for i in range(z.shape[0]):
            for j in range(z.shape[1]):
                m += (
                    Y[i, j]
                    - xsum(
                        Wout_var[j, k] * tot_features[i, k]
                        for k in range(tot_features.shape[1])
                    )
                    <= z[i, j]
                )
                m += (
                    Y[i, j]
                    - xsum(
                        Wout_var[j, k] * tot_features[i, k]
                        for k in range(tot_features.shape[1])
                    )
                    >= -z[i, j]
                )
        m.objective = minimize(
            xsum(z[i, j] for i in range(z.shape[0]) for j in range(z.shape[1]))
        )
        m.optimize()
        Wout = Wout_var.astype(float).to_numpy()
        return Wout

    def __tikhonov_regression(self, lin_features, nlin_features, target):
        """Performs Tikhonov linear regression (with L2 regularization) between
        NVAR features and a target signal to create a readout weight matrix.

        Parameters
        ----------
        lin_features : numpy.ndarray
            NVAR linear features of shape (timesteps, linear_dimension)
        nlin_features : numpy.ndarray
            NVAR non-linear features of shape (timesteps, nonlinear_dimension)
        target : numpy.ndarray
            Target signal, of shape (timesteps, target_dimension)
        alpha : float
            Regularization coefficient
        transients : int
            Number of timesteps to consider as transients (will be discarded before
            linear regression)
        bias : bool
            If True, add a constant term to NVAR features to compute intercept
            during linear regression

        Returns
        -------
            numpy.ndarray
                Readout weights matrix of shape
                (target_dimension, linear_dimension + non_linear_dimension + bias)
        """
        Y = target[self.transients :]

        if Y.ndim < 2:
            Y = Y.reshape(-1, 1)

        tot_features = self.__total_feature(lin_features, nlin_features)

        Wout = np.zeros((Y.shape[1], tot_features.shape[1]))

        # Wout = Y.Otot^T.(Otot.Otot^T + alphaId)^-1
        # (inverted all transpose as we prefer having time on the first axis)
        YXt = np.dot(Y.T, tot_features)
        XXt = np.dot(tot_features.T, tot_features)
        ridge = self.alpha * np.identity(len(XXt), dtype=np.float64)

        Wout[:] = np.dot(YXt, scipy.linalg.pinvh(XXt + ridge))

        return Wout

    def __total_feature(self, lin_features, nlin_features):

        n_steps = len(lin_features) - self.transients

        tot_features = np.c_[lin_features, nlin_features][self.transients :]

        if self.bias:
            c = np.ones((n_steps, 1))
            tot_features = np.c_[c, tot_features]

        return np.c_[lin_features, nlin_features]
