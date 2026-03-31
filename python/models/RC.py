import reservoirpy as rpy
from reservoirpy.nodes import Reservoir, Ridge
import numpy as np

rpy.set_seed(42)


class RC:
    def __init__(self, config):
        self.config = config
        self.reservoir = Reservoir(
            units=config["model"]["params"]["units"],
            lr=config["model"]["params"]["leak_rate"],
            sr=config["model"]["params"]["spectral_radius"],
        )
        self.ridge = Ridge(ridge=config["model"]["params"]["ridge"])
        self.esn_model = self.reservoir >> self.ridge

    def fit(self, train_data, warmup=100):
        X_train = train_data[:-1]
        Y_train = train_data[1:]
        self.esn_model = self.esn_model.fit(
            X_train,
            Y_train,
            warmup,
        )

    def forecast(self, X, steps):
        pred = np.empty((steps, X.shape[1]))
        self.reservoir.reset()
        self.reservoir.run(X)
        x = np.atleast_2d(X[-1])
        for i in range(steps):
            x = self.esn_model.predict(x)
            pred[i] = x
        return pred
