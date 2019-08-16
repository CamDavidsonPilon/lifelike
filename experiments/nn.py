from __future__ import absolute_import, division
from __future__ import print_function
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm

import pandas as pd
from patsy import dmatrix
import numpy as onp
import numpy.random as npr

import jax.numpy as np
from jax import jit, grad, random
from jax.experimental import optimizers
from jax.experimental import stax
from jax.experimental.stax import Dense, Tanh, Exp, randn


BREAKPOINTS = np.concatenate(
    (np.logspace(4, 8, 35, base=np.e), np.array([np.inf]))
)


init_random_params, predict = stax.serial(
    Dense(22, W_init=randn(1e-7)), Tanh,
    Dense(len(BREAKPOINTS), W_init=randn(1e-7)), Exp
)

def _survival_function(params, T):
    return np.exp(-_cumulative_hazard(params, T))

def tiles(n):
    return np.tile(BREAKPOINTS, (n, 1))

def _cumulative_hazard(params, T):
    n = T.shape[0]
    T = T.reshape((n, 1))
    M = np.minimum(tiles(n), T)
    M = np.hstack([M[:, tuple([0])], np.diff(M, axis=1)])
    return (M * params).sum(1)

def _hazard(params, T):
    n = T.shape[0]
    T = T.reshape((n, 1))
    tiles = np.tile(BREAKPOINTS[:-1], (n, 1))

    ix = np.argmin(np.maximum(tiles, T) - tiles, 1)
    #ix = onp.searchsorted(BREAKPOINTS, T)
    # watching issue #1142
    return params[np.arange(n), ix]

def _log_hazard(params, T):
    return np.log(_hazard(params, T))

def log_likelihood(params, T, E):
    n = params.shape[0]
    cum_hz = _cumulative_hazard(params, T)
    log_hz = _log_hazard(params, T)
    ll = 0
    ll = ll + (E * log_hz).sum()
    ll = ll + -(cum_hz).sum()
    return ll / n

def get_dataset():
    df = pd.read_csv("colon.csv", index_col=0).dropna()
    df = df[df["etype"] == 2]


    model_string = """{extent} +
        {rx} +
        {differ} +
        sex + age + obstruct + perfor + adhere + nodes + node4 + surg +
        time + status
    """.format(
        rx="C(rx, Treatment('Obs'))", differ="C(differ, Treatment(1))", extent="C(extent, Treatment(1))"
    )
    df = dmatrix(model_string, df, return_type="dataframe")


    df = df.sample(frac=1., random_state=npr.RandomState(0))
    df_test, df_train = df.iloc[:100], df.iloc[100:]

    T_test = df_test.pop('time').values
    E_test = df_test.pop('status').values
    X_test = df_test.values

    T_train = df_train.pop('time').values
    E_train = df_train.pop('status').values
    X_train = df_train.values

    return X_test, T_test, E_test, X_train, T_train, E_train




if __name__ == '__main__':
    # Model parameters
    L2_reg = .0001
    rng = random.PRNGKey(0)

    # Training parameters
    num_epochs = 100000
    step_size = 0.005


    X_test, T_test, E_test, X_train, T_train, E_train = get_dataset()
    input_shape = X_train.shape[1]

    opt_init, opt_update, get_weights = optimizers.adam(step_size)

    @jit
    def loss(weights, batch):
        X, T, E = batch
        params = predict(weights, X)
        return -log_likelihood(params, T, E) + L2_reg * optimizers.l2_norm(weights)

    @jit
    def update(i, opt_state, batch):
       weights = get_weights(opt_state)
       return opt_update(i, grad(loss)(weights, batch), opt_state)


    _, init_params = init_random_params(rng, (-1, input_shape)) # why -1?
    opt_state = opt_init(init_params)

    plt.ion()
    colors=iter(cm.rainbow(np.linspace(0, 1, num_epochs / 100)))

    print("\nStarting training...")

    for epoch in range(num_epochs):

        opt_state = update(epoch, opt_state, (X_train, T_train, E_train))

        if epoch % 50 == 0:
            weights = get_weights(opt_state)
            train_acc = loss(weights, (X_train, T_train, E_train))
            test_acc = loss(weights, (X_test, T_test, E_test))
            times = np.linspace(1, 3200, 3200)
            y = _survival_function(predict(weights, X_train[[1]]), times)
            plt.plot(times, y, c=next(colors), alpha=0.15)
            plt.draw()
            plt.pause(0.0001)
            print("Training set accuracy {}".format(train_acc))
            print("Test set accuracy {}".format(test_acc))
