# save test

from jax.experimental.stax import Dense, Dropout, Tanh, Relu, randn
from jax.experimental import optimizers
import numpy as np

import lifelike.losses as losses
from lifelike import Model
from lifelike.utils import dump, load
from lifelike.callbacks import *


def get_rossi_dataset():
    from lifelines.datasets import load_rossi

    df = load_rossi()

    T_train = df.pop("week").values
    E_train = df.pop("arrest").values
    X_train = df.values

    return X_train, T_train, E_train


x_train, t_train, e_train = get_rossi_dataset()


model = Model([Dense(18), Relu])

model.compile(
    optimizer=optimizers.adam,
    optimizer_kwargs={"step_size": optimizers.exponential_decay(0.01, 10, 0.999)},
    loss=losses.NonParametric(),
)

model.fit(x_train, t_train, e_train, epochs=2, batch_size=32)

print(model.predict_survival_function(x_train[0], np.arange(0, 10)))

dump(model, "testsavefile")
model = load("testsavefile")


print(model.predict_survival_function(x_train[0], np.arange(0, 10)))

model.fit(
    x_train,
    t_train,
    e_train,
    epochs=100,
    callbacks=[
        Logger(report_every_n_epochs=5),
        ModelCheckpoint("experiments/saved_model.pickle", save_every_n_epochs=500),
    ],
)
