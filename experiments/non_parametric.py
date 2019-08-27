from jax.experimental.stax import Dense, Dropout, Tanh, Relu, randn
from jax.experimental import optimizers
import pandas as pd
import lifelike.losses as losses
from lifelike import Model
from lifelike.callbacks import *
from datasets.loaders import *


x_train, t_train, e_train = get_generated_churn_dataset()

model = Model([Dense(8), Relu, Dense(12), Relu, Dense(16), Relu])

model.compile(
    optimizer=optimizers.adam,
    optimizer_kwargs={"step_size": optimizers.exponential_decay(0.001, 1, 0.9995)},
    l2=0.01,
    loss=losses.NonParametric()
)

model.fit(
    x_train,
    t_train,
    e_train,
    epochs=10000,
    batch_size=50,
    validation_split=0.1,
    callbacks=[
        Logger(report_every_n_epochs=1),
        EarlyStopping(rdelta=1.),
        TerminateOnNaN(),
        PlotSurvivalCurve(individuals=[35, 45, 46, 47, 68], update_every_n_epochs=500),
        ModelCheckpoint("testsavefile.pickle", prefix_timestamp=False, save_every_n_epochs=200)
    ],
)
