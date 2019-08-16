from jax.experimental.stax import Dense, Dropout, Tanh
from jax.experimental import optimizers

import lifelike.losses as losses
from lifelike import Model
from lifelike.callbacks import Logger



def get_dataset():
    import pandas as pd
    from patsy import dmatrix
    df = pd.read_csv("experiments/colon.csv", index_col=0).dropna()
    df = df[df["etype"] == 2]


    model_string = """{extent} +
        {rx} +
        {differ} +
        sex + age + obstruct + perfor + adhere + nodes + node4 + surg +
        time + status""".format(rx="C(rx, Treatment('Obs'))", differ="C(differ, Treatment(1))", extent="C(extent, Treatment(1))")
    df = dmatrix(model_string, df, return_type="dataframe")

    T_train = df.pop('time').values
    E_train = df.pop('status').values
    X_train = df.values

    return X_train, T_train, E_train


x_train, t_train, e_train = get_dataset()


model = Model([
    Dense(20), Tanh,
])

model.compile(optimizer=optimizers.adam, optimizer_kwargs={'step_size': 0.01},
              loss=losses.NonParametric())

model.fit(x_train, t_train, e_train,
    epochs=10000,
    batch_size=32,
    callbacks=[Logger(50)]
)
