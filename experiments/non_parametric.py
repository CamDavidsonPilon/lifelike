from jax.experimental.stax import Dense, Dropout, Tanh, Relu, randn
from jax.experimental import optimizers
import pandas as pd
import lifelike.losses as losses
from lifelike import Model
from lifelike.callbacks import *

def get_generated_churn_dataset():
    df = pd.read_csv("experiments/churn.csv")

    T = df.pop("T").values
    E = df.pop("E").values
    X = df.values

    return X, T, E

def get_telcom_churn_dataset():

    churn_data = pd.read_csv('https://raw.githubusercontent.com/'
                             'treselle-systems/customer_churn_analysis/'
                             'master/WA_Fn-UseC_-Telco-Customer-Churn.csv')

    churn_data = churn_data.set_index('customerID')
    churn_data = churn_data.drop(['TotalCharges'], axis=1)

    churn_data = churn_data.applymap(lambda x: "No" if str(x).startswith("No ") else x)

    df = pd.get_dummies(churn_data,
                        columns=churn_data.columns.difference(['tenure', 'MonthlyCharges']),
                        drop_first=True)

    df = df[df['tenure'] > 0]
    T = df.pop("tenure").values
    E = df.pop("Churn_Yes").values
    X = df.values


    X = (X - X.mean(0)) / X.std(0)

    return X, T, E


def get_colon_dataset():
    from patsy import dmatrix

    df = pd.read_csv("experiments/colon.csv", index_col=0).dropna()

    model_string = """{extent} +
        {rx} +
        {differ} +
        sex + age + obstruct + perfor + adhere + nodes + node4 + surg +
        time + status""".format(
        rx="C(rx, Treatment('Obs'))",
        differ="C(differ, Treatment(1))",
        extent="C(extent, Treatment(1))",
    )
    df = dmatrix(model_string, df, return_type="dataframe")

    T_train = df.pop("time").values
    E_train = df.pop("status").values
    X_train = df.values

    return X_train, T_train, E_train


def get_rossi_dataset():
    from lifelines.datasets import load_rossi

    df = load_rossi()

    T_train = df.pop("week").values
    E_train = df.pop("arrest").values
    X_train = df.values

    return X_train, T_train, E_train


x_train, t_train, e_train = get_generated_churn_dataset()

model = Model([Dense(8), Relu])

model.compile(
    optimizer=optimizers.adam,
    optimizer_kwargs={"step_size": optimizers.exponential_decay(0.001, 1, 0.9995)},
    l2=0.001,
    loss=losses.NonParametric(n_breakpoints=12),
)

model.fit(
    x_train,
    t_train,
    e_train,
    epochs=10000,
    batch_size=50,
    validation_split=0.2,
    callbacks=[
        Logger(report_every_n_epochs=1),
        EarlyStopping(rdelta=1.),
        TerminateOnNaN(),
        PlotSurvivalCurve(individuals=[35,45,46], update_every_n_epochs=1),
        ModelCheckpoint("testsavefile.pickle", prefix_timestamp=False, save_every_n_epochs=200)
    ],
)
