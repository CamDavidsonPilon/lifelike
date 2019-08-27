import pandas as pd

def get_generated_churn_dataset():
    df = pd.read_csv("datasets/churn.csv")

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

    df = pd.read_csv("datasets/colon.csv", index_col=0).dropna()

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