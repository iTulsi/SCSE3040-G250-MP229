from sklearn.metrics import mean_absolute_error

TARGET = "delivery_days"


def mae(model, df, features):
    return float(mean_absolute_error(df[TARGET], model.predict(df[features])))


def mae_by_destination(model, df, features):
    return {
        name: round(mae(model, group, features), 4)
        for name, group in df.groupby("destination")
    }
