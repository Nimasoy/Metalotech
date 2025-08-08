from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd
import joblib


def train_and_save_model():
    df_feats = pd.read_csv("bulk_modulus_features.csv")

    X = df_feats.drop(
        columns=["formula", "composition", "bulk_modulus_vrh", "bulk_modulus_voigt", "bulk_modulus_reuss"])
    y = df_feats["bulk_modulus_vrh"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=46)

    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    print("R^2 score: ", r2_score(y_test, model.predict(X_test)))

    joblib.dump(model, "bulk_model.pkl")
    joblib.dump(list(X.columns), "feature_columns.pkl")


if __name__ == "__main__":
    train_and_save_model()
