"""
Model Trainer Module
====================
Trains and evaluates regression architectures (Random Forest Regressor) for predicting
polycrystalline bulk modulus from Magpie elemental descriptors.
Computes comprehensive metrics: R^2, Root Mean Squared Error (RMSE), and Mean Absolute Error (MAE).
Saves serialized model and feature column definitions via joblib.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def train_and_save_model(
    features_path: str = "bulk_modulus_features.csv",
    model_path: str = "bulk_model.pkl",
    columns_path: str = "feature_columns.pkl",
    random_state: int = 42,
    test_size: float = 0.2
) -> RandomForestRegressor:
    """Train Random Forest regressor on Magpie features and persist serialized artifacts."""
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Features file '{features_path}' not found.")

    print(f"Loading featurized dataset from '{features_path}'...")
    df_feats = pd.read_csv(features_path)

    # Separate target property and metadata columns
    drop_cols = ["formula", "composition", "bulk_modulus_vrh", "bulk_modulus_voigt", "bulk_modulus_reuss"]
    feature_cols = [c for c in df_feats.columns if c not in drop_cols]

    X = df_feats[feature_cols].copy()
    y = df_feats["bulk_modulus_vrh"].copy()

    # Fill any potential NaN values with feature column means
    X = X.fillna(X.mean())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"Dataset split: {len(X_train)} training samples, {len(X_test)} test samples ({test_size*100:.0f}% holdout).")

    # Fit ensemble regressor
    print("Fitting RandomForestRegressor (n_estimators=100, criterion='squared_error')...")
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Predictions & Evaluation
    y_pred_tr = model.predict(X_train)
    y_pred_te = model.predict(X_test)

    r2_tr = r2_score(y_train, y_pred_tr)
    r2_te = r2_score(y_test, y_pred_te)
    rmse_tr = float(np.sqrt(mean_squared_error(y_train, y_pred_tr)))
    rmse_te = float(np.sqrt(mean_squared_error(y_test, y_pred_te)))
    mae_tr = float(mean_absolute_error(y_train, y_pred_tr))
    mae_te = float(mean_absolute_error(y_test, y_pred_te))

    print("\n" + "=" * 50)
    print(" MODEL PERFORMANCE EVALUATION (BULK MODULUS VRH)")
    print("=" * 50)
    print(f"  Training Split : R^2 = {r2_tr:.4f} | RMSE = {rmse_tr:.2f} GPa | MAE = {mae_tr:.2f} GPa")
    print(f"  Testing Split  : R^2 = {r2_te:.4f} | RMSE = {rmse_te:.2f} GPa | MAE = {mae_te:.2f} GPa")
    print("=" * 50)

    # Inspect top features
    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\nTop 5 Most Informative Physical Descriptors:")
    for feat, imp in importances.head(5).items():
        print(f"  - {feat}: {imp*100:.2f}%")

    # Persist model and column order
    joblib.dump(model, model_path)
    joblib.dump(feature_cols, columns_path)
    print(f"\nModel saved to '{model_path}' ({os.path.getsize(model_path) / 1024**2:.1f} MB)")
    print(f"Feature columns saved to '{columns_path}' ({len(feature_cols)} descriptors)")

    return model


if __name__ == "__main__":
    train_and_save_model()
