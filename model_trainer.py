from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd

df_feats = pd.read_csv("bulk_modulus_features.csv")

X = df_feats.drop(columns=["formula", "composition", "bulk_modulus_vrh"])
y = df_feats["bulk_modulus_vrh"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=46)

model = RandomForestRegressor()
model.fit(X_train, y_train)

print("R^2 score: ", r2_score(y_test, model.predict(X_test)))
