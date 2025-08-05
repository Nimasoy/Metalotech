import pandas as pd

df = pd.read_csv("bulk_modulus_dataset_clean.csv")

df = df.dropna()

for col in ["bulk_modulus_voigt", "bulk_modulus_reuss", "bulk_modulus_vrh"]:
    df = df[(df[col] > 0) & (df[col] < 1000)]

df.to_csv("bulk_modulus_dataset_clean.csv", index=False)
print(f"Cleaned dataset saved. Rows left: {len(df)}")
