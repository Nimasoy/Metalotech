"""
Dataset Cleaner Module
======================
Filters raw DFT elasticity records from Materials Project:
- Drops missing values (NaN)
- Removes unphysical non-positive or extreme (>1000 GPa) bulk moduli
- Validates that chemical formulas are valid pymatgen Compositions
"""

import os
import pandas as pd
from pymatgen.core import Composition


def clean_dataset(
    raw_path: str = "bulk_modulus_dataset.csv",
    clean_path: str = "bulk_modulus_dataset_clean.csv"
) -> pd.DataFrame:
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw dataset '{raw_path}' not found.")

    df = pd.read_csv(raw_path)
    initial_count = len(df)
    print(f"Loaded raw dataset with {initial_count} rows.")

    # Drop missing values
    df = df.dropna()

    # Filter physical elasticity bounds (0 < K < 1000 GPa)
    for col in ["bulk_modulus_voigt", "bulk_modulus_reuss", "bulk_modulus_vrh"]:
        if col in df.columns:
            df = df[(df[col] > 0) & (df[col] < 1000)]

    # Validate pymatgen composition parsing
    def is_valid_formula(formula: str) -> bool:
        try:
            comp = Composition(formula)
            return len(comp.elements) > 0
        except Exception:
            return False

    df = df[df["formula"].apply(is_valid_formula)]
    final_count = len(df)
    dropped = initial_count - final_count

    df.to_csv(clean_path, index=False)
    print(f"Cleaned dataset saved to '{clean_path}'.")
    print(f"Rows retained: {final_count} (dropped {dropped} unphysical/invalid rows).")
    return df


if __name__ == "__main__":
    clean_dataset()
