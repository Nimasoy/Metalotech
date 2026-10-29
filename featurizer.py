"""
Featurization Module
====================
Converts chemical formulas into physical, chemical, and electronic feature
vectors using the Magpie (Materials-to-Property) preset via Matminer.
Computes 132 composition-derived features including elemental radii,
electronegativity, valence electron counts, ground state volume, and melting temperatures.
"""

import os
import pandas as pd
from pymatgen.core import Composition
from matminer.featurizers.composition import ElementProperty


def featurize_dataset(
    clean_path: str = "bulk_modulus_dataset_clean.csv",
    output_path: str = "bulk_modulus_features.csv"
) -> pd.DataFrame:
    """Generate Magpie feature representations for clean compositions.

    Args:
        clean_path: Path to cleaned dataset CSV.
        output_path: Destination path for featurized CSV.
    """
    if not os.path.exists(clean_path):
        raise FileNotFoundError(f"Cleaned dataset '{clean_path}' not found.")

    print(f"Loading '{clean_path}' for featurization...")
    df = pd.read_csv(clean_path)

    # Convert formulas to pymatgen Composition objects
    df["composition"] = df["formula"].apply(lambda f: Composition(f))

    # Initialize Magpie preset
    featurizer = ElementProperty.from_preset("magpie")
    print(f"Featurizing {len(df)} compositions across {len(featurizer.feature_labels())} Magpie descriptors...")

    df_feats = featurizer.featurize_dataframe(df, col_id="composition", ignore_errors=False)
    df_feats.to_csv(output_path, index=False)
    print(f"Featurization complete! Saved shape {df_feats.shape} to '{output_path}'.")
    return df_feats


if __name__ == "__main__":
    featurize_dataset()
