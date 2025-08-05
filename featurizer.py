from matminer.featurizers.composition import ElementProperty
from pymatgen.core import Composition
import pandas as pd

if __name__ == "__main__":
    df = pd.read_csv("bulk_modulus_dataset_clean.csv")
    df["composition"] = df["formula"].apply(lambda x: Composition(x))

    featurizer = ElementProperty.from_preset("magpie")
    df_feats = featurizer.featurize_dataframe(df, col_id="composition")

    df_feats.to_csv("bulk_modulus_features.csv", index=False)
    print("Featurization complete.")
