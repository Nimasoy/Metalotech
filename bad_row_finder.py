import pandas as pd
from pymatgen.core import Composition

df = pd.read_csv("bulk_modulus_dataset_clean.csv")

bad_rows = []
for i, f in enumerate(df["formula"]):
    try:
        Composition(f)
    except:
        bad_rows.append((i, f))

print("Bad formulas:", bad_rows)
