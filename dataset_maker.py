from mp_api.client import MPRester
import pandas as pd
import time

chunk_size = 500
max_records = 5000
data = []

with MPRester(api_key="7PvbUyirU0wxBfJT21HhAQmV6Dx4mJm4") as mpr:
    results = mpr.materials.summary.search(
        deprecated=False,
        fields=["material_id", "formula_pretty", "bulk_modulus"],
        chunk_size=chunk_size,
        )

    count = 0
    for r in results:
        if r.bulk_modulus is not None:
            data.append({
                "formula": r.formula_pretty,
                "bulk_modulus_voigt": r.bulk_modulus.get("voigt"),
                "bulk_modulus_reuss": r.bulk_modulus.get("reuss"),
                "bulk_modulus_vrh": r.bulk_modulus.get("vrh")
            })
            count += 1
            if count >= max_records:
                break

            if count % chunk_size == 0:
                print(f"Collected {count} records...")
                time.sleep(1)

df = pd.DataFrame(data)
df.to_csv("bulk_modulus_dataset.csv", index=False)
print(f"Done. Total rows collected: {len(df)}")
