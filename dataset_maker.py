"""
Dataset Acquisition Module
==========================
Queries the Materials Project REST API (via mp-api) for DFT-calculated
elastic tensor properties and Voigt-Reuss-Hill polycrystalline bulk moduli.
"""

import os
import time
import pandas as pd
from mp_api.client import MPRester
from dotenv import load_dotenv

load_dotenv()


def fetch_bulk_modulus_dataset(
    output_path: str = "bulk_modulus_dataset.csv",
    chunk_size: int = 500,
    max_records: int = 5000,
    api_key: str = None
) -> pd.DataFrame:
    """Fetch elasticity data from Materials Project API.

    Args:
        output_path: Destination CSV filepath.
        chunk_size: Pagination chunk size for MP API.
        max_records: Maximum number of elasticity records to query.
        api_key: Materials Project API key (falls back to MP_API_KEY env var).
    """
    key = api_key or os.getenv("MP_API_KEY", "7PvbUyirU0wxBfJT21HhAQmV6Dx4mJm4")
    if not key:
        raise ValueError(
            "Materials Project API key is required. Set MP_API_KEY environment variable "
            "or obtain one at https://materialsproject.org/api"
        )

    print(f"Connecting to Materials Project API (target records: {max_records})...")
    data = []

    with MPRester(api_key=key) as mpr:
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
                    print(f"Collected {count}/{max_records} records...")
                    time.sleep(1)

    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"Done. Collected {len(df)} records and saved to '{output_path}'.")
    return df


if __name__ == "__main__":
    fetch_bulk_modulus_dataset()
