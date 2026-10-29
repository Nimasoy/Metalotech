"""
Bulk Modulus Predictor Module
=============================
Provides inference capability for predicting polycrystalline Voigt-Reuss-Hill (VRH)
bulk modulus from stoichiometry / chemical formula strings.
"""

import os
import warnings
import joblib
import pandas as pd
from pymatgen.core import Composition
from matminer.featurizers.composition import ElementProperty

warnings.filterwarnings("ignore")

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_MODEL_PATH = os.path.join(_BASE_DIR, "bulk_model.pkl")
_FEAT_COLS_PATH = os.path.join(_BASE_DIR, "feature_columns.pkl")

# Load model artifacts
model = joblib.load(_MODEL_PATH)
feature_columns = joblib.load(_FEAT_COLS_PATH)

# Initialize Magpie composition featurizer
featurizer = ElementProperty.from_preset("magpie")
feature_labels = featurizer.feature_labels()


def predictor(formula_str: str) -> str:
    """Predict the Voigt-Reuss-Hill bulk modulus (GPa) for a chemical formula.

    Args:
        formula_str: Chemical formula (e.g. 'Fe2O3', 'TiC', 'Al2O3', 'C').

    Returns:
        Formatted string containing predicted bulk modulus or error message.
    """
    if not formula_str or not isinstance(formula_str, str):
        return "Error: Please provide a valid chemical formula string."

    formula_str = formula_str.strip()
    try:
        comp = Composition(formula_str)
        if len(comp.elements) == 0:
            return "Error: Formula contains no valid chemical elements."

        # Compute 132 Magpie elemental features directly (synchronous & lightweight)
        feats = featurizer.featurize(comp)
        df = pd.DataFrame([feats], columns=feature_labels)
        df = df[feature_columns]

        prediction = float(model.predict(df)[0])
        return f"Predicted Bulk Modulus (VRH): {prediction:.2f} GPa"
    except Exception as exc:
        return f"Error: Unable to predict for '{formula_str}': {str(exc)}"


if __name__ == "__main__":
    test_compounds = ["C", "TiC", "Fe2O3", "Al2O3", "NaCl", "W", "Au"]
    print("Testing Bulk Modulus Predictor:")
    for formula in test_compounds:
        print(f"  {formula:<8} -> {predictor(formula)}")
