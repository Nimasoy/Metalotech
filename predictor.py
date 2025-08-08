import joblib
from matminer.featurizers.composition import ElementProperty
from pymatgen.core import Composition
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

model = joblib.load("bulk_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")

featurizer = ElementProperty.from_preset("magpie")


def predictor(formula_str):
    try:
        comp = Composition(formula_str)
        df = pd.DataFrame([{"composition": comp}])
        df = featurizer.featurize_dataframe(df, "composition")
        df = df[feature_columns]

        prediction = model.predict(df)[0]
        return f"Predicted Bulk Modulus (VRH): {prediction:.2f} GPa"
    except Exception as e:
        return f"Error: {str(e)}"
