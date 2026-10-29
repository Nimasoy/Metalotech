"""
Smoke & Validation Tests for Metalotech
======================================
Verifies model artifact integrity, Magpie descriptor generation,
and prediction sanity bounds on reference materials.
"""

import os
import joblib
import unittest
import predictor as p


class TestMetalotechPipeline(unittest.TestCase):
    def setUp(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(self.base_dir, "bulk_model.pkl")
        self.cols_path = os.path.join(self.base_dir, "feature_columns.pkl")

    def test_artifacts_exist(self):
        self.assertTrue(os.path.exists(self.model_path), "bulk_model.pkl missing")
        self.assertTrue(os.path.exists(self.cols_path), "feature_columns.pkl missing")

    def test_feature_columns_count(self):
        cols = joblib.load(self.cols_path)
        self.assertEqual(len(cols), 132, "Expected 132 Magpie feature descriptors")

    def test_known_compound_predictions(self):
        # Incompressible covalent/metallic carbide: WC or TiC should be high modulus (> 150 GPa)
        tic_pred = p.predictor("TiC")
        self.assertIn("Predicted Bulk Modulus (VRH)", tic_pred)
        tic_val = float(tic_pred.split(":")[1].replace("GPa", "").strip())
        self.assertGreater(tic_val, 150.0)

        # Soft ionic solid: NaCl should be low modulus (< 50 GPa)
        nacl_pred = p.predictor("NaCl")
        self.assertIn("Predicted Bulk Modulus (VRH)", nacl_pred)
        nacl_val = float(nacl_pred.split(":")[1].replace("GPa", "").strip())
        self.assertLess(nacl_val, 50.0)

    def test_invalid_formula_handling(self):
        err = p.predictor("InvalidElement123$$$")
        self.assertTrue(err.startswith("Error"), "Should gracefully handle invalid formulas")

        empty = p.predictor("")
        self.assertTrue(empty.startswith("Error"), "Should gracefully handle empty formula")


if __name__ == "__main__":
    unittest.main()