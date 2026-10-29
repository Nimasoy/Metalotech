<div align="center">

# ⚡ Metalotech
### Machine Learning for High-Throughput Materials Elasticity Prediction

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Framework: Matminer](https://img.shields.io/badge/materials-matminer%20%7C%20pymatgen-orange)](https://hackingmaterials.lbl.gov/matminer/)

</div>

---

## 📌 Executive Summary

**Metalotech** is a machine learning pipeline developed to predict the **Polycrystalline Bulk Modulus ($K_{\text{VRH}}$)** of solid-state materials directly from stoichiometric chemical compositions. By combining Density Functional Theory (DFT) data from the [Materials Project](https://materialsproject.org/) with physical-chemical feature representations ([Magpie](https://doi.org/10.1038/npjcompumats.2016.28)) and ensemble learning, Metalotech delivers instantaneous, sub-second mechanical property estimates, circumventing the massive computational cost of *ab initio* stress-strain tensor simulations.

---

## 🔬 Physical Problem Formulation & Elasticity Theory

### 1. Bulk Modulus ($K$)
The bulk modulus measures a material's intrinsic thermodynamic resistance to isotropic volumetric compression under hydrostatic pressure:

$$K = -V \left(\frac{\partial P}{\partial V}\right)_T = V \left(\frac{\partial^2 E}{\partial V^2}\right)$$

In high-pressure physics, structural metallurgy, and solid-state chemistry, $K$ governs acoustic phonon velocities, shock-wave resistance, interatomic bond stiffness, and hardness.

### 2. Polycrystalline Voigt-Reuss-Hill (VRH) Approximation
For single crystals, elasticity is described by the fourth-rank elastic stiffness tensor $C_{ijkl}$ ($6 \times 6$ Voigt matrix). For untextured polycrystalline aggregates, effective bulk moduli are bounded by:
- **Voigt Bound ($K_V$)** (isostrain assumption):
  $$K_V = \frac{1}{9} \sum_{i=1}^3 \sum_{j=1}^3 C_{ij}$$
- **Reuss Bound ($K_R$)** (isostress assumption):
  $$\frac{1}{K_R} = \sum_{i=1}^3 \sum_{j=1}^3 S_{ij} \quad (S = C^{-1})$$
- **Hill Average ($K_{\text{VRH}}$)**:
  $$K_{\text{VRH}} = \frac{K_V + K_R}{2}$$

The Voigt-Reuss-Hill bulk modulus $K_{\text{VRH}}$ serves as the target regression variable in this project.

---

## 🧠 Feature Representation & Mathematical Architecture

### 1. Stoichiometric Feature Space (Magpie Descriptors)
Rather than requiring expensive 3D relaxed crystal coordinates, the input space maps chemical stoichiometry to a **132-dimensional feature vector** using the *Materials-to-Property* (Magpie) preset (Ward et al., 2016) implemented via `matminer`:

$$\mathbf{x} = \left[ f_1(\mathbf{w}, \mathbf{p}_1), f_2(\mathbf{w}, \mathbf{p}_2), \dots, f_D(\mathbf{w}, \mathbf{p}_K) \right] \in \mathbb{R}^{132}$$

where $\mathbf{w} = (w_1, \dots, w_m)$ are stoichiometric fractions ($\sum w_i = 1$) and $\mathbf{p}_k$ are elemental properties across:
- **Electronic Structure**: Valence electron counts ($s, p, d, f$ shells), unfilled valence orbitals ($N_{\text{unfilled}}$), band gap.
- **Thermodynamic & Cohesive Properties**: Elemental melting temperatures ($T_m$), cohesive energies, Mendeleev numbers.
- **Steric & Geometric Descriptors**: Ground-state atomic volumes ($V_{\text{pa}}$), covalent/ionic radii, electronegativity differences.

For each elemental property $P$, 6 statistical aggregations are computed:
$$\text{Mean}(\mu) = \sum_{i} w_i P_i, \quad \text{Dev}(\Delta) = \sum_{i} w_i |P_i - \mu|, \quad \text{Min}, \quad \text{Max}, \quad \text{Range}, \quad \text{Mode}$$

### 2. Random Forest Regressor & Optimization
The predictive core employs an ensemble of $B = 100$ decorrelated decision trees:

$$\hat{y}(\mathbf{x}) = \frac{1}{B} \sum_{b=1}^{B} T_b(\mathbf{x}; \Theta_b)$$

- **Splitting Criterion**: Minimization of mean squared error (variance reduction):
  $$\mathcal{Q}(S, \theta) = \frac{|S_L|}{|S|} \text{Var}(y_{S_L}) + \frac{|S_R|}{|S|} \text{Var}(y_{S_R})$$
- **Bagging & Random Subspace**: Each tree $T_b$ is trained on a bootstrap sample $S^{(b)} \subset S$, choosing among $m = \sqrt{D}$ random feature subsets per node split to minimize generalization error:
  $$\text{PE}^* \le \bar{\rho} \frac{1 - s^2}{s^2} \sigma^2$$
  where $\bar{\rho}$ is tree correlation and $s$ is individual tree strength.

### 3. Physical Interpretability & Feature Attribution
Feature importance analysis reveals strong physical consistency with condensed matter physics:
- **Mean Ground-State Volume per Atom (`GSvolume_pa`) ~42.6% Importance**: Directly aligns with the Birch-Murnaghan Equation of State ($K_0 \propto V_0^{-\gamma}$). Tightly packed unit cells with smaller atomic volume exhibit steeper repulsive potentials and higher bulk modulus.
- **Maximum & Mean Melting Temperature (`MeltingT`) ~19.0% Combined Importance**: Reflects deep cohesive energy wells and strong covalent/metallic bonding.
- **Unfilled Orbitals & Electronegativity ~8.5% Importance**: Captures directional bond polarity and covalent hybridization.

---

## 📊 Empirical Benchmarks & Model Comparison

To validate model selection, multiple regressors were evaluated under an identical 80/20 train/test split on 4,830 curated materials:

| Architecture / Regressor | Optimization / Loss | Train $R^2$ | Test $R^2$ | Test RMSE (GPa) | Test MAE (GPa) | Inference Time |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Ridge Regression (L2)** | MSE + $\lambda \|\mathbf{w}\|_2^2$ | 0.7406 | 0.7339 | 46.03 | 31.61 | < 1 ms |
| **Multi-Layer Perceptron (MLP)** | Adam, ReLU, early stopping | 0.8897 | 0.8605 | 33.33 | 19.54 | ~ 5 ms |
| **Gradient Boosting Regressor** | Friedman MSE, stagewise additive | 0.9299 | 0.8703 | 32.14 | 18.23 | ~ 2 ms |
| **Random Forest Regressor (Selected)** | Bagging, Variance Reduction | **0.9419** | **0.8723** | **31.89** | **16.61** | **< 3 ms** |

> **Key Takeaway**: Non-linear tree ensembles effectively capture intricate electronic-structure interactions that linear models miss, providing an $R^2 \approx 0.87$ with low mean absolute error (MAE $\approx 16.6$ GPa) across high dynamic range materials ($0.04$ to $508$ GPa).

---

## 📁 Repository Structure

```plaintext
metalotech/
├── app.py                      # Interactive Gradio web application
├── predictor.py                # High-speed single/batch inference engine
├── model_trainer.py            # Model training, metric evaluation, artifact export
├── featurizer.py               # Magpie elemental descriptor generator (via Matminer)
├── dataset_cleaner.py          # Data sanitation, bound filtering, composition validation
├── dataset_maker.py            # Materials Project API client (mp-api) for DFT data collection
├── bad_row_finder.py           # Integrity diagnostic utility for chemical formulas
├── test.py                     # Automated unit test suite & validation checks
├── requirements.txt            # Reproducible dependency specification
├── .gitignore                  # Git exclusions (caches, environments, IDE artifacts)
├── bulk_model.pkl              # Serialized trained RandomForestRegressor
├── feature_columns.pkl         # Serialized 132 Magpie feature schema
├── bulk_modulus_dataset.csv    # Raw Materials Project DFT dataset (5,000 compounds)
├── bulk_modulus_dataset_clean.csv # Cleaned dataset (4,830 validated compounds)
└── bulk_modulus_features.csv   # Featurized dataset (4,830 × 137 matrix)
```

---

## 🚀 Getting Started

### 1. Installation
Clone the repository and install dependencies in a virtual environment:

```bash
git clone https://github.com/Nimasoy/metalotech.git
cd metalotech

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install required dependencies
pip install -r requirements.txt
```

### 2. Verify Pipeline Integrity
Run the automated test suite:

```bash
python test.py
```

### 3. Quick Prediction via Python
```python
import predictor as p

# Predict bulk modulus for any stoichiometry
print(p.predictor("TiC"))    # Titanium Carbide -> ~225 GPa
print(p.predictor("Al2O3"))  # Alumina -> ~172 GPa
print(p.predictor("Fe2O3"))  # Hematite -> ~185 GPa
print(p.predictor("NaCl"))   # Salt -> ~23 GPa
```

### 4. Launch the Interactive Web GUI
```bash
python app.py
```
Open `http://127.0.0.1:7860` in your browser to interactively explore materials.

---

## 🔄 End-to-End Pipeline Reproduction

To re-run the full data acquisition, featurization, and model training pipeline from scratch:

1. **Acquire DFT Data**:
   ```bash
   export MP_API_KEY="your_materials_project_api_key"
   python dataset_maker.py
   ```
2. **Clean & Sanitize**:
   ```bash
   python dataset_cleaner.py
   ```
3. **Generate Magpie Descriptors**:
   ```bash
   python featurizer.py
   ```
4. **Train & Evaluate**:
   ```bash
   python model_trainer.py
   ```

---

## 📚 Key References

1. **Ward, L., Agrawal, A., Choudhary, A., & Wolverton, C.** (2016). *A general-purpose machine learning framework for predicting properties of inorganic materials*. npj Computational Materials, 2(1), 16028.
2. **Jain, A., et al.** (2013). *Commentary: The Materials Project: A materials genome approach to accelerating materials innovation*. APL Materials, 1(1), 011002.
3. **Hill, R.** (1952). *The elastic behaviour of a crystalline aggregate*. Proceedings of the Physical Society. Section A, 65(5), 349.
4. **Breiman, L.** (2001). *Random Forests*. Machine Learning, 45(1), 5-32.