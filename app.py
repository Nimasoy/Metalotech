"""
Metalotech Interactive Web Application
======================================
Gradio web interface for real-time polycrystalline Bulk Modulus (VRH) prediction
using stoichiometry-based Magpie features and Random Forest regression.
"""

import gradio as gr
import predictor as p

EXAMPLES = [
    ["C"],        # Diamond / Carbon allotrope
    ["WC"],       # Tungsten Carbide
    ["TiC"],      # Titanium Carbide
    ["Al2O3"],    # Corundum / Alumina
    ["Fe2O3"],    # Hematite / Iron Oxide
    ["SrTiO3"],   # Strontium Titanate perovskite
    ["NaCl"],     # Halite / Salt (soft ionic)
    ["Au"],       # Gold (ductile metal)
]

DESCRIPTION = """
### 🔬 Predict Polycrystalline Bulk Modulus ($K_{VRH}$) in Gigapascals (GPa)
Enter any valid stoichiometric chemical formula. The model extracts **132 physical-chemical descriptors** 
(Ward et al., *Magpie* preset via Matminer) and predicts the bulk modulus using a trained **Random Forest Regressor** 
benchmarked against Materials Project Density Functional Theory (DFT) calculations.
"""

demo = gr.Interface(
    fn=p.predictor,
    inputs=gr.Textbox(
        label="Chemical Formula",
        placeholder="e.g. Fe2O3, TiC, Al2O3, WC, SrTiO3",
        lines=1
    ),
    outputs=gr.Textbox(
        label="Bulk Modulus Prediction",
        lines=1
    ),
    title="⚡ Metalotech: Materials Bulk Modulus Predictor",
    description=DESCRIPTION,
    examples=EXAMPLES,
    flagging_mode="never"
)

if __name__ == "__main__":
    demo.launch(share=False)
