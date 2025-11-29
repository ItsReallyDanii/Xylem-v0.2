"""
simulate_flow.py
Couples learned xylem structures with a simplified 2D flow simulation.
Outputs conductivity metrics and visual flow maps.
"""

import os, sys, subprocess
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

# --- Auto-install missing packages ---
REQUIRED = ["numpy", "matplotlib", "torch", "torchvision", "Pillow"]
for pkg in REQUIRED:
    try:
        __import__(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

# --- Paths ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "data", "generated_microtubes")
RESULTS_DIR = os.path.join(ROOT_DIR, "results", "flow_simulation")
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- Image loader ---
def load_structure(img_path):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    img = Image.open(img_path)
    tensor = transform(img).squeeze().numpy()
    return tensor

# --- Simple flow solver (steady-state Laplace approximation) ---
def simulate_pressure_field(structure, inlet=1.0, outlet=0.0, iterations=5000):
    """
    structure: 2D numpy array (1=solid, 0=void)
    Simulate a steady-state pressure gradient through open pores.
    """
    mask = (structure < 0.5).astype(float)  # 1=open channel, 0=solid wall
    p = np.zeros_like(mask)
    p[:, 0] = inlet
    p[:, -1] = outlet

    for _ in range(iterations):
        # Jacobi relaxation
        p_new = p.copy()
        p_new[1:-1, 1:-1] = 0.25 * (
            p[2:, 1:-1] + p[:-2, 1:-1] +
            p[1:-1, 2:] + p[1:-1, :-2]
        )
        p = p_new * mask + p * (1 - mask)  # enforce walls
        # boundary conditions
        p[:, 0] = inlet
        p[:, -1] = outlet
    return p, mask

# --- Compute conductivity metric ---
def compute_conductivity(p_field, mask):
    grad = np.gradient(p_field, axis=1)
    flow = -grad * mask
    mean_flow = np.abs(flow).mean()
    return mean_flow

# --- Main routine ---
def main():
    images = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".png")])
    metrics = []
    for f in images:
        path = os.path.join(DATA_DIR, f)
        structure = load_structure(path)
        p_field, mask = simulate_pressure_field(structure)
        cond = compute_conductivity(p_field, mask)
        metrics.append((f, cond))

        # Visualize pressure field
        plt.imshow(p_field, cmap="coolwarm")
        plt.title(f"Pressure field: {f}\nConductivity={cond:.4f}")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"flow_{f}.png"))
        plt.close()

    # Save summary
    metrics.sort(key=lambda x: x[1], reverse=True)
    with open(os.path.join(RESULTS_DIR, "conductivity_metrics.txt"), "w") as f:
        for name, val in metrics:
            f.write(f"{name}\t{val:.6f}\n")

    print(f"✅ Flow simulation complete. Results in {RESULTS_DIR}")

if __name__ == "__main__":
    main()
