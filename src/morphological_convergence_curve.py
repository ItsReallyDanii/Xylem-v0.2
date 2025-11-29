"""
morphological_convergence_curve.py
-------------------------------------
Visualizes morphological convergence metrics (latent distance, overlap, SSIM)
across training epochs, with smoothing and confidence shading.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt

HISTORY_PATH = "results/morphology_metrics/history.json"
SAVE_PATH = "results/morphology_metrics/morphological_convergence_curve.png"

def ema_smooth(values, alpha=0.5):
    """Exponential moving average smoothing."""
    smoothed = []
    s = values[0]
    for v in values:
        s = alpha * v + (1 - alpha) * s
        smoothed.append(s)
    return np.array(smoothed)

if not os.path.exists(HISTORY_PATH):
    print("⚠️ No history.json found — run morphology_metrics.py first.")
    exit()

with open(HISTORY_PATH, "r") as f:
    history = json.load(f)

epochs = np.arange(1, len(history["latent_distance"]) + 1)

plt.figure(figsize=(8, 5))
plt.title("Morphological Convergence Across Training Epochs", fontsize=14)

colors = {
    "latent_distance": "#2ca02c",
    "cluster_overlap": "#1f77b4",
    "ssim": "#d62728"
}

for key, label in [
    ("latent_distance", "Latent Distance ↓"),
    ("cluster_overlap", "Cluster Overlap ↑"),
    ("ssim", "SSIM ↑")
]:
    y = np.array(history[key])
    y_smooth = ema_smooth(y, alpha=0.4)
    plt.plot(epochs, y_smooth, label=label, color=colors[key], linewidth=2)
    plt.fill_between(epochs, y_smooth * 0.95, y_smooth * 1.05, color=colors[key], alpha=0.15)

plt.xlabel("Epochs / Retraining Cycle", fontsize=12)
plt.ylabel("Normalized Metric Value", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(frameon=False)
plt.tight_layout()

plt.savefig(SAVE_PATH, dpi=300, bbox_inches="tight")
plt.close()
print(f"✅ Saved smoothed convergence plot → {SAVE_PATH}")
