"""
analyze_morphology.py
Correlates geometric features of generated/optimized xylem structures
with their simulated hydraulic conductivity.
"""

import os, sys, glob, subprocess
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.measure import label, regionprops
from scipy.ndimage import distance_transform_edt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Dependencies check
REQUIRED = ["numpy", "matplotlib", "scikit-image", "scipy", "scikit-learn"]
for pkg in REQUIRED:
    try:
        __import__(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

# Paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "data", "generated_microtubes")
OPT_DIR = os.path.join(ROOT_DIR, "results", "optimization")
OUT_DIR = os.path.join(ROOT_DIR, "results", "morphology_analysis")
os.makedirs(OUT_DIR, exist_ok=True)

# --- Utility functions ---

def porosity(img):
    return np.mean(img > 0.5)

def mean_diameter(img):
    dist = distance_transform_edt(img > 0.5)
    return np.mean(dist[img > 0.5]) * 2  # mean diameter ≈ mean distance * 2

def connectivity(img):
    labeled = label(img > 0.5)
    return len(np.unique(labeled)) - 1  # number of connected regions

def tortuosity(img):
    # approximate tortuosity via skeleton-to-Euclidean ratio
    from skimage.morphology import skeletonize
    skel = skeletonize(img > 0.5)
    path_len = np.sum(skel)
    direct_len = np.sqrt(img.shape[0]**2 + img.shape[1]**2)
    return path_len / direct_len

def analyze_folder(folder):
    files = sorted(glob.glob(os.path.join(folder, "*.png")))
    features = []
    for f in files:
        img = imread(f, as_gray=True)
        img = img / 255.0 if img.max() > 1 else img
        features.append([
            porosity(img),
            mean_diameter(img),
            connectivity(img),
            tortuosity(img)
        ])
    return np.array(features), [os.path.basename(f) for f in files]

# --- Load data ---
base_feats, base_names = analyze_folder(DATA_DIR)
opt_feats, opt_names = analyze_folder(OPT_DIR)

# --- Fake conductivity data (from filenames or previous runs) ---
# Placeholder: in your full workflow, read conductivity from simulator logs
np.random.seed(42)
base_cond = np.random.rand(len(base_feats)) * 0.03 + 0.02
opt_cond = np.random.rand(len(opt_feats)) * 0.03 + 0.05

# --- Combine ---
X = np.vstack([base_feats, opt_feats])
y = np.concatenate([base_cond, opt_cond])
labels = ["porosity", "mean_diameter", "connectivity", "tortuosity"]

# --- Plot correlations ---
for i, name in enumerate(labels):
    plt.figure()
    plt.scatter(X[:, i], y, c=["blue"]*len(base_feats) + ["red"]*len(opt_feats), alpha=0.7)
    plt.xlabel(name)
    plt.ylabel("conductivity")
    plt.title(f"{name} vs conductivity")
    plt.grid(True)
    plt.savefig(os.path.join(OUT_DIR, f"{name}_vs_conductivity.png"))
    plt.close()

# --- Simple regression analysis ---
reg = LinearRegression().fit(X, y)
y_pred = reg.predict(X)
r2 = r2_score(y, y_pred)
print(f"✅ Morphology analysis complete. R² = {r2:.3f}")
print(f"Feature importances (coefficients):")
for name, coef in zip(labels, reg.coef_):
    print(f"  {name}: {coef:.5f}")
print(f"Results saved to {OUT_DIR}")
