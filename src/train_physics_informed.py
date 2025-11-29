import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
from src.model import XylemAutoencoder
from src.flow_simulation import solve_darcy_flow

# --------------------------
# CONFIG
# --------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "results/model.pth"  # ✅ Change if needed
SAVE_PATH = "results/model_physics_tuned.pth"
REAL_METRICS_PATH = "results/flow_metrics/flow_metrics.csv"
DATA_SYN_DIR = "data/generated_microtubes/"
EPOCHS = 5
ALPHA = 0.3  # weight of physics loss

# --------------------------
# LOAD MODEL
# --------------------------
model = XylemAutoencoder().to(DEVICE)
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE), strict=False)
    print(f"✅ Loaded model from {MODEL_PATH}")
else:
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
model.train()

optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion_recon = nn.MSELoss()

# --------------------------
# LOAD TARGET (REAL) METRICS
# --------------------------
df = pd.read_csv(REAL_METRICS_PATH)
real_means = {
    "Mean_K": np.nanmean(df[df["Type"].str.lower()=="real"]["Mean_K"]),
    "FlowRate": np.nanmean(df[df["Type"].str.lower()=="real"]["FlowRate"]),
    "Porosity": np.nanmean(df[df["Type"].str.lower()=="real"]["Porosity"])
}

# --------------------------
# HELPERS
# --------------------------
to_tensor = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((256, 256)),  # ✅ fix: ensure consistent shape
    transforms.ToTensor()
])
to_pil = transforms.ToPILImage()

def physics_metrics(img_tensor):
    """Compute coarse flow metrics from a single image tensor."""
    img = to_pil(img_tensor.cpu().squeeze(0))
    k_map = np.array(img) / 255.0
    p_field, vx, vy = solve_darcy_flow(k_map)
    mean_k = np.mean(k_map)
    flow_rate = np.mean(np.sqrt(vx**2 + vy**2))
    porosity = np.mean(k_map > 0.5)
    return mean_k, flow_rate, porosity

# --------------------------
# TRAINING LOOP
# --------------------------
for epoch in range(1, EPOCHS + 1):
    recon_loss_total, physics_loss_total = 0, 0
    files = [f for f in os.listdir(DATA_SYN_DIR) if f.lower().endswith(".png")]
    if not files:
        raise RuntimeError(f"No PNG files found in {DATA_SYN_DIR}")

    for fname in files:
        path = os.path.join(DATA_SYN_DIR, fname)
        img = to_tensor(Image.open(path)).unsqueeze(0).to(DEVICE)

        # Forward pass
        z = model.encode(img)
        recon = model.decode(z)

        # Reconstruction loss
        L_recon = criterion_recon(recon, img)

        # Physics loss
        mean_k, flow_rate, porosity = physics_metrics(recon)
        L_phys = (
            (mean_k - real_means["Mean_K"])**2 +
            (flow_rate - real_means["FlowRate"])**2 +
            (porosity - real_means["Porosity"])**2
        )

        # Total loss
        L_total = L_recon + ALPHA * L_phys

        optimizer.zero_grad()
        L_total.backward()
        optimizer.step()

        recon_loss_total += L_recon.item()
        physics_loss_total += L_phys

    print(f"Epoch {epoch}/{EPOCHS} | Recon: {recon_loss_total:.4f} | Phys: {physics_loss_total:.4f}")

# --------------------------
# SAVE MODEL
# --------------------------
os.makedirs("results", exist_ok=True)
torch.save(model.state_dict(), SAVE_PATH)
print(f"✅ Physics-informed model saved → {SAVE_PATH}")
