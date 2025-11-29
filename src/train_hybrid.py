"""
train_hybrid.py
Fine-tunes the XylemAutoencoder on a hybrid dataset:
real xylem (data/real_xylem_preprocessed)
+ synthetic xylem (data/generated_microtubes)
to align latent spaces and increase morphological realism.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from src.model import XylemAutoencoder

# -------------------------------
# Configuration
# -------------------------------
REAL_DIR = "data/real_xylem_preprocessed"
SYN_DIR = "data/generated_microtubes"
SAVE_DIR = "results/hybrid_training"
os.makedirs(SAVE_DIR, exist_ok=True)

torch.backends.cudnn.benchmark = True
BATCH_SIZE = 8
EPOCHS = 20
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "results/model.pth"

# -------------------------------
# Dataset
# -------------------------------
class XylemHybridDataset(Dataset):
    def __init__(self, real_dir, syn_dir, transform=None):
        self.real_files = [os.path.join(real_dir, f)
                           for f in os.listdir(real_dir)
                           if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        self.syn_files = [os.path.join(syn_dir, f)
                          for f in os.listdir(syn_dir)
                          if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        self.all_files = self.real_files + self.syn_files
        self.labels = [0] * len(self.real_files) + [1] * len(self.syn_files)
        self.transform = transform

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        img_path = self.all_files[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert("L")
        if self.transform:
            img = self.transform(img)
        return img, label

# -------------------------------
# Transform
# -------------------------------
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

dataset = XylemHybridDataset(REAL_DIR, SYN_DIR, transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# -------------------------------
# Model Setup
# -------------------------------
model = XylemAutoencoder(latent_dim=32).to(DEVICE)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model_dict = model.state_dict()
compatible_dict = {k: v for k, v in checkpoint.items()
                   if k in model_dict and v.shape == model_dict[k].shape}
model_dict.update(compatible_dict)
model.load_state_dict(model_dict)
print(f"✅ Loaded {len(compatible_dict)} compatible layers from {MODEL_PATH}")

criterion_recon = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# -------------------------------
# Training Loop
# -------------------------------
print(f"🚀 Starting hybrid fine-tuning for {EPOCHS} epochs on {DEVICE}")
for epoch in range(EPOCHS):
    total_loss, total_recon, total_align = 0, 0, 0
    for imgs, labels in dataloader:
        imgs = imgs.to(DEVICE)
        optimizer.zero_grad()

        recon, z = model(imgs)

        # Reconstruction loss
        loss_recon = criterion_recon(recon, imgs)

        # Latent alignment loss (push real & synthetic closer)
        z_real = z[labels == 0]
        z_syn = z[labels == 1]
        if len(z_real) > 0 and len(z_syn) > 0:
            loss_align = torch.mean((z_real.mean(0) - z_syn.mean(0)) ** 2)
        else:
            loss_align = torch.tensor(0.0, device=DEVICE)

        loss = loss_recon + 0.1 * loss_align
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_recon += loss_recon.item()
        total_align += loss_align.item()

    print(f"Epoch {epoch+1}/{EPOCHS} | "
          f"Loss: {total_loss/len(dataloader):.4f} | "
          f"Recon: {total_recon/len(dataloader):.4f} | "
          f"Align: {total_align/len(dataloader):.4f}")

# -------------------------------
# Save Model
# -------------------------------
save_path = os.path.join(SAVE_DIR, "model_hybrid.pth")
torch.save(model.state_dict(), save_path)
print(f"✅ Hybrid fine-tuning complete. Model saved to {save_path}")
