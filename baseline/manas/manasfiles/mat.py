import torch
import torch.nn as nn
import numpy as np
# from datasets import load_dataset
from functools import partial
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, f1_score, roc_auc_score, average_precision_score

# --- IMPORT YOUR MODEL ---
# Ensure model.py is in the directory
from baseline.manas.manasfiles.model import MAE

# --- CONFIGURATION ---
CHECKPOINT_PATH = "/share/sv7577-h200-41/checkpoints/mae_epoch_50.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
LR = 1e-3
N_EPOCHS = 20

# EEGMAT Channels (20 channels)
EEGMAT_CHANNELS = ["Fp1", "Fp2", "F3", "F4", "F7", "F8", "T3", "T4", "C3", "C4", "T5", "T6", "P3", "P4", "O1", "O2", "Fz", "Cz", "Pz", "A2"]

class MyReveClassifier(nn.Module):
    def __init__(self, checkpoint_path, num_classes=2, flat_dim=512):
        super().__init__()

        print(f"Loading checkpoint from {checkpoint_path}...")
        ckpt = torch.load(checkpoint_path, map_location="cpu")

        # Initialize YOUR architecture
        self.mae = MAE(fs=200, embed_dim=512, encoder_depth=12, encoder_heads=8, decoder_depth=4, decoder_heads=8, mask_ratio=0.55)
        self.mae.load_state_dict(ckpt["model_state_dict"])

        # Extract necessary components
        self.patch_embed = self.mae.patch_embed
        self.pos_enc = self.mae.pos_enc
        self.encoder = self.mae.encoder
        self.patch_size = self.mae.patch_size
        self.step = self.mae.step

        # Calculate Flatten Dimension
        # 20 channels, 5 seconds.
        # Patch size 1s, stride 0.9s -> 1s window + 4 * 0.9s = 4.6s...
        # Tutorial says output is [B, 20, 5, 512].
        # Let's trust the tutorial dim: 20 * 5 * 512 = 51200

        # 10s segments gives 512x19x11
        self.flat_dim = flat_dim

        # The Head (Same as Tutorial)
        self.final_layer = nn.Sequential(
            nn.Flatten(),
            nn.RMSNorm(self.flat_dim),  # Tutorial uses RMSNorm
            nn.Dropout(0.1),
            nn.Linear(self.flat_dim, num_classes),
        )

    def prepare_coords(self, xyz, num_patches):
        # Your logic to expand coords to time
        B, C, _ = xyz.shape
        device = xyz.device
        time_idx = torch.arange(num_patches, device=device).float()
        spat = xyz.unsqueeze(2).expand(-1, -1, num_patches, -1)
        time = time_idx.view(1, 1, num_patches, 1).expand(B, C, -1, -1)
        return torch.cat([spat, time], dim=-1).flatten(1, 2)

    def forward(self, x, pos):
        # x: (B, 20, 1000) -> 5s @ 200Hz
        # pos: (B, 20, 3)

        # 1. Patchify
        patches = x.unfold(-1, self.patch_size, self.step)
        num_patches = patches.shape[2]  # Should be 5

        # 2. Embed
        tokens = self.patch_embed.linear(patches).flatten(1, 2)

        # 3. Add PE
        coords = self.prepare_coords(pos, num_patches)
        pe = self.pos_enc(coords)

        # 4. Encode
        x_enc = tokens + pe
        latents, _ = self.encoder(x_enc)  # (B, 100, 512)

        # 5. Classify
        return self.final_layer(latents)
