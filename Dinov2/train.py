# ==========================================================
# DINOv2 training script for CSIRO - Image2Biomass Prediction
# - Image only, no metadata
# - Default uses 5-fold CV (set CFG.n_folds=1 to disable K-Fold)
# - Backbone: DINOv2 from timm, frozen; train only a small regression head
# ==========================================================

import os
import random
import numpy as np
import pandas as pd
from PIL import Image

from sklearn.model_selection import KFold

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import timm
from timm.data import resolve_model_data_config, create_transform


# ---------------- Config ----------------
class CFG:
    train_csv = "/root/dataset/train.csv"     # path to train.csv
    img_root  = "/root/dataset"              # root directory of images

    model_name = "vit_large_patch14_dinov2.lvd142m"  # DINOv2 backbone
    freeze_backbone = True                   # freeze all backbone weights
    hidden_dim = 512                         # dimension for regression head

    batch_size = 8                           # small batch because DINOv2 is large
    epochs = 60
    lr = 3e-4
    weight_decay = 1e-4
    num_workers = 4
    seed = 42
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_folds = 5                              # set to 1 if no K-fold training

    out_dir = "./weights_dinov2"
    patience = 12                            # early stopping patience
    min_delta = 1e-4                         # minimal R2 improvement for saving


# ---------------- Set random seed for reproducibility ----------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(CFG.seed)
os.makedirs(CFG.out_dir, exist_ok=True)


# ---------------- Target definitions (5 outputs) ----------------
TARGET_ORDER = ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g"]

# Weights for competition metric
WEIGHTS = {
    "Dry_Green_g": 0.1,
    "Dry_Dead_g": 0.1,
    "Dry_Clover_g": 0.1,
    "GDM_g": 0.2,
    "Dry_Total_g": 0.5,
}


# ---------------- Load train.csv and pivot into per-image format ----------------
df_raw = pd.read_csv(CFG.train_csv)

pivot = df_raw.pivot_table(
    index="image_path",
    columns="target_name",
    values="target"
).reset_index()

pivot = pivot[["image_path"] + TARGET_ORDER]

print("Total unique images:", len(pivot))


# ---------------- Dataset ----------------
class ImageOnlyDataset(Dataset):
    """Dataset that returns only image tensor and 5 regression targets."""
    def __init__(self, df, img_root, transform):
        self.df = df.reset_index(drop=True)
        self.img_root = img_root
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_rel = row["image_path"]
        img_path = os.path.join(self.img_root, img_rel)

        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        y = row[TARGET_ORDER].values.astype("float32")
        return img, torch.tensor(y)


# ---------------- Weighted R² metric ----------------
def weighted_r2(y_true: torch.Tensor, y_pred: torch.Tensor):
    """Compute weighted R² score used by the competition."""
    ys = y_true.detach().cpu().numpy()
    ps = y_pred.detach().cpu().numpy()

    all_y = []
    all_p = []
    all_w = []

    for i, t in enumerate(TARGET_ORDER):
        w = WEIGHTS[t]
        all_y.append(ys[:, i])
        all_p.append(ps[:, i])
        all_w.append(np.full(len(ys), w, dtype=np.float32))

    y_all = np.concatenate(all_y)
    p_all = np.concatenate(all_p)
    w_all = np.concatenate(all_w)

    y_wbar = np.sum(w_all * y_all) / np.sum(w_all)
    ss_res = np.sum(w_all * (y_all - p_all) ** 2)
    ss_tot = np.sum(w_all * (y_all - y_wbar) ** 2) + 1e-8

    return 1.0 - ss_res / ss_tot


# ---------------- Model: DINOv2 backbone + small regression head ----------------
class DinoRegressor(nn.Module):
    """Frozen backbone + trainable MLP regression head."""
    def __init__(self, backbone_name, num_targets=5, hidden_dim=512, freeze_backbone=True):
        super().__init__()

        # Load DINOv2 pretrained backbone
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=True,
            num_classes=0,    # remove classification head
        )
        feat_dim = self.backbone.num_features

        # Freeze backbone weights (feature extractor only)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # Small regression head
        self.head = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_targets),
        )

    def forward(self, x):
        feat = self.backbone(x)  # [B, feature_dim]
        out = self.head(feat)    # [B, 5]
        return out


# ---------------- DINOv2 recommended preprocessing ----------------
# Use timm official data_config and transforms
tmp_model = timm.create_model(CFG.model_name, pretrained=True, num_classes=0)
data_config = resolve_model_data_config(tmp_model)
train_transform = create_transform(**data_config, is_training=True)
valid_transform = create_transform(**data_config, is_training=False)
del tmp_model


# ---------------- Training & Validation loops ----------------
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    n = 0
    all_true = []
    all_pred = []

    for imgs, targets in loader:
        imgs = imgs.to(CFG.device)
        targets = targets.to(CFG.device)

        optimizer.zero_grad()
        preds = model(imgs)
        loss = criterion(preds, targets)
        loss.backward()
        optimizer.step()

        bs = imgs.size(0)
        total_loss += loss.item() * bs
        n += bs

        all_true.append(targets.detach().cpu())
        all_pred.append(preds.detach().cpu())

    avg_loss = total_loss / n
    all_true = torch.cat(all_true, dim=0)
    all_pred = torch.cat(all_pred, dim=0)
    r2 = weighted_r2(all_true, all_pred)
    return avg_loss, r2


def valid_one_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    n = 0
    all_true = []
    all_pred = []

    with torch.no_grad():
        for imgs, targets in loader:
            imgs = imgs.to(CFG.device)
            targets = targets.to(CFG.device)

            preds = model(imgs)
            loss = criterion(preds, targets)

            bs = imgs.size(0)
            total_loss += loss.item() * bs
            n += bs

            all_true.append(targets.detach().cpu())
            all_pred.append(preds.detach().cpu())

    avg_loss = total_loss / n
    all_true = torch.cat(all_true, dim=0)
    all_pred = torch.cat(all_pred, dim=0)
    r2 = weighted_r2(all_true, all_pred)
    return avg_loss, r2


# ---------------- K-Fold training ----------------
if CFG.n_folds <= 1:
    # If no K-Fold, just use first split from KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=CFG.seed)
    splits = [next(kf.split(pivot))]
else:
    kf = KFold(n_splits=CFG.n_folds, shuffle=True, random_state=CFG.seed)
    splits = list(kf.split(pivot))

fold_best_r2 = []

for fold, (train_idx, valid_idx) in enumerate(splits):
    print(f"\n========== DINOv2 Fold {fold+1}/{len(splits)} ==========")

    train_df = pivot.iloc[train_idx].reset_index(drop=True)
    valid_df = pivot.iloc[valid_idx].reset_index(drop=True)

    train_ds = ImageOnlyDataset(train_df, CFG.img_root, transform=train_transform)
    valid_ds = ImageOnlyDataset(valid_df, CFG.img_root, transform=valid_transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model = DinoRegressor(
        backbone_name=CFG.model_name,
        num_targets=len(TARGET_ORDER),
        hidden_dim=CFG.hidden_dim,
        freeze_backbone=CFG.freeze_backbone,
    ).to(CFG.device)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CFG.lr,
        weight_decay=CFG.weight_decay,
    )
    criterion = nn.MSELoss()

    best_r2 = -1e9
    best_epoch = -1
    bad_epochs = 0

    for epoch in range(1, CFG.epochs + 1):
        train_loss, train_r2 = train_one_epoch(model, train_loader, optimizer, criterion)
        valid_loss, valid_r2 = valid_one_epoch(model, valid_loader, criterion)

        print(
            f"Fold {fold+1} | Epoch {epoch:03d}/{CFG.epochs} "
            f"| train loss: {train_loss:.4f}  train R2: {train_r2:.4f}  "
            f"valid loss: {valid_loss:.4f}  valid R2: {valid_r2:.4f}"
        )

        # Save best checkpoint based on validation R²
        if valid_r2 > best_r2 + CFG.min_delta:
            best_r2 = valid_r2
            best_epoch = epoch
            bad_epochs = 0

            save_path = os.path.join(CFG.out_dir, f"dinov2_fold{fold+1}.pth")
            torch.save(
                {
                    "model": model.state_dict(),
                    "backbone_name": CFG.model_name,
                    "TARGET_ORDER": TARGET_ORDER,
                },
                save_path,
            )
            print(
                f"  → Saved best model for fold {fold+1} at epoch {best_epoch}, "
                f"valid_R2={best_r2:.4f}"
            )
        else:
            bad_epochs += 1
            if bad_epochs >= CFG.patience:
                print(f"  → Early stopping fold {fold+1} at epoch {epoch}")
                break

    fold_best_r2.append(best_r2)
    print(
        f"Fold {fold+1} finished. Best valid R2 = {best_r2:.4f} at epoch {best_epoch}"
    )

print("\nAll folds finished.")
print("Fold R2:", fold_best_r2)
print("Mean R2:", np.mean(fold_best_r2))
