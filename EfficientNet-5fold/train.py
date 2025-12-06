import os
import random
import numpy as np
import pandas as pd
from PIL import Image

from sklearn.model_selection import KFold

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import timm


# ==========================================================
# Config
# ==========================================================
class CFG:
    train_csv = "/root/dataset/train.csv"
    img_root = "/root/dataset"
    model_name = "tf_efficientnetv2_s"

    # image size (keep approx. 2:1 ratio)
    img_height = 256
    img_width = 512

    batch_size = 16
    epochs = 30
    lr = 3e-4
    weight_decay = 1e-2
    num_workers = 4
    seed = 42
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_folds = 5
    pretrained = True  # set to False if no internet
    save_dir = "./weights"
    patience = 6       # early stopping


TARGET_ORDER = ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g"]
WEIGHTS = {
    "Dry_Green_g": 0.1,
    "Dry_Dead_g": 0.1,
    "Dry_Clover_g": 0.1,
    "GDM_g": 0.2,
    "Dry_Total_g": 0.5,
}


# ==========================================================
# Utils
# ==========================================================
def set_seed(seed: int = 42):
    """Reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


os.makedirs(CFG.save_dir, exist_ok=True)
set_seed(CFG.seed)


# ==========================================================
# Build dataframe: one row per image (+ 5 targets)
# ==========================================================
df = pd.read_csv(CFG.train_csv)

pivot = df.pivot_table(
    index="image_path",
    columns="target_name",
    values="target"
)

pivot = pivot[TARGET_ORDER].dropna().reset_index()
print("Num images:", len(pivot))


# ==========================================================
# Dataset: each sample = one image + 5-dim target vector
# ==========================================================
class BiomassDataset(Dataset):
    def __init__(self, df_img, img_root, transform=None):
        self.df = df_img.reset_index(drop=True)
        self.img_root = img_root
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_root, row["image_path"])
        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        y = row[TARGET_ORDER].values.astype("float32")
        return img, torch.tensor(y)


# ==========================================================
# Transforms (train / valid)
# ==========================================================
train_transform = T.Compose([
    T.Resize((CFG.img_height + 32, CFG.img_width + 64)),
    T.RandomResizedCrop(
        size=(CFG.img_height, CFG.img_width),
        scale=(0.8, 1.0),
        ratio=(1.8, 2.2),
    ),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.2),
    T.RandomRotation(degrees=10),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

valid_transform = T.Compose([
    T.Resize((CFG.img_height, CFG.img_width)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])


# ==========================================================
# Model
# ==========================================================
class EffNetV2Regressor(nn.Module):
    def __init__(self, model_name, num_targets=5, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0
        )
        in_features = self.backbone.num_features
        self.head = nn.Linear(in_features, num_targets)

    def forward(self, x):
        feat = self.backbone(x)
        out = self.head(feat)
        return out


# ==========================================================
# Weighted R² (competition metric)
# ==========================================================
def weighted_r2(y_true, y_pred):
    ys = y_true.cpu().numpy()
    ps = y_pred.cpu().detach().numpy()

    all_y = []
    all_p = []
    all_w = []

    for i, t in enumerate(TARGET_ORDER):
        w = WEIGHTS[t]
        all_y.append(ys[:, i])
        all_p.append(ps[:, i])
        all_w.append(np.full(len(ys), w))

    y_all = np.concatenate(all_y)
    p_all = np.concatenate(all_p)
    w_all = np.concatenate(all_w)

    y_wbar = np.sum(w_all * y_all) / np.sum(w_all)
    ss_res = np.sum(w_all * (y_all - p_all) ** 2)
    ss_tot = np.sum(w_all * (y_all - y_wbar) ** 2)

    return 1 - ss_res / ss_tot


# ==========================================================
# Train / Valid loops
# ==========================================================
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    all_true = []
    all_pred = []

    for imgs, targets in loader:
        imgs = imgs.to(CFG.device)
        targets = targets.to(CFG.device)

        optimizer.zero_grad()
        preds = model(imgs)
        loss = criterion(preds, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        all_true.append(targets.detach().cpu())
        all_pred.append(preds.detach().cpu())

    all_true = torch.cat(all_true, dim=0)
    all_pred = torch.cat(all_pred, dim=0)
    r2 = weighted_r2(all_true, all_pred)
    return r2


def valid_one_epoch(model, loader, criterion):
    model.eval()
    all_true = []
    all_pred = []
    valid_loss = 0.0

    with torch.no_grad():
        for imgs, targets in loader:
            imgs = imgs.to(CFG.device)
            targets = targets.to(CFG.device)

            preds = model(imgs)
            loss = criterion(preds, targets)
            valid_loss += loss.item() * imgs.size(0)

            all_true.append(targets.detach().cpu())
            all_pred.append(preds.detach().cpu())

    all_true = torch.cat(all_true, dim=0)
    all_pred = torch.cat(all_pred, dim=0)

    r2 = weighted_r2(all_true, all_pred)
    valid_loss /= len(loader.dataset)
    return r2, valid_loss


# ==========================================================
# K-Fold Training
# ==========================================================
kf = KFold(n_splits=CFG.n_folds, shuffle=True, random_state=CFG.seed)

fold_results = []

for fold, (train_idx, valid_idx) in enumerate(kf.split(pivot)):
    print(f"\n========== Fold {fold+1}/{CFG.n_folds} ==========")

    train_df = pivot.iloc[train_idx].reset_index(drop=True)
    valid_df = pivot.iloc[valid_idx].reset_index(drop=True)

    train_dataset = BiomassDataset(train_df, CFG.img_root, transform=train_transform)
    valid_dataset = BiomassDataset(valid_df, CFG.img_root, transform=valid_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        pin_memory=True
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True
    )

    model = EffNetV2Regressor(
        CFG.model_name,
        num_targets=len(TARGET_ORDER),
        pretrained=CFG.pretrained
    ).to(CFG.device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CFG.lr,
        weight_decay=CFG.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=CFG.epochs,
        eta_min=1e-6
    )
    criterion = nn.MSELoss()

    best_r2 = -1e9
    best_epoch = -1
    no_improve = 0

    for epoch in range(CFG.epochs):
        train_r2 = train_one_epoch(model, train_loader, optimizer, criterion)
        valid_r2, valid_loss = valid_one_epoch(model, valid_loader, criterion)
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Fold {fold+1} | Epoch {epoch+1}/{CFG.epochs} "
            f"| lr: {current_lr:.2e} "
            f"| train R2: {train_r2:.4f}  valid R2: {valid_r2:.4f}  valid loss: {valid_loss:.4f}"
        )

        if valid_r2 > best_r2 + 1e-4:
            best_r2 = valid_r2
            best_epoch = epoch + 1
            no_improve = 0
            save_path = os.path.join(CFG.save_dir, f"effv2s_fold{fold}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"  → Saved best model: {save_path} (R2={best_r2:.4f})")
        else:
            no_improve += 1
            if no_improve >= CFG.patience:
                print(f"  → Early stopping at epoch {epoch+1}")
                break

    fold_results.append(best_r2)
    print(f"Fold {fold+1} finished. Best valid R2 = {best_r2:.4f} at epoch {best_epoch}")

print("\nAll folds finished.")
print("Fold R2:", fold_results)
print("Mean R2:", np.mean(fold_results))
