# ==========================================================
# script_1_stage1_kfold_oof_meta.py
#
# Stage1: 从图像预测 4 个 metadata（NDVI, Height, State, Species）
# - 使用 K-Fold 训练，生成 OOF 预测 (train_meta_oof.csv)
# - 每个 fold 保存一个模型 (stage1_meta_fold{fold}.pth)
#
# 之后 Stage2 读取 train_meta_oof.csv 作为预测 meta 特征。
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
import torchvision.transforms as T
import timm


# ---------------- Config ----------------
class CFG_STAGE1:
    train_csv = "/root/dataset/train.csv"   # 原始长表
    img_root  = "/root/dataset"            # 图像根目录

    model_name = "tf_efficientnetv2_s"
    img_height = 512
    img_width  = 1024

    batch_size = 8
    epochs = 60
    lr = 3e-4
    weight_decay = 1e-4
    num_workers = 4
    seed = 42
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_folds = 5

    out_dir = "./weights_stage1_meta_kfold"
    oof_csv = "/root/dataset/train_meta_oof.csv"

    # 损失权重（可以后面微调）
    w_ndvi = 1.0
    w_height = 1.0
    w_state = 1.0
    w_species = 1.5

    patience = 15   # early stopping patience
    min_delta = 1e-4


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(CFG_STAGE1.seed)
os.makedirs(CFG_STAGE1.out_dir, exist_ok=True)


# ---------------- 读原始 train.csv，构建 per-image + meta ----------------
df_raw = pd.read_csv(CFG_STAGE1.train_csv)

# 每张图像一行，带上 4 个 meta
meta_cols = [
    "image_path",
    "Pre_GSHH_NDVI",
    "Height_Ave_cm",
    "State",
    "Species",
]
df_meta = df_raw[meta_cols].drop_duplicates("image_path").reset_index(drop=True)

# 编码 State / Species 为整数 id
state_codes = {name: i for i, name in enumerate(sorted(df_meta["State"].unique()))}
species_codes = {name: i for i, name in enumerate(sorted(df_meta["Species"].unique()))}

df_meta["state_id"] = df_meta["State"].map(state_codes).astype(int)
df_meta["species_id"] = df_meta["Species"].map(species_codes).astype(int)

num_state = len(state_codes)
num_species = len(species_codes)

print("Stage1 - total images:", len(df_meta))
print("Stage1 - num_state:", num_state)
print("Stage1 - num_species:", num_species)


# ---------------- Dataset ----------------
class MetaNetDataset(Dataset):
    def __init__(self, df, img_root, transform=None):
        self.df = df.reset_index(drop=True)
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

        ndvi = np.float32(row["Pre_GSHH_NDVI"])
        height = np.float32(row["Height_Ave_cm"])
        state_id = np.int64(row["state_id"])
        species_id = np.int64(row["species_id"])

        return (
            img,
            torch.tensor(ndvi),
            torch.tensor(height),
            torch.tensor(state_id),
            torch.tensor(species_id),
            row["image_path"],  # 为了 OOF 记录
        )


# ---------------- Transforms ----------------
train_transform = T.Compose([
    T.Resize((CFG_STAGE1.img_height, CFG_STAGE1.img_width)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(degrees=10),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

valid_transform = T.Compose([
    T.Resize((CFG_STAGE1.img_height, CFG_STAGE1.img_width)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])


# ---------------- Model: MetaNet (image -> 4 meta) ----------------
class MetaNet(nn.Module):
    def __init__(self, backbone_name, num_state, num_species):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=True,
            num_classes=0,
        )
        d = self.backbone.num_features

        self.ndvi_head = nn.Linear(d, 1)
        self.height_head = nn.Linear(d, 1)
        self.state_head = nn.Linear(d, num_state)
        self.species_head = nn.Linear(d, num_species)

    def forward(self, x):
        feat = self.backbone(x)
        ndvi = self.ndvi_head(feat).squeeze(-1)        # [B]
        height = self.height_head(feat).squeeze(-1)    # [B]
        state_logits = self.state_head(feat)           # [B, num_state]
        species_logits = self.species_head(feat)       # [B, num_species]
        return ndvi, height, state_logits, species_logits


# ---------------- Train / Valid ----------------
def train_one_epoch(model, loader, optimizer, criterion_reg, criterion_cls):
    model.train()
    total_loss = 0.0
    n = 0

    for imgs, ndvi, height, state_id, species_id, _ in loader:
        imgs = imgs.to(CFG_STAGE1.device)
        ndvi = ndvi.to(CFG_STAGE1.device)
        height = height.to(CFG_STAGE1.device)
        state_id = state_id.to(CFG_STAGE1.device)
        species_id = species_id.to(CFG_STAGE1.device)

        optimizer.zero_grad()
        pred_ndvi, pred_h, state_logits, species_logits = model(imgs)

        loss_ndvi = criterion_reg(pred_ndvi, ndvi)
        loss_h = criterion_reg(pred_h, height)
        loss_state = criterion_cls(state_logits, state_id)
        loss_species = criterion_cls(species_logits, species_id)

        loss = (
            CFG_STAGE1.w_ndvi * loss_ndvi
            + CFG_STAGE1.w_height * loss_h
            + CFG_STAGE1.w_state * loss_state
            + CFG_STAGE1.w_species * loss_species
        )

        loss.backward()
        optimizer.step()

        bs = imgs.size(0)
        total_loss += loss.item() * bs
        n += bs

    avg_loss = total_loss / n
    return avg_loss


def valid_one_epoch(model, loader, criterion_reg, criterion_cls):
    model.eval()
    total_loss = 0.0
    n = 0

    # 记录一些简单指标（可选）
    ndvi_abs_err = []
    h_abs_err = []
    state_correct = 0
    species_correct = 0
    total_samples = 0

    with torch.no_grad():
        for imgs, ndvi, height, state_id, species_id, _ in loader:
            imgs = imgs.to(CFG_STAGE1.device)
            ndvi = ndvi.to(CFG_STAGE1.device)
            height = height.to(CFG_STAGE1.device)
            state_id = state_id.to(CFG_STAGE1.device)
            species_id = species_id.to(CFG_STAGE1.device)

            pred_ndvi, pred_h, state_logits, species_logits = model(imgs)

            loss_ndvi = criterion_reg(pred_ndvi, ndvi)
            loss_h = criterion_reg(pred_h, height)
            loss_state = criterion_cls(state_logits, state_id)
            loss_species = criterion_cls(species_logits, species_id)

            loss = (
                CFG_STAGE1.w_ndvi * loss_ndvi
                + CFG_STAGE1.w_height * loss_h
                + CFG_STAGE1.w_state * loss_state
                + CFG_STAGE1.w_species * loss_species
            )

            bs = imgs.size(0)
            total_loss += loss.item() * bs
            n += bs

            ndvi_abs_err.append((pred_ndvi - ndvi).abs().cpu())
            h_abs_err.append((pred_h - height).abs().cpu())

            state_pred = state_logits.argmax(dim=1)
            species_pred = species_logits.argmax(dim=1)
            state_correct += (state_pred == state_id).sum().item()
            species_correct += (species_pred == species_id).sum().item()
            total_samples += bs

    avg_loss = total_loss / n
    ndvi_mae = torch.cat(ndvi_abs_err).mean().item()
    h_mae = torch.cat(h_abs_err).mean().item()
    state_acc = state_correct / total_samples
    species_acc = species_correct / total_samples

    return avg_loss, ndvi_mae, h_mae, state_acc, species_acc


# ---------------- K-Fold Training + OOF 预测 ----------------
kf = KFold(n_splits=CFG_STAGE1.n_folds, shuffle=True, random_state=CFG_STAGE1.seed)

# 用来存 OOF 预测
oof_records = []

fold_best_losses = []

for fold, (train_idx, valid_idx) in enumerate(kf.split(df_meta)):
    print(f"\n========== Stage1 Fold {fold+1}/{CFG_STAGE1.n_folds} ==========")

    train_df = df_meta.iloc[train_idx].reset_index(drop=True)
    valid_df = df_meta.iloc[valid_idx].reset_index(drop=True)

    train_ds = MetaNetDataset(train_df, CFG_STAGE1.img_root, transform=train_transform)
    valid_ds = MetaNetDataset(valid_df, CFG_STAGE1.img_root, transform=valid_transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=CFG_STAGE1.batch_size,
        shuffle=True,
        num_workers=CFG_STAGE1.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=CFG_STAGE1.batch_size,
        shuffle=False,
        num_workers=CFG_STAGE1.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model = MetaNet(
        backbone_name=CFG_STAGE1.model_name,
        num_state=num_state,
        num_species=num_species
    ).to(CFG_STAGE1.device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CFG_STAGE1.lr,
        weight_decay=CFG_STAGE1.weight_decay,
    )
    criterion_reg = nn.MSELoss()
    criterion_cls = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_epoch = -1
    bad_epochs = 0

    for epoch in range(1, CFG_STAGE1.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion_reg, criterion_cls)
        val_loss, ndvi_mae, h_mae, state_acc, species_acc = valid_one_epoch(
            model, valid_loader, criterion_reg, criterion_cls
        )

        print(
            f"Fold {fold+1} | Epoch {epoch:03d}/{CFG_STAGE1.epochs} "
            f"| train loss: {train_loss:.4f}  val loss: {val_loss:.4f}  "
            f"NDVI_MAE: {ndvi_mae:.4f}  H_MAE: {h_mae:.4f}  "
            f"State_acc: {state_acc:.4f}  Species_acc: {species_acc:.4f}"
        )

        if val_loss < best_val_loss - CFG_STAGE1.min_delta:
            best_val_loss = val_loss
            best_epoch = epoch
            bad_epochs = 0

            save_path = os.path.join(CFG_STAGE1.out_dir, f"stage1_meta_fold{fold+1}.pth")
            torch.save(
                {
                    "model": model.state_dict(),
                    "state_codes": state_codes,
                    "species_codes": species_codes,
                },
                save_path
            )
            print(f"  → Saved best Stage1 model for fold {fold+1} at epoch {best_epoch}, val_loss={best_val_loss:.4f}")
        else:
            bad_epochs += 1
            if bad_epochs >= CFG_STAGE1.patience:
                print(f"  → Early stopping Stage1 fold {fold+1} at epoch {epoch}")
                break

    fold_best_losses.append(best_val_loss)
    print(f"Stage1 Fold {fold+1} finished. Best val loss = {best_val_loss:.4f} at epoch {best_epoch}")

    # 载入这一折的最佳模型，用来对该折的验证集做 OOF 预测
    ckpt_path = os.path.join(CFG_STAGE1.out_dir, f"stage1_meta_fold{fold+1}.pth")
    ckpt = torch.load(ckpt_path, map_location=CFG_STAGE1.device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    softmax = nn.Softmax(dim=1)

    # 对 valid_df 做预测
    valid_loader_for_oof = DataLoader(
        valid_ds,
        batch_size=CFG_STAGE1.batch_size,
        shuffle=False,
        num_workers=CFG_STAGE1.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    with torch.no_grad():
        for imgs, ndvi, height, state_id, species_id, paths in valid_loader_for_oof:
            imgs = imgs.to(CFG_STAGE1.device)
            pred_ndvi, pred_h, state_logits, species_logits = model(imgs)

            state_prob = softmax(state_logits).cpu().numpy()      # [B, num_state]
            species_prob = softmax(species_logits).cpu().numpy()  # [B, num_species]

            pred_ndvi = pred_ndvi.cpu().numpy()
            pred_h = pred_h.cpu().numpy()

            for i, p in enumerate(paths):
                rec = {
                    "image_path": p,
                    "pred_ndvi": float(pred_ndvi[i]),
                    "pred_height": float(pred_h[i]),
                }
                # state 概率
                for j in range(num_state):
                    rec[f"pstate_{j}"] = float(state_prob[i, j])
                # species 概率
                for j in range(num_species):
                    rec[f"psp_{j}"] = float(species_prob[i, j])

                oof_records.append(rec)

print("\nStage1 All folds finished.")
print("Stage1 Fold best val losses:", fold_best_losses)
print("Stage1 Mean val loss:", np.mean(fold_best_losses))

# ---------------- 保存 OOF 预测到 CSV ----------------
df_oof = pd.DataFrame(oof_records)

# 确保每个 image_path 只出现一次（理论上 OOF 正好覆盖所有样本各一次）
df_oof = df_oof.drop_duplicates("image_path").reset_index(drop=True)

df_oof.to_csv(CFG_STAGE1.oof_csv, index=False)
print("Saved OOF meta predictions to:", CFG_STAGE1.oof_csv)
