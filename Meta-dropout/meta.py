import os
import random
import numpy as np
import pandas as pd
from PIL import Image

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import timm


# ==========================================================
# Config
# ==========================================================
class CFG:
    train_csv = "/root/dataset/train.csv"   # 改成你自己的路径
    img_root  = "/root/dataset"            # 根目录，下边有 images
    model_name = "tf_efficientnetv2_s"
    img_height = 384                       # 稍微放大一点提升分
    img_width  = 768
    batch_size = 16
    epochs = 50
    lr = 3e-4
    weight_decay = 1e-4
    num_workers = 4
    seed = 42
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_folds = 5

    # 三种改进的核心参数
    warmup_epochs = 8          # 前 8 个 epoch 完全不用 meta
    image_only_prob = 0.5      # 之后训练阶段：50% 只用 image
    meta_dropout_p = 0.5       # 在模型里对 meta 做 50% dropout

    out_dir = "./weights_meta_robust"


TARGET_ORDER = ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g"]
WEIGHTS = {
    "Dry_Green_g": 0.1,
    "Dry_Dead_g": 0.1,
    "Dry_Clover_g": 0.1,
    "GDM_g": 0.2,
    "Dry_Total_g": 0.5,
}
TARGET2IDX = {t: i for i, t in enumerate(TARGET_ORDER)}


# ==========================================================
# Utils
# ==========================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(CFG.seed)
os.makedirs(CFG.out_dir, exist_ok=True)


def weighted_r2(y_true, y_pred):
    """
    y_true, y_pred: torch.Tensor on CPU, shape [N, 5]
    """
    ys = y_true.numpy()
    ps = y_pred.numpy()

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
# Build aggregated dataframe: one row per image + meta
# ==========================================================
df_raw = pd.read_csv(CFG.train_csv)

# meta 部分（按 image_path 去重）
meta_cols = [
    "image_path",
    "Sampling_Date",
    "State",
    "Species",
    "Pre_GSHH_NDVI",
    "Height_Ave_cm",
]
df_meta = df_raw[meta_cols].drop_duplicates("image_path").reset_index(drop=True)

# 目标 pivot 到一行 5 列
pivot = df_raw.pivot_table(
    index="image_path",
    columns="target_name",
    values="target"
).reset_index()

pivot = pivot[["image_path"] + TARGET_ORDER]

# 合并
df_all = df_meta.merge(pivot, on="image_path", how="inner").reset_index(drop=True)

# 处理日期为数值
df_all["Sampling_Date"] = pd.to_datetime(df_all["Sampling_Date"])
df_all["date_ordinal"] = df_all["Sampling_Date"].map(pd.Timestamp.toordinal)

# 数值 meta 列
num_cols = ["date_ordinal", "Pre_GSHH_NDVI", "Height_Ave_cm"]

# 类别 one-hot
cat_dummies = pd.get_dummies(
    df_all[["State", "Species"]],
    prefix=["state", "sp"]
)

# 先不归一化，后面在 fold 内做 scaler
meta_features = pd.concat([df_all[num_cols], cat_dummies], axis=1)
META_FEATURES = list(meta_features.columns)

print("Total images:", len(df_all))
print("Metadata dim (before scaling):", len(META_FEATURES))


# ==========================================================
# Dataset
# ==========================================================
class BiomassMetaDataset(Dataset):
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

        # meta 向量
        meta_vec = row[META_FEATURES].values.astype("float32")
        # targets: 按固定顺序
        y = row[TARGET_ORDER].values.astype("float32")

        return img, torch.tensor(meta_vec), torch.tensor(y)


# ==========================================================
# Transforms
# ==========================================================
train_transform = T.Compose([
    T.Resize((CFG.img_height, CFG.img_width)),
    T.RandomHorizontalFlip(p=0.5),
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
# Model with meta + fusion + meta dropout
# ==========================================================
class CSIROModel(nn.Module):
    def __init__(
        self,
        model_name,
        pretrained=True,
        in_chans=3,
        num_classes=5,
        metadata_dim=0,
        fusion_dim=256,
        dropout=0.2,
        meta_dropout_p=0.5,
    ):
        super().__init__()

        self.metadata_dim = metadata_dim
        self.meta_dropout_p = meta_dropout_p

        self.image_model = timm.create_model(
            model_name=model_name,
            pretrained=pretrained,
            in_chans=in_chans,
            num_classes=0,
        )
        self.image_feature_dim = self.image_model.num_features

        if metadata_dim > 0:
            self.metadata_processor = nn.Sequential(
                nn.Linear(metadata_dim, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.metadata_output_dim = 64
        else:
            self.metadata_processor = None
            self.metadata_output_dim = 0

        combined_dim = self.image_feature_dim + self.metadata_output_dim

        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.regressor = nn.Linear(fusion_dim // 2, num_classes)

    def forward(self, image, metadata=None):
        image_features = self.image_model(image)  # [B, C_img]

        if self.metadata_dim > 0:
            # -------- meta dropout（训练时部分 batch 整个 meta 清零） --------
            if self.training and metadata is not None and self.meta_dropout_p > 0:
                if random.random() < self.meta_dropout_p:
                    metadata = torch.zeros_like(metadata)

            # -------- 推理/无 meta 场景：直接用 0 向量 --------
            if metadata is None:
                meta_features = torch.zeros(
                    image_features.size(0),
                    self.metadata_output_dim,
                    device=image_features.device,
                    dtype=image_features.dtype,
                )
            else:
                meta_features = self.metadata_processor(metadata)

            combined = torch.cat([image_features, meta_features], dim=1)
        else:
            combined = image_features

        fused = self.fusion(combined)
        predictions = self.regressor(fused)
        return predictions


# ==========================================================
# Train / Valid loop
# - 训练阶段：三种改进全部融合
#   1) warmup_epochs: 全部 metadata=None
#   2) 之后每个 batch 50% 只图像, 50% 图像+meta
#   3) 模型内部再做 meta_dropout_p
# - 验证 & 线上：始终 metadata=None
# ==========================================================
def train_one_epoch(model, loader, optimizer, criterion, epoch):
    model.train()
    all_true = []
    all_pred = []
    total_loss = 0.0
    n = 0

    for imgs, meta, targets in loader:
        imgs = imgs.to(CFG.device)
        meta = meta.to(CFG.device)
        targets = targets.to(CFG.device)

        optimizer.zero_grad()

        # 1) warmup 期：完全不用 meta
        if epoch < CFG.warmup_epochs:
            preds = model(imgs, metadata=None)

        # 2) 正式期：batch 级 50% 只用 image / 50% image+meta
        else:
            if random.random() < CFG.image_only_prob:
                preds = model(imgs, metadata=None)
            else:
                preds = model(imgs, metadata=meta)

        loss = criterion(preds, targets)
        loss.backward()
        optimizer.step()

        bs = imgs.size(0)
        total_loss += loss.item() * bs
        n += bs

        all_true.append(targets.detach().cpu())
        all_pred.append(preds.detach().cpu())

    all_true = torch.cat(all_true, dim=0)
    all_pred = torch.cat(all_pred, dim=0)
    avg_loss = total_loss / n
    r2 = weighted_r2(all_true, all_pred)
    return r2, avg_loss


def valid_one_epoch(model, loader, criterion):
    model.eval()
    all_true = []
    all_pred = []
    total_loss = 0.0
    n = 0

    with torch.no_grad():
        for imgs, meta, targets in loader:
            imgs = imgs.to(CFG.device)
            targets = targets.to(CFG.device)

            # 验证与线上 test 一致：不传 meta
            preds = model(imgs, metadata=None)
            loss = criterion(preds, targets)

            bs = imgs.size(0)
            total_loss += loss.item() * bs
            n += bs

            all_true.append(targets.detach().cpu())
            all_pred.append(preds.detach().cpu())

    all_true = torch.cat(all_true, dim=0)
    all_pred = torch.cat(all_pred, dim=0)
    avg_loss = total_loss / n
    r2 = weighted_r2(all_true, all_pred)
    return r2, avg_loss


# ==========================================================
# K-Fold Training
# ==========================================================
kf = KFold(n_splits=CFG.n_folds, shuffle=True, random_state=CFG.seed)
fold_results = []

for fold, (train_idx, valid_idx) in enumerate(kf.split(df_all)):
    print(f"\n========== Fold {fold+1}/{CFG.n_folds} ==========")

    train_df = df_all.iloc[train_idx].reset_index(drop=True)
    valid_df = df_all.iloc[valid_idx].reset_index(drop=True)

    # -------- 数值 meta 做标准化（每 fold 单独 scaler） --------
    # 重新构建这一折用的 META_FEATURES 数据
    train_meta = pd.concat(
        [train_df[num_cols], pd.get_dummies(train_df[["State", "Species"]],
                                            prefix=["state", "sp"])],
        axis=1
    )
    valid_meta = pd.concat(
        [valid_df[num_cols], pd.get_dummies(valid_df[["State", "Species"]],
                                            prefix=["state", "sp"])],
        axis=1
    )

    # 对齐 dummy 列
    train_meta, valid_meta = train_meta.align(valid_meta, join="outer", axis=1, fill_value=0.0)
    META_FEATURES = list(train_meta.columns)

    # 标准化数值列（包含日期、高度、NDVI + dummy）
    scaler = StandardScaler()
    train_meta_scaled = scaler.fit_transform(train_meta.values)
    valid_meta_scaled = scaler.transform(valid_meta.values)

    # 把缩放后的 meta 写回 df（这一折专用）
    for i, c in enumerate(META_FEATURES):
        train_df[c] = train_meta_scaled[:, i]
        valid_df[c] = valid_meta_scaled[:, i]

    # 构建 dataset
    train_dataset = BiomassMetaDataset(train_df, CFG.img_root, transform=train_transform)
    valid_dataset = BiomassMetaDataset(valid_df, CFG.img_root, transform=valid_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    metadata_dim = len(META_FEATURES)
    print(f"Fold {fold+1} train size: {len(train_df)}, valid size: {len(valid_df)}, meta_dim: {metadata_dim}")

    model = CSIROModel(
        model_name=CFG.model_name,
        pretrained=True,
        in_chans=3,
        num_classes=len(TARGET_ORDER),
        metadata_dim=metadata_dim,
        fusion_dim=256,
        dropout=0.2,
        meta_dropout_p=CFG.meta_dropout_p,
    ).to(CFG.device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay
    )
    criterion = nn.MSELoss()

    best_r2 = -1e9
    best_epoch = -1
    patience = 15
    bad_epochs = 0

    for epoch in range(CFG.epochs):
        train_r2, train_loss = train_one_epoch(model, train_loader, optimizer, criterion, epoch)
        valid_r2, valid_loss = valid_one_epoch(model, valid_loader, criterion)

        print(
            f"Fold {fold+1} | Epoch {epoch+1}/{CFG.epochs} "
            f"| train R2: {train_r2:.4f}  valid R2(no meta): {valid_r2:.4f}  "
            f"train loss: {train_loss:.4f}  valid loss: {valid_loss:.4f}"
        )

        if valid_r2 > best_r2:
            best_r2 = valid_r2
            best_epoch = epoch + 1
            save_path = os.path.join(CFG.out_dir, f"csiro_meta_robust_fold{fold}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"  → Saved best model for fold {fold+1} at epoch {best_epoch}, R2={best_r2:.4f}")
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"  → Early stopping at epoch {epoch+1}")
                break

    fold_results.append(best_r2)
    print(f"Fold {fold+1} finished. Best valid R2(no meta) = {best_r2:.4f} at epoch {best_epoch}")

print("\nAll folds finished.")
print("Fold R2(no meta):", fold_results)
print("Mean R2:", np.mean(fold_results))
