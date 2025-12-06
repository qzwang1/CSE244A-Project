import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import timm
from timm.data import resolve_model_data_config, create_transform


# ==========================================================
# Paths: update these based on your Kaggle Datasets
# ==========================================================
DATA_DIR   = "/kaggle/input/csiro-biomass"                     # contains train.csv / test.csv / train/ test/ images
WEIGHT_DIR = "/kaggle/input/csiro-dinov2-weights/pytorch/default/1"  # directory with dinov2_fold*.pth

TEST_CSV  = os.path.join(DATA_DIR, "test.csv")
IMG_ROOT  = DATA_DIR   # image_path is relative to this folder


class CFG:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 16
    num_workers = 4
    n_folds = 4           # number of folds trained for DINOv2
    hidden_dim = 512      # must match training


TARGET_ORDER = ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g"]
TARGET2IDX = {t: i for i, t in enumerate(TARGET_ORDER)}


# ==========================================================
# Set all seeds for reproducibility
# ==========================================================
def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)


# ==========================================================
# Dataset for test images
# ==========================================================
class TestImageDataset(Dataset):
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
        return img, idx


# ==========================================================
# DINOv2 Regressor (matches training architecture)
# ==========================================================
class DinoRegressor(nn.Module):
    def __init__(self, backbone_name, num_targets=5, hidden_dim=512, freeze_backbone=True, pretrained=False):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,   # pretrained not needed for inference
            num_classes=0,
        )
        feat_dim = self.backbone.num_features

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.head = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_targets),
        )

    def forward(self, x):
        feat = self.backbone(x)
        out = self.head(feat)
        return out


# ==========================================================
# Step 0: Load test.csv and get unique images
# ==========================================================
test_df = pd.read_csv(TEST_CSV)
print("test_df:", test_df.shape)
print(test_df.head())

df_imgs = test_df[["image_path"]].drop_duplicates().reset_index(drop=True)
N = len(df_imgs)
print("Unique test images:", N)


# ==========================================================
# Step 1: Read backbone_name from fold-1 checkpoint, build transform
# ==========================================================
example_ckpt_path = os.path.join(WEIGHT_DIR, "dinov2_fold1.pth")
print("Loading example checkpoint:", example_ckpt_path)
example_ckpt = torch.load(example_ckpt_path, map_location=CFG.device)

backbone_name = example_ckpt.get("backbone_name", "vit_large_patch14_dinov2.lvd142m")
print("Backbone name:", backbone_name)

# Build inference transform based on backbone
tmp_model = timm.create_model(backbone_name, pretrained=False, num_classes=0)
data_config = resolve_model_data_config(tmp_model)
valid_transform = create_transform(**data_config, is_training=False)
del tmp_model


# ==========================================================
# Step 2: DataLoader
# ==========================================================
test_ds = TestImageDataset(df_imgs, IMG_ROOT, valid_transform)
test_loader = DataLoader(
    test_ds,
    batch_size=CFG.batch_size,
    shuffle=False,
    num_workers=CFG.num_workers,
    pin_memory=True,
)


# ==========================================================
# Step 3: 5-fold ensemble prediction
# ==========================================================
preds_sum = np.zeros((N, len(TARGET_ORDER)), dtype=np.float32)

with torch.no_grad():
    for fold in range(1, CFG.n_folds + 1):
        ckpt_path = os.path.join(WEIGHT_DIR, f"dinov2_fold{fold}.pth")
        print("Loading fold weight:", ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=CFG.device)

        model = DinoRegressor(
            backbone_name=ckpt.get("backbone_name", backbone_name),
            num_targets=len(TARGET_ORDER),
            hidden_dim=CFG.hidden_dim,
            freeze_backbone=True,
            pretrained=False,
        ).to(CFG.device)

        model.load_state_dict(ckpt["model"])
        model.eval()

        for imgs, idxs in test_loader:
            imgs = imgs.to(CFG.device)
            idxs = idxs.numpy()
            out = model(imgs)  # [B, 5]
            preds_sum[idxs] += out.cpu().numpy()

# Average predictions across folds
preds_final = preds_sum / CFG.n_folds

df_pred_img = df_imgs.copy()
for i, t in enumerate(TARGET_ORDER):
    df_pred_img[t] = preds_final[:, i]

print("df_pred_img:", df_pred_img.shape)
print(df_pred_img.head())


# ==========================================================
# Step 4: Flatten to sample_id / target format and save submission.csv
# ==========================================================
df_long = df_pred_img.melt(
    id_vars="image_path",
    value_vars=TARGET_ORDER,
    var_name="target_name",
    value_name="target",
)

sub = test_df.merge(
    df_long,
    on=["image_path", "target_name"],
    how="left",
)[["sample_id", "target"]]

sub.to_csv("submission.csv", index=False)
print("Saved submission.csv:", sub.shape)
print(sub.head())
