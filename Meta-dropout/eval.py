import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as T
import timm

# ==========================================================
# Paths (update according to your Kaggle Datasets)
# ==========================================================
DATA_DIR   = "/kaggle/input/csiro-biomass"   # contains train.csv / test.csv / images
WEIGHT_DIR = "/kaggle/input/effi-meta-v2/pytorch/default/1"  # uploaded weights

TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV  = os.path.join(DATA_DIR, "test.csv")
IMG_ROOT  = DATA_DIR

class CFG:
    model_name = "tf_efficientnetv2_s"
    img_height = 384
    img_width  = 768
    device = "cuda" if torch.cuda.is_available() else "cpu"

TARGET_ORDER = ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g"]
TARGET2IDX = {t: i for i, t in enumerate(TARGET_ORDER)}

# metadata_dim must match the trained model
METADATA_DIM = 22


# ==========================================================
# Model definition (same architecture as training, no pretrained)
# ==========================================================
class CSIROModel(nn.Module):
    def __init__(
        self,
        model_name,
        pretrained=False,
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

        # image backbone
        self.image_model = timm.create_model(
            model_name=model_name,
            pretrained=pretrained,
            in_chans=in_chans,
            num_classes=0,
        )
        self.image_feature_dim = self.image_model.num_features

        # metadata encoder
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

        # fusion + regression head
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
        image_features = self.image_model(image)

        # during inference we use zero-vector for metadata
        if self.metadata_dim > 0:
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


def load_model(weight_path):
    model = CSIROModel(
        model_name=CFG.model_name,
        pretrained=False,
        in_chans=3,
        num_classes=len(TARGET_ORDER),
        metadata_dim=METADATA_DIM,
        fusion_dim=256,
        dropout=0.2,
        meta_dropout_p=0.5,
    ).to(CFG.device)

    state = torch.load(weight_path, map_location=CFG.device)
    model.load_state_dict(state)
    model.eval()
    return model


# ==========================================================
# Load 5-fold weights
# ==========================================================
weight_files = [
    "csiro_meta_robust_fold0.pth",
    "csiro_meta_robust_fold1.pth",
    "csiro_meta_robust_fold2.pth",
    "csiro_meta_robust_fold3.pth",
    "csiro_meta_robust_fold4.pth",
]

print("Weight dir:", WEIGHT_DIR)
print("Files in weight dir:", os.listdir(WEIGHT_DIR))

models = []
for wf in weight_files:
    p = os.path.join(WEIGHT_DIR, wf)
    print("Loading:", p)
    models.append(load_model(p))

print("Loaded", len(models), "models.")


# ==========================================================
# TTA (original + horizontal flip)
# ==========================================================
tta_transforms = [
    T.Compose([
        T.Resize((CFG.img_height, CFG.img_width)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ]),
    T.Compose([
        T.Resize((CFG.img_height, CFG.img_width)),
        T.RandomHorizontalFlip(p=1.0),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ]),
]


# ==========================================================
# Read test.csv (5 rows per image)
# ==========================================================
test_df = pd.read_csv(TEST_CSV)
print("test_df shape:", test_df.shape)
print(test_df.head())

unique_images = test_df["image_path"].unique()
print("Unique test images:", len(unique_images))


# ==========================================================
# 5-fold * TTA averaging for each image
# ==========================================================
pred_dict = {}  # image_path -> (5,)

with torch.no_grad():
    for img_path in unique_images:
        full_path = os.path.join(IMG_ROOT, img_path)
        img = Image.open(full_path).convert("RGB")

        all_preds = []

        for tta in tta_transforms:
            x = tta(img).unsqueeze(0).to(CFG.device)

            for m in models:
                y = m(x, metadata=None)
                all_preds.append(y.cpu().numpy()[0])

        all_preds = np.stack(all_preds, axis=0)
        pred_mean = all_preds.mean(axis=0)

        pred_dict[img_path] = pred_mean

print("Num predicted images:", len(pred_dict))


# ==========================================================
# Convert back to (sample_id, target) rows
# ==========================================================
rows = []
for _, row in test_df.iterrows():
    sid = row["sample_id"]
    ip = row["image_path"]
    tname = row["target_name"]
    t_idx = TARGET2IDX[tname]

    value = float(pred_dict[ip][t_idx])
    rows.append((sid, value))

sub = pd.DataFrame(rows, columns=["sample_id", "target"])
sub.to_csv("submission.csv", index=False)
print("Saved submission.csv, shape:", sub.shape)
print(sub.head())
