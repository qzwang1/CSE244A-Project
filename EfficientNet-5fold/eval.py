import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as T
import timm

# ==========================================================
# Config
# ==========================================================
class CFG:
    model_name = "tf_efficientnetv2_s"
    img_height = 256
    img_width = 512
    device = "cuda" if torch.cuda.is_available() else "cpu"
    folds = [0, 1, 2, 3, 4]   # specify the folds you trained

TARGET_ORDER = ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g"]
TARGET2IDX = {t: i for i, t in enumerate(TARGET_ORDER)}

# ==========================================================
# Detect dataset directory under /kaggle/input
# The directory must contain train.csv / test.csv / sample_submission.csv
# ==========================================================
data_root = None
for dirname, _, filenames in os.walk("/kaggle/input"):
    if "train.csv" in filenames and "test.csv" in filenames:
        data_root = dirname
        break

print("Detected data_root:", data_root)
assert data_root is not None, "Failed to find dataset under /kaggle/input"

TEST_CSV = os.path.join(data_root, "test.csv")

# ==========================================================
# Weight directory: this should be your uploaded dataset name
# Example:
#   if you uploaded a dataset called "effv2s-weights",
#   then set:
#       WEIGHT_DIR = "/kaggle/input/effv2s-weights"
#
# It must contain:
#   effv2s_fold0.pth ... effv2s_fold4.pth
# ==========================================================
WEIGHT_DIR = "/kaggle/input/efficientv2-5flod/pytorch/default/1"

# Check available weight files
print("Available weight files:")
for f in os.listdir(WEIGHT_DIR):
    print("  ", f)

# ==========================================================
# Transforms (same as validation transform used during training)
# ==========================================================
transform = T.Compose([
    T.Resize((CFG.img_height, CFG.img_width)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

# ==========================================================
# Model (must use exactly the same structure as training)
# Pretrained is not needed for inference, since weights are loaded below
# ==========================================================
class EffNetV2Regressor(nn.Module):
    def __init__(self, model_name, num_targets=5):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=False,    # not needed for inference
            num_classes=0
        )
        in_features = self.backbone.num_features
        self.head = nn.Linear(in_features, num_targets)

    def forward(self, x):
        feat = self.backbone(x)
        out = self.head(feat)
        return out


# ==========================================================
# Read test.csv (locally may contain only 1 row, but on Kaggle will be replaced
# with full test set)
# ==========================================================
test_df = pd.read_csv(TEST_CSV)
print("test_df shape:", test_df.shape)
print(test_df.head())

# We need predictions for each unique image_path
image_paths = test_df["image_path"].unique()
print("Unique test images:", len(image_paths))

# ==========================================================
# Make predictions for each image, averaged over folds
# ==========================================================
pred_dict = {}   # img_path -> 5-dim prediction

with torch.no_grad():
    for img_path in image_paths:
        full_path = os.path.join(data_root, img_path)
        img = Image.open(full_path).convert("RGB")
        img = transform(img).unsqueeze(0).to(CFG.device)   # [1,3,H,W]

        fold_sum = np.zeros(len(TARGET_ORDER), dtype="float32")

        # Accumulate predictions across folds
        for fold in CFG.folds:
            weight_file = os.path.join(WEIGHT_DIR, f"effv2s_fold{fold}.pth")
            assert os.path.exists(weight_file), f"Missing weight file: {weight_file}"

            model = EffNetV2Regressor(CFG.model_name, num_targets=len(TARGET_ORDER)).to(CFG.device)
            state = torch.load(weight_file, map_location=CFG.device)
            model.load_state_dict(state)
            model.eval()

            preds = model(img)              # [1,5]
            preds = preds.cpu().numpy()[0]  # (5,)
            fold_sum += preds

        # Average over folds
        fold_mean = fold_sum / len(CFG.folds)
        pred_dict[img_path] = fold_mean

# ==========================================================
# Reconstruct submission: sample_id / target
# ==========================================================
rows = []

for _, row in test_df.iterrows():
    sid = row["sample_id"]
    img_path = row["image_path"]
    tname = row["target_name"]
    t_idx = TARGET2IDX[tname]

    value = float(pred_dict[img_path][t_idx])
    rows.append((sid, value))

sub = pd.DataFrame(rows, columns=["sample_id", "target"])
print(sub.head())

out_path = "/kaggle/working/submission.csv"
sub.to_csv(out_path, index=False)
print("Saved submission to:", out_path)
