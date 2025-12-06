import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import timm

# ==========================================================
# Path configuration: update according to your Kaggle datasets
# ==========================================================
DATA_DIR   = "/kaggle/input/csiro-biomass"                # contains train.csv / test.csv / train/ test/ images
STAGE1_DIR = "/kaggle/input/csiro-stage1-weights/pytorch/default/1"  # contains stage1_meta_fold*.pth
STAGE2_DIR = "/kaggle/input/csiro-stage2-weights/pytorch/default/1"  # contains stage2_main_fold*.pth

TEST_CSV = os.path.join(DATA_DIR, "test.csv")
IMG_ROOT = DATA_DIR   # image_path is relative to this directory


class CFG:
    model_name = "tf_efficientnetv2_s"
    img_height = 512          # must match Stage1/Stage2 training
    img_width = 1024
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 16
    num_workers = 4
    n_folds = 5


TARGET_ORDER = ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g"]
TARGET2IDX = {t: i for i, t in enumerate(TARGET_ORDER)}


# ==========================================================
# Utils
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

test_transform = T.Compose([
    T.Resize((CFG.img_height, CFG.img_width)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])


class TestImageDataset(Dataset):
    def __init__(self, df, img_root, transform):
        self.df = df.reset_index(drop=True)
        self.img_root = img_root
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.loc[idx]
        img_rel = row["image_path"]
        img_path = os.path.join(self.img_root, img_rel)
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img, idx


# ==========================================================
# Stage 1 model (image -> NDVI / Height / State / Species)
# Must match the Stage1 training script
# ==========================================================
class Stage1MetaNet(nn.Module):
    def __init__(self, backbone_name, num_state, num_species):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=False,
            num_classes=0,
            drop_rate=0.2,
            drop_path_rate=0.1,
        )
        d = self.backbone.num_features

        self.ndvi_head = nn.Linear(d, 1)
        self.height_head = nn.Linear(d, 1)
        self.state_head = nn.Linear(d, num_state)
        # Name must match training code: species_head
        self.species_head = nn.Linear(d, num_species)

    def forward(self, x):
        feat = self.backbone(x)
        ndvi = self.ndvi_head(feat).squeeze(-1)
        height = self.height_head(feat).squeeze(-1)
        state_logits = self.state_head(feat)
        species_logits = self.species_head(feat)
        return ndvi, height, state_logits, species_logits


# ==========================================================
# Stage 2 model (image + meta -> 5 biomass targets)
# Must match the Stage2 training script
# ==========================================================
class CSIROStage2Model(nn.Module):
    def __init__(
        self,
        model_name,
        metadata_dim,
        num_classes=5,
        in_chans=3,
        fusion_dim=256,
        dropout=0.3,
        meta_dropout=0.2,
    ):
        super().__init__()

        self.metadata_dim = metadata_dim

        self.image_model = timm.create_model(
            model_name=model_name,
            pretrained=False,
            in_chans=in_chans,
            num_classes=0,
            drop_rate=0.2,
            drop_path_rate=0.1,
        )
        self.image_feature_dim = self.image_model.num_features

        self.meta_net = nn.Sequential(
            nn.Linear(metadata_dim, 128),
            nn.ReLU(),
            nn.Dropout(meta_dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(meta_dropout),
        )
        self.meta_out_dim = 64

        combined_dim = self.image_feature_dim + self.meta_out_dim

        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(fusion_dim // 2, num_classes)

    def forward(self, image, meta):
        img_feat = self.image_model(image)
        meta_feat = self.meta_net(meta)
        feat = torch.cat([img_feat, meta_feat], dim=1)
        fusion = self.fusion(feat)
        out = self.head(fusion)
        return out


# ==========================================================
# Step 0: read test.csv and build unique image list
# ==========================================================
test_df = pd.read_csv(TEST_CSV)
print("test_df:", test_df.shape)
print(test_df.head())

df_imgs = test_df[["image_path"]].drop_duplicates().reset_index(drop=True)
num_test_img = len(df_imgs)
print("unique test images:", num_test_img)


# ==========================================================
# Step 1: Stage1 5-fold ensemble to predict metadata on test
#         Get pred_ndvi, pred_height, pstate_*, psp_*
# ==========================================================
print("\n==== Stage1: predict meta on test images ====")

# Load one checkpoint to inspect number of states/species
example_ckpt = torch.load(
    os.path.join(STAGE1_DIR, "stage1_meta_fold1.pth"),
    map_location=CFG.device,
)
state_codes = example_ckpt["state_codes"]
species_codes = example_ckpt["species_codes"]
num_state = len(state_codes)
num_species = len(species_codes)
print("num_state:", num_state, "num_species:", num_species)

test_ds_s1 = TestImageDataset(df_imgs, IMG_ROOT, test_transform)
test_loader_s1 = DataLoader(
    test_ds_s1,
    batch_size=CFG.batch_size,
    shuffle=False,
    num_workers=CFG.num_workers,
    pin_memory=True,
)

N = num_test_img
ndvi_sum = np.zeros(N, dtype=np.float32)
h_sum = np.zeros(N, dtype=np.float32)
state_logit_sum = np.zeros((N, num_state), dtype=np.float32)
spec_logit_sum = np.zeros((N, num_species), dtype=np.float32)

for fold in range(1, CFG.n_folds + 1):
    ckpt_path = os.path.join(STAGE1_DIR, f"stage1_meta_fold{fold}.pth")
    print("  loading Stage1:", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=CFG.device)

    model_s1 = Stage1MetaNet(
        backbone_name=CFG.model_name,
        num_state=num_state,
        num_species=num_species,
    ).to(CFG.device)
    model_s1.load_state_dict(ckpt["model"])
    model_s1.eval()

    with torch.no_grad():
        for imgs, idxs in test_loader_s1:
            imgs = imgs.to(CFG.device)
            idxs = idxs.numpy()

            pred_ndvi, pred_h, s_logits, sp_logits = model_s1(imgs)

            ndvi_sum[idxs] += pred_ndvi.cpu().numpy()
            h_sum[idxs] += pred_h.cpu().numpy()
            state_logit_sum[idxs] += s_logits.cpu().numpy()
            spec_logit_sum[idxs] += sp_logits.cpu().numpy()

# average over folds
ndvi_pred = ndvi_sum / CFG.n_folds
h_pred = h_sum / CFG.n_folds

state_logits_avg = torch.from_numpy(state_logit_sum / CFG.n_folds)
species_logits_avg = torch.from_numpy(spec_logit_sum / CFG.n_folds)
softmax = nn.Softmax(dim=1)
state_prob = softmax(state_logits_avg).numpy()      # [N, num_state]
species_prob = softmax(species_logits_avg).numpy()  # [N, num_species]

# Build meta feature table consistent with Stage2 training
records = []
for i in range(N):
    rec = {
        "image_path": df_imgs.loc[i, "image_path"],
        "pred_ndvi": float(ndvi_pred[i]),
        "pred_height": float(h_pred[i]),
    }
    for j in range(num_state):
        rec[f"pstate_{j}"] = float(state_prob[i, j])
    for j in range(num_species):
        rec[f"psp_{j}"] = float(species_prob[i, j])
    records.append(rec)

df_meta_test = pd.DataFrame(records)
print("df_meta_test:", df_meta_test.shape)
print(df_meta_test.head())


# ==========================================================
# Step 2: Stage2 5-fold ensemble, image + meta → biomass
# ==========================================================
print("\n==== Stage2: predict biomass on test images ====")

# Load META_FEATURES from one Stage2 checkpoint to keep feature order
example_s2 = torch.load(
    os.path.join(STAGE2_DIR, "stage2_main_fold1.pth"),
    map_location=CFG.device,
)
META_FEATURES = example_s2["META_FEATURES"]
meta_dim = len(META_FEATURES)
print("META_FEATURES dim:", meta_dim)

# Merge image list with test meta
df_test_all = df_imgs.merge(df_meta_test, on="image_path", how="left")
assert not df_test_all[META_FEATURES].isnull().any().any(), "Some meta features are NaN!"

class Stage2TestDataset(Dataset):
    def __init__(self, df, img_root, transform, meta_features):
        self.df = df.reset_index(drop=True)
        self.img_root = img_root
        self.transform = transform
        self.meta_features = meta_features

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.loc[idx]
        img_rel = row["image_path"]
        img_path = os.path.join(self.img_root, img_rel)
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        meta = row[self.meta_features].values.astype("float32")
        return img, torch.tensor(meta), idx


test_ds_s2 = Stage2TestDataset(df_test_all, IMG_ROOT, test_transform, META_FEATURES)
test_loader_s2 = DataLoader(
    test_ds_s2,
    batch_size=CFG.batch_size,
    shuffle=False,
    num_workers=CFG.num_workers,
    pin_memory=True,
)

N2 = len(df_test_all)
preds_sum = np.zeros((N2, len(TARGET_ORDER)), dtype=np.float32)

with torch.no_grad():
    for fold in range(1, CFG.n_folds + 1):
        ckpt_path = os.path.join(STAGE2_DIR, f"stage2_main_fold{fold}.pth")
        print("  loading Stage2:", ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=CFG.device)

        model_s2 = CSIROStage2Model(
            model_name=CFG.model_name,
            metadata_dim=meta_dim,
            num_classes=len(TARGET_ORDER),
        ).to(CFG.device)
        model_s2.load_state_dict(ckpt["model"])
        model_s2.eval()

        for imgs, meta, idxs in test_loader_s2:
            imgs = imgs.to(CFG.device)
            meta = meta.to(CFG.device)
            idxs = idxs.numpy()

            out = model_s2(imgs, meta)  # [B, 5]
            preds_sum[idxs] += out.cpu().numpy()

preds_final = preds_sum / CFG.n_folds

df_pred_img = df_test_all[["image_path"]].copy()
for i, t in enumerate(TARGET_ORDER):
    df_pred_img[t] = preds_final[:, i]

print("df_pred_img:", df_pred_img.shape)
print(df_pred_img.head())


# ==========================================================
# Step 3: expand back to (sample_id, target) and save submission.csv
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
