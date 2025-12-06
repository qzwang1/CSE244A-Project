# DINOv2 – Image Only Baseline

This folder contains the **DINOv2 baseline model** for the CSIRO Image2Biomass challenge.

- Model: `vit_large_patch14_dinov2.lvd142m`
- Input: **image only**
- No metadata
- Training: **5-fold CV**
- All backbone weights **frozen**
- Only a small regression head is trained

Private leaderboard score: **0.62 weighted R²**  
Weights: https://drive.google.com/drive/folders/1Ise3v8TwBIBXmKCaA3toeAiDIWQI2xc8?usp=sharing

---

## What was changed

### 1. Use pretrained DINOv2 as feature extractor

```python
self.backbone = timm.create_model(
    backbone_name,
    pretrained=True,
    num_classes=0   # return features only
)
```

### 2. Freeze all backbone parameters

```python
for p in self.backbone.parameters():
    p.requires_grad = False
```

### 3. Add a small regression head

```python
self.head = nn.Sequential(
    Linear(feat_dim, 512),
    ReLU(),
    Dropout(0.3),
    Linear(512, 5)
)
```

Only this part is trained.

---

## Training

- Loss: `MSELoss`
- Metric: **Weighted R²**
- Early stopping on validation R²

```python
if valid_r2 > best_r2:
    save checkpoint
```

---

## Inference

Use each fold checkpoint:

```python
dinov2_fold{0..4}.pth
```

Average predictions across folds to produce final `submission.csv`.

---

## Notes

- Very simple baseline, **no metadata**
- Good reference to measure improvements from two-stage meta models
- Backbone remains fixed → fast and stable training

