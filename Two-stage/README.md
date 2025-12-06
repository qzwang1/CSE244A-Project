# CSIRO – Two-Stage Pipeline (Overview)

This solution uses a **two-stage model**:

1. Stage-1: **image → metadata**
2. Stage-2: **image + metadata → biomass**

Final private leaderboard score: **0.54 weighted R²**  
Download weights: https://drive.google.com/drive/folders/1eTi6LoNKpohOnJjlZHizbY3mkvGK6OyB?usp=sharing

---

## Stage-1 (Image → Metadata)

**Predicted:**

- NDVI (regression)
- Height (regression)
- State (classification)
- Species (classification)

**Modification:**
Added 4 prediction heads after EfficientNetV2 backbone:

```python
ndvi_head   = Linear(d, 1)
height_head = Linear(d, 1)
state_head  = Linear(d, num_state)
species_head = Linear(d, num_species)
```

Save OOF predictions as `train_meta_oof.csv`:

```
pred_ndvi, pred_height, pstate_*, psp_*
```

---

## Stage-2 (Image + Metadata → Biomass)

**Inputs:**

```
image features + Stage-1 predicted metadata
```

**Modification:**
Small MLP for meta, fused only at the end:

```python
meta -> Linear(128) -> ReLU -> Dropout
     -> Linear(64)  -> ReLU -> Dropout

feat = concat(img_feat, meta_feat)
out  = head(feat)
```

**Targets (5 outputs):**

```
Dry_Green_g
Dry_Dead_g
Dry_Clover_g
GDM_g
Dry_Total_g
```

Training metric = **weighted R²** (same as competition).

---

## Final Pipeline

```
for each image:
    Stage-1 ensemble → metadata
    Stage-2 ensemble → 5 biomass values
    = fold-averaged predictions
```



