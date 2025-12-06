# EfficientNetV2 + Meta-Dropout (5-fold, TTA)

This folder contains the meta-robust version of the EfficientNetV2 model for the CSIRO Image2Biomass challenge.

## Summary

- Backbone: tf_efficientnetv2_s  
- Image size: 384 × 768  
- Training: 5-fold cross validation  
- Inference: 5-fold × TTA (original + horizontal flip)  
- Final output: 5 biomass targets  
- Private LB score: ~0.54 (weighted R²)

## Main modifications over the baseline

Compared to the plain EfficientNetV2 baseline:

- A small MLP is added to encode metadata (22 → 64 dim)
- Image features and metadata features are concatenated and fused in the final layers
- The convolutional backbone is unchanged
- A meta-dropout strategy improves robustness:
  - Warmup: no metadata
  - After warmup: random image-only vs image+metadata batches
  - Stochastic zeroing of metadata inside the model
- Inference uses **image-only mode** (metadata set to zero vector)
- Resolution increased to 384 × 768
- TTA applied at inference (original + horizontal flip)

## Files

- train_meta.py — training with metadata and meta-dropout
- eval_meta.py — inference with 5-fold × TTA
- README.md — this file

## Weights

The 5-fold checkpoints are available via Google Drive:

https://drive.google.com/drive/folders/15Jf9T2f9md5w-EEuOibV1CWECNdpiinU?usp=sharing

File names:

- csiro_meta_robust_fold0.pth
- csiro_meta_robust_fold1.pth
- csiro_meta_robust_fold2.pth
- csiro_meta_robust_fold3.pth
- csiro_meta_robust_fold4.pth


## Inference

Run:

python eval_meta.py


## Notes

This model mainly increases robustness to missing metadata and improves generalization through:
- late fusion
- meta-dropout
- TTA
- higher resolution

It improves upon the baseline EfficientNetV2 results.
