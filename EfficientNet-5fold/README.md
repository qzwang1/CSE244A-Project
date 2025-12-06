# EfficientNetV2 (5-fold Ensemble)

This folder contains a 5-fold ensemble model based on EfficientNetV2 for the CSIRO Image2Biomass challenge.  
The model achieved a **Kaggle private leaderboard score of 0.52 (weighted R²)**.

## Summary

- Backbone: tf_efficientnetv2_s  
- Image size: 256 x 512  
- Loss: MSE  
- Metric: Weighted R²  
- Training: 5-fold cross validation  
- Final prediction: average over folds  
- Outputs 5 biomass targets  
- Only modification: the original classification head was replaced by a 5-dimensional regression layer. No other architectural changes were made to the backbone.

## Files

- train.py — 5-fold training  
- eval.py — inference and submission generation  
- README.md — this file

## Weights

All fold checkpoints are available via Google Drive:

https://drive.google.com/drive/folders/1MK19Br5qc9dErkmwsG9tvbF6lghVGSm4?usp=drive_link

Each file follows the format:

effv2s_fold0.pth  
effv2s_fold1.pth  
effv2s_fold2.pth  
effv2s_fold3.pth  
effv2s_fold4.pth

## Training

All experiments ran on AutoDL GPU servers.  
Datasets used absolute paths under `/root/dataset`.

## Inference

Run eval.py to generate `submission.csv`.  
The script loads each fold weight, predicts per image, and averages results.

Output format:

sample_id,target

## Results

Weighted R² across folds was around 0.50–0.54,  
with a mean of **≈ 0.52** on the Kaggle private leaderboard.

## Notes

This model serves as the baseline for comparison with other approaches in the project.
