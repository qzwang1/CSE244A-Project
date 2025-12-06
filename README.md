# CSIRO - Image2Biomass Prediction (CSE244A)

This repository contains our solution for the CSIRO Image2Biomass Prediction challenge.  
We implemented four different model pipelines, trained on AutoDL GPUs, and used the official Kaggle dataset.

## Folder Structure

- Dinov2  
- EfficientNet-5fold  
- Meta-dropout  
- Two-stage  
- README.md

## Each folder includes

- model description  
- training and evaluation scripts  
- instructions on how to run

## Models

- **Dinov2**
  - pretrained ViT backbone with regression head

- **EfficientNet V2**
  - 5-fold ensemble

- **Meta-dropout**
  - adaptive dropout for robustness

- **Two-stage model**
  - feature extractor + separate regressor

## Training Environment

All experiments were run on AutoDL GPU servers.  
Dataset and logs use absolute paths defined inside each folder.

## Dataset

We used the official Kaggle dataset:

https://www.kaggle.com/competitions/csiro-biomass/data

Code automatically detects the dataset under `/kaggle/input`.

## Weights

Trained model weights are shared via Google Drive.  
Each model folder explains how to download and where to place them.

## Results

We report weighted R² on validation.  
Final submission uses ensemble averaging of models.
