# Siamese ISIC 2020 (Melanoma vs Normal)

**Difficulty:** Hard  
**Goal:** ~0.80 accuracy on a held-out **test** split and report **AUC**.

## Problem & Approach
ISIC-2020 dermoscopic images are imbalanced and visually subtle. Rather than a single-image classifier, I train a **Siamese** network that learns an embedding where *same-class* pairs are close and *different-class* pairs are far. A pairwise head predicts “same vs different” with BCE loss. I then turn this into an **image-level classifier** by comparing each query to a balanced support set and thresholding a similarity score calibrated on the **validation** split.

- **Backbone:** ResNet-18 (ImageNet). Global-pool → 512-D embedding.  
- **Pair head:** |e₁ − e₂| → MLP → sigmoid.  
- **Metrics:** Accuracy, AUC, F1, sensitivity (melanoma recall), specificity.

## Data
- **Dataset:** SIIM-ISIC 2020 (Kaggle).
- **Expected layout**
  ```
  DATA_ROOT/
    ISIC_2020_Training_JPEG/
    ISIC_2020_Training_GroundTruth_v2.csv
    ISIC_2020_Training_Metadata_v2.csv   # optional for patient/lesion-aware split
  ```
- CSV splits are written to `DATA_ROOT/splits/{train,val,test}.csv`.

## How to run (Windows PowerShell examples)

### 1) Make splits
```powershell
python recognition/siamese-isic2020-47525496/train.py split --root "D:\Test Data"
```

### 2) Train (pair-level)
```powershell
python recognition/siamese-isic2020-47525496/train.py train --root "D:\Test Data" `
  --epochs 15 --batch_size 64 --lr 3e-4 --img_size 224 --num_workers 4
```

### 3) Evaluate pair-level (AUC/ACC/F1; optional threshold search)
```powershell
# choose threshold τ on VAL
python recognition/siamese-isic2020-47525496/predict.py eval-pairs --root "D:\Test Data" `
  --ckpt "D:\Test Data\runs\siamese\best.pth" --split val --find_best_threshold

# reuse saved τ on TEST
python recognition/siamese-isic2020-47525496/predict.py eval-pairs --root "D:\Test Data" `
  --ckpt "D:\Test Data\runs\siamese\best.pth" --split test --use_saved_threshold
```

### 4) Prototype image-level classifier (support-set comparison)
```powershell
# pick τ on VAL
python recognition/siamese-isic2020-47525496/predict.py proto --root "D:\Test Data" `
  --ckpt "D:\Test Data\runs\siamese\best.pth" --split val `
  --support_per_class 256 --img_size 256 --l2norm --find_best_threshold

# evaluate TEST with saved τ
python recognition/siamese-isic2020-47525496/predict.py proto --root "D:\Test Data" `
  --ckpt "D:\Test Data\runs\siamese\best.pth" --split test `
  --support_per_class 256 --img_size 256 --l2norm --use_saved_threshold --save_preds
```

### 5) Driver
```powershell
# optional: pick τ on VAL, then always run TEST
python recognition\siamese-isic2020-47525496\driver.py `
  --root "D:\Test Data" `
  --ckpt "D:\Test Data\runs\siamese\best.pth" `
  --find_threshold
python recognition\siamese-isic2020-47525496\driver.py `
  --root "D:\Test Data" `
  --ckpt "D:\Test Data\runs\siamese\best.pth"

# evaluate the pair head directly (optional)
python recognition\siamese-isic2020-47525496\driver.py `
  --root "D:\Test Data" `
  --ckpt "D:\Test Data\runs\siamese\best.pth" `
  --mode pairs `
  --find_threshold
```

## Dependencies
- Python 3.10+
- PyTorch, TorchVision
- pandas, numpy, scikit-learn, Pillow, tqdm, matplotlib

Quick install:
```bash
pip install torch torchvision pandas numpy scikit-learn pillow tqdm matplotlib
```

## Results

### Image-level (support-set prototype)
**Setup:** K = 256 supports/class from TRAIN, `img_size = 256`, L2-normalised embeddings.  
Threshold τ chosen on **VAL** (Youden’s J) and reused on **TEST**.

| Split |  Acc  |  AUC  |  F1   | Sens | Spec |  Thr  |
|-----:|:-----:|:-----:|:-----:|:----:|:----:|:-----:|
| Val  | 0.8098 | 0.9009 | 0.1417 | 0.848 | 0.809 | 0.134 |
| Test | 0.8054 | 0.8642 | 0.1265 | 0.769 | 0.806 | 0.134 |

**Confusion (VAL):** TP = 78, FP = 931, FN = 14, TN = 3946  
**Confusion (TEST):** TP = 70, FP = 946, FN = 21, TN = 3932

Notes: The ≥0.80 accuracy target is met on **TEST**. F1 remains low due to heavy class imbalance; adjusting τ trades sensitivity vs specificity.

## Repository layout
- `modules.py` — model/components  
- `dataset.py` — loaders and split logic  
- `train.py` — split + train entry points  
- `predict.py` — evaluation and prototype image-level classifier  
- `driver.py` — simple driver for VAL threshold + TEST eval  
- `README.md` — this file

