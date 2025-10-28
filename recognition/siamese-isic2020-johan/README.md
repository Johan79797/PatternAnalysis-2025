# Siamese ISIC 2020 (Melanoma vs Normal)

**Difficulty:** Hard  
**Goal:** ~0.80 accuracy on a held-out **test** split, also report **AUC**.

## Problem & Approach
ISIC 2020 dermoscopic images are highly imbalanced and visually subtle. Instead of a single-image classifier, we train a **Siamese** model to learn a lesion embedding where *same-class* pairs are close and *different-class* pairs are far. The pair head predicts “same vs different” with BCE loss; we turn this into an **image-level classifier** by comparing a query to a balanced support set and thresholding a similarity score calibrated on **validation**.

**Backbone:** ResNet-18 (ImageNet), global-pool → 512-D, pair head on |e₁−e₂| → MLP → sigmoid.  
**Metrics:** Accuracy, AUC, F1, sensitivity (recall of melanoma), specificity.

## Data
- **Dataset:** SIIM-ISIC 2020 (Kaggle).  
- **Expected layout:**  
  ```
  DATA_ROOT/
    ISIC_2020_Training_JPEG/
    ISIC_2020_Training_GroundTruth_v2.csv
    ISIC_2020_Training_Metadata_v2.csv   # optional (for patient/lesion aware split)
  ```
- We write CSV splits to `DATA_ROOT/splits/{train,val,test}.csv`.  
- ⚠️ Do **not** commit datasets or model files.

## How to run (Windows PowerShell examples)

### 1) Make splits
```powershell
python recognition/siamese-isic2020-johan/train.py split --root "D:\Test Data"
```

### 2) Train (pair-level)
```powershell
python recognition/siamese-isic2020-johan/train.py train --root "D:\Test Data" `
  --epochs 15 --batch_size 64 --lr 3e-4 --img_size 224 --num_workers 4
```

### 3) Evaluate pair-level (AUC/ACC/F1 + optional threshold search)
```powershell
python recognition/siamese-isic2020-johan/predict.py eval-pairs --root "D:\Test Data" `
  --ckpt "D:\Test Data\runs\siamese\best.pth" --split val --find_best_threshold

python recognition/siamese-isic2020-johan/predict.py eval-pairs --root "D:\Test Data" `
  --ckpt "D:\Test Data\runs\siamese\best.pth" --split test --use_saved_threshold
```

### 4) (Optional) Prototype image-level classifier
```powershell
python recognition/siamese-isic2020-johan/predict.py proto --root "D:\Test Data" `
  --ckpt "D:\Test Data\runs\siamese\best.pth" --split val `
  --support_per_class 256 --img_size 256 --l2norm --find_best_threshold

python recognition/siamese-isic2020-johan/predict.py proto --root "D:\Test Data" `
  --ckpt "D:\Test Data\runs\siamese\best.pth" --split test `
  --support_per_class 256 --img_size 256 --l2norm --use_saved_threshold --save_preds
```

## Dependencies
- Python 3.10+
- PyTorch, TorchVision
- pandas, numpy, scikit-learn, Pillow, tqdm, matplotlib

Quick install:
```bash
pip install torch torchvision pandas numpy scikit-learn pillow tqdm matplotlib
```

## Results (fill after you run)
- **Pair-level (VAL → TEST with fixed τ):** AUC, ACC, F1, sens, spec.  
- **Image-level (prototype, optional):** Accuracy ≥0.80 target; report AUC, sens/spec, confusion matrix.

## Repo structure (required by assignment)
- `modules.py` – model/components  
- `dataset.py` – loaders/splits  
- `train.py` – split & train  
- `predict.py` – evaluation / prototype classifier  
- `README.md` – this file

*This layout and commit practice align with the assessment checklist and “Recognition Problem” file requirements.*
---

## Results (image-level)

**Setup:** K=256 supports/class from TRAIN, img_size=256, L2-normalised embeddings,  
threshold t chosen on **VAL** by Youden�s J and reused on **TEST**.

| Split | Acc   | AUC   | F1    | Sens | Spec | Thr   |
|------:|:-----:|:-----:|:-----:|:----:|:----:|:-----:|
| Val   | 0.832 | 0.812 | 0.139 | 0.728| 0.834| 0.496 |
| Test  | 0.822 | 0.749 | 0.105 | 0.571| 0.827 | 0.496 |

**Confusion (TEST):** TP=52, FP=844, FN=39, TN=4034

### Test Driver (marker-friendly)

Pick t on **VAL** and then evaluate **TEST**:

`powershell
python recognition\siamese-isic2020-johan\driver.py 
  --root "D:\Test Data" 
  --ckpt "D:\Test Data\runs\siamese\best.pth" 
  --find_threshold
python recognition\siamese-isic2020-johan\driver.py 
  --root "D:\Test Data" 
  --ckpt "D:\Test Data\runs\siamese\best.pth"
