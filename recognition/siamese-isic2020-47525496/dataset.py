# recognition/siamese-isic2020-47525496/dataset.py
"""
Dataset & utilities for the ISIC-2020 Siamese pipeline.

Exports:
- index_images(folder)            -> {stem: Path}
- make_group_splits(df, ...)      -> (train_df, val_df, test_df)
- PairISICDataset(csv_path, ...)  -> torch.utils.data.Dataset
- make_loader(csv_path, ...)      -> DataLoader
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from PIL import Image, ImageFile

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import GroupShuffleSplit

# Allow PIL to load truncated/corrupt JPEGs instead of raising (we resample on errors)
ImageFile.LOAD_TRUNCATED_IMAGES = True


# ---------- utilities ----------

def index_images(folder: Path) -> Dict[str, Path]:
    """
    Recursively index image files by their stem (filename without extension).

    Parameters
    ----------
    folder : Path
        Root directory containing images.

    Returns
    -------
    Dict[str, Path]
        Mapping from image stem (e.g., "ISIC_123") to its file path.

    Notes
    -----
    - Accepts common JPG/PNG extensions.
    - If multiple files share the same stem, the last one found wins.
      (ISIC stems are unique, so this is fine for our use.)
    """
    exts = {".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"}
    mapping: Dict[str, Path] = {}
    for p in folder.rglob("*"):
        if p.suffix in exts:
            mapping[p.stem] = p
    return mapping


def make_group_splits(
    df: pd.DataFrame,
    group_col: str | None = None,
    val_pct: float = 0.15,
    test_pct: float = 0.15,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a dataframe into train/val/test with optional group-wise exclusion.

    Parameters
    ----------
    df : pd.DataFrame
        Must include at least the rows representing samples. Other columns are carried through.
    group_col : str | None
        Column name for grouping (e.g., 'patient_id' or 'lesion_id').
        Ensures the same group does not appear across splits. If None, uses row index.
    val_pct : float
        Fraction of the *entire* dataset to allocate to validation.
    test_pct : float
        Fraction of the *entire* dataset to allocate to test.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    (train_df, val_df, test_df)
    """
    # First: carve out test from all data using groups
    groups_all = df[group_col].fillna(df.index) if group_col else df.index
    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_pct, random_state=seed)
    tr_idx, te_idx = next(gss1.split(df, groups=groups_all))
    train_df = df.iloc[tr_idx].copy()
    test_df  = df.iloc[te_idx].copy()

    # Second: carve out val from the remaining train portion (account for prior test removal)
    groups_tr = train_df[group_col].fillna(train_df.index) if group_col else train_df.index
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_pct / (1 - test_pct), random_state=seed)
    tr2_idx, va_idx = next(gss2.split(train_df, groups=groups_tr))
    val_df   = train_df.iloc[va_idx].copy()
    train_df = train_df.iloc[tr2_idx].copy()
    return train_df, val_df, test_df


# ---------- dataset ----------

class PairISICDataset(Dataset):
    """
    Pairwise dataset for Siamese training/evaluation.

    CSV Requirements
    ----------------
    Columns: 'image_name', 'label', 'image_path'
      - 'label' must be {0,1} (benign=0, melanoma=1)
      - 'image_path' is an absolute or repo-local path to the image file

    Returns
    -------
    (x1, x2, target)
      - x1, x2 : torch.FloatTensor [3, H, W], normalized to ImageNet stats
      - target : torch.FloatTensor scalar, 1.0 if same-class else 0.0

    Notes
    -----
    - Balanced sampling of positive/negative pairs (50/50) per __getitem__.
    - Robust to occasional decode errors via retry loop.
    """

    def __init__(
        self,
        csv_path: str | Path,
        img_size: int = 224,
        augment: bool = False,
        max_retries: int = 5,
    ) -> None:
        self.df = pd.read_csv(csv_path)
        must = {"image_name", "label", "image_path"}
        if not must.issubset(self.df.columns):
            raise ValueError(f"CSV at {csv_path} must contain columns: {sorted(must)}")
        self.df["label"] = self.df["label"].astype(int)

        self.img_size = img_size
        self.max_retries = max_retries

        # Pre-index rows by class for fast sampling
        self.by_class = {
            0: self.df.index[self.df.label == 0].tolist(),
            1: self.df.index[self.df.label == 1].tolist(),
        }

        # Build transform pipeline
        aug_ops = [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ] if augment else [transforms.Resize((img_size, img_size))]

        self.t = transforms.Compose(aug_ops + [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std =[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.df)

    def _load(self, path: str | Path) -> torch.Tensor:
        """Open an image, convert to RGB, apply transforms."""
        with Image.open(path) as im:
            return self.t(im.convert("RGB"))

    def _sample_pair(self):
        """Draw one random anchor sample and a positive/negative partner (50/50)."""
        i = random.randrange(len(self.df))
        a = self.df.iloc[i]
        same = (random.random() < 0.5)

        if same:
            pool = self.by_class[int(a.label)]
            # Avoid picking the exact same row unless the class only has one sample
            j = random.choice(pool if len(pool) == 1 else [k for k in pool if k != a.name])
            b = self.df.loc[j]
            target = 1.0
        else:
            j = random.choice(self.by_class[1 - int(a.label)])
            b = self.df.loc[j]
            target = 0.0

        return a, b, target

    def __getitem__(self, _):
        """
        Attempt to load a valid pair a few times before falling back to zeros.
        This avoids DataLoader hangs due to persistent corrupt files.
        """
        for _try in range(self.max_retries):
            try:
                a, b, t = self._sample_pair()
                x1 = self._load(a["image_path"])
                x2 = self._load(b["image_path"])
                return x1, x2, torch.tensor(t, dtype=torch.float32)
            except Exception:
                # If anything fails (decode, missing file, etc.), retry with a fresh sample
                continue

        # Last resort: return black tensors (won't crash training loop)
        x = torch.zeros(3, self.img_size, self.img_size, dtype=torch.float32)
        return x, x.clone(), torch.tensor(0.0, dtype=torch.float32)


def make_loader(
    csv_path: str | Path,
    img_size: int,
    batch: int,
    augment: bool,
    workers: int,
    shuffle: bool,
    timeout: int = 0,
) -> DataLoader:
    """
    Build a standard DataLoader for PairISICDataset.

    Parameters
    ----------
    csv_path : str | Path
        Path to split CSV.
    img_size : int
        Final resize size for images (square).
    batch : int
        Batch size.
    augment : bool
        Enable light augmentation (random flips).
    workers : int
        Number of DataLoader workers.
    shuffle : bool
        Shuffle batches.
    timeout : int
        DataLoader timeout (seconds). 0 disables timeout.

    Returns
    -------
    DataLoader
    """
    ds = PairISICDataset(csv_path, img_size=img_size, augment=augment)
    return DataLoader(
        ds,
        batch_size=batch,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        timeout=timeout,
    )
