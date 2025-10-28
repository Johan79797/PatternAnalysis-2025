import random
from pathlib import Path
import pandas as pd
from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import GroupShuffleSplit

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ---------- utilities ----------
def index_images(folder: Path):
    exts = (".jpg",".jpeg",".JPG",".JPEG",".png",".PNG")
    mapping = {}
    for p in folder.rglob("*"):
        if p.suffix in exts:
            mapping[p.stem] = p
    return mapping

def make_group_splits(df, group_col=None, val_pct=0.15, test_pct=0.15, seed=42):
    groups = df[group_col].fillna(df.index) if group_col else df.index
    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_pct, random_state=seed)
    tr_idx, te_idx = next(gss1.split(df, groups=groups))
    train_df = df.iloc[tr_idx].copy()
    test_df  = df.iloc[te_idx].copy()

    groups_tr = train_df[group_col].fillna(train_df.index) if group_col else train_df.index
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_pct/(1-test_pct), random_state=seed)
    tr2_idx, va_idx = next(gss2.split(train_df, groups=groups_tr))
    val_df   = train_df.iloc[va_idx].copy()
    train_df = train_df.iloc[tr2_idx].copy()
    return train_df, val_df, test_df

# ---------- dataset ----------
class PairISICDataset(Dataset):
    """
    CSV must have: image_name, label (0/1), image_path
    Returns: (x1, x2, target) where target=1 (same) or 0 (different).
    """
    def __init__(self, csv_path, img_size=224, augment=False, max_retries=5):
        self.df = pd.read_csv(csv_path)
        assert {"image_name","label","image_path"}.issubset(self.df.columns)
        self.df["label"] = self.df["label"].astype(int)
        self.img_size = img_size
        self.max_retries = max_retries
        self.by_class = {
            0: self.df.index[self.df.label==0].tolist(),
            1: self.df.index[self.df.label==1].tolist()
        }
        aug_ops = [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ] if augment else [transforms.Resize((img_size, img_size))]
        self.t = transforms.Compose(aug_ops + [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])

    def _load(self, path):
        with Image.open(path) as im:
            return self.t(im.convert("RGB"))

    def _sample_pair(self):
        i = random.randrange(len(self.df))
        a = self.df.iloc[i]
        same = random.random() < 0.5
        if same:
            pool = self.by_class[int(a.label)]
            j = random.choice(pool if len(pool)==1 else [k for k in pool if k!=a.name])
            b = self.df.loc[j]; target = 1
        else:
            j = random.choice(self.by_class[1-int(a.label)])
            b = self.df.loc[j]; target = 0
        return a, b, target

    def __len__(self): return len(self.df)

    def __getitem__(self, _):
        for _ in range(self.max_retries):
            try:
                a,b,t = self._sample_pair()
                return self._load(a["image_path"]), self._load(b["image_path"]), torch.tensor(float(t))
            except Exception:
                continue
        x = torch.zeros(3, self.img_size, self.img_size)
        return x, x.clone(), torch.tensor(0.0)

def make_loader(csv_path, img_size, batch, augment, workers, shuffle, timeout=0):
    ds = PairISICDataset(csv_path, img_size=img_size, augment=augment)
    return DataLoader(
        ds, batch_size=batch, shuffle=shuffle, num_workers=workers, pin_memory=True, timeout=timeout
    )
