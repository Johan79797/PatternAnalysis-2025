# recognition/siamese-isic2020-47525496/train.py
"""
Siamese ISIC-2020 pipeline entry points.

Commands
--------
1) split
   Create patient/lesion-aware train/val/test CSVs under DATA_ROOT/splits/.

2) train
   Train the pair classifier (Siamese head on |e1 - e2|) and save best.pth
   by validation AUC to runs/siamese/ (under DATA_ROOT by default).

This file intentionally stays minimal (no EMA/AMP bells & whistles) to match
the assignment’s “clear test driver” requirement. The model is defined in
modules.py and the dataset/utilities in dataset.py.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, accuracy_score

from dataset import index_images, make_group_splits, make_loader
from modules import SiameseResNet18


# ---------- Defaults / constants ----------

DEFAULT_ROOT = r"D:\Test Data"
TRAIN_DIR_NAME = "ISIC_2020_Training_JPEG"
GT_CSV_NAME    = "ISIC_2020_Training_GroundTruth_v2.csv"
META_V2_NAME   = "ISIC_2020_Training_Metadata_v2.csv"  # optional (patient/lesion groups)
OUT_DIR_NAME   = r"runs\siamese"


# ---------- Helpers ----------

def _stats(tag: str, dframe: pd.DataFrame) -> str:
    """Pretty class-balance string for a split dataframe."""
    pos = int(dframe["label"].sum())
    n = len(dframe)
    return f"{tag}: n={n}  melanoma={pos}  benign={n - pos}"

def _set_seed(seed: int = 42) -> None:
    """Best-effort reproducibility (does not force deterministic cuDNN)."""
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------- Commands ----------

def cmd_split(args: argparse.Namespace) -> None:
    """
    Build CSV splits with optional group exclusion (patient/lesion aware).

    Expects the Kaggle ISIC 2020 assets under:
        DATA_ROOT/
          ISIC_2020_Training_JPEG/
          ISIC_2020_Training_GroundTruth_v2.csv
          ISIC_2020_Training_Metadata_v2.csv   (optional)
    """
    root = Path(args.root or DEFAULT_ROOT)
    img_dir = root / (args.train_dir or TRAIN_DIR_NAME)
    gt_csv  = root / (args.gt_csv or GT_CSV_NAME)
    meta_v2 = root / (args.meta_v2 or META_V2_NAME)

    if not img_dir.exists():
        raise SystemExit(f"[ERROR] Missing image folder: {img_dir}")
    if not gt_csv.exists():
        raise SystemExit(f"[ERROR] Missing labels CSV: {gt_csv}")

    df = pd.read_csv(gt_csv)

    # Resolve the label column name present in various ISIC CSVs.
    label_col = (
        "target" if "target" in df.columns else
        ("melanoma" if "melanoma" in df.columns else None)
    )
    if "image_name" not in df.columns or label_col is None:
        raise SystemExit(
            f"[ERROR] CSV must contain 'image_name' and 'target' (or 'melanoma'). "
            f"Columns present: {df.columns.tolist()}"
        )
    df = df[["image_name", label_col]].rename(columns={label_col: "label"})

    # Map image_name -> file path and drop missing
    mapping = index_images(img_dir)
    df["image_path"] = df["image_name"].map(lambda s: str(mapping.get(str(s), "")))
    missing = (df["image_path"] == "")
    if missing.any():
        nmiss = int(missing.sum())
        print(f"[WARN] {nmiss} images referenced in CSV were not found under {img_dir}. Dropping them.")
        df = df[~missing].copy()

    # Pull grouping column from metadata if available (lesion_id preferred)
    group_key = None
    if meta_v2.exists():
        meta = pd.read_csv(meta_v2)
        gcol = "lesion_id" if "lesion_id" in meta.columns else (
            "patient_id" if "patient_id" in meta.columns else None
        )
        if gcol:
            df = df.merge(meta[["image_name", gcol]].drop_duplicates(),
                          on="image_name", how="left")
            group_key = gcol
            print(f"[INFO] Grouped split using '{gcol}' from Metadata v2.")
        else:
            print("[INFO] Metadata v2 present but no lesion_id/patient_id; proceeding without grouping.")
    else:
        print("[INFO] No Metadata v2 CSV found; proceeding without grouping.")

    # Perform group-aware splitting
    train_df, val_df, test_df = make_group_splits(
        df, group_col=group_key,
        val_pct=args.val_pct, test_pct=args.test_pct, seed=args.seed
    )

    print(_stats("TRAIN", train_df))
    print(_stats("VAL  ", val_df))
    print(_stats("TEST ", test_df))

    out = root / "splits"
    out.mkdir(exist_ok=True)
    train_df.to_csv(out / "train.csv", index=False)
    val_df.to_csv(out / "val.csv", index=False)
    test_df.to_csv(out / "test.csv", index=False)
    print(f"[OK] Wrote splits to: {out.resolve()}")


def cmd_train(args: argparse.Namespace) -> None:
    """
    Train the Siamese pair classifier. Saves best.pth (by VAL AUC).

    Inputs
    ------
    DATA_ROOT/splits/train.csv
    DATA_ROOT/splits/val.csv

    Outputs
    -------
    DATA_ROOT/runs/siamese/best.pth  (dict with 'model' state_dict and 'auc')
    """
    _set_seed(42)

    root = Path(args.root or DEFAULT_ROOT)
    train_csv, val_csv = root / "splits" / "train.csv", root / "splits" / "val.csv"
    if not train_csv.exists() or not val_csv.exists():
        raise SystemExit('Missing splits. Run: python train.py split --root "D:\\Test Data"')

    # Device & backend
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    if device.type == "cuda":
        print(f"[INFO] Using GPU: {torch.cuda.get_device_name(0)} | CUDA {torch.version.cuda}")
    else:
        print("[WARN] CUDA not available; training on CPU will be slow.")

    # Model / optimizer / loss
    model = SiameseResNet18(pretrained=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    # Data
    train_loader = make_loader(train_csv, args.img_size, args.batch_size,
                               augment=True,  workers=args.num_workers,
                               shuffle=True)
    val_loader   = make_loader(val_csv,   args.img_size, args.batch_size,
                               augment=False, workers=args.num_workers,
                               shuffle=False)

    out_dir = Path(args.out or (root / OUT_DIR_NAME))
    out_dir.mkdir(parents=True, exist_ok=True)

    best_auc = -1.0
    steps_per_epoch = len(train_loader)
    print(f"[INFO] steps/epoch = {steps_per_epoch}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        t0 = time.perf_counter()

        # ---- train loop
        for step, (x1, x2, y) in enumerate(train_loader, 1):
            x1, x2, y = x1.to(device, non_blocking=True), x2.to(device, non_blocking=True), y.to(device)

            opt.zero_grad(set_to_none=True)
            logits = model(x1, x2)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()

            running += float(loss) * y.size(0)

            if step == 1 and device.type == "cuda":
                torch.cuda.synchronize()
                alloc = torch.cuda.memory_allocated() / 1e9
                resv  = torch.cuda.memory_reserved() / 1e9
                print(f"[DEBUG] First batch VRAM: alloc={alloc:.2f} GB  reserved={resv:.2f} GB")

        train_loss = running / len(train_loader.dataset)

        # ---- validation
        model.eval()
        preds, targets = [], []
        with torch.inference_mode():
            for x1, x2, y in val_loader:
                p = torch.sigmoid(model(x1.to(device, non_blocking=True),
                                        x2.to(device, non_blocking=True))).cpu()
                preds.append(p)
                targets.append(y)

        preds = torch.cat(preds).numpy()
        targets = torch.cat(targets).numpy()

        # Core metrics
        auc = roc_auc_score(targets, preds)
        acc = accuracy_score(targets, (preds >= 0.5).astype("float32"))

        epoch_min = (time.perf_counter() - t0) / 60.0
        print(f"Epoch {epoch:02d}: train_loss={train_loss:.4f}  val_auc={auc:.4f}  "
              f"val_acc={acc:.4f}  time={epoch_min:.1f} min")

        # ---- checkpoint on best AUC
        if auc > best_auc:
            best_auc = auc
            ckpt = out_dir / "best.pth"
            torch.save({"model": model.state_dict(), "auc": best_auc}, ckpt)
            print(f"[✓] Saved {ckpt} (AUC={best_auc:.4f})")


# ---------- CLI ----------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Siamese ISIC-2020 (split/train).")
    sub = p.add_subparsers(dest="cmd", required=True)

    # split
    sp = sub.add_parser("split", help="Create train/val/test CSVs (patient/lesion-aware if metadata is present).")
    sp.add_argument("--root", type=str, default=DEFAULT_ROOT)
    sp.add_argument("--train_dir", type=str, default=TRAIN_DIR_NAME)
    sp.add_argument("--gt_csv", type=str, default=GT_CSV_NAME)
    sp.add_argument("--meta_v2", type=str, default=META_V2_NAME)
    sp.add_argument("--val_pct", type=float, default=0.15)
    sp.add_argument("--test_pct", type=float, default=0.15)
    sp.add_argument("--seed", type=int, default=42)
    sp.set_defaults(func=cmd_split)

    # train
    tp = sub.add_parser("train", help="Train the Siamese pair classifier.")
    tp.add_argument("--root", type=str, default=DEFAULT_ROOT)
    tp.add_argument("--epochs", type=int, default=15)
    tp.add_argument("--batch_size", type=int, default=64)
    tp.add_argument("--lr", type=float, default=3e-4)
    tp.add_argument("--img_size", type=int, default=224)
    tp.add_argument("--num_workers", type=int, default=4)
    tp.add_argument("--out", type=str, default=None, help="Override output dir; default is DATA_ROOT/runs/siamese")
    tp.set_defaults(func=cmd_train)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.func(args)
