# recognition/siamese-isic2020-47525496/train.py
"""
Siamese training script for ISIC-2020.
- Creates patient/lesion-aware splits (if Metadata v2 present).
- Trains a pairwise BCE-with-logits head over a ResNet-18 encoder.
- Logs train loss, VAL AUC and VAL ACC each epoch.
- Writes training_log.csv + training_curves.png to the output folder.

Usage (PowerShell example):
  python recognition/siamese-isic2020-47525496/train.py split --root "D:\\Test Data"
  python recognition/siamese-isic2020-47525496/train.py train --root "D:\\Test Data" `
      --epochs 15 --batch_size 64 --lr 3e-4 --img_size 224 --num_workers 4 `
      --out "D:\\Test Data\\runs\\siamese_retrain"
"""

from __future__ import annotations
import argparse, time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, accuracy_score

from dataset import index_images, make_group_splits, make_loader
from modules import SiameseResNet18

# ----------------------------
# Defaults (adjust if needed)
# ----------------------------
DEFAULT_ROOT   = r"D:\\Test Data"
TRAIN_DIR_NAME = "ISIC_2020_Training_JPEG"
GT_CSV_NAME    = "ISIC_2020_Training_GroundTruth_v2.csv"
META_V2_NAME   = "ISIC_2020_Training_Metadata_v2.csv"
OUT_DIR_NAME   = r"runs\\siamese"


# ----------------------------
# SPLIT
# ----------------------------
def cmd_split(args: argparse.Namespace) -> None:
    """
    Build train/val/test CSV splits, optionally grouping by patient/lesion if
    Metadata v2 is available. Writes to <root>/splits/{train,val,test}.csv.
    """
    root = Path(args.root or DEFAULT_ROOT)
    img_dir = root / (args.train_dir or TRAIN_DIR_NAME)
    gt_csv  = root / (args.gt_csv or GT_CSV_NAME)
    meta_v2 = root / (args.meta_v2 or META_V2_NAME)

    assert img_dir.exists(), f"Missing folder: {img_dir}"
    assert gt_csv.exists(),  f"Missing CSV: {gt_csv}"

    df = pd.read_csv(gt_csv)
    label_col = "target" if "target" in df.columns else ("melanoma" if "melanoma" in df.columns else None)
    assert "image_name" in df.columns and label_col is not None, "Ground truth CSV must have image_name and (target|melanoma)"
    df = df[["image_name", label_col]].rename(columns={label_col: "label"})

    mapping = index_images(img_dir)
    df["image_path"] = df["image_name"].map(lambda s: str(mapping.get(str(s), "")))
    df = df[df["image_path"] != ""].copy()

    group_key = None
    if meta_v2.exists():
        meta = pd.read_csv(meta_v2)
        gcol = "lesion_id" if "lesion_id" in meta.columns else ("patient_id" if "patient_id" in meta.columns else None)
        if gcol:
            df = df.merge(meta[["image_name", gcol]].drop_duplicates(), on="image_name", how="left")
            group_key = gcol
    else:
        print("[INFO] No Metadata v2 CSV found; proceeding without grouping.")

    train_df, val_df, test_df = make_group_splits(
        df, group_col=group_key, val_pct=args.val_pct, test_pct=args.test_pct, seed=args.seed
    )

    out = root / "splits"
    out.mkdir(exist_ok=True)
    train_df.to_csv(out / "train.csv", index=False)
    val_df.to_csv(out / "val.csv", index=False)
    test_df.to_csv(out / "test.csv", index=False)

    def _counts(name, d):
        n = len(d)
        pos = int((d["label"] == 1).sum())
        neg = n - pos
        print(f"{name:<5}: n={n:<5}  melanoma={pos:<3}  benign={neg}")

    _counts("TRAIN", train_df)
    _counts("VAL",   val_df)
    _counts("TEST",  test_df)
    print(f"[OK] Wrote splits to: {out}")


# ----------------------------
# TRAIN
# ----------------------------
def cmd_train(args: argparse.Namespace) -> None:
    """
    Train the Siamese network (pair classification). Saves the best checkpoint
    by VAL AUC to <out>/best.pth and writes training logs/plots.
    """
    root = Path(args.root or DEFAULT_ROOT)
    train_csv, val_csv = root / "splits" / "train.csv", root / "splits" / "val.csv"
    assert train_csv.exists() and val_csv.exists(), "Run the split command first: train.py split --root ..."

    # Device / info
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[INFO] Using GPU: {gpu_name} | CUDA {torch.version.cuda}")
    else:
        print("[INFO] Using CPU")

    # Model / optim / loss
    model = SiameseResNet18(pretrained=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    # Data
    train_loader = make_loader(train_csv, args.img_size, args.batch_size, augment=True,
                               workers=args.num_workers, shuffle=True)
    val_loader   = make_loader(val_csv,   args.img_size, args.batch_size, augment=False,
                               workers=args.num_workers, shuffle=False)

    steps_per_epoch = len(train_loader)
    print(f"[INFO] steps/epoch = {steps_per_epoch}")

    out_dir = Path(args.out or (root / OUT_DIR_NAME))
    out_dir.mkdir(parents=True, exist_ok=True)

    # History for plots
    history = {"epoch": [], "train_loss": [], "val_auc": [], "val_acc": []}

    best_auc = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        t0 = time.perf_counter()

        # ---- Training loop ----
        for step, (x1, x2, y) in enumerate(train_loader, start=1):
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)

            # VRAM snapshot on first batch (useful for markers)
            if step == 1 and device.type == "cuda":
                alloc = torch.cuda.memory_allocated() / (1024 ** 3)
                reserved = torch.cuda.memory_reserved() / (1024 ** 3)
                print(f"[DEBUG] First batch VRAM: alloc={alloc:.2f} GB  reserved={reserved:.2f} GB")

            opt.zero_grad(set_to_none=True)
            logits = model(x1, x2)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            running += float(loss) * y.size(0)

        train_loss = running / len(train_loader.dataset)

        # ---- Validation ----
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for x1, x2, y in val_loader:
                p = torch.sigmoid(model(x1.to(device), x2.to(device))).cpu()
                preds.append(p)
                targets.append(y)

        preds_t = torch.cat(preds).numpy()
        targets_t = torch.cat(targets).numpy()
        auc = roc_auc_score(targets_t, preds_t)
        acc = accuracy_score(targets_t, (preds_t >= 0.5).astype("float32"))

        mins = (time.perf_counter() - t0) / 60.0
        print(
            f"Epoch {epoch:02d}: train_loss={train_loss:.4f}  "
            f"val_auc={auc:.4f}  val_acc={acc:.4f}  time={mins:.1f} min"
        )

        # ---- Save best ----
        if auc > best_auc:
            best_auc = auc
            ckpt = out_dir / "best.pth"
            torch.save({"model": model.state_dict(), "auc": best_auc}, ckpt)
            print(f"[✓] Saved {ckpt} (AUC={best_auc:.4f})")

        # ---- Log history ----
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_auc"].append(float(auc))
        history["val_acc"].append(float(acc))

    # ----------------------------
    # Write CSV + Plot curves
    # ----------------------------
    log_path = out_dir / "training_log.csv"
    pd.DataFrame(history).to_csv(log_path, index=False)

    # Use non-interactive backend for headless environments
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(history["epoch"], history["train_loss"], label="train_loss")
    plt.plot(history["epoch"], history["val_auc"],   label="val_auc")
    plt.plot(history["epoch"], history["val_acc"],   label="val_acc")
    plt.xlabel("epoch"); plt.legend(); plt.tight_layout()
    fig_path = out_dir / "training_curves.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()

    print(f"[OK] Wrote {log_path} and {fig_path}")


# ----------------------------
# CLI
# ----------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Siamese ISIC 2020")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("split", help="Create train/val/test CSV splits")
    sp.add_argument("--root", type=str, default=DEFAULT_ROOT)
    sp.add_argument("--train_dir", type=str, default=TRAIN_DIR_NAME)
    sp.add_argument("--gt_csv", type=str, default=GT_CSV_NAME)
    sp.add_argument("--meta_v2", type=str, default=META_V2_NAME)
    sp.add_argument("--val_pct", type=float, default=0.15)
    sp.add_argument("--test_pct", type=float, default=0.15)
    sp.add_argument("--seed", type=int, default=42)
    sp.set_defaults(func=cmd_split)

    tp = sub.add_parser("train", help="Train Siamese pair classifier")
    tp.add_argument("--root", type=str, default=DEFAULT_ROOT)
    tp.add_argument("--epochs", type=int, default=15)
    tp.add_argument("--batch_size", type=int, default=64)
    tp.add_argument("--lr", type=float, default=3e-4)
    tp.add_argument("--img_size", type=int, default=224)
    tp.add_argument("--num_workers", type=int, default=4)
    tp.add_argument("--out", type=str, default=None, help="Output dir for checkpoints and logs")
    tp.set_defaults(func=cmd_train)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.func(args)
