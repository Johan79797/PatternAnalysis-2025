# recognition/siamese-isic2020-47525496/predict.py
"""
Evaluation utilities for the Siamese ISIC-2020 project.

Subcommands
-----------
1) eval-pairs
   Evaluate the *pair head* directly on VAL/TEST (AUC/ACC/F1/etc).
   Optional: choose a threshold on VAL (Youden's J) and reuse on TEST.

2) proto
   Prototype an *image-level* classifier by comparing query embeddings to
   balanced class supports from the TRAIN split. Scores are s1 - s0 where
   s_c = -mean( L1 distance to class-c supports ). Threshold on VAL, reuse on TEST.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score, accuracy_score, confusion_matrix, roc_curve
)

from dataset import make_loader
from modules import SiameseResNet18

DEFAULT_ROOT = r"D:\Test Data"


# ---------- checkpoint loading ----------

def _safe_load_checkpoint(ckpt_path: Path, device: torch.device):
    """
    Prefer weights_only=True when available (PyTorch >= 2.4), but gracefully
    fall back to weights_only=False if needed. Expects either:
      - state_dict directly, or
      - {"model": state_dict, ...}
    """
    try:
        bundle = torch.load(ckpt_path, map_location=device, weights_only=True)  # type: ignore[arg-type]
    except TypeError:
        # Older torch without weights_only kw
        bundle = torch.load(ckpt_path, map_location=device)
    except Exception:
        # Some environments may raise UnpicklingError with weights_only=True for older pickled objects
        bundle = torch.load(ckpt_path, map_location=device)

    if isinstance(bundle, dict) and "model" in bundle:
        return bundle["model"], bundle
    return bundle, {"model": bundle}


def _load_model(ckpt_path: Path, device: torch.device) -> SiameseResNet18:
    model = SiameseResNet18(pretrained=False).to(device)
    state_dict, _ = _safe_load_checkpoint(ckpt_path, device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


# ---------- pair-level evaluation ----------

def cmd_eval_pairs(args: argparse.Namespace) -> None:
    """
    Evaluate the Siamese pair head on a split (VAL/TEST).
    Prints AUC and metrics at the chosen threshold.
    If --find_best_threshold is used, writes <ckpt>.thr.txt (pair-level).
    """
    root = Path(args.root or DEFAULT_ROOT)
    csv = root / "splits" / f"{args.split}.csv"
    if not csv.exists():
        raise SystemExit(f"[ERROR] Missing split CSV: {csv}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(Path(args.ckpt), device)

    dl = make_loader(csv, args.img_size, args.batch_size, augment=False,
                     workers=args.num_workers, shuffle=False)
    preds, targs = [], []
    with torch.no_grad():
        for x1, x2, y in dl:
            p = torch.sigmoid(model(x1.to(device), x2.to(device))).cpu()
            preds.append(p); targs.append(y)

    preds = torch.cat(preds).numpy()
    targs = torch.cat(targs).numpy()

    auc = roc_auc_score(targs, preds)

    # Threshold selection
    thr = 0.5
    thr_path = Path(args.ckpt).with_suffix(".thr.txt")
    if args.find_best_threshold:
        fpr, tpr, thr_grid = roc_curve(targs, preds)
        j = np.argmax(tpr - fpr)  # Youden's J
        thr = float(thr_grid[j])
        thr_path.write_text(f"{thr:.6f}\n")
        print(f"[THR] best_thr={thr:.4f}  sens={tpr[j]:.3f}  spec={(1-fpr[j]):.3f}  (saved to {thr_path})")
    elif args.use_saved_threshold and thr_path.exists():
        thr = float(thr_path.read_text().strip())

    # Metrics at threshold
    yhat = (preds >= thr).astype("float32")
    acc = accuracy_score(targs, yhat)
    tn, fp, fn, tp = confusion_matrix(targs, yhat).ravel()
    sens = tp / (tp + fn + 1e-8)
    spec = tn / (tn + fp + 1e-8)
    prec = tp / (tp + fp + 1e-8)
    f1 = 2 * prec * sens / (prec + sens + 1e-8)

    print(
        f"{args.split.upper()}  AUC={auc:.4f}  thr={thr:.3f}  "
        f"ACC={acc:.4f}  F1={f1:.4f}  sens={sens:.3f}  spec={spec:.3f}  "
        f"TP={tp} FP={fp} FN={fn} TN={tn}"
    )


# ---------- image-level prototype classifier ----------

def _embed_paths_in_batches(model: SiameseResNet18,
                            paths: list[str],
                            device: torch.device,
                            img_size: int,
                            batch: int = 64,
                            l2norm: bool = False) -> torch.Tensor:
    """
    Load images from file paths, preprocess, embed with the encoder.
    Returns a tensor [N, D] on CPU. Optionally L2-normalises embeddings.
    """
    from PIL import Image
    from torchvision import transforms
    t = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    embs = []
    with torch.no_grad():
        for i in range(0, len(paths), batch):
            chunk = paths[i:i + batch]
            xs = []
            for p in chunk:
                with Image.open(p) as im:
                    xs.append(t(im.convert("RGB")))
            x = torch.stack(xs).to(device)
            e = model.embed(x)
            if l2norm:
                e = torch.nn.functional.normalize(e, dim=1)
            embs.append(e.cpu())
    return torch.cat(embs, dim=0)


def cmd_proto(args: argparse.Namespace) -> None:
    """
    Prototype an image-level classifier using class supports from TRAIN.

    For each query image:
      1) Embed with the encoder.
      2) Compute mean L1 distance to melanoma supports (class=1) and benign supports (class=0).
      3) Score = s1 - s0 where s_c = -mean(L1 distance to supports of class c).
      4) Threshold the score (chosen on VAL, reused on TEST).
    """
    import pandas as pd
    from tqdm import tqdm

    root = Path(args.root or DEFAULT_ROOT)
    csv = root / "splits" / f"{args.split}.csv"
    train_csv = root / "splits" / "train.csv"
    if not csv.exists() or not train_csv.exists():
        raise SystemExit(f"[ERROR] Missing split CSV(s): {csv} or {train_csv}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(Path(args.ckpt), device)

    # Build balanced supports from TRAIN
    df_tr = pd.read_csv(train_csv)
    S = args.support_per_class
    # Be robust if class counts are smaller than requested S
    sup0 = df_tr[df_tr.label == 0].sample(n=min(S, (df_tr.label == 0).sum()), random_state=0)
    sup1 = df_tr[df_tr.label == 1].sample(n=min(S, (df_tr.label == 1).sum()), random_state=0)

    # Precompute support embeddings (optionally L2-normalised)
    E0 = _embed_paths_in_batches(model, sup0.image_path.tolist(), device,
                                 img_size=args.img_size, batch=64, l2norm=args.l2norm)
    E1 = _embed_paths_in_batches(model, sup1.image_path.tolist(), device,
                                 img_size=args.img_size, batch=64, l2norm=args.l2norm)

    # Evaluate queries
    df = pd.read_csv(csv)
    scores, labels = [], []
    with torch.no_grad():
        for _, r in tqdm(df.iterrows(), total=len(df), desc=f"Eval:{args.split}"):
            e = _embed_paths_in_batches(model, [r.image_path], device,
                                        img_size=args.img_size, batch=1, l2norm=args.l2norm)  # [1, D]
            # Use cdist with p=1 for L1 (Manhattan) distances
            d0 = torch.cdist(e, E0, p=1).mean().item()
            d1 = torch.cdist(e, E1, p=1).mean().item()
            s0 = -d0
            s1 = -d1
            scores.append(s1 - s0)          # larger → more melanoma-like
            labels.append(int(r.label))

    scores = np.asarray(scores)
    labels = np.asarray(labels)

    auc = roc_auc_score(labels, scores)

    thr = 0.0
    thr_path = Path(args.ckpt).with_suffix(".proto.thr.txt")
    if args.find_best_threshold:
        fpr, tpr, thr_grid = roc_curve(labels, scores)
        j = np.argmax(tpr - fpr)
        thr = float(thr_grid[j])
        thr_path.write_text(f"{thr:.6f}\n")
        print(f"[PROTO THR] best_thr={thr:.4f}  sens={tpr[j]:.3f}  spec={(1 - fpr[j]):.3f}  (saved to {thr_path})")
    elif args.use_saved_threshold and thr_path.exists():
        thr = float(thr_path.read_text().strip())

    yhat = (scores >= thr).astype("float32")
    acc = accuracy_score(labels, yhat)
    tn, fp, fn, tp = confusion_matrix(labels, yhat).ravel()
    sens = tp / (tp + fn + 1e-8)
    spec = tn / (tn + fp + 1e-8)
    prec = tp / (tp + fp + 1e-8)
    f1 = 2 * prec * sens / (prec + sens + 1e-8)

    print(
        f"{args.split.upper()} (image-level)  AUC={auc:.4f}  ACC={acc:.4f}  F1={f1:.4f}  "
        f"sens={sens:.3f}  spec={spec:.3f}  TP={tp} FP={fp} FN={fn} TN={tn}  thr={thr:.3f}"
    )

    if args.save_preds:
        out = Path(args.ckpt).with_suffix(f".{args.split}.proto.preds.csv")
        import pandas as pd
        pd.DataFrame(
            {"image_path": df.image_path, "label": labels, "score": scores, "yhat": yhat}
        ).to_csv(out, index=False)
        print(f"[OK] Saved proto predictions to {out}")


# ---------- CLI ----------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Evaluate Siamese (pairs) and prototype image-level classifier.")
    sub = p.add_subparsers(dest="cmd", required=True)

    ep = sub.add_parser("eval-pairs", help="Evaluate the pair head on VAL/TEST.")
    ep.add_argument("--root", type=str, default=DEFAULT_ROOT)
    ep.add_argument("--ckpt", type=str, required=True)
    ep.add_argument("--split", type=str, default="val", choices=["val", "test"])
    ep.add_argument("--batch_size", type=int, default=64)
    ep.add_argument("--img_size", type=int, default=224)
    ep.add_argument("--num_workers", type=int, default=4)
    ep.add_argument("--find_best_threshold", action="store_true")
    ep.add_argument("--use_saved_threshold", action="store_true")
    ep.set_defaults(func=cmd_eval_pairs)

    pp = sub.add_parser("proto", help="Prototype image-level classifier via support-set comparison.")
    pp.add_argument("--root", type=str, default=DEFAULT_ROOT)
    pp.add_argument("--ckpt", type=str, required=True)
    pp.add_argument("--split", type=str, default="val", choices=["val", "test"])
    pp.add_argument("--img_size", type=int, default=256)
    pp.add_argument("--support_per_class", type=int, default=128)
    pp.add_argument("--l2norm", action="store_true", help="L2-normalise the *embeddings* before distance.")
    pp.add_argument("--find_best_threshold", action="store_true")
    pp.add_argument("--use_saved_threshold", action="store_true")
    pp.add_argument("--save_preds", action="store_true")
    pp.set_defaults(func=cmd_proto)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.func(args)
