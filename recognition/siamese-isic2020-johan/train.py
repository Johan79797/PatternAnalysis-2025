import argparse, time
from pathlib import Path
import pandas as pd
import torch, torch.nn as nn
from sklearn.metrics import roc_auc_score, accuracy_score
from dataset import index_images, make_group_splits, make_loader
from modules import SiameseResNet18

DEFAULT_ROOT = r"D:\\Test Data"
TRAIN_DIR_NAME = "ISIC_2020_Training_JPEG"
GT_CSV_NAME    = "ISIC_2020_Training_GroundTruth_v2.csv"
META_V2_NAME   = "ISIC_2020_Training_Metadata_v2.csv"
OUT_DIR_NAME   = r"runs\\siamese"

def cmd_split(args):
    root = Path(args.root or DEFAULT_ROOT)
    img_dir = root / (args.train_dir or TRAIN_DIR_NAME)
    gt_csv  = root / (args.gt_csv or GT_CSV_NAME)
    meta_v2 = root / (args.meta_v2 or META_V2_NAME)

    assert img_dir.exists(), f"Missing folder: {img_dir}"
    assert gt_csv.exists(),  f"Missing CSV: {gt_csv}"

    df = pd.read_csv(gt_csv)
    label_col = "target" if "target" in df.columns else ("melanoma" if "melanoma" in df.columns else None)
    assert "image_name" in df.columns and label_col is not None
    df = df[["image_name", label_col]].rename(columns={label_col:"label"})

    mapping = index_images(img_dir)
    df["image_path"] = df["image_name"].map(lambda s: str(mapping.get(str(s), "")))
    df = df[df["image_path"]!=""].copy()

    group_key = None
    if meta_v2.exists():
        meta = pd.read_csv(meta_v2)
        gcol = "lesion_id" if "lesion_id" in meta.columns else ("patient_id" if "patient_id" in meta.columns else None)
        if gcol:
            df = df.merge(meta[["image_name", gcol]].drop_duplicates(), on="image_name", how="left" )
            group_key = gcol

    train_df, val_df, test_df = make_group_splits(df, group_col=group_key,
                                                  val_pct=args.val_pct, test_pct=args.test_pct, seed=args.seed)
    out = root / "splits"; out.mkdir(exist_ok=True)
    train_df.to_csv(out/"train.csv", index=False)
    val_df.to_csv(out/"val.csv", index=False)
    test_df.to_csv(out/"test.csv", index=False)
    print(f"[OK] Wrote splits to {out.resolve()}")

def cmd_train(args):
    root = Path(args.root or DEFAULT_ROOT)
    train_csv, val_csv = root/"splits"/"train.csv", root/"splits"/"val.csv"
    assert train_csv.exists() and val_csv.exists(), "Run: python train.py split --root ..."

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SiameseResNet18(pretrained=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    train_loader = make_loader(train_csv, args.img_size, args.batch_size, True,  args.num_workers, True)
    val_loader   = make_loader(val_csv,   args.img_size, args.batch_size, False, args.num_workers, False)

    out_dir = Path(args.out or (root/OUT_DIR_NAME)); out_dir.mkdir(parents=True, exist_ok=True)
    best_auc = -1.0
    for epoch in range(1, args.epochs+1):
        model.train()
        running = 0.0
        t0 = time.perf_counter()
        for x1, x2, y in train_loader:
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x1, x2)
            loss = loss_fn(logits, y)
            loss.backward(); opt.step()
            running += float(loss) * y.size(0)

        train_loss = running / len(train_loader.dataset)
        model.eval(); preds, targets = [], []
        with torch.no_grad():
            for x1, x2, y in val_loader:
                p = torch.sigmoid(model(x1.to(device), x2.to(device))).cpu()
                preds.append(p); targets.append(y)
        import torch as _t
        preds = _t.cat(preds).numpy(); targets = _t.cat(targets).numpy()
        auc = roc_auc_score(targets, preds)
        acc = accuracy_score(targets, (preds>=0.5).astype("float32"))
        print(f"Epoch {epoch:02d}: train_loss={train_loss:.4f}  val_auc={auc:.4f}  val_acc={acc:.4f}  "
              f"time={(time.perf_counter()-t0)/60:.1f} min")

        if auc > best_auc:
            best_auc = auc
            ckpt = out_dir/"best.pth"
            torch.save({"model": model.state_dict(), "auc": best_auc}, ckpt)
            print(f"[✓] Saved {ckpt} (AUC={best_auc:.4f})")

def build_parser():
    p = argparse.ArgumentParser("Siamese ISIC 2020")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("split")
    sp.add_argument("--root", type=str, default=DEFAULT_ROOT)
    sp.add_argument("--train_dir", type=str, default=TRAIN_DIR_NAME)
    sp.add_argument("--gt_csv", type=str, default=GT_CSV_NAME)
    sp.add_argument("--meta_v2", type=str, default=META_V2_NAME)
    sp.add_argument("--val_pct", type=float, default=0.15)
    sp.add_argument("--test_pct", type=float, default=0.15)
    sp.add_argument("--seed", type=int, default=42)
    sp.set_defaults(func=cmd_split)

    tp = sub.add_parser("train")
    tp.add_argument("--root", type=str, default=DEFAULT_ROOT)
    tp.add_argument("--epochs", type=int, default=15)
    tp.add_argument("--batch_size", type=int, default=64)
    tp.add_argument("--lr", type=float, default=3e-4)
    tp.add_argument("--img_size", type=int, default=224)
    tp.add_argument("--num_workers", type=int, default=4)
    tp.add_argument("--out", type=str, default=None)
    tp.set_defaults(func=cmd_train)
    return p

if __name__ == "__main__":
    args = build_parser().parse_args()
    args.func(args)
