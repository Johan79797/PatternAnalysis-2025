import argparse
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, roc_curve
from dataset import make_loader
from modules import SiameseResNet18

DEFAULT_ROOT = r"D:\\Test Data"

def _load_model(ckpt_path, device):
    model = SiameseResNet18(pretrained=False).to(device)
    bundle = torch.load(ckpt_path, map_location=device)
    state = bundle["model"] if isinstance(bundle, dict) and "model" in bundle else bundle
    model.load_state_dict(state)
    model.eval()
    return model

# -------- pair-level evaluation --------
def cmd_eval_pairs(args):
    root = Path(args.root or DEFAULT_ROOT)
    csv = root/"splits"/f"{args.split}.csv"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(Path(args.ckpt), device)

    dl = make_loader(csv, args.img_size, args.batch_size, False, args.num_workers, False)
    preds, targs = [], []
    with torch.no_grad():
        for x1, x2, y in dl:
            p = torch.sigmoid(model(x1.to(device), x2.to(device))).cpu()
            preds.append(p); targs.append(y)
    preds = torch.cat(preds).numpy(); targs = torch.cat(targs).numpy()

    auc = roc_auc_score(targs, preds)
    thr = 0.5
    if args.find_best_threshold:
        fpr, tpr, thr_grid = roc_curve(targs, preds)
        j = np.argmax(tpr - fpr)
        thr = float(thr_grid[j])
        Path(args.ckpt).with_suffix(".thr.txt").write_text(f"{thr:.6f}\n")
        print(f"[THR] best_thr={thr:.4f}  sens={tpr[j]:.3f}  spec={(1-fpr[j]):.3f}")
    elif args.use_saved_threshold:
        p = Path(args.ckpt).with_suffix(".thr.txt")
        thr = float(p.read_text().strip())

    yhat = (preds >= thr).astype("float32")
    acc = accuracy_score(targs, yhat)
    tn, fp, fn, tp = confusion_matrix(targs, yhat).ravel()
    sens = tp/(tp+fn+1e-8); spec = tn/(tn+fp+1e-8)
    prec = tp/(tp+fp+1e-8); f1 = 2*prec*sens/(prec+sens+1e-8)
    print(f"{args.split.upper()}  AUC={auc:.4f}  thr={thr:.3f}  ACC={acc:.4f}  F1={f1:.4f}  sens={sens:.3f}  spec={spec:.3f}  "
          f"TP={tp} FP={fp} FN={fn} TN={tn}")

# -------- image-level prototype classifier (optional) --------
def cmd_proto(args):
    root = Path(args.root or DEFAULT_ROOT)
    csv = root/"splits"/f"{args.split}.csv"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(Path(args.ckpt), device)

    # Build a balanced support set from TRAIN split
    import pandas as pd
    from PIL import Image
    from torchvision import transforms

    train_csv = root/"splits"/"train.csv"
    df_tr = pd.read_csv(train_csv)
    S = args.support_per_class
    sup0 = df_tr[df_tr.label==0].sample(S, random_state=0)
    sup1 = df_tr[df_tr.label==1].sample(S, random_state=0)

    t = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    def load(path):
        with Image.open(path) as im:
            x = t(im.convert("RGB"))
            return torch.nn.functional.normalize(x, dim=0) if args.l2norm else x

    # Precompute support embeddings
    with torch.no_grad():
        def embed_many(paths):
            xs = torch.stack([load(p) for p in paths]).to(device)
            return model.embed(xs).cpu()
        E0 = embed_many(sup0.image_path.values)
        E1 = embed_many(sup1.image_path.values)

    # Score each VAL/TEST image by similarity gap to class supports
    import pandas as pd
    from tqdm import tqdm
    df = pd.read_csv(csv)
    scores, labels = [], []
    with torch.no_grad():
        for _, r in tqdm(df.iterrows(), total=len(df), desc=f"Eval:{args.split}"):
            x = load(r.image_path).unsqueeze(0).to(device)
            e = model.embed(x).cpu()      # [1,512]
            s0 = (-(e - E0).abs().mean(1)).mean().item()
            s1 = (-(e - E1).abs().mean(1)).mean().item()
            scores.append(s1 - s0); labels.append(int(r.label))
    scores = np.asarray(scores); labels = np.asarray(labels)

    from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, roc_curve
    auc = roc_auc_score(labels, scores)
    thr = 0.0
    if args.find_best_threshold:
        fpr, tpr, thr_grid = roc_curve(labels, scores)
        j = np.argmax(tpr - fpr)
        thr = float(thr_grid[j])
        Path(args.ckpt).with_suffix(".proto.thr.txt").write_text(f"{thr:.6f}\n")
        print(f"[PROTO THR] best_thr={thr:.4f}  sens={tpr[j]:.3f}  spec={(1-fpr[j]):.3f}")
    elif args.use_saved_threshold:
        p = Path(args.ckpt).with_suffix(".proto.thr.txt")
        thr = float(p.read_text().strip())

    yhat = (scores >= thr).astype("float32")
    acc = accuracy_score(labels, yhat)
    tn, fp, fn, tp = confusion_matrix(labels, yhat).ravel()
    sens = tp/(tp+fn+1e-8); spec = tn/(tn+fp+1e-8)
    prec = tp/(tp+fp+1e-8); f1 = 2*prec*sens/(prec+sens+1e-8)
    print(f"{args.split.upper()} (image-level)  AUC={auc:.4f}  ACC={acc:.4f}  F1={f1:.4f}  sens={sens:.3f}  spec={spec:.3f}  "
          f"TP={tp} FP={fp} FN={fn} TN={tn}  thr={thr:.3f}")
    if args.save_preds:
        out = Path(args.ckpt).with_suffix(f".{args.split}.proto.preds.csv")
        import pandas as pd
        pd.DataFrame({"image_path":df.image_path,"label":labels,"score":scores,"yhat":yhat}).to_csv(out,index=False)
        print(f"[OK] Saved proto predictions to {out}")

def build_parser():
    p = argparse.ArgumentParser("Evaluate Siamese")
    sub = p.add_subparsers(dest="cmd", required=True)

    ep = sub.add_parser("eval-pairs")
    ep.add_argument("--root", type=str, default=DEFAULT_ROOT)
    ep.add_argument("--ckpt", type=str, required=True)
    ep.add_argument("--split", type=str, default="val", choices=["val","test"])
    ep.add_argument("--batch_size", type=int, default=64)
    ep.add_argument("--img_size", type=int, default=224)
    ep.add_argument("--num_workers", type=int, default=4)
    ep.add_argument("--find_best_threshold", action="store_true")
    ep.add_argument("--use_saved_threshold", action="store_true")
    ep.set_defaults(func=cmd_eval_pairs)

    pp = sub.add_parser("proto")
    pp.add_argument("--root", type=str, default=DEFAULT_ROOT)
    pp.add_argument("--ckpt", type=str, required=True)
    pp.add_argument("--split", type=str, default="val", choices=["val","test"])
    pp.add_argument("--img_size", type=int, default=256)
    pp.add_argument("--support_per_class", type=int, default=128)
    pp.add_argument("--l2norm", action="store_true")
    pp.add_argument("--find_best_threshold", action="store_true")
    pp.add_argument("--use_saved_threshold", action="store_true")
    pp.add_argument("--save_preds", action="store_true")
    pp.set_defaults(func=cmd_proto)
    return p

if __name__ == "__main__":
    args = build_parser().parse_args()
    args.func(args)
