import argparse, sys, subprocess
from pathlib import Path

HERE = Path(__file__).parent

def run(args):
    py = sys.executable
    # Pick t on VAL if requested
    if args.find_threshold:
        subprocess.run([
            py, str(HERE / "predict.py"),
            "--root", args.root,
            "--ckpt", args.ckpt,
            "--split", "val",
            "--support_per_class", str(args.support_per_class),
            "--img_size", str(args.img_size),
            "--l2norm",
            "--use_ema",
            "--find_best_threshold",
        ], check=True)

    # Evaluate TEST with saved t
    subprocess.run([
        py, str(HERE / "predict.py"),
        "--root", args.root,
        "--ckpt", args.ckpt,
        "--split", "test",
        "--support_per_class", str(args.support_per_class),
        "--img_size", str(args.img_size),
        "--l2norm",
        "--use_ema",
        "--use_saved_threshold",
        "--save_preds",
    ], check=True)

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Siamese ISIC2020 driver")
    p.add_argument("--root", required=True, help="Dataset root (contains ISIC_2020_Training_JPEG, CSVs, splits/)")
    p.add_argument("--ckpt", required=True, help=r"Path to trained checkpoint (e.g., D:\Test Data\runs\siamese\best.pth)")
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--support_per_class", type=int, default=256)
    p.add_argument("--find_threshold", action="store_true", help="Pick t on VAL before running TEST")
    run(p.parse_args())
