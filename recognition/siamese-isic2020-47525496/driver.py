# recognition/siamese-isic2020-47525496/driver.py
"""
Simple driver to run the evaluation workflow.

Default mode: 'proto' (image-level classifier via support-set comparison)
  - Optionally pick a threshold (tau) on VAL.
  - Then evaluate TEST with the saved threshold and save predictions.

Alternate mode: 'pairs' (evaluate Siamese pair head directly)

Examples (PowerShell)
---------------------
# Pick tau on VAL, then run TEST (image-level)
python recognition\siamese-isic2020-47525496\driver.py `
  --root "D:\Test Data" `
  --ckpt "D:\Test Data\runs\siamese\best.pth" `
  --find_threshold

# Run TEST only, using whatever tau was already saved
python recognition\siamese-isic2020-47525496\driver.py `
  --root "D:\Test Data" `
  --ckpt "D:\Test Data\runs\siamese\best.pth"
"""

from __future__ import annotations

import argparse
import sys
import subprocess
from pathlib import Path

HERE = Path(__file__).parent


def run_cmd(cmd: list[str]) -> None:
    print("[RUN]", " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Siamese ISIC-2020 evaluation driver")
    p.add_argument("--root", required=True,
                   help="Dataset root (contains ISIC_2020_Training_JPEG, CSVs, splits/)")
    p.add_argument("--ckpt", required=True,
                   help=r"Path to trained checkpoint (e.g., D:\Test Data\runs\siamese\best.pth)")
    p.add_argument("--mode", choices=["proto", "pairs"], default="proto",
                   help="Evaluation mode: 'proto' (image-level) or 'pairs' (pair head)")
    p.add_argument("--img_size", type=int, default=256, help="Image size for embedding/eval")
    p.add_argument("--support_per_class", type=int, default=256,
                   help="Supports per class for 'proto' mode")
    p.add_argument("--l2norm", action="store_true",
                   help="L2-normalise embeddings before distance (recommended)")
    p.add_argument("--find_threshold", action="store_true",
                   help="Pick tau on VAL before running TEST")
    args = p.parse_args()

    root = Path(args.root)
    ckpt = Path(args.ckpt)
    if not root.exists():
        raise SystemExit(f"[ERROR] --root not found: {root}")
    if not ckpt.exists():
        raise SystemExit(f"[ERROR] --ckpt not found: {ckpt}")

    py = sys.executable

    if args.mode == "proto":
        # 1) (optional) choose tau on VAL
        if args.find_threshold:
            run_cmd([
                py, str(HERE / "predict.py"), "proto",
                "--root", str(root),
                "--ckpt", str(ckpt),
                "--split", "val",
                "--support_per_class", str(args.support_per_class),
                "--img_size", str(args.img_size),
                *(["--l2norm"] if args.l2norm else []),
                "--find_best_threshold",
            ])

        # 2) TEST with saved tau; also save predictions CSV
        run_cmd([
            py, str(HERE / "predict.py"), "proto",
            "--root", str(root),
            "--ckpt", str(ckpt),
            "--split", "test",
            "--support_per_class", str(args.support_per_class),
            "--img_size", str(args.img_size),
            *(["--l2norm"] if args.l2norm else []),
            "--use_saved_threshold",
            "--save_preds",
        ])

    else:  # args.mode == "pairs"
        if args.find_threshold:
            run_cmd([
                py, str(HERE / "predict.py"), "eval-pairs",
                "--root", str(root),
                "--ckpt", str(ckpt),
                "--split", "val",
                "--img_size", str(args.img_size),
                "--find_best_threshold",
            ])

        run_cmd([
            py, str(HERE / "predict.py"), "eval-pairs",
            "--root", str(root),
            "--ckpt", str(ckpt),
            "--split", "test",
            "--img_size", str(args.img_size),
            "--use_saved_threshold",
        ])


if __name__ == "__main__":
    main()
