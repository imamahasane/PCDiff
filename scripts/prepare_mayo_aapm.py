#!/usr/bin/env python
"""Build Mayo-AAPM train/val/test .npz splits: resize, normalize to [-1,1],
forward-project, and inject Poisson low-dose noise at a given photon count.

--i0 and --angles are already exposed as CLI args, which is what makes the
dose sweep (E-005) and projection-count sweep (E-006) tractable without new
code: generate additional test-set variants at different --i0/--angles
against the SAME trained checkpoint (see scripts/run_dose_sweep.py) rather
than retraining per dose level.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.ct.operators import CTProjector, ParallelBeamGeometry
from src.ct.poisson import add_poisson_noise_to_sinogram
from src.utils.io import ensure_dir
from src.utils.seed import seed_everything


def load_slices(input_dir: Path) -> List[Path]:
    files = sorted(list(input_dir.glob("*.npy")) + list(input_dir.glob("*.npz")))
    if not files:
        raise FileNotFoundError(f"No .npy/.npz files found in {input_dir}")
    return files


def read_slice(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path).astype(np.float32)
    d = np.load(path)
    if "image" in d:
        return d["image"].astype(np.float32)
    return d[list(d.keys())[0]].astype(np.float32)


def normalize_to_minus1_1(x: torch.Tensor) -> torch.Tensor:
    """Per-slice min/max rescale to [-1,1] (common in CT preprocessing when,
    unlike LoDoPaB, there's no dataset-wide fixed intensity convention to
    rely on)."""
    x_min = x.amin(dim=(-2, -1), keepdim=True)
    x_max = x.amax(dim=(-2, -1), keepdim=True)
    x = (x - x_min) / (x_max - x_min + 1e-8)
    return x * 2.0 - 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--image_size", type=int, default=256)
    ap.add_argument("--angles", type=int, default=180)
    ap.add_argument("--det_count", type=int, default=256)
    ap.add_argument("--i0", type=float, default=1e4)
    ap.add_argument("--train_frac", type=float, default=0.8)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--splits", type=str, default="train,val,test", help="Comma-separated subset of splits to (re)generate -- dose/projection sweeps only need 'test'.")
    args = ap.parse_args()

    seed_everything(args.seed, deterministic=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    requested_splits = set(args.splits.split(","))
    for s in requested_splits:
        ensure_dir(out_dir / s)

    files = load_slices(in_dir)
    random.Random(args.seed).shuffle(files)

    n = len(files)
    n_train = int(n * args.train_frac)
    n_val = int(n * args.val_frac)
    all_splits = {
        "train": files[:n_train],
        "val": files[n_train : n_train + n_val],
        "test": files[n_train + n_val :],
    }
    splits = {k: v for k, v in all_splits.items() if k in requested_splits}

    geom = ParallelBeamGeometry(image_size=args.image_size, angles=args.angles, det_count=args.det_count)
    projector = CTProjector(geom, device=device)

    for split, flist in splits.items():
        for p in tqdm(flist, desc=f"prepare {split} (i0={args.i0}, angles={args.angles})"):
            img = read_slice(p)
            x = torch.from_numpy(img)[None, None].to(device)
            x = F.interpolate(x, size=(args.image_size, args.image_size), mode="bilinear", align_corners=False)
            x = normalize_to_minus1_1(x)

            y_clean = projector.A(x)
            y_ld = add_poisson_noise_to_sinogram(y_clean, i0=args.i0)

            out_path = out_dir / split / f"{p.stem}.npz"
            np.savez_compressed(out_path, x=x.detach().cpu().numpy(), y=y_ld.detach().cpu().numpy())


if __name__ == "__main__":
    main()
