#!/usr/bin/env python
"""Dose-level (E-005) and projection-count (E-006) robustness sweep.

Generates additional Mayo-AAPM *test-set only* variants at different photon
counts (--i0 values) or view counts (--angles values), then evaluates a
SINGLE already-trained checkpoint (zero-shot, no retraining) against each
variant. This is what a "robustness sweep" should mean: does one trained
model degrade gracefully outside its training operating point, not "train N
separate models."

Baseline config (per ARCHITECTURE.md / EXPERIMENTS.md): Mayo-AAPM uses
I0=1e4, 180 views. This orchestrates re-running prepare_mayo_aapm.py (which
already exposes --i0/--angles as CLI args) plus evaluate.py across a list of
additional values, and collects everything into one summary table.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from src.utils.io import ensure_dir, save_json


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep_type", choices=["dose", "projection"], required=True)
    ap.add_argument("--values", type=float, nargs="+", required=True, help="I0 values (dose sweep) or view counts (projection sweep). Include the baseline value for a like-for-like reference point.")
    ap.add_argument("--raw_slices_dir", type=str, required=True, help="Directory of raw Mayo-AAPM test slices, as consumed by prepare_mayo_aapm.py --input_dir.")
    ap.add_argument("--work_dir", type=str, required=True)
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--image_size", type=int, default=256)
    ap.add_argument("--det_count", type=int, default=256)
    ap.add_argument("--base_i0", type=float, default=1e4)
    ap.add_argument("--base_angles", type=int, default=180)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--max_items", type=int, default=None)
    ap.add_argument("--out_json", type=str, required=True)
    args = ap.parse_args()

    work_dir = ensure_dir(Path(args.work_dir))
    summary = []

    for value in args.values:
        tag = f"i0_{value:.0e}" if args.sweep_type == "dose" else f"angles_{int(value)}"
        variant_dir = work_dir / tag
        i0 = value if args.sweep_type == "dose" else args.base_i0
        angles = int(value) if args.sweep_type == "projection" else args.base_angles

        run([
            sys.executable, "scripts/prepare_mayo_aapm.py",
            "--input_dir", args.raw_slices_dir,
            "--output_dir", str(variant_dir),
            "--image_size", str(args.image_size),
            "--angles", str(angles),
            "--det_count", str(args.det_count),
            "--i0", str(i0),
            "--splits", "test",
        ])

        eval_out = variant_dir / "eval_results.json"
        run([
            sys.executable, "scripts/evaluate.py",
            "--config", args.config,
            "--model", args.model,
            "--ckpt", args.ckpt,
            "--split", "test",
            "--K", str(args.K),
            *(["--max_items", str(args.max_items)] if args.max_items is not None else []),
            "--out_json", str(eval_out),
        ])

        result = json.loads(eval_out.read_text())
        summary.append({
            "sweep_type": args.sweep_type,
            "value": value,
            "i0": i0,
            "angles": angles,
            "aggregate": result["aggregate"],
            "n_items": len(result["per_item"]),
            "result_file": str(eval_out),
        })

    save_json({"sweep_type": args.sweep_type, "checkpoint": args.ckpt, "runs": summary}, args.out_json)
    print(f"Wrote sweep summary ({len(summary)} points) to {args.out_json}")


if __name__ == "__main__":
    main()
