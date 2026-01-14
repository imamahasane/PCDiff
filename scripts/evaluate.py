#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from src.utils.seed import seed_everything
from src.ct.operators import CTProjector, ParallelBeamGeometry
from src.data.lodopab import LoDoPaBDataset
from src.data.mayo_aapm import MayoAAPMDataset
from src.diffusion.schedule import make_ddpm_schedule
from src.diffusion.ddpm import PhysicsConditionedDDPM
from src.models.unet import UNetConfig, UNetModel
from src.metrics.image import psnr, ssim_torch
from src.metrics.uncertainty import mean_and_std, pearson_r, coverage_and_ece

def build_dataset(cfg):
    if cfg.data.dataset == "lodopab":
        return LoDoPaBDataset(cfg.data.root, cfg.data.split)
    if cfg.data.dataset == "mayo_aapm":
        return MayoAAPMDataset(cfg.data.root, cfg.data.split)
    raise ValueError(cfg.data.dataset)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--max_items", type=int, default=200)
    ap.add_argument("--out_json", type=str, default="eval_results.json")
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    cfg.data.split = args.split

    seed_everything(int(cfg.experiment.seed), deterministic=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    H, W = cfg.data.image_size
    geom = ParallelBeamGeometry(image_size=int(H), angles=int(cfg.ct.angles), det_count=int(cfg.ct.det_count))
    projector = CTProjector(geom, device=device)

    schedule = make_ddpm_schedule(
        T=int(cfg.diffusion.T),
        beta_schedule=str(cfg.diffusion.beta_schedule),
        beta_start=float(cfg.diffusion.beta_start),
        beta_end=float(cfg.diffusion.beta_end),
        device=device,
    )

    unet_cfg = UNetConfig(
        in_channels=int(cfg.model.in_channels),
        out_channels=int(cfg.model.out_channels),
        base_channels=int(cfg.model.base_channels),
        channel_mult=tuple(cfg.model.channel_mult),
        num_res_blocks=int(cfg.model.num_res_blocks),
        attention_resolutions=tuple(cfg.model.attention_resolutions),
        num_heads=int(cfg.model.num_heads),
        dropout=float(cfg.model.dropout),
    )
    denoiser = UNetModel(unet_cfg, image_size=int(H)).to(device)
    ddpm = PhysicsConditionedDDPM(denoiser, schedule).to(device)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    ddpm.load_state_dict(ckpt["model"])
    ddpm.eval()

    ds = build_dataset(cfg)

    psnrs, ssims = [], []
    rs, eces, covs = [], [], []

    for idx in tqdm(range(min(len(ds), int(args.max_items))), desc="eval"):
        item = ds[idx]
        x_gt = item["x"].unsqueeze(0).to(device)  # (1,1,H,W)
        y = item["y"].unsqueeze(0).to(device)
        c = projector.AT(y)

        # K samples for uncertainty
        samples = []
        for _ in range(int(args.K)):
            x_hat = ddpm.sample(c, shape=(1,1,int(H),int(W)))
            samples.append(x_hat)
        samples = torch.stack(samples, dim=0)  # (K,1,1,H,W)
        mean, std = mean_and_std(samples)

        psnrs.append(psnr(mean[0], x_gt[0]))
        ssims.append(ssim_torch(mean[0], x_gt[0]))

        # uncertainty alignment metrics (paper style)
        err = (mean - x_gt).abs()
        rs.append(pearson_r(std, err))
        cal = coverage_and_ece(err, std)
        covs.append(cal.coverage95)
        eces.append(cal.ece)

    results = {
        "N": len(psnrs),
        "PSNR_mean": float(np.mean(psnrs)),
        "PSNR_std": float(np.std(psnrs)),
        "SSIM_mean": float(np.mean(ssims)),
        "SSIM_std": float(np.std(ssims)),
        "Pearson_r_mean": float(np.mean(rs)),
        "ECE_mean": float(np.mean(eces)),
        "Coverage95_mean": float(np.mean(covs)),
    }

    Path(args.out_json).write_text(json_dumps(results))
    print(results)

def json_dumps(obj):
    import json
    return json.dumps(obj, indent=2, sort_keys=True)

if __name__ == "__main__":
    main()
