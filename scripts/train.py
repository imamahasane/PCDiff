#!/usr/bin/env python
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from tqdm import tqdm

from src.utils.seed import seed_everything, worker_init_fn
from src.utils.io import ensure_dir, git_commit_hash, save_yaml
from src.utils.logging import setup_logger, make_tb_writer

from src.ct.operators import CTProjector, ParallelBeamGeometry
from src.ct.poisson import add_poisson_noise_to_sinogram
from src.data.lodopab import LoDoPaBDataset
from src.data.mayo_aapm import MayoAAPMDataset
from src.data.transforms import AugmentConfig, random_augment

from src.diffusion.schedule import make_ddpm_schedule
from src.diffusion.ddpm import PhysicsConditionedDDPM
from src.models.unet import UNetConfig, UNetModel
from src.losses.physics import PhysicsConsistencyLoss
from src.losses.perceptual import VGGPerceptualLoss

def build_dataset(cfg) :
    if cfg.data.dataset == "lodopab":
        return LoDoPaBDataset(cfg.data.root, cfg.data.split)
    if cfg.data.dataset == "mayo_aapm":
        return MayoAAPMDataset(cfg.data.root, cfg.data.split)
    raise ValueError(f"Unknown dataset: {cfg.data.dataset}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--resume", type=str, default=None)
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)

    seed_everything(int(cfg.experiment.seed), deterministic=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(cfg.experiment.out_dir) / cfg.experiment.name
    ensure_dir(out_dir)
    log_dir = ensure_dir(out_dir / "logs")
    ckpt_dir = ensure_dir(out_dir / "checkpoints")

    logger = setup_logger(log_dir)
    writer = make_tb_writer(log_dir)

    # log config + git hash
    cfg_meta = OmegaConf.to_container(cfg, resolve=True)
    cfg_meta["git_commit"] = git_commit_hash()
    save_yaml(cfg_meta, out_dir / "config_resolved.yaml")

    # dataset / loader
    train_cfg = cfg.train
    ds_train = build_dataset(OmegaConf.merge(cfg, {"data": {"split": "train"}}))
    ds_val = build_dataset(OmegaConf.merge(cfg, {"data": {"split": "val"}}))

    dl_train = DataLoader(
        ds_train,
        batch_size=int(train_cfg.batch_size),
        shuffle=True,
        num_workers=int(cfg.data.num_workers),
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=1,
        shuffle=False,
        num_workers=int(cfg.data.num_workers),
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )

    # CT operators
    H, W = cfg.data.image_size
    geom = ParallelBeamGeometry(image_size=int(H), angles=int(cfg.ct.angles), det_count=int(cfg.ct.det_count))
    projector = CTProjector(geom, device=device)

    # model + diffusion
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

    # losses
    phys_loss_fn = PhysicsConsistencyLoss()
    perceptual = None
    if bool(cfg.loss.perceptual.enabled):
        perceptual = VGGPerceptualLoss(list(cfg.loss.perceptual.layers)).to(device)

    # optimizer/scheduler
    opt = torch.optim.AdamW(
        ddpm.parameters(),
        lr=float(cfg.optim.lr),
        betas=tuple(cfg.optim.betas),
        weight_decay=float(cfg.optim.weight_decay),
    )
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode=str(cfg.optim.scheduler.mode),
        factor=float(cfg.optim.scheduler.factor),
        patience=int(cfg.optim.scheduler.patience),
        min_lr=float(cfg.optim.scheduler.min_lr),
    )

    scaler = torch.cuda.amp.GradScaler(enabled=bool(train_cfg.amp))

    start_epoch = 0
    best_val = float("inf")
    epochs_no_improve = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        ddpm.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        sched.load_state_dict(ckpt["sched"])
        scaler.load_state_dict(ckpt.get("scaler", scaler.state_dict()))
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val = ckpt.get("best_val", best_val)
        logger.info(f"Resumed from {args.resume} (epoch={start_epoch})")

    aug_cfg = AugmentConfig()

    grad_accum = int(cfg.optim.grad_accum_steps)
    T = int(cfg.diffusion.T)
    physics_w = float(cfg.loss.physics_weight)
    perc_w = float(cfg.loss.perceptual.weight)

    def run_val(epoch: int) -> float:
        ddpm.eval()
        losses = []
        with torch.no_grad():
            for batch in tqdm(dl_val, desc=f"val {epoch}", leave=False):
                x0 = batch["x"].to(device)  # (B,1,H,W)
                y = batch["y"].to(device)   # (B,angles,det) or (angles,det)
                if y.ndim == 2:
                    y = y.unsqueeze(0)

                c = projector.AT(y)  # conditioning c=A*(y)
                t = torch.randint(0, T, (x0.shape[0],), device=device, dtype=torch.long)
                eps = torch.randn_like(x0)
                x_t = ddpm.q_sample(x0, t, eps)
                out = ddpm.predict_eps_and_x0(x_t, c, t)

                loss_diff = F.mse_loss(out.eps_pred, eps)
                loss_phys = phys_loss_fn(projector, out.x0_pred, y)
                loss = loss_diff + physics_w * loss_phys
                if perceptual is not None:
                    loss = loss + perc_w * perceptual(out.x0_pred, x0)

                losses.append(loss.item())
        ddpm.train()
        return float(sum(losses) / max(1, len(losses)))

    max_epochs = int(cfg.optim.max_epochs)
    for epoch in range(start_epoch, max_epochs):
        ddpm.train()
        opt.zero_grad(set_to_none=True)

        pbar = tqdm(dl_train, desc=f"train {epoch}")
        for step, batch in enumerate(pbar):
            x0 = batch["x"].to(device)
            y = batch["y"].to(device)
            if y.ndim == 2:
                y = y.unsqueeze(0)

            # augmentations on x0 (paper)
            x0_aug = random_augment(x0, aug_cfg)

            c = projector.AT(y)

            t = torch.randint(0, T, (x0.shape[0],), device=device, dtype=torch.long)
            eps = torch.randn_like(x0_aug)
            x_t = ddpm.q_sample(x0_aug, t, eps)

            with torch.cuda.amp.autocast(enabled=bool(train_cfg.amp)):
                out = ddpm.predict_eps_and_x0(x_t, c, t)
                loss_diff = F.mse_loss(out.eps_pred, eps)
                loss_phys = phys_loss_fn(projector, out.x0_pred, y)
                loss = loss_diff + physics_w * loss_phys
                if perceptual is not None:
                    loss = loss + perc_w * perceptual(out.x0_pred, x0_aug)

            scaler.scale(loss / grad_accum).backward()

            if (step + 1) % grad_accum == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

            pbar.set_postfix({
                "L": f"{loss.item():.4f}",
                "Ldiff": f"{loss_diff.item():.4f}",
                "Lphys": f"{loss_phys.item():.4f}",
                "lr": opt.param_groups[0]["lr"],
            })

            global_step = epoch * len(dl_train) + step
            if global_step % int(train_cfg.log_every) == 0:
                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/loss_diff", loss_diff.item(), global_step)
                writer.add_scalar("train/loss_phys", loss_phys.item(), global_step)
                writer.add_scalar("train/lr", opt.param_groups[0]["lr"], global_step)

        # validation
        if (epoch + 1) % int(train_cfg.val_every_epochs) == 0:
            val_loss = run_val(epoch)
            writer.add_scalar("val/loss", val_loss, epoch)
            logger.info(f"Epoch {epoch}: val_loss={val_loss:.6f}")
            sched.step(val_loss)

            if val_loss < best_val:
                best_val = val_loss
                epochs_no_improve = 0
                torch.save({
                    "model": ddpm.state_dict(),
                    "opt": opt.state_dict(),
                    "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(),
                    "epoch": epoch,
                    "best_val": best_val,
                    "config": cfg_meta,
                }, ckpt_dir / "best.pt")
                logger.info(f"Saved best checkpoint (val_loss={best_val:.6f})")
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= int(cfg.optim.early_stop_patience):
                logger.info(f"Early stopping at epoch {epoch} (no improvement for {epochs_no_improve} epochs).")
                break

        # periodic checkpoint
        if (epoch + 1) % int(cfg.optim.save_every_epochs) == 0:
            torch.save({
                "model": ddpm.state_dict(),
                "opt": opt.state_dict(),
                "sched": sched.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "best_val": best_val,
                "config": cfg_meta,
            }, ckpt_dir / f"epoch_{epoch:04d}.pt")

    logger.info("Training finished.")

if __name__ == "__main__":
    main()
