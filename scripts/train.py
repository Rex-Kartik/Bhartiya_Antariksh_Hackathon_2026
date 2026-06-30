"""
src/train.py — Fine-tuning Loop for RIFE TIR Satellite Interpolation
=====================================================================
Usage:
    python src/train.py                          # defaults
    python src/train.py --epochs 20 --batch 4
    python src/train.py --resume checkpoints/epoch_003.pth
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")


import argparse
import gc
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.dataset import DataConfig, build_dataloaders
from src.model   import ModelConfig, RIFEGrayscale, build_model, generate_three_frames_train
from src.metrics import compute_metrics_batch

try:
    from pytorch_msssim import ssim as ms_ssim
    _HAS_MSSSIM = True
except ImportError:
    _HAS_MSSSIM = False
    print("  ⚠️  pytorch_msssim not found — using pure MSE loss")

# ── Paths ─────────────────────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
LOG_DIR        = PROJECT_ROOT / "logs"
SAMPLE_DIR     = LOG_DIR / "samples"

for _d in [CHECKPOINT_DIR, LOG_DIR, SAMPLE_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────
@dataclass
class TrainConfig:
    epochs:            int   = 10
    lr:                float = 1e-5
    lr_min:            float = 1e-7
    grad_clip:         float = 1.0
    lambda_mse:        float = 1.0
    lambda_ssim:       float = 0.5
    save_sample_every: int   = 500
    validate_every:    int   = 1
    preprocessed_dir:  str   = str(PROJECT_ROOT / "data" / "preprocessed")
    checkpoint_dir:    str   = str(CHECKPOINT_DIR)
    rife_repo_path:    str   = str(PROJECT_ROOT / "ECCV2022-RIFE")
    pretrained_ckpt:   str   = str(CHECKPOINT_DIR / "flownet.pkl")
    resume_from:       str   = ""
    device:            str   = "cuda"
    batch_size:        int   = 2
    accumulation_steps: int  = 4
    num_workers:       int   = 0
    use_amp:           bool  = False
    min_batch_size:    int   = 1


# ── Loss ──────────────────────────────────────────────────────────────
class InterpolationLoss(nn.Module):
    def __init__(self, lambda_mse: float = 1.0, lambda_ssim: float = 0.5) -> None:
        super().__init__()
        self.lambda_mse  = lambda_mse
        self.lambda_ssim = lambda_ssim
        self.mse = nn.MSELoss()
        
        self.frame_weights = {
            "t025": 1.0, 
            "t050": 2.0,  # Double the penalty for mistakes on the midpoint
            "t075": 1.0
        }
        self.weight_sum = sum(self.frame_weights.values())

    def forward(
        self,
        preds:   Dict[str, torch.Tensor],
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        frame_map = {
            "t025": targets[:, 0:1],
            "t050": targets[:, 1:2],
            "t075": targets[:, 2:3],
        }
        total_loss = torch.tensor(0.0, device=targets.device)
        per_frame: Dict[str, float] = {}

        for key, gt in frame_map.items():
            pred = preds[key]
            mse_val  = self.mse(pred, gt)
            
            if _HAS_MSSSIM:
                ssim_val = ms_ssim(pred.float(), gt.float(), data_range=1.0, size_average=True)
            else:
                ssim_val = torch.tensor(0.0, device=targets.device)
                
            frame_loss = self.lambda_mse * mse_val + self.lambda_ssim * (1.0 - ssim_val)
            frame_loss = torch.clamp(frame_loss, min=0.0) 
            
            weighted_loss = frame_loss * self.frame_weights[key]
            total_loss = total_loss + weighted_loss
            
            # 3. Log the RAW (unweighted) loss so your terminal metrics stay readable
            per_frame[key] = frame_loss.item()

        # 4. Divide by the weight sum (4.0) instead of 3.0 to keep gradients normalized
        return total_loss / self.weight_sum, per_frame


# ── Checkpoint helpers ────────────────────────────────────────────────
def save_checkpoint(model, optimizer, scheduler, scaler, epoch, best_ssim, metrics, path):
    torch.save({
        "epoch":     epoch,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler":    scaler.state_dict(),
        "best_ssim": best_ssim,
        "metrics":   metrics,
    }, str(path))


def load_checkpoint(path, model, optimizer, scheduler, scaler):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    scaler.load_state_dict(ckpt["scaler"])
    tqdm.write(f"  ✅ Resumed from epoch {ckpt['epoch']}, best SSIM={ckpt.get('best_ssim', 0):.4f}")
    return ckpt["epoch"], ckpt.get("best_ssim", 0.0), ckpt.get("metrics", {})


# ── Sample saver ─────────────────────────────────────────────────────
def save_sample(frame0, frame1, preds, targets, step):
    out = {
        "f0":     frame0[0, 0].cpu().numpy(),
        "f1":     frame1[0, 0].cpu().numpy(),
        "t025":   preds["t025"][0, 0].detach().cpu().numpy(),
        "t050":   preds["t050"][0, 0].detach().cpu().numpy(),
        "t075":   preds["t075"][0, 0].detach().cpu().numpy(),
        "gt_025": targets[0, 0].cpu().numpy(),
        "gt_050": targets[0, 1].cpu().numpy(),
        "gt_075": targets[0, 2].cpu().numpy(),
    }
    np.savez_compressed(SAMPLE_DIR / f"step_{step:07d}.npz", **out)


# ── Validation ───────────────────────────────────────────────────────
def validate(model, val_loader, loss_fn, device, use_amp):
    model.eval()
    running = {k: [] for k in ["loss", "ssim_025", "ssim_050", "ssim_075",
                                        "psnr_025", "psnr_050", "psnr_075"]}

    val_bar = tqdm(val_loader, desc="  Validating", unit="batch", leave=False,
                   bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]{postfix}")

    with torch.no_grad():
        for frame0, frame1, targets in val_bar:
            frame0  = frame0.to(device, non_blocking=True)
            frame1  = frame1.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with autocast(enabled=use_amp):
                preds = generate_three_frames_train(model, frame0, frame1)
                loss, _ = loss_fn(preds, targets)

            running["loss"].append(loss.item())

            frame_map = {
                "025": (preds["t025"].float().cpu(), targets[:, 0:1].float().cpu()),
                "050": (preds["t050"].float().cpu(), targets[:, 1:2].float().cpu()),
                "075": (preds["t075"].float().cpu(), targets[:, 2:3].float().cpu()),
            }
            for tag, (pred_f, gt_f) in frame_map.items():
                m = compute_metrics_batch(pred_f, gt_f)
                running[f"ssim_{tag}"].append(m["ssim"])
                running[f"psnr_{tag}"].append(m["psnr"])

            val_bar.set_postfix({
                "loss":    f"{loss.item():.4f}",
                "ssim050": f"{running['ssim_050'][-1]:.4f}" if running["ssim_050"] else "─",
            })

    val_bar.close()
    return {k: float(np.mean(v)) for k, v in running.items()}


# ── Training loop ─────────────────────────────────────────────────────
def train(cfg: TrainConfig) -> None:
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  Bhartiya Antariksh Hackathon 2026 — PS12 Training")
    print(f"  Device : {device}")
    print(f"  Epochs : {cfg.epochs}   Batch : {cfg.batch_size}   LR : {cfg.lr:.1e}")
    print(f"  AMP    : {cfg.use_amp}   Workers : {cfg.num_workers}")
    print(f"{'='*60}\n")

    model_cfg = ModelConfig(
        checkpoint_path=cfg.pretrained_ckpt,
        rife_repo_path=cfg.rife_repo_path,
        device=str(device),
    )
    model     = build_model(model_cfg)
    optimizer = Adam(model.parameters(), lr=cfg.lr, betas=(0.9, 0.999))
    scaler    = GradScaler(enabled=(cfg.use_amp and device.type == "cuda"))
    loss_fn   = InterpolationLoss(cfg.lambda_mse, cfg.lambda_ssim)

    start_epoch = 0
    best_ssim   = 0.0
    all_metrics: Dict = {"train_loss": [], "val": []}

    if cfg.resume_from and Path(cfg.resume_from).exists():
        # Load model/optimizer/scaler weights only — do NOT restore scheduler.
        # Scheduler is rebuilt fresh below with T_max=remaining epochs so LR
        # anneals correctly from resume point instead of oscillating.  # <- UPDATED
        ckpt = torch.load(cfg.resume_from, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_ssim   = ckpt.get("best_ssim", 0.0)
        all_metrics = ckpt.get("metrics", {"train_loss": [], "val": []})
        tqdm.write(f"  Resumed from epoch {ckpt['epoch']}, best SSIM={best_ssim:.4f}")

    # Build scheduler AFTER resume so T_max covers remaining epochs only  # <- UPDATED
    remaining = cfg.epochs - start_epoch
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, remaining), eta_min=cfg.lr_min)
    tqdm.write(f"  Scheduler: CosineAnnealingLR T_max={max(1,remaining)} epochs")

    data_cfg = DataConfig(
        preprocessed_dir=cfg.preprocessed_dir,
        train_batch=cfg.batch_size,
        val_batch=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    train_loader, val_loader, _ = build_dataloaders(data_cfg)

    run_id   = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"train_{run_id}.jsonl"
    print(f"  Log : {log_path}\n")

    def _log(record: dict) -> None:
        with open(log_path, "a") as fh:
            fh.write(json.dumps(record) + "\n")

    _log({"event": "start", "config": asdict(cfg), "run_id": run_id})

    # ── Epoch bar — outer loop ────────────────────────────────────────
    epoch_bar = tqdm(
        range(start_epoch, cfg.epochs),
        desc="  Overall",
        unit="epoch",
        leave=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} epochs  [elapsed {elapsed}  ETA {remaining}]{postfix}"
    )

    for epoch in epoch_bar:
        model.train()
        epoch_losses: list = []
        t_epoch      = time.time()
        global_step  = epoch * len(train_loader)

        # ── Batch bar — inner loop ────────────────────────────────────
        batch_bar = tqdm(
            train_loader,
            desc=f"  Ep {epoch+1:>2}/{cfg.epochs}  train",
            unit="batch",
            leave=False,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}  [{elapsed}<{remaining}, {rate_fmt}]{postfix}"
        )

        # Move zero_grad HERE, before the loop starts
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, (frame0, frame1, targets) in enumerate(batch_bar):
            frame0  = frame0.to(device, non_blocking=True)
            frame1  = frame1.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            try:
                with autocast(enabled=(cfg.use_amp and device.type == "cuda")):
                    preds        = generate_three_frames_train(model, frame0, frame1)
                    loss, pf_lss = loss_fn(preds, targets)
                    
                    # 1. Scale the loss down by accumulation steps
                    loss = loss / cfg.accumulation_steps

                # 2. Backward pass (accumulates gradients automatically)
                if cfg.use_amp:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                # 3. Only step weights when accumulation window hits, or at end of epoch
                if (batch_idx + 1) % cfg.accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                    if cfg.use_amp:
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                        optimizer.step()
                    
                    # Clear gradients for the next window
                    optimizer.zero_grad(set_to_none=True)

            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                gc.collect()
                new_bs = max(cfg.min_batch_size, data_cfg.train_batch // 2)
                if new_bs < data_cfg.train_batch:
                    tqdm.write(f"  ⚠️  CUDA OOM — halving batch {data_cfg.train_batch}→{new_bs}")
                    data_cfg.train_batch = new_bs
                    train_loader, val_loader, _ = build_dataloaders(data_cfg)
                    optimizer.zero_grad(set_to_none=True) # Reset tracking on crash
                    batch_bar.close()
                    break
                else:
                    tqdm.write("  ❌ CUDA OOM — min batch size reached.")
                    raise

            # Recalculate original loss magnitude for reporting/logging accuracy
            unscaled_loss_val = loss.item() * cfg.accumulation_steps
            epoch_losses.append(unscaled_loss_val)
            step    = global_step + batch_idx
            lr_now  = optimizer.param_groups[0]["lr"]

            if step % cfg.save_sample_every == 0:
                save_sample(frame0, frame1, preds, targets, step)

            batch_bar.set_postfix(ordered_dict={
                "loss":  f"{unscaled_loss_val:.4f}",
                "T0.25": f"{pf_lss.get('t025', 0):.4f}",
                "T0.50": f"{pf_lss.get('t050', 0):.4f}",
                "T0.75": f"{pf_lss.get('t075', 0):.4f}",
                "lr":    f"{lr_now:.1e}",
            })

            if batch_idx % 50 == 0:
                _log({"event": "batch", "epoch": epoch+1, "batch": batch_idx,
                    "loss": unscaled_loss_val, "lr": lr_now,
                    **{f"loss_{k}": v for k, v in pf_lss.items()}})

        batch_bar.close()
        scheduler.step()
        mean_train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        all_metrics["train_loss"].append(mean_train_loss)

        # ── Validation ───────────────────────────────────────────────
        if (epoch + 1) % cfg.validate_every == 0:
            val_m = validate(model, val_loader, loss_fn, device, cfg.use_amp)
            all_metrics["val"].append({**val_m, "epoch": epoch+1})

            mean_val_ssim = (val_m["ssim_025"] + val_m["ssim_050"] + val_m["ssim_075"]) / 3.0

            # Print validation summary cleanly above the epoch bar
            tqdm.write(
                f"\n  ╔══ Epoch {epoch+1}/{cfg.epochs} ══════════════════════════════╗\n"
                f"  ║  train_loss : {mean_train_loss:.5f}\n"
                f"  ║  val_loss   : {val_m['loss']:.5f}\n"
                f"  ║  SSIM  T0.25={val_m['ssim_025']:.4f}  T0.50={val_m['ssim_050']:.4f}  T0.75={val_m['ssim_075']:.4f}  mean={mean_val_ssim:.4f}\n"
                f"  ║  PSNR  T0.25={val_m['psnr_025']:.2f}    T0.50={val_m['psnr_050']:.2f}    T0.75={val_m['psnr_075']:.2f}\n"
                f"  ║  time  {time.time()-t_epoch:.1f}s\n"
                f"  ╚{'═'*48}╝"
            )

            _log({"event": "val", "epoch": epoch+1, "train_loss": mean_train_loss,
                  **val_m, "ssim_mean": mean_val_ssim,
                  "elapsed_s": time.time()-t_epoch})

            if mean_val_ssim > best_ssim:
                best_ssim = mean_val_ssim
                best_path = Path(cfg.checkpoint_dir) / "best_model.pth"
                save_checkpoint(model, optimizer, scheduler, scaler,
                                epoch, best_ssim, all_metrics, best_path)
                tqdm.write(f"  🏆 New best SSIM {best_ssim:.4f} → {best_path}")

            # Update outer epoch bar with latest val metrics
            epoch_bar.set_postfix({
                "loss":      f"{mean_train_loss:.4f}",
                "val_ssim":  f"{mean_val_ssim:.4f}",
                "best_ssim": f"{best_ssim:.4f}",
            })

        # ── Per-epoch checkpoint ──────────────────────────────────────
        epoch_path = Path(cfg.checkpoint_dir) / f"epoch_{epoch+1:03d}.pth"
        save_checkpoint(model, optimizer, scheduler, scaler,
                        epoch, best_ssim, all_metrics, epoch_path)
        tqdm.write(f"  💾 Saved {epoch_path.name}")

    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE — Best val SSIM : {best_ssim:.4f}")
    print(f"  Best model : {Path(cfg.checkpoint_dir) / 'best_model.pth'}")
    print(f"{'='*60}\n")
    _log({"event": "done", "best_ssim": best_ssim})


# ── CLI ───────────────────────────────────────────────────────────────
def _parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="Train RIFE on Himawari-8 TIR")
    
    p.add_argument("--epochs",      type=int,   default=10)
    p.add_argument("--batch",       type=int,   default=2,    help="Physical batch size (Default: 2 for 6GB VRAM)")
    p.add_argument("--accum-steps", type=int,   default=4,    help="Gradient accumulation steps (2 * 4 = 8 effective)")
    p.add_argument("--lr",          type=float, default=1e-5)
    
    # Set workers to 0 by default to prevent Windows RAM freezing
    p.add_argument("--workers",     type=int,   default=0,    help="Dataloader workers (0 recommended for Windows)")
    
    p.add_argument("--resume",      type=str,   default="")
    
    # Flipped logic: AMP is now OFF by default to protect SSIM accuracy
    p.add_argument("--use-amp",     action="store_true",      help="Enable Mixed Precision (Not recommended for TIR data)")
    p.add_argument("--device",      type=str,   default="cuda")
    
    args = p.parse_args()
    
    return TrainConfig(
        epochs=args.epochs, 
        batch_size=args.batch, 
        accumulation_steps=args.accum_steps,
        lr=args.lr,
        num_workers=args.workers, 
        resume_from=args.resume,
        use_amp=args.use_amp, 
        device=args.device,
    )


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    train(_parse_args())