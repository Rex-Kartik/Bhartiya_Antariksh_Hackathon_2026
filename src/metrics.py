"""
src/metrics.py — SSIM, PSNR, MSE for TIR Frame Interpolation
=============================================================
All metrics computed on normalized [0,1] float32 tensors.
data_range=1.0 explicitly set everywhere — critical for correct SSIM values.

compute_metrics_batch: works on CPU tensors (B, 1, H, W)
eval_split: runs full split through model, returns per-frame and mean table
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from torch import Tensor

try:
    from skimage.metrics import (
        structural_similarity as sk_ssim,
        peak_signal_noise_ratio as sk_psnr,
    )
    _HAS_SKIMAGE = True
except ImportError:
    _HAS_SKIMAGE = False
    print("  ⚠️  scikit-image not found — SSIM/PSNR will be approximate")


# ── Per-batch metrics (numpy, on CPU float32) ────────────────────────
def compute_metrics_batch(
    pred:   Tensor,   # (B, 1, H, W)  float32 CPU
    target: Tensor,   # (B, 1, H, W)  float32 CPU
) -> Dict[str, float]:
    """
    Returns mean SSIM, PSNR, MSE over the batch.
    Inputs must be in [0,1]; data_range=1.0 is always passed explicitly.
    """
    assert pred.shape == target.shape, f"Shape mismatch: {pred.shape} vs {target.shape}"

    pred_np   = pred.numpy()    # (B, 1, H, W)
    target_np = target.numpy()

    B = pred_np.shape[0]
    ssim_list, psnr_list, mse_list = [], [], []

    for i in range(B):
        p = pred_np[i, 0]     # (H, W)
        t = target_np[i, 0]   # (H, W)

        mse_val = float(np.mean((p - t) ** 2))
        mse_list.append(mse_val)

        if _HAS_SKIMAGE:
            # data_range MUST be set — skimage default assumes uint8 range otherwise
            ssim_val = float(sk_ssim(t, p, data_range=1.0))
            psnr_val = float(sk_psnr(t, p, data_range=1.0))
        else:
            # Fallback: approximate SSIM
            ssim_val = float(1.0 - mse_val * 4)  # rough proxy
            psnr_val = float(-10.0 * np.log10(mse_val + 1e-10))

        ssim_list.append(ssim_val)
        psnr_list.append(psnr_val)

    return {
        "ssim": float(np.mean(ssim_list)),
        "psnr": float(np.mean(psnr_list)),
        "mse":  float(np.mean(mse_list)),
    }


# ── Full split evaluation ─────────────────────────────────────────────
def eval_split(
    model,
    loader,
    device:  torch.device,
    use_amp: bool = True,
) -> Dict[str, Dict[str, float]]:
    """
    Runs model on the given loader (val or test).
    Returns per-frame metrics and overall mean.

    Returns:
        {
          "t025": {"ssim": ..., "psnr": ..., "mse": ...},
          "t050": {...},
          "t075": {...},
          "mean": {...},
        }
    """
    from torch.cuda.amp import autocast
    from src.model import generate_three_frames

    model.eval()
    accum: Dict[str, List[float]] = {
        k: [] for k in [
            "ssim_025", "psnr_025", "mse_025",
            "ssim_050", "psnr_050", "mse_050",
            "ssim_075", "psnr_075", "mse_075",
        ]
    }

    with torch.no_grad():
        for frame0, frame1, targets in loader:
            frame0  = frame0.to(device)
            frame1  = frame1.to(device)

            with autocast(enabled=use_amp):
                preds = generate_three_frames(model, frame0, frame1)

            frame_map = {
                "025": (preds["t025"].float().cpu(), targets[:, 0:1].float()),
                "050": (preds["t050"].float().cpu(), targets[:, 1:2].float()),
                "075": (preds["t075"].float().cpu(), targets[:, 2:3].float()),
            }
            for tag, (pred_f, gt_f) in frame_map.items():
                m = compute_metrics_batch(pred_f, gt_f)
                accum[f"ssim_{tag}"].append(m["ssim"])
                accum[f"psnr_{tag}"].append(m["psnr"])
                accum[f"mse_{tag}"].append(m["mse"])

    # Aggregate
    means = {k: float(np.mean(v)) for k, v in accum.items()}

    result = {
        "t025": {"ssim": means["ssim_025"], "psnr": means["psnr_025"], "mse": means["mse_025"]},
        "t050": {"ssim": means["ssim_050"], "psnr": means["psnr_050"], "mse": means["mse_050"]},
        "t075": {"ssim": means["ssim_075"], "psnr": means["psnr_075"], "mse": means["mse_075"]},
    }
    result["mean"] = {
        "ssim": float(np.mean([result[k]["ssim"] for k in ["t025","t050","t075"]])),
        "psnr": float(np.mean([result[k]["psnr"] for k in ["t025","t050","t075"]])),
        "mse":  float(np.mean([result[k]["mse"]  for k in ["t025","t050","t075"]])),
    }
    return result


# ── Linear interpolation baseline ────────────────────────────────────
def linear_interpolation_baseline(
    loader,
    device: torch.device,
) -> Dict[str, Dict[str, float]]:
    """
    Baseline: simple weighted average of T0 and T1.
      T0.25 = 0.75*T0 + 0.25*T1
      T0.50 = 0.50*T0 + 0.50*T1
      T0.75 = 0.25*T0 + 0.75*T1

    Returns same structure as eval_split — allows direct comparison table.
    """
    accum: Dict[str, List[float]] = {
        k: [] for k in [
            "ssim_025", "psnr_025", "mse_025",
            "ssim_050", "psnr_050", "mse_050",
            "ssim_075", "psnr_075", "mse_075",
        ]
    }

    for frame0, frame1, targets in loader:
        f0, f1 = frame0.float(), frame1.float()

        preds_lin = {
            "025": (0.75 * f0 + 0.25 * f1),
            "050": (0.50 * f0 + 0.50 * f1),
            "075": (0.25 * f0 + 0.75 * f1),
        }
        frame_map = {
            "025": (preds_lin["025"], targets[:, 0:1].float()),
            "050": (preds_lin["050"], targets[:, 1:2].float()),
            "075": (preds_lin["075"], targets[:, 2:3].float()),
        }
        for tag, (pred_f, gt_f) in frame_map.items():
            m = compute_metrics_batch(pred_f, gt_f)
            accum[f"ssim_{tag}"].append(m["ssim"])
            accum[f"psnr_{tag}"].append(m["psnr"])
            accum[f"mse_{tag}"].append(m["mse"])

    means = {k: float(np.mean(v)) for k, v in accum.items()}
    result = {
        "t025": {"ssim": means["ssim_025"], "psnr": means["psnr_025"], "mse": means["mse_025"]},
        "t050": {"ssim": means["ssim_050"], "psnr": means["psnr_050"], "mse": means["mse_050"]},
        "t075": {"ssim": means["ssim_075"], "psnr": means["psnr_075"], "mse": means["mse_075"]},
    }
    result["mean"] = {
        "ssim": float(np.mean([result[k]["ssim"] for k in ["t025","t050","t075"]])),
        "psnr": float(np.mean([result[k]["psnr"] for k in ["t025","t050","t075"]])),
        "mse":  float(np.mean([result[k]["mse"]  for k in ["t025","t050","t075"]])),
    }
    return result


# ── Pretty-print comparison table ────────────────────────────────────
def print_comparison_table(
    model_metrics:    Dict[str, Dict[str, float]],
    baseline_metrics: Optional[Dict[str, Dict[str, float]]] = None,
) -> None:
    from typing import Optional
    frames = ["t025", "t050", "t075", "mean"]
    header = f"{'Frame':<8}  {'SSIM':>8}  {'PSNR':>8}  {'MSE':>10}"
    if baseline_metrics:
        header += f"  {'|':>2}  {'Base SSIM':>9}  {'Base PSNR':>9}  {'Base MSE':>10}"

    print(f"\n{'─'*len(header)}")
    print(header)
    print(f"{'─'*len(header)}")

    for f in frames:
        m = model_metrics[f]
        row = f"{f:<8}  {m['ssim']:>8.4f}  {m['psnr']:>8.2f}  {m['mse']:>10.6f}"
        if baseline_metrics and f in baseline_metrics:
            b = baseline_metrics[f]
            row += f"  {'|':>2}  {b['ssim']:>9.4f}  {b['psnr']:>9.2f}  {b['mse']:>10.6f}"
        print(row)

    print(f"{'─'*len(header)}\n")