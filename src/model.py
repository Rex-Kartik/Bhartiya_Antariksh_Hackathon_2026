"""
src/model.py - RIFE v6 Wrapper for Grayscale TIR Satellite Interpolation
=========================================================================
Adapted for the ECCV2022-RIFE IFNet.py forward signature:
  forward(x, scale=[4,2,1], timestep=0.5)
    x = cat(img0, img1, gt)   shape (B, 6, H, W) at inference (gt = zeros)
  returns (flow_list, mask, merged, flow_teacher, merged_teacher, loss_distill)
    merged[2] is the final interpolated frame  shape (B, 3, H, W)

Grayscale strategy: 1ch -> 3ch via replication, output averaged back to 1ch.
"""

from __future__ import annotations

import importlib.util
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn

# ── Constants ─────────────────────────────────────────────────────────
RIFE_REPO_PATH = Path(__file__).resolve().parent.parent / "ECCV2022-RIFE"


# ── Config ────────────────────────────────────────────────────────────
@dataclass
class ModelConfig:
    checkpoint_path: str  = "checkpoints/flownet.pkl"
    rife_repo_path:  str  = str(RIFE_REPO_PATH)
    device:          str  = "cuda"


# ── RIFE import via importlib (avoids sys.path naming conflicts) ───────
def _import_rife(rife_repo_path: str):
    repo = Path(rife_repo_path)
    if not repo.exists():
        raise ImportError(
            f"RIFE repo not found at {repo}\n"
            "Run: git clone https://github.com/hzwer/ECCV2022-RIFE"
        )

    candidates = [
        repo / "model" / "IFNet.py",
        repo / "model" / "IFNet_HDv3.py",
        repo / "IFNet.py",
        repo / "IFNet_HDv3.py",
    ]
    ifnet_path = next((p for p in candidates if p.exists()), None)
    if ifnet_path is None:
        raise ImportError(
            f"No IFNet file found in RIFE repo.\n"
            f"Checked: {[str(c) for c in candidates]}"
        )
    print(f"  Using IFNet: {ifnet_path}")

    # IFNet.py uses absolute imports: "from model.warplayer import warp"
    # So the REPO ROOT (parent of model/) must be on sys.path permanently.
    repo_root = str(ifnet_path.parent.parent)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    spec   = importlib.util.spec_from_file_location("IFNet_rife", str(ifnet_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module.IFNet


# ── Main model wrapper ────────────────────────────────────────────────
class RIFEGrayscale(nn.Module):
    """
    Wraps RIFE IFNet for single-channel TIR interpolation.

    forward(img0, img1, timestep) -> (B, 1, H, W)
      img0, img1 : (B, 1, H, W) float32 in [0,1]
      timestep   : float in (0,1)

    Internally:
      1. Replicate 1ch -> 3ch
      2. cat(img0_3ch, img1_3ch, zeros_3ch) -> (B, 9, H, W)  [gt=zeros at inference]
         NOTE: IFNet expects (B, 6, H, W) for img0+img1, plus optional gt.
         At inference gt has shape (B, 0, H, W) — pass zeros with 0 channels.
      3. IFNet returns merged[2] shape (B, 3, H, W)
      4. Average 3ch -> 1ch
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg    = cfg
        IFNet       = _import_rife(cfg.rife_repo_path)
        self.ifnet  = IFNet()
        self._load_checkpoint(cfg.checkpoint_path)

    def _load_checkpoint(self, ckpt_path: str) -> None:
        p = Path(ckpt_path)
        if not p.exists():
            warnings.warn(
                f"Checkpoint {p} not found - using random weights.\n"
                "Download flownet.pkl from the RIFE releases page."
            )
            return
        state = torch.load(str(p), map_location="cpu", weights_only=True)
        state = {k.replace("module.", ""): v for k, v in state.items()}
        missing, unexpected = self.ifnet.load_state_dict(state, strict=False)
        if missing:
            print(f"  Warning: {len(missing)} missing keys (e.g. {missing[0]})")
        print(f"  Loaded RIFE checkpoint: {p}")

    def forward(
        self,
        img0:     torch.Tensor,   # (B, 1, H, W)
        img1:     torch.Tensor,   # (B, 1, H, W)
        timestep: float = 0.5,
    ) -> torch.Tensor:            # (B, 1, H, W)

        # 1ch -> 3ch via replication
        i0 = img0.repeat(1, 3, 1, 1)   # (B, 3, H, W)
        i1 = img1.repeat(1, 3, 1, 1)   # (B, 3, H, W)

        # IFNet expects x = cat(img0, img1, gt) where gt has 0 channels at inference
        # gt.shape[1] == 3 triggers teacher distillation — we pass gt with 0 channels
        # to skip it (see IFNet.forward: `if gt.shape[1] == 3`)
        B, _, H, W = i0.shape
        gt = torch.zeros(B, 0, H, W, dtype=i0.dtype, device=i0.device)
        x  = torch.cat([i0, i1, gt], dim=1)   # (B, 6, H, W)

        # IFNet returns: (flow_list, mask, merged, flow_teacher, merged_teacher, loss_distill)
        # merged[2] is the final refined output, shape (B, 3, H, W), clamped [0,1]
        _, _, merged, _, _, _ = self.ifnet(x, scale=[4, 2, 1], timestep=timestep)
        out_3ch = merged[2]   # (B, 3, H, W)

        # 3ch -> 1ch via mean
        return out_3ch.mean(dim=1, keepdim=True)   # (B, 1, H, W)


# ── Iterative 3-frame generation (inference, no grad) ─────────────────
@torch.no_grad()
def generate_three_frames(
    model:  RIFEGrayscale,
    frame0: torch.Tensor,
    frame1: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    T0.50 = RIFE(T0,    T1,    t=0.5)
    T0.25 = RIFE(T0,    T0.50, t=0.5)
    T0.75 = RIFE(T0.50, T1,    t=0.5)
    """
    model.eval()
    t050 = model(frame0, frame1, timestep=0.5)
    t025 = model(frame0, t050,   timestep=0.5)
    t075 = model(t050,  frame1,  timestep=0.5)
    return {"t025": t025, "t050": t050, "t075": t075}


# ── Same with gradient tracking (for training) ────────────────────────
def generate_three_frames_train(
    model:  RIFEGrayscale,
    frame0: torch.Tensor,
    frame1: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    model.train()
    t050 = model(frame0, frame1, timestep=0.5)
    t025 = model(frame0, t050,   timestep=0.5)
    t075 = model(t050,  frame1,  timestep=0.5)
    return {"t025": t025, "t050": t050, "t075": t075}


# ── Model factory ─────────────────────────────────────────────────────
def build_model(cfg: ModelConfig) -> RIFEGrayscale:
    model  = RIFEGrayscale(cfg)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    model  = model.to(device)
    print(f"  Model on: {device}")
    return model


# ── Sanity check ──────────────────────────────────────────────────────
if __name__ == "__main__":
    cfg   = ModelConfig(device="cpu")
    model = build_model(cfg)
    f0    = torch.rand(1, 1, 256, 256)
    f1    = torch.rand(1, 1, 256, 256)
    preds = generate_three_frames(model, f0, f1)
    for k, v in preds.items():
        print(f"  {k}: {v.shape}  [{v.min():.3f}, {v.max():.3f}]")
    print("OK")