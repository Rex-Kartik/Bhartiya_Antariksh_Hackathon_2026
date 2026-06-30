"""
src/dataset.py — Himawari-8 TIR Frame Interpolation Dataset
============================================================
Loads .npz files (5, H, W) float32 from preprocessed splits.
Each .npz contains: [T0, T10, T20, T30, T40] normalized to [0,1].
Training task: (T0, T40) → (T10, T20, T30)  i.e. T0.25, T0.50, T0.75

Augmentations (train only):
  - Random 256×256 crop (same crop applied to all 5 frames)
  - Random horizontal flip (same flip applied to all 5 frames)
  NO rotation, NO vertical flip — see decisions_made.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# ── Constants ────────────────────────────────────────────────────────
PATCH_SIZE    = 256
BT_MIN_K      = 180.0
BT_MAX_K      = 320.0

# ── Config ───────────────────────────────────────────────────────────
@dataclass
class DataConfig:
    preprocessed_dir: str = "data/preprocessed"
    patch_size:       int  = PATCH_SIZE
    num_workers:      int  = 4
    pin_memory:       bool = True
    # batch sizes
    train_batch: int = 8
    val_batch:   int = 8
    test_batch:  int = 4


# ── Dataset ──────────────────────────────────────────────────────────
class HimawariInterpolationDataset(Dataset):
    """
    Returns:
        frame0 : (1, H, W)   T0   normalized float32
        frame1 : (1, H, W)   T40  normalized float32
        targets: (3, H, W)   [T10, T20, T30]  normalized float32
                              i.e. T0.25, T0.50, T0.75
    """

    def __init__(
        self,
        npz_paths:  List[str],
        augment:    bool = False,
        patch_size: int  = PATCH_SIZE,
    ) -> None:
        super().__init__()
        self.paths      = npz_paths
        self.augment    = augment
        self.patch_size = patch_size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        path = self.paths[idx]

        try:
            with np.load(path, allow_pickle=False) as npz:
                # shape (5, H, W) float32 in [0,1]
                frames: np.ndarray = npz['frames']
        except Exception as e:
            # Return a zero sample on corruption — training will ignore low-loss steps
            # but won't crash.  Logged for awareness.
            print(f"  ⚠️  corrupt npz {Path(path).name}: {e}")
            z = torch.zeros(1, self.patch_size, self.patch_size, dtype=torch.float32)
            return z, z, torch.zeros(3, self.patch_size, self.patch_size)

        assert frames.ndim == 3 and frames.shape[0] == 5, \
            f"Expected (5,H,W), got {frames.shape} in {path}"

        H, W = frames.shape[1], frames.shape[2]

        # ── Random crop (same window for all 5 frames) ────────────────
        if self.augment:
            max_r = H - self.patch_size
            max_c = W - self.patch_size
            r = random.randint(0, max(0, max_r))
            c = random.randint(0, max(0, max_c))
        else:
            # centre crop for val/test — deterministic
            r = (H - self.patch_size) // 2
            c = (W - self.patch_size) // 2

        r = max(0, min(r, H - self.patch_size))
        c = max(0, min(c, W - self.patch_size))
        frames = frames[:, r:r + self.patch_size, c:c + self.patch_size]

        # ── Random horizontal flip (all 5 frames together) ← UPDATED ──
        if self.augment and random.random() < 0.5:
            frames = frames[:, :, ::-1].copy()   # .copy() needed for torch conversion

        # ── To tensor ─────────────────────────────────────────────────
        frames_t = torch.from_numpy(frames)   # (5, H, W)

        frame0  = frames_t[0:1]               # (1, H, W)  T0
        frame1  = frames_t[4:5]               # (1, H, W)  T40
        targets = frames_t[[1, 2, 3]]         # (3, H, W)  T10, T20, T30

        return frame0, frame1, targets


# ── Path collection helpers ──────────────────────────────────────────
def collect_npz_paths(preprocessed_dir: str, split: str) -> List[str]:
    """
    Returns sorted list of .npz paths for the given split (train/val/test).
    Skips any path that fails a quick header check.
    """
    split_dir = Path(preprocessed_dir) / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    paths = sorted(split_dir.glob("*.npz"))
    valid = []
    for p in paths:
        try:
            with np.load(str(p), allow_pickle=False) as npz:
                if 'frames' in npz.files:
                    valid.append(str(p))
        except Exception:
            pass   # corrupted — skip silently
    return valid


# ── DataLoader factory ───────────────────────────────────────────────
def build_dataloaders(cfg: DataConfig) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Returns (train_loader, val_loader, test_loader).
    TEMPORAL split is guaranteed by the preprocessor — do NOT shuffle across dates.
    Within a split, shuffle=True for train is fine (shuffles within the split only).
    """
    train_paths = collect_npz_paths(cfg.preprocessed_dir, "train")
    val_paths   = collect_npz_paths(cfg.preprocessed_dir, "val")
    test_paths  = collect_npz_paths(cfg.preprocessed_dir, "test")

    print(f"  Dataset sizes — train: {len(train_paths)}  val: {len(val_paths)}  test: {len(test_paths)}")

    train_ds = HimawariInterpolationDataset(train_paths, augment=True,  patch_size=cfg.patch_size)
    val_ds   = HimawariInterpolationDataset(val_paths,   augment=False, patch_size=cfg.patch_size)
    test_ds  = HimawariInterpolationDataset(test_paths,  augment=False, patch_size=cfg.patch_size)

    # Windows: num_workers > 0 requires if __name__ == '__main__' guard in train script.
    # Set to 0 here if running on Windows; override via cfg.
    nw = cfg.num_workers

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train_batch,
        shuffle=True,               # safe — split is already temporally bounded
        num_workers=nw,
        pin_memory=cfg.pin_memory,
        persistent_workers=(nw > 0),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.val_batch,
        shuffle=False,
        num_workers=nw,
        pin_memory=cfg.pin_memory,
        persistent_workers=(nw > 0),
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.test_batch,
        shuffle=False,
        num_workers=0,              # test: never drop, never shuffle, no workers needed
        pin_memory=False,
        drop_last=False,
    )
    return train_loader, val_loader, test_loader


# ── Quick sanity check ───────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root))

    cfg = DataConfig(
        preprocessed_dir=str(root / "data" / "preprocessed"),
        num_workers=0,
    )
    train_loader, val_loader, _ = build_dataloaders(cfg)

    f0, f1, tgt = next(iter(train_loader))
    print(f"frame0  : {f0.shape}  [{f0.min():.3f}, {f0.max():.3f}]")
    print(f"frame1  : {f1.shape}  [{f1.min():.3f}, {f1.max():.3f}]")
    print(f"targets : {tgt.shape} [{tgt.min():.3f}, {tgt.max():.3f}]")
    print("✅ Dataset OK")