"""
api/visualize_demo.py - See the actual interpolated images
=============================================================
Calls /interpolate/demo and saves all 5 frames (T0, T0.25, T0.50,
T0.75, T1) as PNG images side by side, so you can visually confirm
the model is producing sensible interpolated cloud motion.

Usage:
    python api/visualize_demo.py <frame_t0.nc> <frame_t1.nc> [var_name] [crop_row] [crop_col]

Example:
    python api/visualize_demo.py t0.nc t1.nc __xarray_dataarray_variable__ 2000 2000

Requires: pip install matplotlib requests --break-system-packages
"""

import sys
import requests
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_URL = "http://localhost:8000"


def main():
    if len(sys.argv) < 3:
        print("Usage: python api/visualize_demo.py <frame_t0.nc> <frame_t1.nc> [var_name] [crop_row] [crop_col]")
        sys.exit(1)

    file0_path = sys.argv[1]
    file1_path = sys.argv[2]
    var_name   = sys.argv[3] if len(sys.argv) > 3 else "__xarray_dataarray_variable__"
    crop_row   = int(sys.argv[4]) if len(sys.argv) > 4 else 2000
    crop_col   = int(sys.argv[5]) if len(sys.argv) > 5 else 2000

    print(f"Requesting patch at row={crop_row}, col={crop_col} with var_name='{var_name}' ...")

    with open(file0_path, "rb") as f0, open(file1_path, "rb") as f1:
        r = requests.post(
            f"{BASE_URL}/interpolate/demo",
            files={"file0": f0, "file1": f1},
            params={"var_name": var_name, "crop_row": crop_row, "crop_col": crop_col},
        )

    if r.status_code != 200:
        print(f"FAILED: {r.status_code}")
        print(r.json())
        sys.exit(1)

    result = r.json()
    print(f"Got response in {result['elapsed_seconds']}s on {result['device']}")

    frames = {
        "T0 (input)":    np.array(result["frame_t0"]),
        "T0.25 (pred)":  np.array(result["frame_t025"]),
        "T0.50 (pred)":  np.array(result["frame_t050"]),
        "T0.75 (pred)":  np.array(result["frame_t075"]),
        "T1 (input)":    np.array(result["frame_t1"]),
    }

    # Use the same color scale across all 5 panels so brightness is comparable
    vmin = min(f.min() for f in frames.values())
    vmax = max(f.max() for f in frames.values())
    print(f"Brightness temperature range across all frames: {vmin:.1f}K - {vmax:.1f}K")

    fig, axes = plt.subplots(1, 5, figsize=(20, 4.5))
    for ax, (title, arr) in zip(axes, frames.items()):
        # TIR convention: cold = high cloud = should appear bright -> use reversed grayscale
        im = ax.imshow(arr, cmap="gray_r", vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=12)
        ax.axis("off")

    fig.colorbar(im, ax=axes, orientation="horizontal", fraction=0.04, pad=0.08, label="Brightness temperature (K)")
    fig.suptitle(
        f"Patch at row={crop_row}, col={crop_col}  |  inference={result['elapsed_seconds']}s on {result['device']}",
        fontsize=11, y=1.02
    )

    out_path = "interpolation_result.png"
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"\nSaved visualization to: {out_path}")
    print("Open that file to see T0 -> T0.25 -> T0.50 -> T0.75 -> T1")


if __name__ == "__main__":
    main()