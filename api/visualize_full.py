"""
api/visualize_full.py - Full-globe interpolation, downloaded and visualized
==============================================================================
Calls /interpolate/full (the real pipeline, not the demo patch), polls
until the job completes, downloads the resulting .nc file, and renders
all 5 frames (T0, T0.25, T0.50, T0.75, T1) at full resolution.

NaN handling: pixels that are NaN in the ORIGINAL T0 or T1 (fill value,
no data, space background outside the Earth disk) stay NaN in every
output frame -- rendered as transparent, not black or any fabricated
brightness temperature. This is enforced server-side in main.py
(nan_mask captured before normalization, reapplied after denorm).

Usage:
    python api/visualize_full.py <frame_t0.nc> <frame_t1.nc> [var_name]

This can take several minutes on CPU, ~1-2 minutes on GPU for a full
5500x5500 Himawari frame (only non-fill tiles are actually run through
the model -- ocean/space tiles are skipped, see run_tiled_inference).

Requires: pip install matplotlib requests netCDF4 --break-system-packages
"""

import sys
import time
import requests
import numpy as np
import netCDF4 as nc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_URL = "http://localhost:8000"
POLL_INTERVAL_S = 5


def submit_job(file0_path: str, file1_path: str, var_name: str) -> str:
    print(f"Submitting full-globe job with var_name='{var_name}' ...")
    with open(file0_path, "rb") as f0, open(file1_path, "rb") as f1:
        r = requests.post(
            f"{BASE_URL}/interpolate/full",
            files={"file0": f0, "file1": f1},
            params={"var_name": var_name},
        )
    r.raise_for_status()
    job = r.json()
    print(f"Job queued: {job['job_id']}")
    return job["job_id"]


def poll_until_done(job_id: str) -> dict:
    while True:
        r = requests.get(f"{BASE_URL}/interpolate/full/{job_id}")
        r.raise_for_status()
        status = r.json()
        shape_info = f" shape={status['shape']}" if "shape" in status else ""
        print(f"  status: {status['status']}{shape_info}")

        if status["status"] == "completed":
            return status
        if status["status"] == "failed":
            print(f"  ERROR: {status.get('error')}")
            sys.exit(1)

        time.sleep(POLL_INTERVAL_S)


def download_result(job_id: str, out_path: str = "full_result.nc") -> str:
    print(f"Downloading result ...")
    r = requests.get(f"{BASE_URL}/interpolate/full/{job_id}/download")
    r.raise_for_status()
    with open(out_path, "wb") as f:
        f.write(r.content)
    print(f"Saved: {out_path}")
    return out_path


def load_original_frame(path: str, var_name: str) -> np.ndarray:
    """Loads T0 or T1 raw frame for side-by-side comparison, masking fill values as NaN."""
    with nc.Dataset(path, "r") as ds:
        arr = ds.variables[var_name][:]
        if hasattr(arr, "filled"):
            arr = arr.filled(np.nan)
        return np.asarray(arr, dtype=np.float32)


def main():
    if len(sys.argv) < 3:
        print("Usage: python api/visualize_full.py <frame_t0.nc> <frame_t1.nc> [var_name]")
        sys.exit(1)

    file0_path = sys.argv[1]
    file1_path = sys.argv[2]
    var_name   = sys.argv[3] if len(sys.argv) > 3 else "__xarray_dataarray_variable__"

    job_id = submit_job(file0_path, file1_path, var_name)
    status = poll_until_done(job_id)
    nc_path = download_result(job_id)

    print("\nLoading frames for visualization ...")
    t0 = load_original_frame(file0_path, var_name)
    t1 = load_original_frame(file1_path, var_name)

    with nc.Dataset(nc_path, "r") as ds:
        t025 = np.asarray(ds.variables["t025"][:], dtype=np.float32)
        t050 = np.asarray(ds.variables["t050"][:], dtype=np.float32)
        t075 = np.asarray(ds.variables["t075"][:], dtype=np.float32)
        # NaN where fill_value was used -- masked arrays auto-convert on read
        if hasattr(t025, "filled"):
            t025 = t025.filled(np.nan)
        bt_min = float(ds.bt_min)
        bt_max = float(ds.bt_max)

    frames = {
        "T0 (input)":   t0,
        "T0.25 (pred)": t025,
        "T0.50 (pred)": t050,
        "T0.75 (pred)": t075,
        "T1 (input)":   t1,
    }

    # Report NaN coverage so you can confirm masking is sane (Earth disk
    # vs space background -- typically ~20-25% NaN per known_pitfalls)
    for name, arr in frames.items():
        nan_pct = 100.0 * np.isnan(arr).sum() / arr.size
        valid = arr[~np.isnan(arr)]
        rng = f"{valid.min():.1f}-{valid.max():.1f}K" if valid.size else "n/a"
        print(f"  {name}: {nan_pct:.1f}% NaN, valid range {rng}")

    vmin, vmax = bt_min, min(bt_max, 320.0)

    # Use masked arrays so NaN renders as transparent, not black or white
    cmap = matplotlib.colormaps["gray_r"].copy()
    cmap.set_bad(color="#3a6ea5")  # NaN (space/no-data) shown as blue, not confused with cloud

    fig, axes = plt.subplots(1, 5, figsize=(24, 5.2))
    im = None
    for ax, (title, arr) in zip(axes, frames.items()):
        masked = np.ma.masked_invalid(arr)
        im = ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=12)
        ax.axis("off")

    fig.colorbar(im, ax=axes, orientation="horizontal", fraction=0.035, pad=0.06,
                 label="Brightness temperature (K)  |  blue = no data / outside Earth disk")
    fig.suptitle(
        f"Full-globe interpolation  |  shape={t0.shape}  |  job={job_id}",
        fontsize=12, y=1.02
    )

    out_png = "full_interpolation_result.png"
    plt.savefig(out_png, dpi=110, bbox_inches="tight")
    print(f"\nSaved visualization to: {out_png}")
    print(f"Raw .nc file with t025/t050/t075 variables: {nc_path}")


if __name__ == "__main__":
    main()