"""
api/main.py - INSAT-3DS / Himawari-8 Frame Interpolation API
===============================================================
Bhartiya Antariksh Hackathon 2026 - PS12

Two modes:
  POST /interpolate/demo  - single 256x256 patch, fast (~1-2s), for frontend dev
  POST /interpolate/full  - full-globe tiled inference, slow, async job pattern

Run locally:
    cd D:\\machine_learning
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

Team frontend devs hit: http://<your-laptop-ip>:8000/docs for interactive API docs.
"""

from __future__ import annotations

import gc
import json
import os
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import netCDF4 as nc
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.model import build_model, ModelConfig, generate_three_frames

# Setup
app = FastAPI(
    title="PS12 Frame Interpolation API",
    description="Generate T0.25, T0.50, T0.75 intermediate satellite TIR frames",
    version="0.1.0",
)

# NOTE: "*" is fine for local team development. Restrict this to your
# actual frontend origin (e.g. http://localhost:3000) before any
# public-facing deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
JOBS_DIR     = PROJECT_ROOT / "api" / "jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)

# Model loading - GPU if available, else CPU. Uses your fine-tuned weights.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT = str(PROJECT_ROOT / "checkpoints" / "best_model.pth")

cfg = ModelConfig(checkpoint_path=CHECKPOINT, device=DEVICE)
model: Optional[torch.nn.Module] = None
model_load_error: Optional[str] = None

try:
    model = build_model(cfg)
    model.eval()
    print(f"  Model loaded on {DEVICE} from {CHECKPOINT}")
except Exception as e:
    model_load_error = str(e)
    print(f"  WARNING: model failed to load: {e}")

# Normalization constants - MUST match scripts/preprocess.py exactly
BT_MIN = 180.0
BT_MAX = 320.0
BT_SCALE = 1.0 / (BT_MAX - BT_MIN)
PATCH_SIZE = 256

# Himawari-8 full-disk crop box (from decisions_made). Do NOT reuse for
# INSAT-3DS without first checking its actual array shape via /inspect.
HIMAWARI_CROP = {"row_min": 42, "row_max": 5457, "col_min": 33, "col_max": 5466}


def normalize_bt(arr: np.ndarray) -> np.ndarray:
    """Normalize brightness temperature to [0,1]. NaN -> 0."""
    arr = arr.astype(np.float32, copy=True)
    arr -= BT_MIN
    arr *= BT_SCALE
    np.clip(arr, 0.0, 1.0, out=arr)
    np.nan_to_num(arr, nan=0.0, copy=False)
    return arr


def denormalize_bt(arr: np.ndarray) -> np.ndarray:
    return (arr * (BT_MAX - BT_MIN)) + BT_MIN


def pad_to_patch(arr: np.ndarray, patch_size: int) -> Tuple[np.ndarray, int, int]:
    """
    Pads a 2D array on bottom/right with zeros so both dims are multiples
    of patch_size. Returns (padded, orig_h, orig_w) for cropping back later.
    """
    h, w = arr.shape
    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size
    if pad_h == 0 and pad_w == 0:
        return arr, h, w
    padded = np.zeros((h + pad_h, w + pad_w), dtype=arr.dtype)
    padded[:h, :w] = arr
    return padded, h, w


def run_patch_inference(p0: np.ndarray, p1: np.ndarray) -> Dict[str, np.ndarray]:
    """Runs the model on a single (patch_size, patch_size) normalized pair."""
    t0 = torch.from_numpy(p0).unsqueeze(0).unsqueeze(0).to(DEVICE)
    t1 = torch.from_numpy(p1).unsqueeze(0).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        preds = generate_three_frames(model, t0, t1)
    out = {
        "t025": preds["t025"][0, 0].cpu().numpy(),
        "t050": preds["t050"][0, 0].cpu().numpy(),
        "t075": preds["t075"][0, 0].cpu().numpy(),
    }
    del t0, t1, preds
    return out


def run_tiled_inference(
    crop0: np.ndarray,
    crop1: np.ndarray,
    patch_size: int = PATCH_SIZE,
) -> Dict[str, np.ndarray]:
    """
    Pads crop to a patch_size multiple, runs inference tile by tile,
    crops back to original size. Skips tiles that are entirely fill
    value (0.0 after normalization) to save compute over ocean/space.
    """
    padded0, orig_h, orig_w = pad_to_patch(crop0, patch_size)
    padded1, _, _ = pad_to_patch(crop1, patch_size)
    H, W = padded0.shape

    out_025 = np.zeros((H, W), dtype=np.float32)
    out_050 = np.zeros((H, W), dtype=np.float32)
    out_075 = np.zeros((H, W), dtype=np.float32)

    n_tiles = (H // patch_size) * (W // patch_size)
    n_processed = 0
    t_start = time.time()

    for y in range(0, H, patch_size):
        for x in range(0, W, patch_size):
            p0 = padded0[y:y + patch_size, x:x + patch_size]
            p1 = padded1[y:y + patch_size, x:x + patch_size]

            if np.all(p0 == 0.0) or np.all(p1 == 0.0):
                continue   # skip fill-value tiles - speeds up ocean/space regions

            preds = run_patch_inference(p0, p1)
            out_025[y:y + patch_size, x:x + patch_size] = preds["t025"]
            out_050[y:y + patch_size, x:x + patch_size] = preds["t050"]
            out_075[y:y + patch_size, x:x + patch_size] = preds["t075"]
            n_processed += 1

    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    elapsed = time.time() - t_start
    print(f"  Tiled inference: {n_processed}/{n_tiles} tiles in {elapsed:.1f}s "
          f"({elapsed/max(n_processed,1):.2f}s/tile)")

    return {
        "t025": out_025[:orig_h, :orig_w],
        "t050": out_050[:orig_h, :orig_w],
        "t075": out_075[:orig_h, :orig_w],
    }


def read_nc_variable(path: str, var_name: str) -> np.ndarray:
    """Reads a 2D variable from a NetCDF file, raising a clear error if missing."""
    with nc.Dataset(path, "r") as ds:
        if var_name not in ds.variables:
            available = list(ds.variables.keys())
            raise ValueError(
                f"Variable '{var_name}' not found. Available variables: {available}"
            )
        arr = ds.variables[var_name][:]
        if hasattr(arr, "filled"):
            arr = arr.filled(np.nan)
        return np.asarray(arr, dtype=np.float32)


def write_nc_output(
    out_path: str,
    pred_025: np.ndarray,
    pred_050: np.ndarray,
    pred_075: np.ndarray,
) -> None:
    with nc.Dataset(out_path, "w", format="NETCDF4") as ds:
        ds.createDimension("y", pred_025.shape[0])
        ds.createDimension("x", pred_025.shape[1])

        for name, arr in [("t025", pred_025), ("t050", pred_050), ("t075", pred_075)]:
            v = ds.createVariable(name, "f4", ("y", "x"), fill_value=np.nan, zlib=True)
            v[:] = arr
            v.units = "K"
            v.long_name = f"Interpolated brightness temperature at {name}"

        ds.bt_min = BT_MIN
        ds.bt_max = BT_MAX
        ds.source = "PS12 RIFE interpolation model"


def cleanup_file(path: str) -> None:
    if os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass


def job_path(job_id: str) -> Path:
    return JOBS_DIR / f"{job_id}.json"


def set_job_status(job_id: str, status: str, **extra) -> None:
    data = {"job_id": job_id, "status": status, "updated": time.time(), **extra}
    with open(job_path(job_id), "w") as f:
        json.dump(data, f)


def get_job_status(job_id: str) -> Optional[dict]:
    p = job_path(job_id)
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


@app.get("/health")
def health():
    return {
        "status": "ok" if model is not None else "model_unavailable",
        "device": DEVICE,
        "checkpoint": CHECKPOINT,
        "checkpoint_exists": os.path.exists(CHECKPOINT),
        "model_load_error": model_load_error,
    }


@app.post("/inspect")
async def inspect_nc(file: UploadFile = File(...)):
    """
    Upload any .nc file to see its variable names, shapes, and dtypes.
    Use this BEFORE running /interpolate on INSAT-3DS data to find the
    correct variable name - do not guess it.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".nc") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        with nc.Dataset(tmp_path, "r") as ds:
            info = {
                "variables": {
                    name: {
                        "shape": list(var.shape),
                        "dtype": str(var.dtype),
                        "attrs": {k: str(getattr(var, k)) for k in var.ncattrs()},
                    }
                    for name, var in ds.variables.items()
                },
                "dimensions": {k: len(v) for k, v in ds.dimensions.items()},
                "global_attrs": {k: str(getattr(ds, k)) for k in ds.ncattrs()},
            }
        return JSONResponse(info)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read NetCDF: {e}")
    finally:
        cleanup_file(tmp_path)


@app.post("/interpolate/demo")
async def interpolate_demo(
    file0: UploadFile = File(...),
    file1: UploadFile = File(...),
    var_name: str = "__xarray_dataarray_variable__",
    crop_row: int = 2000,
    crop_col: int = 2000,
):
    """
    Fast demo endpoint - extracts ONE 256x256 patch from a fixed location
    and interpolates it. Returns JSON with raw arrays (not a .nc file) so
    the frontend can render it directly with a colormap.

    Runs in ~1-2s on GPU, a few seconds on CPU. Use this while building
    the frontend so you're not waiting on full-globe inference.
    """
    if model is None:
        raise HTTPException(status_code=500, detail=f"Model not loaded: {model_load_error}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".nc") as tmp0, \
         tempfile.NamedTemporaryFile(delete=False, suffix=".nc") as tmp1:
        tmp0.write(await file0.read())
        tmp1.write(await file1.read())
        t0_path, t1_path = tmp0.name, tmp1.name

    try:
        arr0 = read_nc_variable(t0_path, var_name)
        arr1 = read_nc_variable(t1_path, var_name)

        if arr0.shape != arr1.shape:
            raise HTTPException(
                status_code=400,
                detail=f"Shape mismatch: file0={arr0.shape} file1={arr1.shape}"
            )

        r0, r1 = crop_row, crop_row + PATCH_SIZE
        c0, c1 = crop_col, crop_col + PATCH_SIZE

        if r1 > arr0.shape[0] or c1 > arr0.shape[1]:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"crop_row/crop_col out of bounds for array shape {arr0.shape}. "
                    f"Requested patch [{r0}:{r1}, {c0}:{c1}]"
                )
            )

        patch0 = normalize_bt(arr0[r0:r1, c0:c1])
        patch1 = normalize_bt(arr1[r0:r1, c0:c1])

        t_start = time.time()
        preds = run_patch_inference(patch0, patch1)
        elapsed = time.time() - t_start

        return JSONResponse({
            "elapsed_seconds": round(elapsed, 3),
            "device": DEVICE,
            "patch_location": {"row": crop_row, "col": crop_col, "size": PATCH_SIZE},
            "frame_t0":   denormalize_bt(patch0).tolist(),
            "frame_t1":   denormalize_bt(patch1).tolist(),
            "frame_t025": denormalize_bt(preds["t025"]).tolist(),
            "frame_t050": denormalize_bt(preds["t050"]).tolist(),
            "frame_t075": denormalize_bt(preds["t075"]).tolist(),
            "units": "kelvin",
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Processing error: {e}")
    finally:
        cleanup_file(t0_path)
        cleanup_file(t1_path)


def _run_full_job(job_id: str, t0_path: str, t1_path: str, var_name: str) -> None:
    """Background worker - runs full tiled inference and writes output .nc."""
    try:
        set_job_status(job_id, "reading_input")
        arr0 = read_nc_variable(t0_path, var_name)
        arr1 = read_nc_variable(t1_path, var_name)

        if arr0.shape != arr1.shape:
            set_job_status(job_id, "failed", error=f"Shape mismatch: {arr0.shape} vs {arr1.shape}")
            return

        set_job_status(job_id, "normalizing", shape=list(arr0.shape))
        nan_mask = np.isnan(arr0) | np.isnan(arr1)
        norm0 = normalize_bt(arr0)
        norm1 = normalize_bt(arr1)

        set_job_status(job_id, "running_inference", shape=list(arr0.shape))
        preds = run_tiled_inference(norm0, norm1, patch_size=PATCH_SIZE)

        set_job_status(job_id, "denormalizing")
        pred_025 = denormalize_bt(preds["t025"])
        pred_050 = denormalize_bt(preds["t050"])
        pred_075 = denormalize_bt(preds["t075"])

        pred_025[nan_mask] = np.nan
        pred_050[nan_mask] = np.nan
        pred_075[nan_mask] = np.nan

        out_path = str(JOBS_DIR / f"{job_id}_output.nc")
        set_job_status(job_id, "writing_output")
        write_nc_output(out_path, pred_025, pred_050, pred_075)

        set_job_status(job_id, "completed", output_path=out_path)

    except Exception as e:
        set_job_status(job_id, "failed", error=str(e))
    finally:
        cleanup_file(t0_path)
        cleanup_file(t1_path)


@app.post("/interpolate/full")
async def interpolate_full(
    background_tasks: BackgroundTasks,
    file0: UploadFile = File(...),
    file1: UploadFile = File(...),
    var_name: str = "__xarray_dataarray_variable__",
):
    """
    Full tiled inference over the entire uploaded image. Runs as a
    background job since it can take minutes on CPU. Returns a job_id -
    poll GET /interpolate/full/{job_id} for status, then download from
    GET /interpolate/full/{job_id}/download when complete.
    """
    if model is None:
        raise HTTPException(status_code=500, detail=f"Model not loaded: {model_load_error}")

    job_id = str(uuid.uuid4())[:8]

    with tempfile.NamedTemporaryFile(delete=False, suffix=".nc") as tmp0, \
         tempfile.NamedTemporaryFile(delete=False, suffix=".nc") as tmp1:
        tmp0.write(await file0.read())
        tmp1.write(await file1.read())
        t0_path, t1_path = tmp0.name, tmp1.name

    set_job_status(job_id, "queued")
    background_tasks.add_task(_run_full_job, job_id, t0_path, t1_path, var_name)

    return JSONResponse({
        "job_id": job_id,
        "status": "queued",
        "poll_url": f"/interpolate/full/{job_id}",
        "download_url": f"/interpolate/full/{job_id}/download",
    })


@app.get("/interpolate/full/{job_id}")
def get_job(job_id: str):
    status = get_job_status(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return JSONResponse(status)


@app.get("/interpolate/full/{job_id}/download")
def download_job(job_id: str, background_tasks: BackgroundTasks):
    status = get_job_status(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if status["status"] != "completed":
        raise HTTPException(status_code=409, detail=f"Job not ready, status={status['status']}")

    out_path = status["output_path"]
    if not os.path.exists(out_path):
        raise HTTPException(status_code=410, detail="Output file no longer exists")

    background_tasks.add_task(cleanup_file, out_path)
    background_tasks.add_task(cleanup_file, str(job_path(job_id)))
    return FileResponse(out_path, media_type="application/x-netcdf",
                        filename=f"interpolated_{job_id}.nc")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)