# Dockerfile — INSAT-3DS / Himawari-8 Frame Interpolation API (inference only)
# ================================================================================
# Builds a container with ONLY what's needed to run api/main.py:
#   - src/model.py (the RIFE wrapper)
#   - api/ (the FastAPI app)
#   - ECCV2022-RIFE/ (the RIFE IFNet source, imported at runtime via importlib)
#   - checkpoints/best_model.pth (your fine-tuned weights)
#
# Deliberately does NOT copy: data/, logs/, src/dataset.py, src/train.py,
# src/metrics.py, scripts/preprocess.py, or any training-only files.
#
# Build:
#   docker build -t ps12-api .
#
# Run (GPU — requires nvidia-container-toolkit on host):
#   docker run --gpus all -p 8000:8000 ps12-api
#
# Run (CPU fallback — just omit --gpus all, main.py auto-detects via
# torch.cuda.is_available() and falls back to CPU automatically):
#   docker run -p 8000:8000 ps12-api

FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

WORKDIR /app

# System dependency for netCDF4 (libnetcdf) — the pip wheel bundles its own
# binary so apt install is usually unneeded, but kept as a safety net for
# platforms where the wheel doesn't ship a prebuilt binary.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libnetcdf-dev \
    && rm -rf /var/lib/apt/lists/*

# Install only inference dependencies — NOT requirements.txt (which would
# pull in xarray, h5py, matplotlib, tqdm, scikit-image, etc. for training)
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy ONLY what inference needs — explicit allowlist, not `COPY . .`
COPY src/model.py            src/model.py
COPY api/main.py             api/main.py
COPY api/__init__.py         api/__init__.py
# src/__init__.py is required for `from src.model import ...` to work.
# Created here rather than copied, in case it's missing from your repo —
# harmless if you also have one locally, this just guarantees it exists.
RUN touch src/__init__.py
# ECCV2022-RIFE/model/ contains IFNet.py, warplayer.py, refine.py — these use
# absolute imports like "from model.warplayer import warp", so ECCV2022-RIFE/
# (the parent of model/) is added to sys.path at runtime by src/model.py.
# No __init__.py needed — it's added via sys.path.insert, not import as a package.
COPY ECCV2022-RIFE/model/    ECCV2022-RIFE/model/
COPY checkpoints/best_model.pth checkpoints/best_model.pth

# api/jobs/ is created at runtime by main.py — pre-create with correct
# permissions so the non-root user (set below) can write to it
RUN mkdir -p api/jobs && chmod 777 api/jobs

# Run as non-root for basic container hygiene
RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Healthcheck mirrors the /health endpoint so `docker ps` shows real status
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()" || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]