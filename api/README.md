# PS12 Frame Interpolation API

For frontend / presentation team — this exposes the trained RIFE model over HTTP so you can build a UI without touching PyTorch or model internals.

## Run it (whoever has the trained model + GPU)

```bash
cd D:\machine_learning
pip install fastapi uvicorn python-multipart --break-system-packages
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Then open `http://localhost:8000/docs` — interactive Swagger UI, you can upload files and try every endpoint directly in the browser without writing any code.

To let teammates on the same WiFi hit it from their own laptop: find your IP with `ipconfig` (Windows), then they use `http://<your-ip>:8000` instead of `localhost`.

## Endpoints

### `GET /health`
Check if the model loaded correctly and which device (cuda/cpu) it's running on.

### `POST /inspect`
Upload any `.nc` file — returns its variable names, shapes, and metadata. **Use this first on any new INSAT-3DS file** to find the correct variable name before calling interpolate endpoints.

### `POST /interpolate/demo`
Fast, single-patch interpolation. Returns JSON arrays directly (no file download) — ideal for the frontend to fetch and render with a colormap (e.g. Chart.js, D3, or a `<canvas>` heatmap).

Params:
- `file0`, `file1` — two `.nc` files (T0 and T1 frames)
- `var_name` — the NetCDF variable name (check via `/inspect` first; defaults to `"TIR"`)
- `crop_row`, `crop_col` — top-left corner of the 256×256 patch to interpolate (defaults to a mid-image region)

Returns in ~1-2 seconds on GPU:
```json
{
  "elapsed_seconds": 1.2,
  "device": "cuda",
  "patch_location": {"row": 2000, "col": 2000, "size": 256},
  "frame_t0": [[...]],
  "frame_t1": [[...]],
  "frame_t025": [[...]],
  "frame_t050": [[...]],
  "frame_t075": [[...]],
  "units": "kelvin"
}
```
Each `frame_*` is a 256×256 array of brightness temperature in Kelvin. Frontend can normalize to 0-255 or use a colormap library to render as an image.

### `POST /interpolate/full`
Full-image tiled inference. Slow (minutes), so it's async:

1. POST returns `{"job_id": "...", "poll_url": "...", "download_url": "..."}`
2. GET the `poll_url` repeatedly until `status: "completed"`
3. GET the `download_url` to fetch the resulting `.nc` file with `t025`, `t050`, `t075` variables

## What the frontend should build first

Since INSAT-3DS sample data isn't available yet, **build against `/interpolate/demo` using the Himawari-8 test files** (`data/preprocessed/test/*.npz` — note these are `.npz` not `.nc`, see below). Once INSAT-3DS data arrives, switching is just a different `var_name` and file — the API contract doesn't change.

For now, sample `.nc` test files are at: `<ask the model trainer for 2 raw .nc files from the Himawari test set, e.g. data/raw/.../HS_H08_20220720_..._B13_FLDK.nc>`

## Known limitations (current scope)

- `var_name="TIR"` is a placeholder default — actual INSAT-3DS variable name is unverified, use `/inspect` first
- CORS is wide open (`allow_origins=["*"]`) — fine for local team dev, must be restricted before any public deployment
- Jobs are stored as local JSON files in `api/jobs/` — fine for one laptop, won't survive a server restart or scale to multiple users
- No deployment to AWS yet — this README covers local-only for now