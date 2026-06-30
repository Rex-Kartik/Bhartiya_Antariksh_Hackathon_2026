"""
api/test_api.py - Quick smoke test for the running API
=========================================================
Run this AFTER starting `uvicorn api.main:app` to confirm everything
works before handing the API off to the frontend team.

Usage:
    python api/test_api.py path/to/frame_t0.nc path/to/frame_t1.nc
"""

import sys
import time
import requests

BASE_URL = "http://localhost:8000"


def main():
    if len(sys.argv) != 3:
        print("Usage: python api/test_api.py <frame_t0.nc> <frame_t1.nc>")
        sys.exit(1)

    file0_path, file1_path = sys.argv[1], sys.argv[2]

    print("1. Checking /health ...")
    r = requests.get(f"{BASE_URL}/health")
    r.raise_for_status()
    health = r.json()
    print(f"   status={health['status']}  device={health['device']}")
    if health["status"] != "ok":
        print(f"   ERROR: {health.get('model_load_error')}")
        sys.exit(1)

    print("\n2. Inspecting frame_t0 variables ...")
    with open(file0_path, "rb") as f:
        r = requests.post(f"{BASE_URL}/inspect", files={"file": f})
    r.raise_for_status()
    info = r.json()
    print(f"   Variables: {list(info['variables'].keys())}")
    for name, meta in info["variables"].items():
        print(f"     {name}: shape={meta['shape']} dtype={meta['dtype']}")

    var_name = input("\n   Enter the TIR variable name to use (or press Enter for 'TIR'): ").strip() or "TIR"

    print(f"\n3. Running /interpolate/demo with var_name='{var_name}' ...")
    t_start = time.time()
    with open(file0_path, "rb") as f0, open(file1_path, "rb") as f1:
        r = requests.post(
            f"{BASE_URL}/interpolate/demo",
            files={"file0": f0, "file1": f1},
            params={"var_name": var_name},
        )
    elapsed = time.time() - t_start

    if r.status_code != 200:
        print(f"   FAILED: {r.status_code} {r.json()}")
        sys.exit(1)

    result = r.json()
    print(f"   Success in {elapsed:.2f}s (server reported {result['elapsed_seconds']}s)")
    print(f"   Device: {result['device']}")
    print(f"   Patch: {result['patch_location']}")
    t025 = result["frame_t025"]
    print(f"   T0.25 sample value [0][0]: {t025[0][0]:.2f} K")
    print(f"   T0.25 array shape: {len(t025)}x{len(t025[0])}")

    print("\nAll checks passed. API is ready for frontend integration.")


if __name__ == "__main__":
    main()