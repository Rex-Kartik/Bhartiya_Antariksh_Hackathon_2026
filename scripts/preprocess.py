"""
Himawari-8 Preprocessing — Local Day‑by‑Day Staging from Google Drive
======================================================================
Exactly replicates the Colab v4.1 pipeline:
  - Copies only the current day's .nc files from Drive to staging.
  - Processes that day while, every time a file is deleted,
    ONE file from the next day is immediately downloaded.
  - Shows separate progress bars for copying, processing, and
    the number of next‑day files already cached.
  - Resumes safely: skips already completed days and sequences.

Before running:
  - Install Google Drive for desktop (Stream mode).
  - Adjust DRIVE_MOUNT_PATH and LOCAL_PROJECT_ROOT below.
"""

import os, re, shutil, json, gc, time, threading, warnings
import numpy as np
import pickle
from datetime import datetime, timedelta
from collections import defaultdict, OrderedDict
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
warnings.filterwarnings('ignore')

try:
    import h5py
    _HAS_H5PY = True
except ImportError:
    _HAS_H5PY = False

try:
    import xarray as xr
except ImportError:
    xr = None

# =====================================================================
#  USER PATHS – ADAPT THESE TWO
# =====================================================================
DRIVE_MOUNT_PATH    = Path(r"G:\My Drive\Colab Notebooks\Data\Himawari_full_disk")
LOCAL_PROJECT_ROOT  = Path(__file__).resolve().parent.parent   # BAH2026_PS12/

# Derived paths (do not change)
RAW_SOURCE_DIR = DRIVE_MOUNT_PATH
LOCAL_STAGE_DIR = LOCAL_PROJECT_ROOT / 'data' / 'stage'
OUTPUT_DIR      = LOCAL_PROJECT_ROOT / 'data' / 'preprocessed'
PROGRESS_FILE   = OUTPUT_DIR / 'progress_manifest.json'

CONFIG = {
    'bt_min': 180.0, 'bt_max': 320.0,
    'row_min': 42,   'row_max': 5457,
    'col_min': 33,   'col_max': 5466,
    'min_file_bytes':  50_000_000,
    'max_workers':     2,
    'frame_cache_size': 12,
    'gc_every_n':      20,
    'disk_min_free_mb': 2000,
}

for d in [LOCAL_STAGE_DIR, OUTPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

_BT_SCALE = np.float32(1.0 / (CONFIG['bt_max'] - CONFIG['bt_min']))
_BT_MIN   = np.float32(CONFIG['bt_min'])

print(f"\n📁 PREPROCESSING (on‑deletion download from Google Drive)")
print(f"   Drive source : {RAW_SOURCE_DIR}")
print(f"   Local stage  : {LOCAL_STAGE_DIR}")
print(f"   Output dir   : {OUTPUT_DIR}")
print(f"   Progress     : {PROGRESS_FILE}\n")

# ───────────────────────── helpers ─────────────────────────────
def free_disk_mb(path=None):  # ← UPDATED: fixed for Windows (os.statvfs is POSIX-only)
    try:
        usage = shutil.disk_usage(str(LOCAL_PROJECT_ROOT))
        return usage.free / 1e6
    except Exception:
        return 9999.0

def disk_guard():
    if free_disk_mb() < CONFIG['disk_min_free_mb']:
        print("   ⚠️  Low disk – waiting…")
        time.sleep(30)
        gc.collect()

def safe_delete(path):
    p = Path(path)
    if p.exists():
        try:
            p.unlink()
        except Exception as e:
            print(f"   ⚠️  delete {p.name}: {e}")

def is_complete_on_drive(path):
    p = Path(path)
    if not p.exists():
        return False
    try:
        with np.load(p, allow_pickle=False) as npz:
            _ = npz.files
        return True
    except:
        return False

# ───────────────────── progress manifest ───────────────────────
def load_progress():
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            data = json.load(f)
        for s in ("train", "val", "test"):
            data.setdefault("completed_dates", {}).setdefault(s, [])
        return data
    return {"completed_dates": {"train": [], "val": [], "test": []}}

def save_progress(progress):
    tmp = str(PROGRESS_FILE) + ".tmp"
    with open(tmp, 'w') as f:
        json.dump(progress, f, indent=2)
    os.replace(tmp, str(PROGRESS_FILE))

def is_date_done(progress, split, date_str):
    return date_str in progress["completed_dates"].get(split, [])

def mark_date_done(progress, split, date_str):
    lst = progress["completed_dates"].setdefault(split, [])
    if date_str not in lst:
        lst.append(date_str)
    save_progress(progress)
    print(f"   📌 {split}/{date_str} marked complete.")

# ───────────────────── scan Drive ──────────────────────────────
print("📂 Scanning .nc files on Google Drive...")
ts_pat = re.compile(r'HS_H08_(\d{8})_(\d{4})_B13_FLDK\.nc')
date_to_files = defaultdict(list)

for dirpath, _, filenames in os.walk(RAW_SOURCE_DIR):
    for fname in filenames:
        if not fname.endswith('.nc'):
            continue
        full = os.path.join(dirpath, fname)
        try:
            if os.path.getsize(full) < CONFIG['min_file_bytes']:
                continue
        except OSError:
            continue
        m = ts_pat.match(fname)
        if m:
            date_str = m.group(1)
            ts_key   = f"{date_str}_{m.group(2)}"
            date_to_files[date_str].append((ts_key, full))

for d in date_to_files:
    date_to_files[d].sort(key=lambda x: x[0])

all_dates = sorted(date_to_files.keys())
print(f"Found {len(all_dates)} dates, {sum(len(v) for v in date_to_files.values())} valid files\n")

# ───────────────────── startup cache maps ──────────────────────
# Built ONCE at launch. Used by process_day to skip copies and
# skip sequence processing without touching disk again.

print("🗂️  Mapping already-staged .nc files...")
# fname → local staged path, for every file already in stage that passes size check
_STAGED_MAP: dict[str, Path] = {}
for _p in LOCAL_STAGE_DIR.glob("*.nc"):
    try:
        if _p.stat().st_size >= CONFIG['min_file_bytes']:
            _STAGED_MAP[_p.name] = _p
    except OSError:
        pass
print(f"   {len(_STAGED_MAP)} .nc files already in stage/\n")

print("🗂️  Mapping already-preprocessed .npz files...")
# stem (e.g. '20220701_0023') → Path, for every valid .npz in preprocessed/
_PREPROCESSED_MAP: dict[str, Path] = {}
for _split in ("train", "val", "test"):
    _sd = OUTPUT_DIR / _split
    if not _sd.exists():
        continue
    for _p in _sd.glob("*.npz"):
        try:
            with np.load(str(_p), allow_pickle=False) as _npz:
                if 'frames' in _npz.files:
                    _PREPROCESSED_MAP[_p.stem] = _p
        except Exception:
            pass   # corrupted — not counted
_npz_by_split = {s: sum(1 for p in _PREPROCESSED_MAP.values()
                         if p.parent.name == s)
                 for s in ("train", "val", "test")}
print(f"   {len(_PREPROCESSED_MAP)} valid .npz files already preprocessed "
      f"(train={_npz_by_split['train']} val={_npz_by_split['val']} test={_npz_by_split['test']})\n")

# ───────────────────── temporal split ──────────────────────────
n = len(all_dates)
train_dates = all_dates[:int(n*0.70)]
val_dates   = all_dates[int(n*0.70):int(n*0.85)]
test_dates  = all_dates[int(n*0.85):]

# ───────────────────── sliding window ──────────────────────────
def generate_sequences(ts_path_list):
    ts_to_path = dict(ts_path_list)
    timestamps = [t for t, _ in ts_path_list]
    seqs = []
    for i in range(len(timestamps)-4):
        try:
            start_dt  = datetime.strptime(timestamps[i], "%Y%m%d_%H%M")
            needed_ts = [(start_dt + timedelta(minutes=off)).strftime("%Y%m%d_%H%M")
                         for off in [0,10,20,30,40]]
        except: continue
        if all(ts in ts_to_path for ts in needed_ts):
            seqs.append([ts_to_path[ts] for ts in needed_ts])
    return seqs

# ───────────────────── fast frame loader ───────────────────────
def load_frame_fast(local_path):
    r_min, r_max = CONFIG['row_min'], CONFIG['row_max']
    c_min, c_max = CONFIG['col_min'], CONFIG['col_max']
    if _HAS_H5PY:
        try:
            with h5py.File(local_path, 'r') as f:
                varname = next((k for k in f.keys() if f[k].ndim==2 and f[k].shape[0]>1000), None)
                if varname is None: raise ValueError("No 2D variable")
                arr = f[varname][r_min:r_max+1, c_min:c_max+1].astype(np.float32)
        except:
            arr = _load_frame_xarray(local_path)
    else:
        arr = _load_frame_xarray(local_path)
    if arr is None: return None
    arr -= _BT_MIN
    arr *= _BT_SCALE
    np.clip(arr, 0.0, 1.0, out=arr)
    np.nan_to_num(arr, copy=False, nan=0.0)
    return arr

def _load_frame_xarray(local_path):
    if xr is None: return None
    try:
        engine = 'h5netcdf' if _HAS_H5PY else 'scipy'
        ds = xr.open_dataset(local_path, engine=engine, mask_and_scale=False)
        varname = next((v for v in ds.data_vars if ds[v].ndim==2), list(ds.data_vars)[0])
        arr = ds[varname].values[CONFIG['row_min']:CONFIG['row_max']+1,
                                  CONFIG['col_min']:CONFIG['col_max']+1].astype(np.float32)
        ds.close()
        return arr
    except Exception as e:
        print(f"   ⚠️  xarray load failed {Path(local_path).name}: {e}")
        return None

# ───────────────────── LRU cache ───────────────────────────────
class FrameCache:
    def __init__(self, maxsize=12):
        self.maxsize = maxsize
        self.cache = OrderedDict()
        self.lock  = threading.Lock()
    def get(self, path):
        with self.lock:
            if path in self.cache:
                self.cache.move_to_end(path)
                return self.cache[path]
        frame = load_frame_fast(path)
        if frame is not None:
            with self.lock:
                self.cache[path] = frame
                while len(self.cache) > self.maxsize:
                    self.cache.popitem(last=False)
        return frame
    def evict(self, path):
        with self.lock:
            self.cache.pop(path, None)
    def clear(self):
        with self.lock:
            self.cache.clear()

# ───────────────────── write .npz ──────────────────────────────
def write_npz(stacked, final_path, name):
    final_path = Path(final_path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = final_path.parent / (name.replace(".npz", "_tmp.npz"))
    try:
        np.savez_compressed(str(tmp), frames=stacked)
        del stacked   # ← move deletion HERE, inside the worker thread
        if not tmp.exists():
            raise FileNotFoundError(f"savez_compressed produced no file at {tmp}")
        if tmp.stat().st_size < 100:
            raise ValueError(f"tmp file suspiciously small ({tmp.stat().st_size} bytes)")
        os.replace(str(tmp), str(final_path))
        _PREPROCESSED_MAP[final_path.stem] = final_path
        return True
    except Exception as e:
        print(f"   ⚠️  write {name}: {e}")
        if tmp.exists():
            safe_delete(tmp)
        return False

# =====================================================================
#  ON‑DELETION DOWNLOADER
# =====================================================================
class OnDeletionDownloader:
    def __init__(self, next_day_files, source_dir, stage_dir):
        self._all = list(dict.fromkeys(p for _, p in next_day_files))
        self._source_dir = source_dir
        self._stage_dir  = stage_dir
        self._idx = 0
        self.downloaded = 0
        self.total = len(self._all)

    def download_one(self):
        while self._idx < len(self._all):
            drive_path = self._all[self._idx]
            self._idx += 1
            fname  = Path(drive_path).name
            # ← UPDATED: check startup map first
            if fname in _STAGED_MAP:
                self.downloaded += 1
                continue
            staged = self._stage_dir / fname
            if staged.exists() and staged.stat().st_size >= CONFIG['min_file_bytes']:
                _STAGED_MAP[fname] = staged   # add to map retroactively
                self.downloaded += 1
                continue
            disk_guard()
            try:
                shutil.copy2(drive_path, staged)
                _STAGED_MAP[fname] = staged
                self.downloaded += 1
                return True
            except Exception as e:
                print(f"   ⚠️  On‑deletion copy {fname}: {e}")
                return False
        return False

# =====================================================================
# PROCESS ONE DAY
# =====================================================================
def process_day(date_str, ts_path_list, split_name, executor, progress, next_downloader=None):
    split_dir = OUTPUT_DIR / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    sequences = generate_sequences(ts_path_list)
    n_total   = len(sequences)

    # Use startup map — avoids re-scanning disk for every sequence  ← UPDATED
    done_set = set()
    for seq_idx in range(n_total):
        stem = f"{date_str}_{seq_idx:04d}"
        if stem in _PREPROCESSED_MAP:
            done_set.add(str(_PREPROCESSED_MAP[stem]))

    n_already_done = len(done_set)
    if n_already_done == n_total:
        print(f"  ⏭  {date_str}: all {n_total} already done.")
        for _, src_path in ts_path_list:
            staged = LOCAL_STAGE_DIR / Path(src_path).name
            safe_delete(staged)
        return 0, n_total, n_total, True

    print(f"  📅 {date_str}: {n_total - n_already_done} to process ({n_already_done} already done)")

    unique_sources = list(dict.fromkeys(p for _, p in ts_path_list))
    drive_to_local = {}
    n_already_staged = sum(1 for p in unique_sources if Path(p).name in _STAGED_MAP)
    missing = [p for p in unique_sources if Path(p).name not in _STAGED_MAP]

    # Fast path: resolve already-staged files instantly (no tqdm needed)
    for drive_path in unique_sources:
        fname = Path(drive_path).name
        if fname in _STAGED_MAP:
            drive_to_local[drive_path] = str(_STAGED_MAP[fname])

    print(f"     ↩️  {n_already_staged}/{len(unique_sources)} already staged — skipped copy")

    # Only copy the missing files (usually 0 or 1 after an interruption)
    if missing:
        print(f"     📥 Copying {len(missing)} missing file(s) from Drive...")
        copy_pbar = tqdm(missing, desc="     Copying missing", unit="file", leave=False)
        for drive_path in copy_pbar:
            fname  = Path(drive_path).name
            staged = LOCAL_STAGE_DIR / fname
            if staged.exists() and staged.stat().st_size >= CONFIG['min_file_bytes']:
                # appeared on disk since startup scan — accept it
                _STAGED_MAP[fname] = staged
                drive_to_local[drive_path] = str(staged)
                continue
            disk_guard()
            # ← threaded copy with timeout so a hung Drive stream doesn't block forever
            copy_result = [None]
            copy_exc    = [None]
            def _do_copy(src=drive_path, dst=staged, out=copy_result, exc=copy_exc):
                try:
                    shutil.copy2(src, dst)
                    out[0] = dst
                except Exception as e:
                    exc[0] = e
            t = threading.Thread(target=_do_copy, daemon=True)
            t.start()
            t.join(timeout=120)   # 2-minute timeout per file
            if t.is_alive():
                print(f"     ⚠️  Copy timeout for {fname} — skipping (sequences needing it will be skipped)")
                drive_to_local[drive_path] = None   # sequences using this file will be skipped gracefully
            elif copy_exc[0]:
                print(f"     ⚠️  Copy error {fname}: {copy_exc[0]} — skipping")
                drive_to_local[drive_path] = None
            else:
                _STAGED_MAP[fname] = staged
                drive_to_local[drive_path] = str(staged)
            if next_downloader:
                copy_pbar.set_postfix({"next_cached": f"{next_downloader.downloaded}/{next_downloader.total}"})
        copy_pbar.close()

    local_sequences = [[drive_to_local.get(dp) for dp in seq] for seq in sequences]

    last_use = {}
    for seq_idx, seq in enumerate(sequences):
        for dp in seq:
            lp = drive_to_local.get(dp)
            if lp:
                last_use[lp] = seq_idx
    delete_after = defaultdict(set)
    for lp, idx in last_use.items():
        delete_after[idx].add(lp)

    cache = FrameCache(maxsize=CONFIG['frame_cache_size'])
    n_saved = 0
    n_skipped = n_already_done
    all_ok = True
    pending_write = None

    seq_pbar = tqdm(local_sequences, desc=f"     Processing {date_str}", unit="seq", leave=False)
    for seq_idx, local_seq in enumerate(seq_pbar):
        final_name = f"{date_str}_{seq_idx:04d}.npz"
        final_path = split_dir / final_name

        if str(final_path) in done_set:
            for lp in delete_after.get(seq_idx, set()):
                cache.evict(lp)
                safe_delete(lp)
                if next_downloader:
                    next_downloader.download_one()
            if next_downloader:
                seq_pbar.set_postfix({
                    "saved": n_saved,
                    "skip": "already",
                    "next": f"{next_downloader.downloaded}/{next_downloader.total}"
                })
            continue

        if pending_write:
            ok, _ = _wait(pending_write, f"write {seq_idx-1}")
            if not ok:
                all_ok = False
            pending_write = None

        frames = []
        ok = True
        for lp in local_seq:
            if lp is None:
                ok = False
                break
            frame = cache.get(lp)
            if frame is None:
                ok = False
                break
            frames.append(frame)
        if not ok:
            n_total -= 1
            for lp in delete_after.get(seq_idx, set()):
                cache.evict(lp)
                safe_delete(lp)
                if next_downloader:
                    next_downloader.download_one()
            del frames
            seq_pbar.set_postfix({"saved": n_saved, "error": "frame load"})
            continue

        stacked = np.stack(frames, axis=0)
        del frames
        disk_guard()
        pending_write = executor.submit(write_npz, stacked, final_path, final_name)
        n_saved += 1

        for lp in delete_after.get(seq_idx, set()):
            cache.evict(lp)
            safe_delete(lp)
            if next_downloader:
                next_downloader.download_one()

        seq_pbar.set_postfix({
            "saved": n_saved,
            "next_cached": f"{next_downloader.downloaded}/{next_downloader.total}" if next_downloader else "-"
        })

        if (seq_idx+1) % CONFIG['gc_every_n'] == 0:
            gc.collect()

    seq_pbar.close()

    if pending_write:
        ok, _ = _wait(pending_write, "final write")
        if not ok:
            all_ok = False

    cache.clear()
    gc.collect()

    for _, drive_path in ts_path_list:
        staged = LOCAL_STAGE_DIR / Path(drive_path).name
        safe_delete(staged)

    n_done = n_saved + n_skipped
    fully_done = all_ok and n_done >= n_total
    if fully_done and progress is not None:
        mark_date_done(progress, split_name, date_str)

    return n_saved, n_skipped, n_total, fully_done

def _wait(future, label, timeout=600):
    try:
        result = future.result(timeout=timeout)
        return True, result
    except Exception as e:
        print(f"   ⚠️  {label}: {e}")
        return False, None

# =====================================================================
# SPLIT ORCHESTRATOR
# =====================================================================
def process_split(dates, split_name, progress):
    total_saved = total_skipped = 0
    with ThreadPoolExecutor(max_workers=CONFIG['max_workers']) as executor:
        pbar = tqdm(total=len(dates), desc=f"Split: {split_name}", leave=True)
        next_downloader = None
        for idx, date_str in enumerate(dates):
            if is_date_done(progress, split_name, date_str):
                print(f"  ⏭  {split_name}/{date_str} fully done (manifest).")
                pbar.update(1)
                continue

            next_date = dates[idx+1] if idx+1 < len(dates) else None
            if next_date and not is_date_done(progress, split_name, next_date):
                next_downloader = OnDeletionDownloader(
                    date_to_files[next_date],
                    RAW_SOURCE_DIR,
                    LOCAL_STAGE_DIR
                )
            else:
                next_downloader = None

            print(f"\n{'─'*58}")
            print(f"  {split_name.upper()} | {idx+1}/{len(dates)}: {date_str}")
            print(f"{'─'*58}")

            t0 = time.time()
            n_s, n_k, n_t, fully = process_day(
                date_str, date_to_files[date_str], split_name, executor, progress,
                next_downloader=next_downloader
            )
            elapsed = time.time() - t0
            total_saved   += n_s
            total_skipped += n_k
            print(f"  ✅ {date_str}: saved={n_s} skipped={n_k} total={n_t} | {elapsed:.1f}s")
            pbar.update(1)

        pbar.close()
    print(f"\n🎉 {split_name.upper()}: saved={total_saved} skipped={total_skipped}")
    return total_saved, total_skipped

# =====================================================================
# MAIN
# =====================================================================
if __name__ == "__main__":
    print("=" * 58)
    print("STARTING PREPROCESSING (on‑deletion download from Google Drive)")
    print("=" * 58)

    progress = load_progress()
    print(f"Manifest: train={len(progress['completed_dates']['train'])} "
          f"val={len(progress['completed_dates']['val'])} "
          f"test={len(progress['completed_dates']['test'])} day(s) done.\n")

    process_split(train_dates, 'train', progress)
    process_split(val_dates,   'val',   progress)
    process_split(test_dates,  'test',  progress)

    metadata = {
        'splits': {'train': train_dates, 'val': val_dates, 'test': test_dates},
        'config': CONFIG,
        'note': 'frames shape (5,H,W) float32'
    }
    with open(OUTPUT_DIR / 'metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)

    print(f"\n{'='*58}")
    print("✅ ALL DONE")
    print(f"{'='*58}")