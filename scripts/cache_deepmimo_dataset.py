"""
Script to download, generate, and cache DeepMIMO v4 scenario data for wireless ML research.
- Downloads and generates the 'asu_campus_3p5' scenario if not already cached
- Exports a simplified .npz file with 'csi' and 'positions' for reuse
- Safe to rerun: will not redownload or regenerate if cache exists
"""
import sys
from pathlib import Path
import numpy as np

CACHE_PATH = Path("data/raw/deepmimo/deepmimo_export.npz")
SCENARIO = "asu_campus_3p5"
RAW_DIR = Path("data/raw/deepmimo/")


def print_shapes(data):
    for k in data:
        v = data[k]
        try:
            shape = v.shape
        except Exception:
            shape = f"type: {type(v)}"
        print(f"  {k}: {shape}")

def main():
    if CACHE_PATH.exists():
        print(f"[INFO] Cached DeepMIMO export found at {CACHE_PATH}")
        arr = np.load(CACHE_PATH)
        print("[INFO] Cached data keys:", list(arr.keys()))
        print("[INFO] Shapes:")
        for key in ["csi", "positions"]:
            if key in arr:
                print(f"  {key}: {arr[key].shape}")
            else:
                print(f"  {key}: [NOT FOUND]")
        return

    print(f"[INFO] No cache found. Preparing DeepMIMO scenario '{SCENARIO}'...")
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    import deepmimo as dm
    print("[INFO] Downloading scenario (if needed)...")
    dm.download(SCENARIO)
    print("[INFO] Generating dataset...")
    dataset = dm.generate(SCENARIO)

    print("[INFO] Inspecting dataset top-level keys:")
    if hasattr(dataset, 'keys') and callable(dataset.keys):
        keys = list(dataset.keys())
    elif isinstance(dataset, dict):
        keys = list(dataset.keys())
    else:
        try:
            keys = list(dataset)
        except Exception:
            raise RuntimeError(f"Cannot determine keys of dataset. Type: {type(dataset)}")
    print("  Keys:", keys)

    # Print type/shape for each key
    for k in keys:
        try:
            v = dataset[k]
            try:
                shape = v.shape
            except Exception:
                shape = f"type: {type(v)}"
            print(f"  {k}: {shape}")
        except Exception as e:
            print(f"  {k}: [ERROR accessing: {e}]")

    # Try to extract CSI and positions
    csi = None
    positions = None
    csi_keys = ["csi", "channel", "channels"]
    pos_keys = ["rx_pos", "positions"]

    for k in csi_keys:
        try:
            if k in dataset:
                csi = dataset[k]
                print(f"[INFO] Found CSI under key '{k}'")
                break
        except Exception:
            continue
    for k in pos_keys:
        try:
            if k in dataset:
                positions = dataset[k]
                print(f"[INFO] Found positions under key '{k}'")
                break
        except Exception:
            continue

    if csi is None or positions is None:
        raise RuntimeError(f"Could not find required keys. Available: {keys}\n" +
                           f"Found csi: {csi is not None}, positions: {positions is not None}")

    print(f"[INFO] Exporting to {CACHE_PATH}")
    np.savez_compressed(CACHE_PATH, csi=csi, positions=positions)
    print("[INFO] Done. Cached export ready.")

if __name__ == "__main__":
    main()
