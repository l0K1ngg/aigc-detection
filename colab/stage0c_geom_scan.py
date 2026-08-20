"""
Stage 0c: Dataset geometry census and color mode inspection.
Generates geometry distribution parquets and verifies alpha channels.
"""
import os
import pandas as pd
from PIL import Image
from tqdm import tqdm

def scan_geom(root_dir: str, out_parquet: str):
    rows = []
    for root, _, files in os.walk(root_dir):
        for f in tqdm(files, desc=os.path.basename(root)):
            if not f.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            p = os.path.join(root, f)
            rel = os.path.relpath(p, root_dir)
            parts = rel.split(os.sep)
            subset = parts[0] if len(parts) > 1 else "unknown"
            label = "fake" if "ai" in rel.lower() or "fake" in rel.lower() else "real"
            
            try:
                with Image.open(p) as img:
                    w, h = img.size
                    mode = img.mode
                    fmt = img.format
                rows.append({
                    "path": rel, "subset": subset, "label": label,
                    "width": w, "height": h, "short_edge": min(w, h),
                    "long_edge": max(w, h), "mode": mode, "format": fmt
                })
            except Exception:
                continue

    df = pd.DataFrame(rows)
    df.to_parquet(out_parquet, index=False)
    print(f"[+] Saved census to {out_parquet} ({len(df)} rows)")

if __name__ == "__main__":
    os.makedirs("/content/_reports", exist_ok=True)
    scan_geom("/content/raw_extracted/sdv4/train", "/content/_reports/geom_train.parquet")
    scan_geom("/content/raw_extracted", "/content/_reports/geom_val.parquet")