"""
Metadata baseline: Evaluates container format & short-edge shortcuts on validation set.
Quantifies the dataset construction bias (PNG/JPEG & fixed dimension shortcuts).
"""
import os
import json
import pandas as pd
from sklearn.metrics import accuracy_score

def eval_shortcuts(parquet_path: str, out_report: str):
    df = pd.read_parquet(parquet_path)
    df["y_true"] = df["label"].apply(lambda x: 1 if x == "fake" else 0)
    
    # R1: PNG 格式即判假
    df["r1_pred"] = df["format"].apply(lambda x: 1 if str(x).upper() == "PNG" else 0)
    
    # R2: 短边命中生成器固定尺寸集
    df["r2_pred"] = df["short_edge"].apply(lambda x: 1 if x in [128, 256, 512, 1024] else 0)
    
    # R1 ∪ R2 联合判定
    df["union_pred"] = ((df["r1_pred"] == 1) | (df["r2_pred"] == 1)).astype(int)
    
    res = {
        "R1_PNG_acc": float(accuracy_score(df["y_true"], df["r1_pred"])),
        "R2_Edge_acc": float(accuracy_score(df["y_true"], df["r2_pred"])),
        "Union_acc": float(accuracy_score(df["y_true"], df["union_pred"]))
    }
    
    print("[*] Metadata Shortcuts Baseline:")
    print(json.dumps(res, indent=2))
    
    with open(out_report, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)

if __name__ == "__main__":
    eval_shortcuts("/content/_reports/geom_val.parquet", "/content/_reports/metadata_baseline.json")