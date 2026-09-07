"""Probe script to inspect dataset schemas, timestamps, and availability for v3 research."""
from pathlib import Path
import pandas as pd
import numpy as np

def probe_datasets():
    files = {
        "price_history": Path("data/history/price_history.parquet"),
        "condition_history": Path("data/history/condition_history_cleaned.parquet"),
        "archive": Path("data/history/archive.parquet"),
    }
    
    for name, p in files.items():
        if p.exists():
            df = pd.read_parquet(p)
            print(f"=== {name} ===")
            print(f"Shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            if "date" in df.columns:
                print(f"Date min: {df['date'].min()}, max: {df['date'].max()}, unique days: {df['date'].nunique()}")
            if "snapshot_timestamp" in df.columns:
                print(f"snapshot_timestamp sample: {df['snapshot_timestamp'].dropna().head(3).tolist()}")
            print(df.head(2))
            print()
        else:
            print(f"=== {name} MISSING ===")

    # Check intraday
    intra_dir = Path("data/history/intraday")
    if intra_dir.exists():
        files_1m = list(intra_dir.glob("**/*.parquet"))
        print(f"=== Intraday 1m files count: {len(files_1m)} ===")
        if files_1m:
            df_intra = pd.read_parquet(files_1m[0])
            print(f"Sample 1m file: {files_1m[0].name}, shape: {df_intra.shape}, cols: {list(df_intra.columns)}")
            print(df_intra.head(2))

    # Check altdata
    alt_dir = Path("data/history/altdata")
    if alt_dir.exists():
        for ap in alt_dir.glob("*.parquet"):
            df_alt = pd.read_parquet(ap)
            print(f"=== Altdata: {ap.name}, shape: {df_alt.shape}, cols: {list(df_alt.columns)} ===")

if __name__ == "__main__":
    probe_datasets()
