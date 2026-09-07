"""Probe script to measure 15:19/15:20 vs 15:30 EOD price difference and universe drift."""
from pathlib import Path
import pandas as pd
import numpy as np

def probe_drift():
    p = Path("data/history/intraday/1m/regular")
    files = sorted(p.glob("**/*.parquet"))
    print(f"Total 1m files: {len(files)}")
    
    drifts = []
    u_match = []
    
    for f in files[:50]: # Sample 50 files
        df = pd.read_parquet(f)
        # stck_cntg_hour: string or int e.g. 151900, 152000, 153000
        # Check hour format
        df["hour"] = df["stck_cntg_hour"].astype(str).str.zfill(6)
        
        # For each symbol
        for sym, g in df.groupby("종목코드"):
            g = g.sort_values("hour")
            # find bar at or immediately before 152000
            pre_20 = g[g["hour"] <= "152000"]
            bar_30 = g[g["hour"] >= "153000"]
            if not pre_20.empty and not bar_30.empty:
                p_20 = float(pre_20.iloc[-1]["stck_prpr"])
                p_30 = float(bar_30.iloc[-1]["stck_prpr"])
                if p_20 > 0 and p_30 > 0:
                    drift_bp = (p_30 / p_20 - 1.0) * 10000.0
                    drifts.append(drift_bp)

    drifts = np.array(drifts)
    print(f"Drift count: {len(drifts)}")
    print(f"Mean drift: {np.mean(drifts):.2f} bp")
    print(f"Median drift: {np.median(drifts):.2f} bp")
    print(f"Std drift: {np.std(drifts):.2f} bp")
    print(f"Min drift: {np.min(drifts):.2f} bp, Max drift: {np.max(drifts):.2f} bp")
    print(f"Abs drift > 50bp: {np.mean(np.abs(drifts) > 50.0)*100:.1f}%")
    print(f"Abs drift > 100bp: {np.mean(np.abs(drifts) > 100.0)*100:.1f}%")

if __name__ == "__main__":
    probe_drift()
