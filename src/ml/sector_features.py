"""Point-in-time sector clustering and market-wide sector returns."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

SECTOR_FEATURE_COLUMNS: tuple[str, ...] = (
    "sector_rel_mkt",
    "sector_mkt_ret",
    "sector_member_n",
    "sector_breadth",
    "sector_mom_5d",
)

_UNASSIGNED_CLUSTER: int = -1


def build_pit_sector_map(
    price_history: pd.DataFrame,
    *,
    n_clusters: int = 20,
    lookback_days: int = 550,
    min_obs: int = 120,
    random_state: int = 42,
) -> pd.DataFrame:
    ph = price_history.copy()
    ph["date"] = pd.to_datetime(ph["date"])
    ph["symbol"] = ph["symbol"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    ph = ph.sort_values(["symbol", "date"]).reset_index(drop=True)

    years = sorted(int(y) for y in ph["date"].dt.year.unique())
    all_symbols = sorted(ph["symbol"].unique().tolist())

    rows: list[pd.DataFrame] = []
    for y in years:
        jan1 = pd.Timestamp(f"{y}-01-01")
        window_start = jan1 - pd.Timedelta(days=int(lookback_days))
        window_end = jan1
        mask = (ph["date"] >= window_start) & (ph["date"] < window_end)
        ph_win = ph.loc[mask].copy()
        # default all -1
        year_df = pd.DataFrame({"year": y, "symbol": all_symbols, "cluster_id": _UNASSIGNED_CLUSTER})
        year_df["cluster_id"] = year_df["cluster_id"].astype(np.int64)

        if ph_win.empty:
            rows.append(year_df)
            continue

        # compute daily log returns per symbol
        # groupby shift requires sorted by symbol,date already
        close = ph_win["close"].astype(np.float64)
        labels = ph_win["symbol"]
        # pct_change per symbol
        pct = close.groupby(labels.to_numpy(), sort=False).pct_change()
        log_ret = np.log1p(pct.to_numpy(dtype=np.float64))
        ph_win["_log_ret"] = log_ret

        pivot = ph_win.pivot_table(index="date", columns="symbol", values="_log_ret", aggfunc="first")
        # 창(window) 내 거래이력이 있는 종목만 후보. 창에 없는 종목은 year_df 기본값(-1) 유지.
        valid_counts = pivot.notna().sum(axis=0)
        qualified = valid_counts[valid_counts >= int(min_obs)].index.tolist()

        if len(qualified) < int(n_clusters):
            rows.append(year_df)
            continue

        pivot_q = pivot[qualified]
        # correlation matrix
        corr = pivot_q.corr()
        # corr is DataFrame qualified x qualified
        corr_mat = corr.to_numpy(dtype=np.float64)
        corr_mat = np.nan_to_num(corr_mat, nan=0.0, posinf=0.0, neginf=0.0)

        kmeans = KMeans(n_clusters=int(n_clusters), random_state=int(random_state), n_init=10)
        labels_arr = kmeans.fit_predict(corr_mat)

        mapping = dict(zip(qualified, labels_arr.tolist(), strict=True))
        # assign
        cluster_ids = []
        for sym in all_symbols:
            if sym in mapping:
                cluster_ids.append(int(mapping[sym]))
            else:
                cluster_ids.append(_UNASSIGNED_CLUSTER)
        year_df["cluster_id"] = np.asarray(cluster_ids, dtype=np.int64)
        rows.append(year_df)

    if not rows:
        return pd.DataFrame({"year": pd.Series(dtype="int64"), "symbol": pd.Series(dtype="object"), "cluster_id": pd.Series(dtype="int64")})
    out = pd.concat(rows, ignore_index=True)
    out["year"] = out["year"].astype(np.int64)
    out["symbol"] = out["symbol"].astype(str)
    out["cluster_id"] = out["cluster_id"].astype(np.int64)
    out = out.sort_values(["year", "symbol"]).reset_index(drop=True)
    # replace inf with NaN safety (though cluster_id is int)
    return out[["year", "symbol", "cluster_id"]]


def compute_market_sector_returns(
    price_history: pd.DataFrame,
    sector_map: pd.DataFrame,
) -> pd.DataFrame:
    ph = price_history.copy()
    ph["date"] = pd.to_datetime(ph["date"])
    ph["symbol"] = ph["symbol"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    ph = ph.sort_values(["symbol", "date"]).reset_index(drop=True)

    sm = sector_map.copy()
    sm["symbol"] = sm["symbol"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    sm["year"] = sm["year"].astype(np.int64)
    sm["cluster_id"] = sm["cluster_id"].astype(np.int64)

    ph["year"] = ph["date"].dt.year.astype(np.int64)
    merged = ph.merge(sm[["year", "symbol", "cluster_id"]], on=["year", "symbol"], how="left")
    # exclude unassigned
    merged = merged[merged["cluster_id"] != _UNASSIGNED_CLUSTER].copy()
    merged = merged.dropna(subset=["cluster_id"])
    if merged.empty:
        return pd.DataFrame(
            {
                "date": pd.Series(dtype="datetime64[ns]"),
                "cluster_id": pd.Series(dtype="int64"),
                "sector_mkt_ret": pd.Series(dtype="float64"),
                "sector_member_n": pd.Series(dtype="float64"),
                "sector_breadth": pd.Series(dtype="float64"),
                "sector_mom_5d": pd.Series(dtype="float64"),
            }
        )
    merged["cluster_id"] = merged["cluster_id"].astype(np.int64)
    merged = merged.sort_values(["symbol", "date"]).reset_index(drop=True)
    close_sorted = merged["close"].astype(np.float64)
    labels_sorted = merged["symbol"].to_numpy()
    pct = close_sorted.groupby(labels_sorted, sort=False).pct_change()
    merged["daily_ret"] = pct.to_numpy(dtype=np.float64) * 100.0
    # replace inf with NaN
    merged["daily_ret"] = merged["daily_ret"].replace([np.inf, -np.inf], np.nan).astype(np.float64)

    grouped = merged.groupby(["date", "cluster_id"], sort=True)
    agg2 = grouped.agg(
        sector_mkt_ret=("daily_ret", "mean"),
        sector_member_n=("daily_ret", "count"),
    ).reset_index()
    merged["is_positive"] = np.where(merged["daily_ret"].notna(), (merged["daily_ret"] > 0).astype(np.float64), np.nan)
    breadth = merged.groupby(["date", "cluster_id"], sort=True)["is_positive"].mean().reset_index()
    breadth = breadth.rename(columns={"is_positive": "sector_breadth"})
    agg2 = agg2.merge(breadth, on=["date", "cluster_id"], how="left")

    agg2["sector_mkt_ret"] = agg2["sector_mkt_ret"].replace([np.inf, -np.inf], np.nan).astype(np.float64)
    agg2["sector_member_n"] = agg2["sector_member_n"].astype(np.float64)
    agg2["sector_breadth"] = agg2["sector_breadth"].replace([np.inf, -np.inf], np.nan).astype(np.float64)

    # sector_mom_5d per cluster
    agg2 = agg2.sort_values(["cluster_id", "date"]).reset_index(drop=True)
    # compute per cluster shift then rolling sum
    # Use groupby transform
    def _mom(s: pd.Series) -> pd.Series:
        shifted = s.shift(1)
        return shifted.rolling(window=5, min_periods=3).sum()

    agg2["sector_mom_5d"] = agg2.groupby("cluster_id", sort=False)["sector_mkt_ret"].transform(_mom)
    agg2["sector_mom_5d"] = agg2["sector_mom_5d"].replace([np.inf, -np.inf], np.nan).astype(np.float64)
    agg2["cluster_id"] = agg2["cluster_id"].astype(np.int64)
    agg2 = agg2.sort_values(["date", "cluster_id"]).reset_index(drop=True)
    return agg2[["date", "cluster_id", "sector_mkt_ret", "sector_member_n", "sector_breadth", "sector_mom_5d"]]


def attach_sector_features(
    panel: pd.DataFrame,
    price_history: pd.DataFrame,
    *,
    date_col: str = "trade_date",
    code_col: str = "stock_code",
    change_col: str = "change_rate",
    n_clusters: int = 20,
) -> pd.DataFrame:
    if date_col not in panel.columns:
        raise ValueError(f"panel missing date_col {date_col!r}")
    if code_col not in panel.columns:
        raise ValueError(f"panel missing code_col {code_col!r}")
    if change_col not in panel.columns:
        raise ValueError(f"panel missing change_col {change_col!r}")

    out = panel.copy()
    origin_index = out.index
    # Build maps
    sector_map = build_pit_sector_map(price_history, n_clusters=int(n_clusters))
    agg = compute_market_sector_returns(price_history, sector_map)

    # Prepare join keys for sector_map
    out["_join_symbol"] = out[code_col].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    out["_join_date"] = pd.to_datetime(out[date_col])
    out["_join_year"] = out["_join_date"].dt.year.astype(np.int64)

    sm = sector_map.copy()
    sm["symbol"] = sm["symbol"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    # Merge cluster_id
    tmp = out.merge(
        sm[["year", "symbol", "cluster_id"]],
        left_on=["_join_year", "_join_symbol"],
        right_on=["year", "symbol"],
        how="left",
    )
    # tmp has duplicate symbol/year columns; drop them
    # cluster_id NaN -> unassigned
    tmp["cluster_id"] = tmp["cluster_id"].fillna(_UNASSIGNED_CLUSTER).astype(np.int64)
    tmp["sector_cluster_id"] = tmp["cluster_id"].astype(np.int64)

    # Now join agg on date + cluster_id
    # agg date is datetime, _join_date is datetime normalize to date (without time)
    # Ensure date matching at day level
    tmp["_join_date_norm"] = pd.to_datetime(tmp["_join_date"]).dt.normalize()
    agg2 = agg.copy()
    agg2["date"] = pd.to_datetime(agg2["date"]).dt.normalize()

    result = tmp.merge(
        agg2[["date", "cluster_id", "sector_mkt_ret", "sector_member_n", "sector_breadth", "sector_mom_5d"]],
        left_on=["_join_date_norm", "cluster_id"],
        right_on=["date", "cluster_id"],
        how="left",
    )

    # compute sector_rel_mkt = change_col - sector_mkt_ret
    change_vals = result[change_col].to_numpy(dtype=np.float64)
    mkt_vals = result["sector_mkt_ret"].to_numpy(dtype=np.float64)
    rel = change_vals - mkt_vals
    # where mkt is NaN, rel becomes NaN automatically
    result["sector_rel_mkt"] = rel.astype(np.float64)

    # Cast SECTOR_FEATURE_COLUMNS to float64 and replace inf
    for col in SECTOR_FEATURE_COLUMNS:
        if col in result.columns:
            result[col] = result[col].replace([np.inf, -np.inf], np.nan).astype(np.float64)
        else:
            result[col] = np.nan

    # For unassigned cluster rows, ensure all sector features NaN (they already are NaN after left join, but ensure)
    unassigned_mask = result["sector_cluster_id"] == _UNASSIGNED_CLUSTER
    for col in SECTOR_FEATURE_COLUMNS:
        result.loc[unassigned_mask, col] = np.nan

    # Also where agg missing, features remain NaN

    # Reconstruct output preserving original index/order and dropping helpers
    # The tmp merge may have reordered? Use left join preserves left order
    result.index = origin_index
    # Ensure original columns order preserved plus new
    # Keep all original panel columns plus new ones
    # Drop helpers
    helpers = ["_join_symbol", "_join_date", "_join_year", "year", "symbol", "cluster_id", "date", "_join_date_norm"]
    result = result.drop(columns=[c for c in helpers if c in result.columns])

    # Ensure sector_cluster_id is int64
    result["sector_cluster_id"] = result["sector_cluster_id"].astype(np.int64)

    # Replace inf in feature cols
    for col in SECTOR_FEATURE_COLUMNS:
        result[col] = result[col].replace([np.inf, -np.inf], np.nan).astype(np.float64)

    # Preserve original index and column order: original columns first, then sector columns
    # But spec says preserve index/order; just return with same index and new columns appended
    # Ensure we return in same row order as input (which left merge preserves)
    result = result.copy()
    result.index = origin_index
    return result
