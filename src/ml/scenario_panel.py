"""Scenario action panel port."""
from __future__ import annotations

from typing import Any

import pandas as pd

SCENARIO_ONE_HOT_FEATURES: tuple[str, ...] = (
    "scenario_is_sangtta",
    "scenario_is_120_breakout",
    "scenario_is_volume_surge",
    "scenario_is_new_high",
    "scenario_is_near_new_high",
    "scenario_is_limitup_next_day",
    "scenario_is_rising_bearish",
    "scenario_other",
)

SCENARIO_CONTEXT_FEATURES: tuple[str, ...] = (
    "scenario_count_for_stock_date",
    "has_sangtta_for_stock_date",
    "is_multi_scenario_stock_date",
)

_SCENARIO_NAME_TO_FEATURE: dict[str, str] = {
    "상따": "scenario_is_sangtta",
    "120 돌파": "scenario_is_120_breakout",
    "거래량 폭증": "scenario_is_volume_surge",
    "신고가": "scenario_is_new_high",
    "신고가 근접": "scenario_is_near_new_high",
    "상한가 다음날": "scenario_is_limitup_next_day",
    "상승형 음봉": "scenario_is_rising_bearish",
}

_REJECT_REASON_COL = "reject_reason"
_REJECT_REASON_VALUE = "conflicting_duplicate_action"


def build_scenario_action_panel(
    df: pd.DataFrame,
    date_col: str = "trade_date",
    stock_col: str = "stock_code",
    scenario_col: str = "chart_analysis",
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """(date, stock, scenario) 행동 패널과 충돌 reject list 를 반환합니다."""
    key_cols = [date_col, stock_col, scenario_col]
    missing = [col for col in key_cols if col not in df.columns]
    if missing:
        raise ValueError(f"missing required key columns for scenario action panel: {missing}")
    null_cols = [col for col in key_cols if df[col].isna().any()]
    if null_cols:
        raise ValueError(f"key columns contain nulls: {null_cols}")

    work = df.copy()
    dup_mask = work.duplicated(subset=key_cols, keep=False)
    keep_rows: list[pd.DataFrame] = []
    reject_rows: list[pd.DataFrame] = []
    if dup_mask.any():
        for _, group in work.loc[dup_mask].groupby(key_cols, sort=False, dropna=False):
            if group.drop_duplicates().shape[0] == 1:
                keep_rows.append(group.iloc[:1])
            else:
                reject = group.copy()
                reject[_REJECT_REASON_COL] = _REJECT_REASON_VALUE
                reject_rows.append(reject)

    panel = pd.concat([work.loc[~dup_mask], *keep_rows], axis=0).sort_index()

    if reject_rows:
        rejects_df = pd.concat(reject_rows, axis=0)
        rejects: list[dict[str, Any]] = rejects_df.to_dict(orient="records")  # type: ignore[return-value]
    else:
        rejects = []

    scenario_feature = (
        panel[scenario_col]
        .astype(str)
        .map(_SCENARIO_NAME_TO_FEATURE)
        .fillna("scenario_other")
    )
    for feature in SCENARIO_ONE_HOT_FEATURES:
        panel[feature] = scenario_feature.eq(feature).astype("int64")

    group_key = [date_col, stock_col]
    scenario_counts = panel.groupby(group_key, sort=False)[scenario_col].transform("size")
    panel["scenario_count_for_stock_date"] = scenario_counts
    panel["is_multi_scenario_stock_date"] = (scenario_counts > 1).astype("int64")
    panel["has_sangtta_for_stock_date"] = (
        panel.groupby(group_key, sort=False)["scenario_is_sangtta"].transform("max").astype("int64")
    )
    return panel, rejects
