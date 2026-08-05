"""시나리오 행동(action) 패널 정규화.

`docs/specs/scenario_action_panel_resolution.md` 의 1단계(시나리오 행동 패널)를
구현합니다. 원천의 ``(trade_date, stock_code)`` 중복을 임의로 한 행으로 합치지
않고 각 행을 서로 다른 시나리오 행동으로 보존합니다. 같은 행동 key
``(date, stock, scenario)`` 의 완전 동일 행만 한 행으로 축소하고, 실행값이나
피처가 충돌하는 중복은 reject table 로 이동합니다. 임의의 first/last, 평균,
최대/최소 수익률 선택을 사용하지 않습니다.
"""

from __future__ import annotations

import pandas as pd

# 고정 시나리오 one-hot 수치 피처: LightGBM 입력에는 ``chart_analysis`` 원문 대신
# 이 컬럼들을 사용합니다. 순서는 피처셋 명세에 고정되어 있습니다.
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

# 날짜-종목 수준 시나리오 context 피처 (날짜-종목당 실행 가능 행동이 여러 개일 때
# 모델이 행동 가용성을 학습할 수 있도록 합니다).
SCENARIO_CONTEXT_FEATURES: tuple[str, ...] = (
    "scenario_count_for_stock_date",
    "has_sangtta_for_stock_date",
    "is_multi_scenario_stock_date",
)

# 알 수 없는 시나리오는 ``scenario_other=1`` 로 보냅니다.
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
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(date, stock, scenario) 행동 패널과 충돌 reject table 을 반환합니다.

    - 서로 다른 시나리오의 같은 날짜-종목 행은 모두 보존합니다.
    - 같은 행동 key 의 완전 동일 행만 한 행으로 축소합니다 (결정적).
    - 같은 행동 key 에서 실행/피처가 충돌하면 해당 key 의 모든 행을
      ``conflicting_duplicate_action`` 으로 reject table 로 이동합니다.
    - 허용된 행에 고정 시나리오 one-hot/context 수치 피처를 추가합니다.

    Args:
        df: ``date_col``/``stock_col``/``scenario_col`` 을 포함한 정제된 데이터.
        date_col: 거래일 key 컬럼.
        stock_col: 종목 key 컬럼.
        scenario_col: 시나리오(차트분석) 컬럼.

    Returns:
        ``(panel, rejects)``: ``panel`` 은 유일 행동 key 의 행에 시나리오 수치
        피처가 추가된 DataFrame, ``rejects`` 는 충돌 중복 행으로 ``reject_reason``
        컬럼(``conflicting_duplicate_action``)으로 식별됩니다.

    Raises:
        ValueError: key 컬럼이 없거나 key 컬럼에 null 이 있으면 발생합니다.
    """
    key_cols = [date_col, stock_col, scenario_col]
    missing = [col for col in key_cols if col not in df.columns]
    if missing:
        raise ValueError(f"missing required key columns for scenario action panel: {missing}")
    null_cols = [col for col in key_cols if df[col].isna().any()]
    if null_cols:
        raise ValueError(f"key columns contain nulls: {null_cols}")

    work = df.copy()

    # 같은 행동 key 가 둘 이상인 행만 후보로 취급합니다.
    dup_mask = work.duplicated(subset=key_cols, keep=False)
    keep_rows: list[pd.DataFrame] = []
    reject_rows: list[pd.DataFrame] = []
    if dup_mask.any():
        for _, group in work.loc[dup_mask].groupby(key_cols, sort=False, dropna=False):
            if group.drop_duplicates().shape[0] == 1:
                # 완전 동일 행만 한 행으로 축소 (첫 번째 행 유지).
                keep_rows.append(group.iloc[:1])
            else:
                reject = group.copy()
                reject[_REJECT_REASON_COL] = _REJECT_REASON_VALUE
                reject_rows.append(reject)

    panel = pd.concat([work.loc[~dup_mask], *keep_rows], axis=0).sort_index()

    if reject_rows:
        rejects = pd.concat(reject_rows, axis=0)
    else:
        rejects = pd.DataFrame(columns=[*work.columns, _REJECT_REASON_COL])

    # 고정 시나리오 one-hot 수치 피처 (벡터 연산, pd.apply 미사용).
    scenario_feature = (
        panel[scenario_col]
        .astype(str)
        .map(_SCENARIO_NAME_TO_FEATURE)
        .fillna("scenario_other")
    )
    for feature in SCENARIO_ONE_HOT_FEATURES:
        panel[feature] = scenario_feature.eq(feature).astype("int64")

    # 날짜-종목 수준 context 피처.
    group_key = [date_col, stock_col]
    scenario_counts = panel.groupby(group_key, sort=False)[scenario_col].transform("size")
    panel["scenario_count_for_stock_date"] = scenario_counts
    panel["is_multi_scenario_stock_date"] = (scenario_counts > 1).astype("int64")
    panel["has_sangtta_for_stock_date"] = (
        panel.groupby(group_key, sort=False)["scenario_is_sangtta"].transform("max").astype("int64")
    )
    return panel, rejects
