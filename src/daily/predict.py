import logging

import pandas as pd

logger = logging.getLogger(__name__)

from src.daily.archive import fetch_archive_snapshot
from src.serving.realtime.artifacts import load_model_bundle
from src.utils.display import Colors, print_table


def load_daily_snapshot(decision_date: pd.Timestamp) -> pd.DataFrame:
    """당일 wide 스냅샷을 아카이브 저장소에서 읽는다.

    Args:
        decision_date: 조회 대상 일자.

    Returns:
        종목코드가 6자리 zero-fill 문자열로 정규화된 wide 단면. 금액 단위
        환산은 수행하지 않는다(저장소가 억 단위 원본을 그대로 보관).
    """
    df = fetch_archive_snapshot(snapshot_date=decision_date.strftime("%Y-%m-%d"))
    if "종목코드" in df.columns:
        df["종목코드"] = df["종목코드"].astype(str).str.zfill(6)
    return df


def run_topk_ranker_sleeve(decision_date: pd.Timestamp) -> pd.DataFrame:
    """자동 top-3 리랭커 슬리브를 실행한다.

    Args:
        decision_date: 리랭커 피처에 찍는 결정 일자.

    Returns:
        등가중 top-k 선정 결과. 저장소에 기록된 ``admitted`` 컬럼을 그대로
        사용하며 재계산하지 않는다. 번들이 없거나 wide 단면이 비정상이면
        경고를 남기고 빈 프레임을 반환한다.
    """
    try:
        from src.ml.costaware_topk import MIN_TOP_K
        from src.ml.topk_ranker_research import TOPK_RANKER_BUNDLE_DIR, select_topk_equal_weight
        from src.serving.realtime.features import build_topk_ranker_features

        wide = load_daily_snapshot(decision_date)
        features_df = build_topk_ranker_features(wide, decision_date)
        features_df["admitted"] = wide["admitted"].to_numpy()
        bundle = load_model_bundle(import_dir=TOPK_RANKER_BUNDLE_DIR)
        picks = select_topk_equal_weight(
            features_df, bundle, top_k=int(bundle.get("top_k", MIN_TOP_K))
        )
        name_map = dict(
            wide[["종목코드", "종목명"]].itertuples(index=False, name=None)
        )
        picks["name"] = picks["symbol"].map(name_map)
        return picks
    except (FileNotFoundError, ValueError) as exc:
        logger.warning(
            f"{Colors.YELLOW}[Warning] top-k ranker sleeve yielded no decision (미참여): {exc}{Colors.RESET}"
        )
        return pd.DataFrame()


def run_automated_topk_decision(decision_date: pd.Timestamp) -> None:
    """Print the single automated-mode top-3 decision table, if any.

    Args:
        decision_date: Decision date for the reranker sleeve.
    """
    sleeve_df = run_topk_ranker_sleeve(decision_date)
    if sleeve_df.empty:
        logger.warning("오늘 자동 유니버스 기준 진입 후보 없음(미참여)")
        return
    rows = [
        {
            "Code": str(row.get("symbol", "")),
            "Name": row.get("name", ""),
            "Pred": round(float(row.get("pred", 0.0)), 4),
            "Alloc%": round(float(row.get("allocation", 0.0)) * 100.0, 2),
        }
        for _, row in sleeve_df.iterrows()
    ]
    print_table(rows, "Top-3 Cost-Aware Decision (Equal-Weight)")


def main() -> None:
    decision_date = pd.Timestamp.today().normalize()
    run_automated_topk_decision(decision_date)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
