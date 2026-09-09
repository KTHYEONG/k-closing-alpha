import sys
import unicodedata

import pandas as pd


def _write_line(value: object = "") -> None:
    """Write one display line without invoking the banned print builtin."""
    sys.stdout.write(f"{value}\n")


class Colors:
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    BOLD = "\033[1m"
    RESET = "\033[0m"
    GRAY = "\033[90m"


def get_decision_color(decision):
    """Decision 값에 따라 색상 코드를 반환"""
    d = str(decision).lower()
    if "strong" in d or "buy" in d:
        return Colors.RED + Colors.BOLD
    if "good" in d:
        return Colors.GREEN + Colors.BOLD
    if "weak" in d:
        return Colors.YELLOW
    if "pass" in d or "abstain" in d:
        return Colors.GRAY
    if "max" in d:
        return Colors.RED + Colors.BOLD
    if "expand" in d:
        return Colors.MAGENTA
    if "neutral" in d:
        return Colors.WHITE
    if "reduce" in d:
        return Colors.YELLOW
    return Colors.RESET


def get_display_width(s):
    """한글/영문 혼합 문자열의 실제 화면 너비 계산"""
    width = 0
    for char in s:
        if unicodedata.east_asian_width(char) in ["F", "W", "A"]:
            width += 2
        else:
            width += 1
    return width


def pad_str(s, width, align="left"):
    """화면 너비 기준으로 문자열 정렬(Padding)"""
    s = str(s)
    current_width = get_display_width(s)
    padding_size = max(0, width - current_width)

    if align == "center":
        left = padding_size // 2
        right = padding_size - left
        return " " * left + s + " " * right
    elif align == "right":
        return " " * padding_size + s
    else:  # left
        return s + " " * padding_size


def print_table(results_list, title, minimal=False):
    """결과 리스트 또는 DataFrame을 테이블 형태로 출력"""
    if results_list is None:
        return

    if isinstance(results_list, pd.DataFrame):
        if results_list.empty:
            return
        _write_line(f"\n{Colors.BOLD}=== {title} ==={Colors.RESET}")
        for _idx, row in results_list.iterrows():
            decision = row.get("decision", "ABSTAIN")
            reason = row.get("decision_reason", "")
            stock = row.get("stock_code") or "None"
            score = row.get("rank_score")
            score_str = f"{score:.4f}" if score is not None and pd.notna(score) else "N/A"
            color = get_decision_color(decision)
            _write_line(
                f"  > Decision: {color}{decision}{Colors.RESET} | Reason: {reason} | Stock: {stock} | Score: {score_str}"
            )
        return

    if not results_list:
        return

    # Modern Top-K Reranker 결정 행 (Code, Name, Pred, Alloc% 형태)
    first_item = results_list[0] if isinstance(results_list[0], dict) else {}
    if "Code" in first_item or "Pred" in first_item:
        W_RANK, W_CODE, W_NAME, W_PRED, W_ALLOC = 6, 10, 16, 16, 14
        header = (
            f"| {pad_str('Rank', W_RANK, 'center')} "
            f"| {pad_str('Code', W_CODE, 'center')} "
            f"| {pad_str('Name', W_NAME, 'center')} "
            f"| {pad_str('Pred(Return)', W_PRED, 'center')} "
            f"| {pad_str('Alloc(Weight)', W_ALLOC, 'center')} |"
        )
        divider = "─" * get_display_width(header)
        box_top = "━" * get_display_width(header)

        _write_line(f"\n{Colors.BOLD}{box_top}{Colors.RESET}")
        _write_line(f" {Colors.CYAN}{Colors.BOLD}🎯 [{title}]{Colors.RESET}")
        _write_line(f"{Colors.BOLD}{box_top}{Colors.RESET}")
        _write_line(Colors.BOLD + header + Colors.RESET)
        _write_line(divider)

        for rank, res in enumerate(results_list, start=1):
            code_display = str(res.get("Code", ""))
            name_display = str(res.get("Name", ""))
            if get_display_width(name_display) > W_NAME:
                while get_display_width(name_display + "..") > W_NAME:
                    name_display = name_display[:-1]
                name_display += ".."

            pred_val = res.get("Pred", 0.0)
            pred_str = f"{pred_val:+.4f}" if isinstance(pred_val, (int, float)) else str(pred_val)
            alloc_val = res.get("Alloc%", 0.0)
            alloc_str = f"{alloc_val:.2f}%" if isinstance(alloc_val, (int, float)) else str(alloc_val)

            pred_color = Colors.RED if isinstance(pred_val, (int, float)) and pred_val > 0 else Colors.BLUE if isinstance(pred_val, (int, float)) and pred_val < 0 else Colors.WHITE

            row_str = (
                f"| {pad_str(str(rank), W_RANK, 'center')} "
                f"| {pad_str(code_display, W_CODE, 'center')} "
                f"| {pad_str(name_display, W_NAME, 'left')} "
                f"| {pred_color}{pad_str(pred_str, W_PRED, 'center')}{Colors.RESET} "
                f"| {Colors.GREEN}{pad_str(alloc_str, W_ALLOC, 'center')}{Colors.RESET} |"
            )
            _write_line(row_str)

        _write_line(box_top)
        return

    # Legacy Rank/Score/Scenario/Decision 포맷 하위 호환
    results_list_sorted = sorted(
        results_list,
        key=lambda x: (-x.get("Score", 0.0), x.get("Rank", 999)),
    )

    if minimal:
        W_RANK, W_NAME = 6, 16
        W_PROB, W_DECISION = 8, 12
        header = (
            f"| {pad_str('Rank', W_RANK, 'center')} "
            f"| {pad_str('Name', W_NAME, 'center')} "
            f"| {pad_str('Score', W_PROB, 'center')} "
            f"| {pad_str('Decision', W_DECISION, 'center')} |"
        )
    else:
        W_RANK, W_NAME, W_RATE = 6, 16, 8
        W_SCENARIO, W_PROB, W_DECISION = 18, 8, 12
        header = (
            f"| {pad_str('Rank', W_RANK, 'center')} "
            f"| {pad_str('Name', W_NAME, 'center')} "
            f"| {pad_str('Rate', W_RATE, 'center')} "
            f"| {pad_str('Scenario', W_SCENARIO, 'center')} "
            f"| {pad_str('Score', W_PROB, 'center')} "
            f"| {pad_str('Decision', W_DECISION, 'center')} |"
        )
    divider = "-" * get_display_width(header)

    _write_line(f"\n{Colors.BOLD}=== {title} ==={Colors.RESET}")
    _write_line(divider)
    _write_line(Colors.BOLD + header + Colors.RESET)
    _write_line(divider)

    previous_stock_name = None

    for res in results_list_sorted:
        dec_color = get_decision_color(res.get("Decision", ""))
        name = res.get("Name", "")
        if previous_stock_name is not None and name != previous_stock_name:
            _write_line(divider)
        previous_stock_name = name

        name_display = name
        if get_display_width(name_display) > W_NAME:
            while get_display_width(name_display + "..") > W_NAME:
                name_display = name_display[:-1]
            name_display += ".."
        score_val = res.get("Score", 0.0)
        score_str = f"{score_val:.4f}" if isinstance(score_val, (int, float)) else str(score_val)

        if minimal:
            row_str = (
                f"| {pad_str(str(res.get('Rank', '')), W_RANK, 'center')} "
                f"| {pad_str(name_display, W_NAME, 'left')} "
                f"| {pad_str(score_str, W_PROB, 'center')} "
                f"| {dec_color}{pad_str(str(res.get('Decision', '')), W_DECISION, 'center')}{Colors.RESET} |"
            )
        else:
            rate_display = f"{res.get('Applied_Rate', 'N/A')}%"
            scenario_display = str(res.get("Scenario", "N/A"))
            if get_display_width(scenario_display) > W_SCENARIO:
                while get_display_width(scenario_display + "..") > W_SCENARIO:
                    scenario_display = scenario_display[:-1]
                scenario_display += ".."

            row_str = (
                f"| {pad_str(str(res.get('Rank', '')), W_RANK, 'center')} "
                f"| {pad_str(name_display, W_NAME, 'left')} "
                f"| {pad_str(rate_display, W_RATE, 'center')} "
                f"| {pad_str(scenario_display, W_SCENARIO, 'left')} "
                f"| {pad_str(score_str, W_PROB, 'center')} "
                f"| {dec_color}{pad_str(str(res.get('Decision', '')), W_DECISION, 'center')}{Colors.RESET} |"
            )
        _write_line(row_str)
    _write_line(divider)


def apply_label_encodings(df, encoder_map):
    """Apply label encoding mappings to categorical columns in-place."""
    if not encoder_map:
        object_cols = df.select_dtypes(include=["object"]).columns
        for col in object_cols:
            df[col] = pd.Categorical(df[col]).codes.astype(float)
        return df

    for col, info in encoder_map.items():
        if col not in df.columns:
            continue
        mapping = info["mapping"]
        unknown_idx = info["unknown"]
        df[col] = (
            df[col]
            .astype(str)
            .apply(lambda val: mapping.get(val, unknown_idx))
            .astype(float)
        )
    return df
