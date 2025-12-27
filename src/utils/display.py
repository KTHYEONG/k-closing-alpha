import unicodedata
import pandas as pd


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
    """decision 값에 따라 색상 코드를 반환"""
    d = decision.lower()
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
    """결과 리스트를 테이블 형태로 출력"""
    if not results_list:
        return

    # Rank 오름차순, 동일 Rank 내에서는 Score 내림차순 정렬
    results_list.sort(key=lambda x: (x["Rank"], -x["Score"]))

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

    print(f"\n{Colors.BOLD}=== {title} ==={Colors.RESET}")
    print(divider)
    print(Colors.BOLD + header + Colors.RESET)
    print(divider)

    previous_stock_name = None

    for res in results_list:
        dec_color = get_decision_color(res["Decision"])
        if previous_stock_name is not None and res["Name"] != previous_stock_name:
            print(divider)
        previous_stock_name = res["Name"]

        name_display = res["Name"]
        # 이름이 설정된 너비보다 길 경우에만 최소한의 말줄임 적용
        if get_display_width(name_display) > W_NAME:
            while get_display_width(name_display + "..") > W_NAME:
                name_display = name_display[:-1]
            name_display += ".."
        score_str = f"{res['Score']:.4f}"

        if minimal:
            row_str = (
                f"| {pad_str(res['Rank'], W_RANK, 'center')} "
                f"| {pad_str(name_display, W_NAME, 'left')} "
                f"| {pad_str(score_str, W_PROB, 'center')} "
                f"| {dec_color}{pad_str(res['Decision'], W_DECISION, 'center')}{Colors.RESET} |"
            )
        else:
            rate_display = f"{res['Applied_Rate']}%"
            scenario_display = res["Scenario"]
            if get_display_width(scenario_display) > W_SCENARIO:
                while get_display_width(scenario_display + "..") > W_SCENARIO:
                    scenario_display = scenario_display[:-1]
                scenario_display += ".."

            row_str = (
                f"| {pad_str(res['Rank'], W_RANK, 'center')} "
                f"| {pad_str(name_display, W_NAME, 'left')} "
                f"| {pad_str(rate_display, W_RATE, 'center')} "
                f"| {pad_str(scenario_display, W_SCENARIO, 'left')} "
                f"| {pad_str(score_str, W_PROB, 'center')} "
                f"| {dec_color}{pad_str(res['Decision'], W_DECISION, 'center')}{Colors.RESET} |"
            )
        print(row_str)
    print(divider)


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
