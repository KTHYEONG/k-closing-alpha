"""터미널 유틸리티 순수 함수 테스트."""

from __future__ import annotations

from src.utils.display import get_display_width, pad_str


def test_get_display_width_ascii() -> None:
    assert get_display_width("abc") == 3
    assert get_display_width("") == 0


def test_get_display_width_hangul_is_double() -> None:
    assert get_display_width("가") == 2
    assert get_display_width("가나다") == 6


def test_pad_str_left() -> None:
    assert pad_str("가", 4, align="left") == "가  "


def test_pad_str_right() -> None:
    assert pad_str("가", 4, align="right") == "  가"


def test_pad_str_center() -> None:
    # 너비 4, 표시폭 2 → 좌측 1, 우측 1
    assert pad_str("가", 4, align="center") == " 가 "


def test_pad_str_no_truncation_when_too_long() -> None:
    assert pad_str("longer", 3) == "longer"


def test_print_table_modern_topk_rows(capsys) -> None:
    from src.utils.display import print_table

    rows = [
        {"Code": "005930", "Name": "삼성전자", "Pred": 0.0234, "Alloc%": 33.33},
        {"Code": "000660", "Name": "SK하이닉스", "Pred": -0.0125, "Alloc%": 33.33},
    ]
    print_table(rows, "Top-3 Cost-Aware Decision (Equal-Weight)")
    captured = capsys.readouterr().out

    assert "Top-3 Cost-Aware Decision (Equal-Weight)" in captured
    assert "005930" in captured
    assert "삼성전자" in captured
    assert "+0.0234" in captured
    assert "33.33%" in captured
    assert "-0.0125" in captured


def test_print_table_handles_empty_or_none(capsys) -> None:
    from src.utils.display import print_table

    print_table(None, "Empty")
    print_table([], "Empty")
    captured = capsys.readouterr().out
    assert captured == ""

