from __future__ import annotations


def _seed_round_trip(tmp_path):
    import pandas as pd

    from src.execution.paper_broker import PaperLedger, refresh_trade_ledgers

    ledger = PaperLedger(root=tmp_path)
    buy = {"order_id": "2026-09-10:005930:entry", "symbol": "005930", "side": "buy", "qty": 10,
           "fill_price": 70_000, "filled_at": pd.Timestamp("2026-09-10 15:30:20", tz="Asia/Seoul"),
           "decision_date": "2026-09-10", "trigger": "auction_close", "entry_order_id": None}
    sell = {"order_id": "2026-09-10:005930:entry:exit:2026-09-11", "symbol": "005930", "side": "sell", "qty": 10,
            "fill_price": 73_600, "filled_at": pd.Timestamp("2026-09-11 09:00:00", tz="Asia/Seoul"),
            "decision_date": "2026-09-11", "trigger": "auction_open",
            "entry_order_id": "2026-09-10:005930:entry"}
    ledger.record([buy, sell], kind="fills")
    refresh_trade_ledgers(ledger, seed_capital=10_000_000, as_of_date="2026-09-11")
    return buy, sell


def test_cli_void_fill_refreshes_derived_ledgers(tmp_path, monkeypatch) -> None:
    import pandas as pd
    import pytest

    from src.tools import paper_ledger_correct
    from src.execution.paper_broker import PaperLedger

    buy, sell = _seed_round_trip(tmp_path)
    monkeypatch.setattr(paper_ledger_correct.settings, "PAPER_DIR", tmp_path)
    monkeypatch.setattr(paper_ledger_correct.settings, "PAPER_SEED_CAPITAL", 10_000_000)

    # When
    paper_ledger_correct.main(
        ["void-fill", "--order-id", sell["order_id"], "--reason", "holiday_phantom_exit",
         "--evidence", "oracle says holiday", "--operator", "tester", "--as-of", "2026-09-11"]
    )

    # Then: correction 행이 남고, trades는 비고, 해당 일자 NAV가 재진술된다
    corrections = pd.read_parquet(tmp_path / "corrections.parquet")
    assert len(corrections) == 1
    assert corrections.iloc[0]["correction_id"] == f"2026-09-11:VOID_FILL:{sell['order_id']}"
    assert pd.read_parquet(tmp_path / "trades.parquet").empty
    nav = pd.read_parquet(tmp_path / "nav.parquet").set_index("as_of_date")
    assert int(nav.loc["2026-09-11", "cash"]) == 10_000_000 - 700_025
    assert PaperLedger(root=tmp_path).load_open_positions()["entry_order_id"].tolist() == [buy["order_id"]]

    # And: 같은 void를 다시 실행하면 exit 1로 실패하고 corrections는 그대로다
    before = pd.read_parquet(tmp_path / "corrections.parquet")
    with pytest.raises(SystemExit) as exc:
        paper_ledger_correct.main(
            ["void-fill", "--order-id", sell["order_id"], "--reason", "holiday_phantom_exit",
             "--evidence", "oracle says holiday", "--operator", "tester", "--as-of", "2026-09-11"]
        )
    assert exc.value.code == 1
    pd.testing.assert_frame_equal(pd.read_parquet(tmp_path / "corrections.parquet"), before)


def test_cli_note_is_idempotent_safe(tmp_path, monkeypatch) -> None:
    import pandas as pd
    import pytest

    from src.tools import paper_ledger_correct

    _seed_round_trip(tmp_path)
    monkeypatch.setattr(paper_ledger_correct.settings, "PAPER_DIR", tmp_path)
    argv = ["note", "--targets", "ghost:1,ghost:2", "--reason", "retro_doc",
            "--evidence", "manual repair log", "--operator", "tester", "--as-of", "2026-09-24"]

    # When
    paper_ledger_correct.main(argv)

    # Then: NOTE 행이 남는다
    assert len(pd.read_parquet(tmp_path / "corrections.parquet")) == 1

    # And: 같은 note를 다시 실행하면 exit 1이고 corrections는 그대로다
    before = pd.read_parquet(tmp_path / "corrections.parquet")
    with pytest.raises(SystemExit) as exc:
        paper_ledger_correct.main(argv)
    assert exc.value.code == 1
    pd.testing.assert_frame_equal(pd.read_parquet(tmp_path / "corrections.parquet"), before)


def _void_argv(order_id, *, reason="holiday_phantom_exit", operator="tester"):
    return [
        "void-fill", "--order-id", order_id, "--reason", reason,
        "--evidence", "oracle says holiday", "--operator", operator, "--as-of", "2026-09-11",
    ]


def _snapshot_bytes(tmp_path):
    from pathlib import Path

    return {p.name: Path(p).read_bytes() for p in sorted(tmp_path.iterdir()) if p.is_file()}


def test_cli_void_fill_leaves_fill_evidence_untouched(tmp_path, monkeypatch) -> None:
    from src.tools import paper_ledger_correct

    _, sell = _seed_round_trip(tmp_path)
    monkeypatch.setattr(paper_ledger_correct.settings, "PAPER_DIR", tmp_path)
    monkeypatch.setattr(paper_ledger_correct.settings, "PAPER_SEED_CAPITAL", 10_000_000)
    before = (tmp_path / "fills.parquet").read_bytes()

    # When
    paper_ledger_correct.main(_void_argv(sell["order_id"]))

    # Then: append-only 증거는 그대로, 정정은 corrections에만 산다
    assert (tmp_path / "fills.parquet").read_bytes() == before


def test_cli_void_fill_of_referenced_buy_fails_closed(tmp_path, monkeypatch) -> None:
    import pytest

    from src.tools import paper_ledger_correct

    buy, _ = _seed_round_trip(tmp_path)
    monkeypatch.setattr(paper_ledger_correct.settings, "PAPER_DIR", tmp_path)
    monkeypatch.setattr(paper_ledger_correct.settings, "PAPER_SEED_CAPITAL", 10_000_000)
    trades_before = (tmp_path / "trades.parquet").read_bytes()
    nav_before = (tmp_path / "nav.parquet").read_bytes()

    # When: 매도 정산에 참조 중인 매수를 void하면
    with pytest.raises(SystemExit) as exc:
        paper_ledger_correct.main(_void_argv(buy["order_id"]))

    # Then: exit 1, 파생 원장은 손대지 않는다
    assert exc.value.code == 1
    assert not (tmp_path / "corrections.parquet").exists()
    assert (tmp_path / "trades.parquet").read_bytes() == trades_before
    assert (tmp_path / "nav.parquet").read_bytes() == nav_before


def test_cli_void_fill_of_unknown_order_fails_closed(tmp_path, monkeypatch) -> None:
    import pytest

    from src.tools import paper_ledger_correct

    _seed_round_trip(tmp_path)
    monkeypatch.setattr(paper_ledger_correct.settings, "PAPER_DIR", tmp_path)
    monkeypatch.setattr(paper_ledger_correct.settings, "PAPER_SEED_CAPITAL", 10_000_000)

    # When / Then: 모르는 주문 id는 exit 1, corrections 없음
    with pytest.raises(SystemExit) as exc:
        paper_ledger_correct.main(_void_argv("nope"))
    assert exc.value.code == 1
    assert not (tmp_path / "corrections.parquet").exists()


def test_cli_void_fill_with_blank_reason_or_operator_fails_closed(tmp_path, monkeypatch) -> None:
    import pytest

    from src.tools import paper_ledger_correct

    _, sell = _seed_round_trip(tmp_path)
    monkeypatch.setattr(paper_ledger_correct.settings, "PAPER_DIR", tmp_path)
    monkeypatch.setattr(paper_ledger_correct.settings, "PAPER_SEED_CAPITAL", 10_000_000)

    # When / Then: 공백 reason도, 공백 operator도 exit 1이며 정정 행이 남지 않는다
    with pytest.raises(SystemExit) as exc:
        paper_ledger_correct.main(_void_argv(sell["order_id"], reason="  "))
    assert exc.value.code == 1
    with pytest.raises(SystemExit) as exc:
        paper_ledger_correct.main(_void_argv(sell["order_id"], operator="  "))
    assert exc.value.code == 1
    assert not (tmp_path / "corrections.parquet").exists()


def test_cli_full_void_restores_seed_capital_exactly(tmp_path, monkeypatch) -> None:
    import pandas as pd

    from src.execution.paper_broker import PaperLedger
    from src.tools import paper_ledger_correct

    buy, sell = _seed_round_trip(tmp_path)
    monkeypatch.setattr(paper_ledger_correct.settings, "PAPER_DIR", tmp_path)
    monkeypatch.setattr(paper_ledger_correct.settings, "PAPER_SEED_CAPITAL", 10_000_000)
    fills_before = (tmp_path / "fills.parquet").read_bytes()

    # When: 매도 void 후 매수 void (as-of 2026-09-11)
    paper_ledger_correct.main(_void_argv(sell["order_id"]))
    paper_ledger_correct.main(_void_argv(buy["order_id"]))

    # Then: 09-11 NAV가 시드머니로 정확히 복원된다
    nav = pd.read_parquet(tmp_path / "nav.parquet").set_index("as_of_date")
    assert int(nav.loc["2026-09-11", "cash"]) == 10_000_000
    assert int(nav.loc["2026-09-11", "nav"]) == 10_000_000
    assert int(nav.loc["2026-09-11", "cumulative_cost"]) == 0
    assert PaperLedger(root=tmp_path).load_open_positions().empty
    assert pd.read_parquet(tmp_path / "trades.parquet").empty
    corrections = pd.read_parquet(tmp_path / "corrections.parquet")
    assert corrections["correction_id"].tolist() == [
        f"2026-09-11:VOID_FILL:{sell['order_id']}",
        f"2026-09-11:VOID_FILL:{buy['order_id']}",
    ]
    assert (tmp_path / "fills.parquet").read_bytes() == fills_before


def test_cli_note_normalizes_targets_without_economic_effect(tmp_path, monkeypatch) -> None:
    import pandas as pd

    from src.tools import paper_ledger_correct

    _seed_round_trip(tmp_path)
    monkeypatch.setattr(paper_ledger_correct.settings, "PAPER_DIR", tmp_path)
    monkeypatch.setattr(paper_ledger_correct.settings, "PAPER_SEED_CAPITAL", 10_000_000)
    before = {k: (tmp_path / f).read_bytes() for k, f in
              (("fills", "fills.parquet"), ("trades", "trades.parquet"), ("nav", "nav.parquet"))}

    # When
    paper_ledger_correct.main(
        ["note", "--targets", " a, ,b ", "--reason", "retro_doc",
         "--evidence", "manual repair log", "--operator", "tester", "--as-of", "2026-09-24"]
    )

    # Then: 타깃 정규화 + NOTE 기록, 경제 상태 불변
    row = pd.read_parquet(tmp_path / "corrections.parquet").iloc[0]
    assert row["target_order_ids"] == "a,b"
    assert row["action"] == "NOTE"
    for kind, raw in before.items():
        assert (tmp_path / f"{kind}.parquet").read_bytes() == raw


def test_cli_note_propagates_held_ledger_lock_timeout(tmp_path, monkeypatch) -> None:
    import pytest

    from src.execution.paper_broker import PaperLedger
    from src.tools import paper_ledger_correct

    _seed_round_trip(tmp_path)
    monkeypatch.setattr(
        paper_ledger_correct, "PaperLedger",
        lambda *a, **k: PaperLedger(root=tmp_path, lock_timeout_seconds=0.1),
    )

    # When: 다른 홀더가 락을 쥔 채 note를 실행하면
    with PaperLedger(root=tmp_path).exclusive(), pytest.raises(TimeoutError):
        paper_ledger_correct.main(
            ["note", "--targets", "ghost:1", "--reason", "retro_doc",
             "--evidence", "manual repair log", "--operator", "tester", "--as-of", "2026-09-24"]
        )

    # Then: SystemExit이 아닌 TimeoutError이며 corrections 없음
    assert not (tmp_path / "corrections.parquet").exists()


def test_cli_argument_errors_touch_nothing(tmp_path, monkeypatch) -> None:
    import pytest

    from src.tools import paper_ledger_correct

    _seed_round_trip(tmp_path)
    monkeypatch.setattr(paper_ledger_correct.settings, "PAPER_DIR", tmp_path)
    before = _snapshot_bytes(tmp_path)

    # When / Then: 필수 플래그 누락은 exit 2, 디렉터리 내용 불변
    with pytest.raises(SystemExit) as exc:
        paper_ledger_correct.main(["void-fill", "--order-id", "x"])
    assert exc.value.code == 2
    assert _snapshot_bytes(tmp_path) == before


def test_cli_void_fill_logs_correlation_ids(tmp_path, monkeypatch, caplog) -> None:
    import logging

    from src.tools import paper_ledger_correct

    _, sell = _seed_round_trip(tmp_path)
    monkeypatch.setattr(paper_ledger_correct.settings, "PAPER_DIR", tmp_path)
    monkeypatch.setattr(paper_ledger_correct.settings, "PAPER_SEED_CAPITAL", 10_000_000)

    # When
    with caplog.at_level(logging.INFO):
        paper_ledger_correct.main(_void_argv(sell["order_id"]))

    # Then: 상관 id를 담은 성공 로그 1건
    records = [r for r in caplog.records if "stage=paper_ledger_correction" in r.getMessage()]
    assert len(records) == 1
    message = records[0].getMessage()
    assert "action=VOID_FILL" in message
    assert f"correction_id=2026-09-11:VOID_FILL:{sell['order_id']}" in message
