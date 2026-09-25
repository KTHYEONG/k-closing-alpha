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
