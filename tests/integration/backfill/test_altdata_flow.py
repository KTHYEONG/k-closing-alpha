import json

import pandas as pd

import src.backfill.altdata.runner as runner
from src.backfill.altdata.config import AltDataFetchConfig


def test_run_altdata_backfill_is_fail_soft_per_source(monkeypatch, tmp_path) -> None:
    good = pd.DataFrame({"date": pd.to_datetime(["2024-01-02"]), "symbol": ["005930"], "per": [10.0], "pbr": [1.0], "eps": [1.0], "bps": [1.0], "div_yield": [1.0], "dps": [1.0]})
    monkeypatch.setattr(runner, "collect_fundamental", lambda *a, **k: good)

    def _boom(*a, **k):
        raise RuntimeError("krx blocked")

    monkeypatch.setattr(runner, "collect_shorting", _boom)
    cfg = AltDataFetchConfig(
        start=pd.Timestamp("2024-01-02"), end=pd.Timestamp("2024-01-04"),
        out_dir=tmp_path, sources=("shorting", "fundamental", "disclosure"),
        dart_api_key="", retries=1, retry_sleep_sec=0.0,
    )
    manifest = runner.run_altdata_backfill(cfg)
    assert manifest["panels"]["shorting"]["status"] == "unavailable"
    assert manifest["panels"]["fundamental"]["status"] == "ok"
    assert manifest["panels"]["disclosure"]["status"] == "skipped_no_key"
    assert (tmp_path / "fundamental.parquet").exists()
    assert not (tmp_path / "shorting.parquet").exists()
    saved = json.loads((tmp_path / "_manifest.json").read_text())
    assert saved["panels"]["fundamental"]["availability_rule"] == "eod_release_next_decision"


import src.backfill.backfill_altdata as cli


def test_backfill_altdata_main_wires_runner(monkeypatch, tmp_path) -> None:
    seen = {}

    def _fake_runner(cfg):
        seen["cfg"] = cfg
        return {"panels": {}}

    monkeypatch.setattr(cli, "run_altdata_backfill", _fake_runner)
    cli.main([
        "--source", "shorting",
        "--start", "2024-01-02",
        "--end", "2024-01-10",
        "--out-dir", str(tmp_path),
    ])
    cfg = seen["cfg"]
    assert cfg.sources == ("shorting",)
    assert cfg.start == pd.Timestamp("2024-01-02")
    assert str(cfg.out_dir) == str(tmp_path)
