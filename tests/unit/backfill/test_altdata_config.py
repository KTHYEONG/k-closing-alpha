from pathlib import Path

import pandas as pd
import pytest

from src.backfill.altdata.config import AltDataFetchConfig


def test_altdata_config_rejects_invalid_domains() -> None:
    ok = AltDataFetchConfig(start=pd.Timestamp("2020-01-01"), end=pd.Timestamp("2020-02-01"), out_dir=Path("x"))
    assert ok.sources[0] == "shorting"
    with pytest.raises(ValueError, match="start"):
        AltDataFetchConfig(start=pd.Timestamp("2020-02-01"), end=pd.Timestamp("2020-01-01"), out_dir=Path("x"))
    with pytest.raises(ValueError, match="source"):
        AltDataFetchConfig(start=pd.Timestamp("2020-01-01"), end=pd.Timestamp("2020-02-01"), out_dir=Path("x"), sources=("orderbook",))
    with pytest.raises(ValueError, match="market"):
        AltDataFetchConfig(start=pd.Timestamp("2020-01-01"), end=pd.Timestamp("2020-02-01"), out_dir=Path("x"), markets=("NASDAQ",))
    with pytest.raises(ValueError, match="krx_requests_per_sec"):
        AltDataFetchConfig(start=pd.Timestamp("2020-01-01"), end=pd.Timestamp("2020-02-01"), out_dir=Path("x"), krx_requests_per_sec=0.0)


def test_altdata_config_carries_krx_api_key() -> None:
    cfg = AltDataFetchConfig(
        start=pd.Timestamp("2020-01-01"), end=pd.Timestamp("2020-02-01"),
        out_dir=Path("x"), krx_api_key="AUTHKEY", krx_requests_per_sec=3.0,
    )
    assert cfg.krx_api_key == "AUTHKEY"
    assert cfg.krx_requests_per_sec == 3.0


def test_altdata_config_defaults_to_single_key() -> None:
    cfg = AltDataFetchConfig(start=pd.Timestamp("2020-01-01"), end=pd.Timestamp("2020-02-01"), out_dir=Path("x"))
    assert cfg.extra_client_kwargs == ()


def test_altdata_config_accepts_valid_extra_keys() -> None:
    cfg = AltDataFetchConfig(
        start=pd.Timestamp("2020-01-01"),
        end=pd.Timestamp("2020-02-01"),
        out_dir=Path("x"),
        extra_client_kwargs=(("key2", "sec2", "hts2"),),
    )
    assert cfg.extra_client_kwargs == (("key2", "sec2", "hts2"),)


def test_altdata_config_rejects_empty_app_key() -> None:
    with pytest.raises(ValueError, match="non-empty app_key"):
        AltDataFetchConfig(
            start=pd.Timestamp("2020-01-01"),
            end=pd.Timestamp("2020-02-01"),
            out_dir=Path("x"),
            extra_client_kwargs=(("", "sec2", "hts2"),),
        )


def test_altdata_config_rejects_bad_tuple_length() -> None:
    with pytest.raises(ValueError, match="extra_client_kwargs"):
        AltDataFetchConfig(
            start=pd.Timestamp("2020-01-01"),
            end=pd.Timestamp("2020-02-01"),
            out_dir=Path("x"),
            extra_client_kwargs=(("key2", "sec2"),),
        )


import pathlib


def test_altdata_package_has_no_ml_or_realtime_imports() -> None:
    bad = []
    for p in pathlib.Path("src/backfill/altdata").rglob("*.py"):
        t = p.read_text(encoding="utf-8")
        if "src.ml" in t or "src.serving" in t:
            bad.append((str(p), "ml/serving import"))
        if "inquire-asking-price" in t or "websocket" in t.lower():
            bad.append((str(p), "realtime endpoint"))
    assert bad == [], bad


def test_short_code_accepts_numeric_and_alphanumeric() -> None:
    from src.backfill.altdata.config import is_krx_short_code

    assert is_krx_short_code("005930")
    assert is_krx_short_code("0009K0")
    assert is_krx_short_code("00088K")


def test_short_code_rejects_non_short_forms() -> None:
    from src.backfill.altdata.config import is_krx_short_code

    for bad in ("A005930", "12345", "Q500001", "0009k0", ""):
        assert not is_krx_short_code(bad)
    assert not is_krx_short_code(None)


def test_config_accepts_alphanumeric_universe() -> None:
    from pathlib import Path

    import pandas as pd

    from src.backfill.altdata.config import AltDataFetchConfig

    cfg = AltDataFetchConfig(
        start=pd.Timestamp("2020-01-01"),
        end=pd.Timestamp("2020-02-01"),
        out_dir=Path("x"),
        universe_symbols=frozenset({"005930", "0013V0"}),
    )

    assert cfg.universe_symbols == frozenset({"005930", "0013V0"})


def test_config_rejects_malformed_universe_symbol() -> None:
    from pathlib import Path

    import pandas as pd
    import pytest

    from src.backfill.altdata.config import AltDataFetchConfig

    with pytest.raises(ValueError, match="short code"):
        AltDataFetchConfig(
            start=pd.Timestamp("2020-01-01"),
            end=pd.Timestamp("2020-02-01"),
            out_dir=Path("x"),
            universe_symbols=frozenset({"12345"}),
        )


def _altdata_cfg(**kw):
    return AltDataFetchConfig(start=pd.Timestamp("2026-01-02"), end=pd.Timestamp("2026-01-05"), out_dir=Path("altdata-out"), **kw)


def test_altdata_config_invalid_extra_entry_message_is_index_only() -> None:
    with pytest.raises(ValueError, match="extra_client_kwargs") as excinfo:
        _altdata_cfg(extra_client_kwargs=(("k1", "s1", "h1"), ("", "SUPER-SECRET-VALUE", "h2")))
    message = str(excinfo.value)
    assert "#1" in message
    assert "extra_client_kwargs" in message
    assert "non-empty app_key" in message
    for leaked in ("SUPER-SECRET-VALUE", "h2", "k1"):
        assert leaked not in message


def test_altdata_config_wrong_arity_message_is_index_only() -> None:
    with pytest.raises(ValueError, match="extra_client_kwargs") as excinfo:
        _altdata_cfg(extra_client_kwargs=(("KEY-ONLY-VALUE", "SECRET-ONLY-VALUE"),))
    message = str(excinfo.value)
    assert "#0" in message
    assert "KEY-ONLY-VALUE" not in message
    assert "SECRET-ONLY-VALUE" not in message


def test_altdata_config_repr_hides_credentials() -> None:
    import dataclasses

    cfg = _altdata_cfg(
        dart_api_key="DART-SECRET-KEY-1",
        krx_api_key="KRX-SECRET-KEY-1",
        extra_client_kwargs=(("AK-EXTRA-1", "AS-EXTRA-1", "hts"),),
    )
    text = repr(cfg)
    for leaked in ("DART-SECRET-KEY-1", "KRX-SECRET-KEY-1", "AK-EXTRA-1", "AS-EXTRA-1"):
        assert leaked not in text
    assert "altdata-out" in text
    replaced = dataclasses.replace(cfg, page_count=50)
    assert replaced.dart_api_key == "DART-SECRET-KEY-1"
    assert replaced.krx_api_key == "KRX-SECRET-KEY-1"
    assert replaced.extra_client_kwargs == (("AK-EXTRA-1", "AS-EXTRA-1", "hts"),)
