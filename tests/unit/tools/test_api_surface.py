"""Invariants for the cross-project API surface manifest."""

from __future__ import annotations

import dataclasses
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from src.tools import api_surface as sf

NOW = datetime(2026, 10, 20, tzinfo=UTC)


def _write(root: Path, rel: str, text: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _tree(base: Path) -> None:
    _write(base, "proj/src/api/ls/client.py", 'TR = "t8412"  # not t84120x\nP = "/stock/chart"\n')
    _write(base, "proj/src/api/kis/client.py", 'ID = "FHKST01010100"\nU = "/uapi/domestic-stock/v1/quotations/inquire-price"\nW = "H0STCNT0"')
    _write(base, "proj/src/api/dbsec/client.py", 'U = "/api/v1/quote/kr-chart/min"\n')
    _write(base, "proj/src/api/shared.py", 'U = "/oauth2/token"\n')
    _write(base, "proj/config/providers.toml", 'tr = "ka10079"\n')


SPEC = sf.ProjectSpec("proj", "proj", ("src/api", "config/providers.toml"))


def test_scan_attributes_trs_by_pattern_and_paths_by_prefix_or_file_hint(tmp_path: Path) -> None:
    _tree(tmp_path)
    tokens = sf.scan_project(SPEC, tmp_path)
    assert tokens["t8412"].brokers == {"ls"} and tokens["t8412"].files == ("src/api/ls/client.py",)
    assert "t84120x" not in tokens
    assert tokens["fhkst01010100"].brokers == {"kis"} and tokens["h0stcnt0"].brokers == {"kis"}
    assert tokens["/uapi/domestic-stock/v1/quotations/inquire-price"].brokers == {"kis"}
    assert tokens["/api/v1/quote/kr-chart/min"].brokers == {"dbsec"}
    assert tokens["/oauth2/token"].brokers == frozenset()
    assert tokens["ka10079"].brokers == {"kiwoom"} and tokens["ka10079"].files == ("config/providers.toml",)
    assert "/stock/chart" not in tokens


def test_extra_tokens_are_attributed_and_must_exist_in_sources(tmp_path: Path) -> None:
    _write(tmp_path, "proj/src/rt/ls.py", 'MAP = {"x": "S3_"}\n')
    spec = sf.ProjectSpec("proj", "proj", ("src/rt",), {"ls": ("S3_",)})
    entry = sf.scan_project(spec, tmp_path)["s3_"]
    assert entry.brokers == {"ls"} and entry.files == ("src/rt/ls.py",)
    stale = sf.ProjectSpec("proj", "proj", ("src/rt",), {"ls": ("ZZ_",)})
    with pytest.raises(ValueError, match="ZZ_"):
        sf.scan_project(stale, tmp_path)


def test_missing_root_or_scan_path_fails_loudly(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="scan path missing"):
        sf.scan_project(SPEC, tmp_path)


def test_surface_queries_split_broker_specific_and_unattributed_tokens(tmp_path: Path) -> None:
    _tree(tmp_path)
    surface = sf.build_surface(tmp_path, [SPEC], now=NOW)
    ls_used = surface.used("ls")["proj"]
    assert {"t8412", "/oauth2/token"} <= ls_used and "fhkst01010100" not in ls_used
    assert surface.attributed("ls") == {"t8412"}
    assert surface.projects_using("kis") == ("proj",) and surface.projects_using("toss") == ()
    assert surface.files("proj", "t8412") == ("src/api/ls/client.py",) and surface.files("nope", "t8412") == ()
    assert surface.age_days(datetime(2026, 10, 30, tzinfo=UTC)) == 10


def test_manifest_round_trips_and_rejects_corruption(tmp_path: Path) -> None:
    _tree(tmp_path)
    surface = sf.build_surface(tmp_path, [SPEC], now=NOW)
    target = tmp_path / "m.json"
    target.write_text(json.dumps(sf.surface_to_json(surface)), encoding="utf-8")
    loaded = sf.load_surface(target)
    assert loaded == surface
    assert sf._comparable(loaded) == sf._comparable(surface)
    for bad in ("{broken", json.dumps({"schema_version": 99}), json.dumps({"schema_version": 1, "generated_at": "x"})):
        target.write_text(bad, encoding="utf-8")
        with pytest.raises(OSError, match="Unreadable"):
            sf.load_surface(target)
    with pytest.raises(OSError, match="Unreadable"):
        sf.load_surface(tmp_path / "missing.json")


def test_committed_manifest_matches_live_kca_sources() -> None:
    """kca is the only project present on every checkout; regenerate with `python -m src.tools.api_surface --sync`."""
    kca = dataclasses.replace(next(spec for spec in sf.PROJECTS if spec.name == "kca"), root=sf.KCA_ROOT.name)
    live = sf.scan_project(kca, sf.KCA_ROOT.parent)
    committed = sf.load_surface().projects["kca"]
    assert {t: (e.brokers, e.files) for t, e in committed.items()} == {t: (e.brokers, e.files) for t, e in live.items()}
