"""Theme auto-resolver for unmapped stocks.

Implements 5-tier resolution:
Tier1 cache -> Tier2 name rules -> Tier3 Naver content -> Tier4 KIS upjong -> Tier5 fallback
"""

from __future__ import annotations

import logging
import re
import sqlite3
from typing import Any

import pandas as pd

from src import settings

logger = logging.getLogger(__name__)

# 44 user theme categories
_THEME_CATEGORIES: tuple[str, ...] = (
    "2차전지",
    "ai",
    "iot",
    "lcd",
    "si",
    "가상화폐",
    "강관",
    "건설",
    "게임",
    "교육",
    "금융",
    "기타",
    "대북",
    "드론",
    "로봇",
    "메타버스",
    "물류",
    "바이오",
    "반도체",
    "방산",
    "방송",
    "보안",
    "석유",
    "스마트폰",
    "식량",
    "엔터",
    "여행",
    "원자력",
    "의료기기",
    "의류",
    "자동차",
    "저출산",
    "전력",
    "정치",
    "제지",
    "조선",
    "지주사",
    "초전도체",
    "친환경",
    "통신",
    "핀테크",
    "항공",
    "해운",
    "화장품",
)

# Additional keywords per theme for mapping
_THEME_ALIASES: dict[str, list[str]] = {
    "2차전지": ["배터리", "전지", "battery"],
    "ai": ["인공지능"],
    "iot": ["사물인터넷"],
    "si": ["시스템통합", "it서비스"],
    "가상화폐": ["코인", "블록체인", "가상자산"],
    "강관": ["파이프"],
    "건설": ["건축"],
    "금융": ["은행", "증권", "보험", "리츠", "finance"],
    "대북": ["남북", "북한", "개성"],
    "바이오": ["제약", "의약", "헬스케어", "의약품"],
    "반도체": ["반도체장비", "팹리스", "파운드리"],
    "방산": ["방위산업", "무기", "방위"],
    "방송": ["미디어"],
    "보안": ["보안솔루션"],
    "석유": ["정유", "에너지"],
    "스마트폰": ["휴대폰", "핸드폰", "모바일"],
    "식량": ["식품", "음식료"],
    "엔터": ["연예", "콘텐츠", "음악"],
    "여행": ["관광"],
    "원자력": ["원전", "원자로"],
    "의료기기": ["의료장비", "헬스케어기기"],
    "의류": ["섬유", "의복", "패션"],
    "자동차": ["모빌리티", "차량", "자동차부품"],
    "저출산": ["출산", "육아"],
    "전력": ["전기"],
    "정치": ["정치인", "테마주"],
    "제지": ["종이", "펄프"],
    "조선": ["선박", "조선해양"],
    "지주사": ["지주", "홀딩스"],
    "초전도체": ["초전도"],
    "친환경": ["환경", "신재생", "탄소", "친환경"],
    "통신": ["통신장비", "텔레콤"],
    "핀테크": ["금융기술"],
    "항공": ["공항", "항공사"],
    "해운": ["선박", "물류해운"],
    "화장품": ["뷰티", "cosmetic", "화장"],
}


def _normalize_market(kis_market: str) -> str:
    """Normalize market name to KOSPI or KOSDAQ."""
    if not kis_market:
        return "KOSDAQ"
    upper = kis_market.upper()
    raw = str(kis_market)
    if "KOSPI" in upper or "유가" in raw:
        return "KOSPI"
    if "KOSDAQ" in upper or "코스닥" in raw or "KSQ" in upper:
        return "KOSDAQ"
    # Fallback heuristic: if contains 1001 or similar?
    if "KOSDAQ" in upper or "KSQ" in upper:
        return "KOSDAQ"
    return "KOSDAQ" if "KOSDAQ" in upper else "KOSPI" if "KOSPI" in upper else "KOSDAQ"


def _normalize_code(code: str) -> str:
    return str(code).strip().split(".")[0].zfill(6)


def _load_existing_theme_map() -> dict[str, str]:
    try:
        from src.data.data_loader import load_theme

        return load_theme()
    except Exception as e:
        logger.debug("Failed to load existing theme map: %s", e)
        return {}


def _match_name_rule(name: str) -> str | None:
    if not name:
        return None
    upper = name.upper()
    # SPAC
    if "스팩" in name or "SPAC" in upper:
        return "기타"
    if "홀딩스" in name or "지주" in name:
        return "지주사"
    if "리츠" in name or "REIT" in upper:
        return "금융"
    return None


def _map_text_to_theme(text: str) -> str | None:
    if not text:
        return None
    lower = text.lower()
    for theme in _THEME_CATEGORIES:
        if theme == "기타":
            continue
        keywords = [theme.lower()] + [k.lower() for k in _THEME_ALIASES.get(theme, [])]
        for kw in keywords:
            if kw and kw in lower:
                return theme
    return None


def _map_content_to_theme(upjong: str, summary: str, kis_upjong: str) -> str | None:
    combined = f"{upjong} {summary} {kis_upjong}"
    return _map_text_to_theme(combined)


def _fetch_naver_stock_metadata(code: str, timeout: float = 3.0) -> dict[str, str]:
    """Fetch Naver Finance WICS upjong and summary."""
    try:
        import requests

        url = f"https://finance.naver.com/item/main.naver?code={code}"
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        html = resp.text
        # Extract upjong
        upjong = ""
        m = re.search(r"업종.*?<a[^>]*>([^<]+)</a>", html, re.DOTALL)
        if m:
            upjong = m.group(1).strip()
        # Summary fallback: search for summary section
        summary = ""
        m2 = re.search(r'<p[^>]*class="summary"[^>]*>(.*?)</p>', html, re.DOTALL)
        if not m2:
            m2 = re.search(r'<div[^>]*id="summary"[^>]*>(.*?)</div>', html, re.DOTALL)
        if m2:
            summary = re.sub(r"<[^>]+>", "", m2.group(1)).strip()
            summary = re.sub(r"\s+", " ", summary)
        return {"upjong": upjong, "summary": summary}
    except Exception as e:
        logger.debug("Naver fetch failed for %s: %s", code, e)
        return {}


def resolve_stock_theme_and_market(
    code: str,
    name: str = "",
    kis_market: str = "",
    kis_upjong: str = "",
    timeout: float = 3.0,
    custom_theme_map: dict[str, str] | None = None,
) -> tuple[str, str]:
    """Resolve theme and market for a stock via 5-tier logic."""
    code_norm = _normalize_code(code)
    market_norm = _normalize_market(kis_market)

    # Tier 1 Cache
    if custom_theme_map is not None:
        if code_norm in custom_theme_map:
            return custom_theme_map[code_norm], market_norm
    else:
        existing = _load_existing_theme_map()
        if existing.get(code_norm):
            return existing[code_norm], market_norm

    # Tier 2 Name Rules
    rule_theme = _match_name_rule(name)
    if rule_theme is not None:
        return rule_theme, market_norm

    # Tier 3 Naver Content
    try:
        meta = _fetch_naver_stock_metadata(code_norm, timeout=timeout)
    except Exception:
        meta = {}
    upjong = str(meta.get("upjong", "")) if meta else ""
    summary = str(meta.get("summary", "")) if meta else ""
    mapped = _map_content_to_theme(upjong, summary, kis_upjong)
    if mapped is not None and mapped in _THEME_CATEGORIES:
        return mapped, market_norm

    # Tier 4 KIS Upjong (already combined in tier3, but try isolated kis_upjong)
    if kis_upjong:
        kis_mapped = _map_text_to_theme(kis_upjong)
        if kis_mapped is not None and kis_mapped in _THEME_CATEGORIES:
            return kis_mapped, market_norm

    # Tier 5 Fallback
    return "기타", market_norm


def batch_resolve_missing_themes(
    stocks: list[dict[str, Any]], sync_gsheet: bool = False
) -> list[dict[str, Any]]:
    """Batch resolve missing themes and persist atomically to local Parquet and SQLite DB."""
    resolved: list[dict[str, Any]] = []
    for stock in stocks:
        code = _normalize_code(str(stock.get("종목코드") or stock.get("code") or ""))
        name = str(stock.get("종목명") or stock.get("name") or "")
        kis_market = str(stock.get("시장구분") or stock.get("market") or stock.get("kis_market") or "")
        kis_upjong = str(stock.get("업종") or stock.get("industry") or stock.get("kis_upjong") or "")
        # Batch inputs are pre-filtered as missing, so bypass cache to force tier2-5 resolution
        theme, market = resolve_stock_theme_and_market(code, name, kis_market, kis_upjong, custom_theme_map={})
        new_entry = dict(stock)
        new_entry["종목코드"] = code
        new_entry["종목명"] = name
        new_entry["테마"] = theme
        new_entry["시장구분"] = market
        resolved.append(new_entry)

    if not resolved:
        return resolved

    # Persistence: parquet
    try:
        parquet_path = settings.THEME_PARQUET_PATH
        if parquet_path.exists():
            df_existing = pd.read_parquet(parquet_path)
        else:
            df_existing = pd.DataFrame(columns=["종목코드", "테마", "시장구분"])
        df_new = pd.DataFrame(
            [{"종목코드": r["종목코드"], "테마": r["테마"], "시장구분": r["시장구분"]} for r in resolved]
        )
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined["종목코드"] = df_combined["종목코드"].astype(str).str.strip().str.split(".").str[0].str.zfill(6)
        df_combined = df_combined.drop_duplicates(subset=["종목코드"], keep="last")
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = parquet_path.with_suffix(".tmp.parquet")
        df_combined.to_parquet(tmp_path, index=False)
        import os

        os.replace(tmp_path, parquet_path)
        logger.info("Updated theme parquet with %d entries", len(resolved))
    except Exception as e:
        logger.warning("Failed to update theme parquet: %s", e)

    # Persistence: SQLite
    try:
        db_path = settings.STOCK_DB_PATH
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path))
        try:
            cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='table_theme'")
            if cur.fetchone() is None:
                conn.execute("CREATE TABLE table_theme (종목코드 TEXT PRIMARY KEY, 테마 TEXT, 시장구분 TEXT)")
            else:
                cols = [row[1] for row in conn.execute("PRAGMA table_info(table_theme)")]
                if "시장구분" not in cols:
                    conn.execute("ALTER TABLE table_theme ADD COLUMN 시장구분 TEXT")
                if "테마" not in cols:
                    conn.execute("ALTER TABLE table_theme ADD COLUMN 테마 TEXT")
            for r in resolved:
                conn.execute(
                    "INSERT OR REPLACE INTO table_theme (종목코드, 테마, 시장구분) VALUES (?, ?, ?)",
                    (r["종목코드"], r["테마"], r["시장구분"]),
                )
            conn.commit()
        finally:
            conn.close()
    except Exception as e:
        logger.warning("Failed to update theme DB: %s", e)

    # Optional GSheet sync (only if explicitly enabled)
    if sync_gsheet:
        try:
            from src.data.gsheet_loader import append_stocks_to_gsheet

            key_path = str(settings.GOOGLE_KEY_PATH)
            sheet_name = settings.GOOGLE_SHEET_NAME
            ws_name = settings.THEME_WORKSHEET_NAME
            append_stocks_to_gsheet(key_path, sheet_name, ws_name, resolved)
        except Exception as e:
            logger.warning("Failed to sync gsheet: %s", e)

    return resolved
