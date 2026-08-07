"""이동평균(SMA/EMA) 및 변동성(historical volatility) 계산 헬퍼."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta

import aiohttp

from src.api.kis.client import KisApiClient

logger = logging.getLogger(__name__)


async def fetch_index_and_calculate_volatility(index_code="1028", session=None):
    """지수 코드를 받아 최근 데이터를 가져와 역사적 변동성(HV)을 계산합니다.
    기본값 1028은 KOSPI 200입니다. KOSDAQ 150은 2203(예상)입니다.
    
    Returns:
        tuple: (hv_today, hv_change)

    """
    import numpy as np
    import pandas as pd
    
    client = KisApiClient()
    
    # 최근 30일 데이터 (영업일 기준 약 21일)
    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=34)).strftime("%Y%m%d") # 여유있게 34일로 늘림
    
    # 세션 관리: 전달받은 세션이 있으면 사용, 없으면 생성
    local_session = False
    if session is None:
        from aiohttp.resolver import ThreadedResolver
        connector = aiohttp.TCPConnector(resolver=ThreadedResolver())
        session = aiohttp.ClientSession(connector=connector)
        local_session = True
        
    try:
        await client.ensure_token(session)
        
        resp = await client.get_market_index_history(
            session, index_code, start_date, end_date
        )
        
        if resp.get('rt_cd') == '0':
            items = resp.get('output2', [])
            
            if len(items) >= 2:
                # 데이터 정리 및 정렬
                records = []
                for item in items:
                    date = item.get('stck_bsop_date')
                    close = float(item.get('bstp_nmix_prpr') or 0)
                    if date and close > 0:
                        records.append({'date': date, 'close': close})
                
                df = pd.DataFrame(records).sort_values('date').reset_index(drop=True)
                
                if len(df) >= 2:
                    # 로그 수익률 계산
                    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
                    
                    # 최근 20일 표준편차 (마지막 행 기준)
                    if len(df) >= 21:
                        recent_returns = df['log_ret'].iloc[-20:]
                    else:
                        recent_returns = df['log_ret'].dropna()
                    
                    std = recent_returns.std()
                    
                    # 연율화 HV
                    hv_today = std * np.sqrt(252) * 100

                    # 어제 HV 계산 (전일 대비 변화율용)
                    if len(df) >= 22:
                        prev_returns = df['log_ret'].iloc[-21:-1]
                        prev_std = prev_returns.std()
                        hv_yesterday = prev_std * np.sqrt(252) * 100
                        hv_change = (hv_today - hv_yesterday) / hv_yesterday if hv_yesterday != 0 else 0
                    else:
                        hv_change = 0

                    return hv_today, hv_change

        return 0.0, 0.0

    finally:
        if local_session:
            await session.close()

async def fetch_kospi200_and_calculate_vkospi():
    """KOSPI 200(1028) 기반 V-KOSPI 계산 (Legacy Wrapper)"""
    return await fetch_index_and_calculate_volatility("1028")


async def calculate_stock_sma(stock_code, sma_period=120, lookback_days=200, session=None):
    """특정 종목의 SMA (단순이동평균)를 계산합니다.
    
    한국투자증권 API는 한 번에 약 100일의 데이터만 반환하므로,
    충분한 데이터를 확보하기 위해 여러 번 호출합니다.
    """
    import pandas as pd
    
    client = KisApiClient()
    all_records = []
    
    local_session = False
    if session is None:
        from aiohttp.resolver import ThreadedResolver
        connector = aiohttp.TCPConnector(resolver=ThreadedResolver())
        session = aiohttp.ClientSession(connector=connector)
        local_session = True
    
    try:
        await client.ensure_token(session)
        
        # 여러 번 API 호출하여 충분한 데이터 확보
        # 한 번에 약 100일씩, 최대 3번 호출 (총 300일치)
        for chunk in range(3):
            # 각 청크의 종료일과 시작일 계산
            end_date = (datetime.now() - timedelta(days=100 * chunk)).strftime("%Y%m%d")
            start_date = (datetime.now() - timedelta(days=100 * (chunk + 1) + 50)).strftime("%Y%m%d")
            
            # API 호출
            resp = await client.get_stock_ohlcv_history(
                session, stock_code, start_date, end_date
            )
            
            if resp.get('rt_cd') != '0':
                if chunk == 0:
                    # 첫 번째 호출 실패면 전체 실패
                    logger.warning(
                        "[SMA Debug] %s 일봉 조회 실패: rt_cd=%s, msg=%s, range=%s~%s, chunk=%s",
                        stock_code,
                        resp.get('rt_cd'),
                        resp.get('msg1', 'N/A'),
                        start_date,
                        end_date,
                        chunk,
                    )
                    return 0.0, False
                else:
                    # 이후 호출 실패는 무시 (이미 충분한 데이터가 있을 수 있음)
                    break
            
            items = resp.get('output2', [])
            
            # 데이터 파싱
            for item in items:
                date = item.get('stck_bsop_date')
                close = float(item.get('stck_clpr') or 0)
                if date and close > 0:
                    all_records.append({'date': date, 'close': close})
            
            # 충분한 데이터를 확보했으면 중단
            if len(all_records) >= sma_period + 10:  # 여유분 10일 추가
                break
            
            # API 부하 방지를 위한 짧은 대기
            await asyncio.sleep(0.1)
        
        if len(all_records) < sma_period:
            # 데이터가 부족하면 실패 (상장 초기 종목 등)
            return 0.0, False
        
        # 데이터프레임 생성 및 정렬
        df = pd.DataFrame(all_records)
        # 중복 제거 (날짜 기준)
        df = df.drop_duplicates(subset=['date'], keep='first')
        df = df.sort_values('date').reset_index(drop=True)
        
        if len(df) < sma_period:
            return 0.0, False
        
        # SMA 계산 (최근 N일의 평균)
        sma_value = df['close'].tail(sma_period).mean()
        
        return float(sma_value), True
            
    except Exception as e:
        logger.warning(
            "[SMA Debug] %s SMA 계산 예외: %s: %s", stock_code, type(e).__name__, e
        )
        return 0.0, False
    finally:
        if local_session:
            await session.close()


async def calculate_stock_ema(stock_code, ema_period=20, lookback_days=60, session=None):
    """특정 종목의 EMA (지수이동평균)를 계산합니다."""
    import pandas as pd
    
    client = KisApiClient()
    
    local_session = False
    if session is None:
        from aiohttp.resolver import ThreadedResolver
        connector = aiohttp.TCPConnector(resolver=ThreadedResolver())
        session = aiohttp.ClientSession(connector=connector)
        local_session = True
    
    try:
        await client.ensure_token(session)
        
        # 과거 데이터 조회
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y%m%d")
        
        # API 호출
        resp = await client.get_stock_ohlcv_history(
            session, stock_code, start_date, end_date
        )
        
        if resp.get('rt_cd') != '0':
            logger.warning(
                "[EMA Debug] %s 일봉 조회 실패: rt_cd=%s, msg=%s, range=%s~%s",
                stock_code,
                resp.get('rt_cd'),
                resp.get('msg1', 'N/A'),
                start_date,
                end_date,
            )
            return 0.0, False, 0
        
        items = resp.get('output2', [])
        if not items:
            logger.warning(
                "[EMA Debug] %s 일봉 응답이 비어 있음: rt_cd=%s, msg=%s, range=%s~%s",
                stock_code,
                resp.get('rt_cd'),
                resp.get('msg1', 'N/A'),
                start_date,
                end_date,
            )
        
        # 데이터 파싱
        records = []
        for item in items:
            date = item.get('stck_bsop_date')
            close = float(item.get('stck_clpr') or 0)
            if date and close > 0:
                records.append({'date': date, 'close': close})
        
        if len(records) < ema_period:
            # 데이터가 부족하면 실패 (상장 초기 종목)
            return 0.0, False, len(records)
        
        # 데이터프레임 생성 및 정렬
        df = pd.DataFrame(records)
        df = df.drop_duplicates(subset=['date'], keep='first')
        df = df.sort_values('date').reset_index(drop=True)
        
        if len(df) < ema_period:
            return 0.0, False, len(df)
        
        # EMA 계산 (지수이동평균)
        ema_value = df['close'].ewm(span=ema_period, adjust=False).mean().iloc[-1]
        
        return float(ema_value), True, len(df)
            
    except Exception as e:
        logger.warning(
            "[EMA Debug] %s EMA 계산 예외: %s: %s", stock_code, type(e).__name__, e
        )
        return 0.0, False, 0
    finally:
        if local_session:
            await session.close()


async def calculate_multiple_emas(stock_code, periods=[5, 10, 20], lookback_days=120, session=None):
    """한 번의 데이터 조회로 여러 EMA를 계산합니다.
    """
    import pandas as pd
    
    client = KisApiClient()
    results = {}
    
    local_session = False
    if session is None:
        from aiohttp.resolver import ThreadedResolver
        connector = aiohttp.TCPConnector(resolver=ThreadedResolver())
        session = aiohttp.ClientSession(connector=connector)
        local_session = True
    
    try:
        await client.ensure_token(session)
        
        # 충분한 과거 데이터 조회
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y%m%d")
        
        resp = await client.get_stock_ohlcv_history(
            session, stock_code, start_date, end_date
        )
        
        if resp.get('rt_cd') != '0':
            return {}
        
        items = resp.get('output2', [])
        records = []
        for item in items:
            date = item.get('stck_bsop_date')
            close = float(item.get('stck_clpr') or 0)
            if date and close > 0:
                records.append({'date': date, 'close': close})
        
        if not records:
            return {}
            
        df = pd.DataFrame(records)
        df = df.drop_duplicates(subset=['date'], keep='first')
        df = df.sort_values('date').reset_index(drop=True)
        
        if len(df) < max(periods):
            return {}
        
        for period in periods:
            if len(df) >= period:
                val = df['close'].ewm(span=period, adjust=False).mean().iloc[-1]
                results[period] = round(float(val), 2)
            else:
                results[period] = 0.0
                
        return results
            
    except Exception:
        return {}
    finally:
        if local_session:
            await session.close()


async def prefetch_ohlcv_for_sma120(
    codes: list[str],
    session: aiohttp.ClientSession,
    client: KisApiClient,
) -> dict[str, list[dict[str, str]]]:
    """SMA120 계산용 OHLCV 이력을 사전 일괄 병렬 선조회한다.

    최근 200역일 범위를 단일 청크로 종목당 1회 호출하여 150건 이상의 데이터를
    확보한다. 실패한 종목은 반환 dict에서 key로 제외되며 예외를 전파하지 않는다.
    """
    if not codes:
        return {}

    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=200)).strftime("%Y%m%d")

    async def _fetch(code: str) -> tuple[str, list[dict[str, str]]] | None:
        try:
            await client.rate_limiter.acquire()
            resp = await client.get_stock_ohlcv_history(
                session, code, start_date, end_date
            )
            if resp.get("rt_cd") != "0":
                return None
            records = [
                {"date": item.get("stck_bsop_date", ""), "close": item.get("stck_clpr", "")}
                for item in resp.get("output2", [])
            ]
            records = [r for r in records if r["date"] and r["close"]]
            return (code, records) if records else None
        except Exception as e:
            logger.warning("[OHLCV Prefetch] %s: %s: %s", code, type(e).__name__, e)
            return None

    results = await asyncio.gather(*[_fetch(code) for code in codes])
    pairs = [item for item in results if item is not None]
    return dict(pairs)


async def calculate_all_moving_averages(
    stock_code: str,
    session: aiohttp.ClientSession | None = None,
    client: KisApiClient | None = None,
    prefetched_records: list[dict[str, str]] | None = None,
) -> tuple:
    """한 종목의 여러 이동평균(EMA 5/10/20, SMA 60/120)을 최소한의 API 호출로 통합 계산함.
    TPS 부하를 줄이기 위해 중복되는 OHLCV 데이터를 한 번에 가져와서 메모리에서 계산함.
    ``prefetched_records``가 주어지면 API 호출 없이 전달된 이력 레코드로만 계산한다.
    """
    from datetime import datetime, timedelta

    import pandas as pd

    all_records = []
    local_session = False

    if prefetched_records is not None:
        for r in prefetched_records:
            date = r.get("date")
            close = r.get("close")
            if not date:
                continue
            try:
                close_val = float(close)
            except (TypeError, ValueError):
                continue
            if close_val > 0:
                all_records.append({"date": date, "close": close_val})
    elif session is None:
        from aiohttp.resolver import ThreadedResolver
        connector = aiohttp.TCPConnector(resolver=ThreadedResolver())
        session = aiohttp.ClientSession(connector=connector)
        local_session = True

    try:
        if prefetched_records is None:
            if client is None:
                client = KisApiClient()
                await client.ensure_token(session)

            # SMA 120까지 계산하기 위해 충분한 데이터 확보
            for chunk in range(2):
                end_dt = datetime.now() - timedelta(days=100 * chunk)
                start_dt = end_dt - timedelta(days=120)
                end_date = end_dt.strftime("%Y%m%d")
                start_date = start_dt.strftime("%Y%m%d")

                resp = await client.get_stock_ohlcv_history(session, stock_code, start_date, end_date)

                if resp.get("rt_cd") != "0":
                    if chunk == 0:
                        return {}, (0.0, False, 0), (0.0, False), (0.0, False)
                    break

                items = resp.get("output2", [])
                for item in items:
                    date = item.get("stck_bsop_date")
                    close = float(item.get("stck_clpr") or 0)
                    if date and close > 0:
                        all_records.append({"date": date, "close": close})

                if len(all_records) >= 150:
                    break

                if chunk < 1:
                    await asyncio.sleep(0.05)

        if not all_records:
            return {}, (0.0, False, 0), (0.0, False), (0.0, False)

        df = pd.DataFrame(all_records).drop_duplicates(subset=["date"], keep="first").sort_values("date").reset_index(drop=True)
        data_count = len(df)

        ema_res = {}
        for period in [5, 10, 20]:
            if data_count >= period:
                val = df["close"].ewm(span=period, adjust=False).mean().iloc[-1]
                ema_res[period] = round(float(val), 2)
            else:
                ema_res[period] = 0.0

        ema20_val = ema_res.get(20, 0.0)
        ema_success = data_count >= 20

        sma60_val = round(float(df["close"].tail(60).mean()), 2) if data_count >= 60 else 0.0
        sma60_ok = data_count >= 60

        sma120_val = round(float(df["close"].tail(120).mean()), 2) if data_count >= 120 else 0.0
        sma120_ok = data_count >= 120

        return ema_res, (ema20_val, ema_success, data_count), (sma60_val, sma60_ok), (sma120_val, sma120_ok)

    except Exception as e:
        logger.warning(
            "[MA Calc Error] %s: %s: %s", stock_code, type(e).__name__, e
        )
        return {}, (0.0, False, 0), (0.0, False), (0.0, False)
    finally:
        if local_session:
            await session.close()
