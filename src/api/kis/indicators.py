"""Historical-volatility helper for KIS index series (V-KOSPI proxy attached by collect)."""

from __future__ import annotations

from datetime import datetime, timedelta

import aiohttp

from src.api.kis.client import KisApiClient, kis_data_client_kwargs


async def fetch_index_and_calculate_volatility(index_code="1028", session=None):
    """지수 코드를 받아 최근 데이터를 가져와 역사적 변동성(HV)을 계산합니다.
    기본값 1028은 KOSPI 200입니다. KOSDAQ 150은 2203(예상)입니다.
    
    Returns:
        tuple: (hv_today, hv_change)

    """
    import numpy as np
    import pandas as pd
    
    client = KisApiClient(**kis_data_client_kwargs())
    
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
