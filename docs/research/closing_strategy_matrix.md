# 국내주식 종가매매 차세대 알파 연구 매트릭스 (Next-Gen Alpha Matrix)

> **문서 상태:** 활성 연구 청사진 (Active Research Specification)  
> **핵심 규약:** 100% 무누출 Point-in-Time (PIT) 15:20 가용 정보 + Combinatorial Purged Cross-Validation (CPCV 8,2) 검증  
> **체결 현실성:** 15:30 종가 매수 $\rightarrow$ 익일 09:00 시초가 기계적 전량 청산 (사후 갭 필터 등 사후 편향 전면 배제)  
> **비용 모델:** PIT KRX 법정거래세(18~23bp) + 왕복 2.0틱 호가 스프레드 크로싱 비용 전액 엄밀 차감  
> **총 탐색 전략수:** 118개 실전 무누출 전략 (Core 35 + Edge 32 + Granular 51) / 검증 기간: 2023-01-25 ~ 2026-09-10 (886 거래일, 68,493 단면)

---

## 1. 베이스라인 및 거버넌스 불변 팩트

| 항목 | 프로덕션 채택 규격 | 근거 및 실증 검증 결과 |
| :--- | :--- | :--- |
| **틱비용 상한** | `MAX_TICK_COST_BP = 12.0bp` | 7.5bp 대비 후보군 2.6배 확대, Sharpe +0.84 우위, K=1~8 전 구간 파레토 지배 |
| **번들 스크린 가드** | `assert_bundle_screen_parity` | 모델 번들과 라이브 유니버스 스펙 간 1bp라도 괴리 발생 시 Fail-Closed 차단 |
| **청산 프로토콜** | 09:00 시초가 기계적 청산 | 장중 익절/오후 청산 대비 일평균 +38bp ~ +58bp 우위 (CPCV 28/28 무패 검증) |
| **체결 현실성 규약** | 사후 갭 필터 전면 금지 | 15:20 이후 익일 시초가 정보 참조는 실전 체결 불가능한 환각 편향으로 전면 배제 |

---

## 2. 100% 무누출 핵심 탐색 매트릭스 (Core 35 Strategies)

| ID | 카테고리 | 전략명 및 핵심 조건 | Net (bp/일) | Sharpe | t-stat | 승률 (%) | MDD (%) | 활동일 (%) |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `DIM1-01` | 1. Novel ML | ElasticNet (L1+L2 Regularized) | -12.27 | -1.13 | -2.12 | 41.8% | 76.18% | 99.7% |
| `DIM1-02` | 1. Novel ML | Huber Regressor (Robust Loss) | -8.81 | -0.87 | -1.63 | 43.0% | 73.38% | 99.7% |
| `DIM1-03` | 1. Novel ML | ExtraTrees Ensemble (Extreme Random Forest) | +17.98 | 1.48 | 2.77 | 53.9% | 26.39% | 99.7% |
| `DIM1-04` | 1. Novel ML | LGBM Regression L1 (Median/MAE Optimizer) | +26.77 | 2.39 | 4.49 | 57.3% | 30.72% | 99.7% |
| `DIM1-05` | 1. Novel ML | LGBM Quantile Q10 (Downside Floor Maximizer) | -1.06 | -0.14 | -0.27 | 46.5% | 45.63% | 99.7% |
| `DIM1-06` | 1. Novel ML | LGBM Quantile Q90 (Explosive Upside Hunter) | -16.56 | -1.11 | -2.08 | 41.3% | 89.97% | 99.7% |
| `DIM2-01` | 2. Blended Loss | Safe EV Blend (Base EV + 0.5*Q10 Floor) | +28.27 | 2.63 | 4.94 | 58.2% | 25.87% | 99.7% |
| `DIM2-02` | 2. Blended Loss | Cross-Sectional Point Sharpe Score (EV / Spread) | +28.20 | 2.44 | 4.58 | 58.2% | 21.28% | 99.7% |
| `DIM2-03` | 2. Blended Loss | Asymmetric Defense Blend (Base 70% + Asym 30%) | +31.41 | 2.68 | 5.02 | 58.2% | 21.27% | 99.7% |
| `DIM2-04` | 2. Blended Loss | Quad-Model Multi-Family Consensus | +17.78 | 1.70 | 3.20 | 51.9% | 17.82% | 99.7% |
| `DIM3-01` | 3. Weighting | Base Top-3 Linear Rank Tilt (50%-33%-17%) | +32.79 | 2.49 | 4.66 | 55.5% | 26.68% | 99.7% |
| `DIM3-02` | 3. Weighting | **Base Top-3 Quadratic Tilt (Conviction Focus)** | **+33.90** | **2.35** | 4.40 | 53.4% | 27.89% | 99.7% |
| `DIM3-03` | 3. Weighting | **Base Top-3 Friction-Adjusted Kelly Sizing** | **+34.81** | **2.62** | 4.91 | 56.2% | 31.64% | 99.7% |
| `DIM3-04` | 3. Weighting | Base Top-3 Liquidity Proportional Weighting | +30.10 | 2.13 | 3.65 | 56.0% | 32.99% | 99.7% |
| `DIM3-05` | 3. Weighting | Quad-Model Consensus + Inv-Tick Weighting | +22.88 | 2.11 | 3.96 | 51.8% | 15.95% | 99.7% |
| `DIM3-06` | 3. Weighting | Quad-Model Consensus + Linear Rank Tilt | +20.57 | 1.95 | 3.66 | 51.7% | 20.30% | 99.7% |
| `DIM4-01` | 4. Microstructure | Base Top-3 + Program Net Buy Density > 0 | +25.14 | 2.15 | 4.04 | 54.3% | 29.53% | 97.4% |
| `DIM4-02` | 4. Microstructure | Base Top-3 + Smart Money Net Inflow | +30.69 | 2.70 | 5.06 | 55.8% | 22.38% | 98.8% |
| `DIM4-03` | 4. Microstructure | **Base Top-3 + Positive Morning Gap (Open >= PrevClose)** | **+33.61** | **3.07** | 5.75 | 57.2% | 18.60% | 98.1% |
| `DIM4-04` | 4. Microstructure | Base Top-3 + Clean Candle (Upper Shadow < 35%) | +13.33 | 1.15 | 2.15 | 53.6% | 42.50% | 96.8% |
| `DIM4-05` | 4. Microstructure | Base Top-3 + Institutional Relative Rank >= 0.5 | +31.33 | 2.64 | 4.95 | 57.3% | 28.13% | 99.3% |
| `DIM5-01` | 5. Macro Brake | Base Top-3 + Fast Trend Gate (KOSPI > SMA5) | +22.24 | 2.51 | 4.70 | 27.7% | 20.24% | 48.3% |
| `DIM5-02` | 5. Macro Brake | Base Top-3 + Dual Index Breadth (KOSPI & KOSDAQ > 0) | +19.73 | 2.49 | 4.66 | 23.9% | 13.90% | 42.3% |
| `DIM5-03` | 5. Macro Brake | Base Top-3 + Low Vol Regime Gate (VKOSPI <= 20) | +15.89 | 2.00 | 3.75 | 36.6% | 25.03% | 66.0% |
| `DIM5-04` | 5. Macro Brake | Base Top-3 + Moderate Dip Buying (-1.5% <= KOSPI < 0%) | +3.88 | 0.49 | 0.91 | 24.3% | 21.07% | 43.1% |
| `DIM6-01` | 6. Universe Screen | Tight Tick Friction Screen (Cap = 6.0 bp) | +22.96 | 2.39 | 4.47 | 40.5% | 18.89% | 71.2% |
| `DIM6-02` | 6. Universe Screen | Medium Tick Friction Screen (Cap = 9.0 bp) | +28.78 | 2.65 | 4.96 | 46.4% | 25.55% | 82.7% |
| `DIM6-03` | 6. Universe Screen | **Expanded Tick Friction Screen (Cap = 15.0 bp)** | **+34.04** | **2.88** | 5.40 | 47.2% | 26.45% | 83.3% |
| `DIM6-04` | 6. Universe Screen | Uncapped Tick Friction Screen (Cap = None) | +28.69 | 2.42 | 4.54 | 45.6% | 26.96% | 83.3% |
| `DIM6-05` | 6. Universe Screen | Medium Cap (9bp) + Inv-Tick Weighting | +29.68 | 2.71 | 5.09 | 46.4% | 24.75% | 82.7% |
| `DIM6-06` | 6. Universe Screen | **Current Cap (12bp) + Inv-Tick Weighting** | **+34.58** | **2.73** | 5.12 | 56.3% | 28.98% | 99.7% |
| `DIM7-01` | 7. Pure Synergy | Quad-Consensus + Clean Candle + Inv-Tick | +0.73 | 0.08 | 0.15 | 46.7% | 52.71% | 96.8% |
| `DIM7-02` | 7. Pure Synergy | Quad-Consensus + Smart Money + Linear Tilt | +16.05 | 1.65 | 3.10 | 51.0% | 20.82% | 98.8% |
| `DIM7-03` | 7. Pure Synergy | Safe EV + Top-2 Inv-Tick Weighting | +30.32 | 2.23 | 4.18 | 57.1% | 32.00% | 99.8% |
| `DIM7-04` | 7. Pure Synergy | Safe EV + Linear Tilt (K=3) | +28.62 | 2.48 | 4.66 | 58.1% | 27.01% | 99.7% |

---

## 3. 구조적 엣지 케이스 매트릭스 (Seasonality, Market Split & Friction - 32 Strategies)

| ID | 카테고리 | 전략명 및 핵심 조건 | Net (bp/일) | Sharpe | t-stat | 승률 (%) | MDD (%) | 활동일 (%) |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `EDGE-01` | 1. Calendar Seasonality | Skip Friday (Weekend Gap Risk Elimination) | +25.04 | 2.26 | 4.24 | 45.1% | 22.06% | 79.5% |
| `EDGE-02` | 1. Calendar Seasonality | Friday Only Overnight (Weekend Arbitrage) | +7.32 | 1.67 | 3.13 | 12.3% | 8.17% | 20.2% |
| `EDGE-03` | 1. Calendar Seasonality | Monday Only (Post-Weekend Rebound Trade) | +6.04 | 1.00 | 1.87 | 11.7% | 16.49% | 19.2% |
| `EDGE-04` | 1. Calendar Seasonality | Mid-Week Core (Tue-Thu Only) | +19.00 | 2.04 | 3.83 | 33.4% | 19.51% | 60.3% |
| `EDGE-05` | 1. Calendar Seasonality | Month-End Window Dressing Days (Day >= 27) | +5.60 | 1.34 | 2.51 | 8.1% | 9.37% | 14.6% |
| `EDGE-06` | 1. Calendar Seasonality | Normal Days Only (Exclude Month-End) | +26.77 | 2.40 | 4.51 | 49.3% | 19.95% | 85.1% |
| `EDGE-07` | 2. Market Segment | KOSPI Only Candidates (Large-Cap Institutional Focus) | +12.16 | 1.15 | 2.15 | 49.9% | 41.82% | 96.8% |
| `EDGE-08` | 2. Market Segment | KOSDAQ Only Candidates (High Beta Retail Momentum) | +24.64 | 1.98 | 3.70 | 55.2% | 26.85% | 99.2% |
| `EDGE-09` | 2. Market Segment | Market Pairing Guard (Force 1 KOSPI + 2 KOSDAQ) | +27.51 | 2.28 | 4.27 | 55.4% | 24.17% | 99.7% |
| `EDGE-10` | 2. Market Segment | KOSDAQ Only + Inv-Tick Weighting | +28.20 | 2.15 | 4.04 | 54.4% | 30.23% | 99.2% |
| `EDGE-11` | 2. Market Segment | KOSPI Only + Inv-Tick Weighting | +11.47 | 1.04 | 1.96 | 49.3% | 47.82% | 96.8% |
| `EDGE-12` | 3. Volume & Momentum | Volume Explosion Surge (TradeValue / MA5 >= 2.0x) | +20.83 | 1.96 | 3.68 | 45.3% | 25.11% | 83.1% |
| `EDGE-13` | 3. Volume & Momentum | Ultra High Liquidity (TradeValue >= 500억원) | +9.18 | 0.88 | 1.66 | 39.8% | 39.14% | 79.8% |
| `EDGE-14` | 3. Volume & Momentum | Mid-Range Liquidity (100억 <= TV < 300억) | +13.19 | 1.46 | 2.74 | 42.8% | 22.79% | 81.7% |
| `EDGE-15` | 3. Volume & Momentum | Strong Intraday Drive ((Close - Open)/Range >= 0.6) | +10.28 | 1.05 | 1.96 | 47.2% | 23.48% | 90.0% |
| `EDGE-16` | 3. Volume & Momentum | Low Intraday Volatility Squeeze (Range < 5%) | +1.27 | 0.18 | 0.33 | 38.4% | 38.08% | 74.8% |
| `EDGE-17` | 3. Volume & Momentum | High Range Breakout (Range >= 8%) | +16.88 | 1.26 | 2.37 | 50.9% | 47.74% | 99.5% |
| `EDGE-18` | 4. Conviction Margin | Conviction Margin >= 10bp (1st-3rd Spread) | +22.59 | 2.18 | 4.08 | 40.6% | 21.23% | 71.3% |
| `EDGE-19` | 4. Conviction Margin | Conviction Margin >= 20bp (Strong Leader Spread) | +12.26 | 1.45 | 2.72 | 21.7% | 16.49% | 36.2% |
| `EDGE-20` | 4. Conviction Margin | Conviction Margin >= 30bp (Dominant Leader) | +3.09 | 0.46 | 0.87 | 9.9% | 20.65% | 17.3% |
| `EDGE-21` | 5. Drawdown Braking | **Consecutive Bear Market Gate (Skip if KOSPI down 3+ days)** | **+32.58** | **2.81** | 5.27 | 53.6% | 21.14% | 92.1% |
| `EDGE-22` | 5. Drawdown Braking | Dip Buying Relief (Trade ONLY when KOSPI down 2+ days) | +1.92 | 0.48 | 0.89 | 9.4% | 11.10% | 18.5% |
| `EDGE-23` | 5. Drawdown Braking | Extreme Volatility Escape (Skip if VKOSPI > 25) | +19.84 | 2.41 | 4.53 | 42.0% | 23.29% | 73.6% |
| `EDGE-24` | 6. Friction Sensitivity | **Optimal PA Execution (1.0 Round-Trip Tick)** | **+41.92** | **3.54** | 6.64 | 60.2% | 18.92% | 99.7% |
| `EDGE-25` | 6. Friction Sensitivity | Standard AA Baseline (2.0 Round-Trip Ticks) | +32.36 | 2.73 | 5.12 | 57.5% | 20.63% | 99.7% |
| `EDGE-26` | 6. Friction Sensitivity | 1.5x Friction Stress (3.0 Round-Trip Ticks) | +22.80 | 1.92 | 3.61 | 55.1% | 24.00% | 99.7% |
| `EDGE-27` | 6. Friction Sensitivity | 2.0x Friction Severe Stress (4.0 Round-Trip Ticks) | +13.24 | 1.12 | 2.09 | 52.4% | 36.44% | 99.7% |
| `EDGE-28` | 6. Friction Sensitivity | 3.0x Extreme Stress (6.0 Round-Trip Ticks) | -5.88 | -0.50 | -0.93 | 45.1% | 72.66% | 99.7% |
| `EDGE-29` | 7. Refined Combos | Mid-Week + Inv-Tick + Day Gap >= 0 | +16.53 | 1.90 | 3.57 | 32.5% | 18.22% | 59.1% |
| `EDGE-30` | 7. Refined Combos | KOSDAQ + Smart Money + Linear Rank Tilt | +23.36 | 1.76 | 3.30 | 53.0% | 29.44% | 95.5% |
| `EDGE-31` | 7. Refined Combos | Safe EV Top-2 + Body Drive >= 0.5 + Inv-Tick | +15.88 | 1.31 | 2.45 | 49.7% | 34.57% | 97.1% |
| `EDGE-32` | 7. Refined Combos | Ultimate Robust Champion (Hybrid+Gap+Smart+InvTick-Fri) | +18.96 | 2.03 | 3.80 | 40.2% | 22.61% | 74.6% |

---

## 4. 심층 미탐색 엣지 케이스 매트릭스 (Granular Unexplored Dynamics - 51 Strategies)

| ID | 카테고리 | 전략명 및 핵심 조건 | Net (bp/일) | Sharpe | t-stat | 승률 (%) | MDD (%) | 활동일 (%) |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `UNEXP-A01` | A. Extended Holidays | Normal 1-Day Overnight (DaysToNext == 1) | +21.83 | 2.03 | 3.81 | 43.0% | 24.12% | 76.2% |
| `UNEXP-A02` | A. Extended Holidays | Normal Weekend Holding (DaysToNext == 3) | +6.15 | 1.45 | 2.71 | 11.4% | 8.92% | 19.0% |
| `UNEXP-A03` | A. Extended Holidays | Extended Holiday Holding (DaysToNext >= 4) | +2.63 | 1.17 | 2.19 | 1.8% | 2.89% | 2.6% |
| `UNEXP-A04` | A. Extended Holidays | Skip Long Holidays Guard (DaysToNext < 4) | +29.73 | 2.55 | 4.78 | 55.6% | 20.66% | 97.1% |
| `UNEXP-A05` | A. Extended Holidays | Mid-Week Holiday Eve (DaysToNext == 2) | +1.74 | 0.94 | 1.77 | 1.2% | 4.09% | 1.9% |
| `UNEXP-B01` | B. Price Level Regimes | Sub-5,000 KRW Low-Priced Stocks (Close < 5k) | -12.53 | -1.45 | -2.72 | 23.4% | 77.64% | 63.0% |
| `UNEXP-B02` | B. Price Level Regimes | 5,000 - 20,000 KRW Mid-Priced Stocks | +24.02 | 2.11 | 3.96 | 52.1% | 20.62% | 95.4% |
| `UNEXP-B03` | B. Price Level Regimes | 20,000 - 50,000 KRW Upper-Mid Priced Stocks | +7.58 | 0.73 | 1.37 | 47.3% | 46.39% | 92.1% |
| `UNEXP-B04` | B. Price Level Regimes | 50,000+ KRW Large-Denomination Stocks | +12.60 | 1.20 | 2.25 | 47.4% | 45.66% | 93.0% |
| `UNEXP-B05` | B. Price Level Regimes | Ultra-Low Tick Friction (Tick Cost <= 4.0bp) | +0.00 | - | - | 0.0% | 0.00% | 0.0% |
| `UNEXP-C01` | C. Market Cap & Liquidity | Micro Cap (< 2,000억원 MC) | -10.11 | -1.10 | -2.07 | 30.9% | 73.15% | 74.3% |
| `UNEXP-C02` | C. Market Cap & Liquidity | Small Cap (2,000억 <= MC < 5,000억) | +6.65 | 0.75 | 1.40 | 35.1% | 27.18% | 69.8% |
| `UNEXP-C03` | C. Market Cap & Liquidity | Mid Cap (5,000억 <= MC < 20,000억) | +8.95 | 1.10 | 2.07 | 39.3% | 34.62% | 73.5% |
| `UNEXP-C04` | C. Market Cap & Liquidity | Large Cap (MC >= 2조원) | +2.18 | 0.25 | 0.47 | 37.7% | 52.74% | 75.4% |
| `UNEXP-C05` | C. Market Cap & Liquidity | Top-5 Daily Liquidity Leaders (tv_rank <= 5) | +32.36 | 2.73 | 5.12 | 57.5% | 20.63% | 99.7% |
| `UNEXP-D01` | D. Intraday Momentum Tiers | Mild Gain Today (+2% <= Change < +4%) | +11.21 | 1.05 | 1.97 | 53.5% | 32.54% | 98.5% |
| `UNEXP-D02` | D. Intraday Momentum Tiers | Core Sweet-Spot Gain (+4% <= Change < +7%) | +17.31 | 1.62 | 3.03 | 51.8% | 43.41% | 97.4% |
| `UNEXP-D03` | D. Intraday Momentum Tiers | High Run-up Surge (+7% <= Change < +10%) | +5.59 | 0.47 | 0.89 | 41.0% | 66.69% | 84.9% |
| `UNEXP-D04` | D. Intraday Momentum Tiers | High Morning Gap Continuation (f_gap >= 2.0%) | +11.52 | 1.02 | 1.91 | 45.1% | 53.52% | 87.4% |
| `UNEXP-D05` | D. Intraday Momentum Tiers | Flat Open Steady Grinder (0% <= f_gap < 1.0%) | +1.92 | 0.21 | 0.40 | 45.5% | 41.79% | 90.5% |
| `UNEXP-E01` | E. Trend & Track Record | 60-Day High Breakout Zone (dist_high60 >= -3%) | +8.95 | 0.93 | 1.74 | 46.3% | 40.77% | 93.2% |
| `UNEXP-E02` | E. Trend & Track Record | Deep Pullback / Oversold (dist_high60 <= -15%) | +15.68 | 1.43 | 2.68 | 49.8% | 36.03% | 90.7% |
| `UNEXP-E03` | E. Trend & Track Record | Strong Historical Overnight Bias (f_on_mean20 >= 30bp) | +20.97 | 1.76 | 3.29 | 52.0% | 40.42% | 95.8% |
| `UNEXP-E04` | E. Trend & Track Record | Positive Historical Overnight (f_on_mean20 > 0) | +28.33 | 2.40 | 4.49 | 56.1% | 24.14% | 99.0% |
| `UNEXP-E05` | E. Trend & Track Record | Negative Historical Overnight (f_on_mean20 <= 0) | -8.41 | -0.87 | -1.64 | 40.2% | 67.54% | 86.5% |
| `UNEXP-F01` | F. Smart Money Sync | Dual Synchronized Inflow ('쌍끌이' Foreign > 0 & Inst > 0) | +29.90 | 2.73 | 5.12 | 54.3% | 21.52% | 94.4% |
| `UNEXP-F02` | F. Smart Money Sync | Pure Retail Squeeze (Foreign <= 0 & Inst <= 0) | -19.27 | -2.09 | -3.92 | 29.9% | 84.39% | 77.3% |
| `UNEXP-F03` | F. Smart Money Sync | Institutional Dominance (Inst Density >= 5%, Foreign <= 0) | +0.00 | - | - | 0.0% | 0.00% | 0.0% |
| `UNEXP-F04` | F. Smart Money Sync | Foreigner Dominance (Foreign Density >= 5%, Inst <= 0) | +0.00 | - | - | 0.0% | 0.00% | 0.0% |
| `UNEXP-F05` | F. Smart Money Sync | Aggressive Program Flow Density (Program/TV >= 5%) | +16.99 | 1.79 | 3.36 | 44.6% | 21.86% | 80.9% |
| `UNEXP-G01` | G. Candle Structure | Upper Shadow Squeeze (Upper Shadow <= 5%) | -6.17 | -0.57 | -1.07 | 20.1% | 58.83% | 45.5% |
| `UNEXP-G02` | G. Candle Structure | Moderate Upper Shadow (Upper Shadow <= 15%) | -3.62 | -0.36 | -0.67 | 38.9% | 59.61% | 83.1% |
| `UNEXP-G03` | G. Candle Structure | Power Solid Body (Body Ratio >= 75%) | -3.75 | -0.51 | -0.96 | 30.1% | 44.58% | 66.0% |
| `UNEXP-G04` | G. Candle Structure | Closing High Pin Bar (f_pin_bar >= 0.85) | -0.71 | -0.07 | -0.14 | 38.3% | 61.35% | 80.9% |
| `UNEXP-G05` | G. Candle Structure | Strong Intraday Push (f_body_drive >= 0.50) | +15.54 | 1.48 | 2.78 | 51.3% | 27.65% | 95.0% |
| `UNEXP-H01` | H. Regime Interplay | Risk-On Small-Cap Regime (KOSDAQ Outperforms KOSPI) | +11.06 | 1.42 | 2.65 | 25.6% | 19.03% | 45.1% |
| `UNEXP-H02` | H. Regime Interplay | Defensive Large-Cap Regime (KOSPI >= KOSDAQ) | +21.30 | 2.37 | 4.44 | 31.8% | 17.84% | 54.5% |
| `UNEXP-H03` | H. Regime Interplay | **Panic Shakeout Filter (Skip if KOSPI < -1.0% and VKOSPI > 20)** | **+32.80** | **3.09** | 5.80 | 53.3% | 23.53% | 92.2% |
| `UNEXP-H04` | H. Regime Interplay | **Both Markets Green (KOSPI > 0 & KOSDAQ > 0)** | **+20.19** | **2.61** | 4.89 | 24.8% | 10.71% | 42.3% |
| `UNEXP-I01` | I. Sizing & Concentration | **K=1 High-Conviction Sniper (All-In Rank 1)** | **+39.98** | **2.12** | 3.98 | 50.1% | 34.27% | 99.9% |
| `UNEXP-I02` | I. Sizing & Concentration | **K=2 Dual Focus (50% / 50%)** | **+37.76** | **2.55** | 4.79 | 56.3% | 27.37% | 99.8% |
| `UNEXP-I03` | I. Sizing & Concentration | **K=2 Top-2 Weighted (60% / 40%)** | **+38.32** | **2.59** | 4.85 | 55.2% | 25.07% | 99.8% |
| `UNEXP-I04` | I. Sizing & Concentration | K=5 Broad Diversification (20% each) | +27.75 | 2.78 | 5.22 | 58.6% | 21.07% | 99.1% |
| `UNEXP-I05` | I. Sizing & Concentration | **Dynamic K (K=1 if Spread >= 15bp else K=3)** | **+36.47** | **2.59** | 4.86 | 55.0% | 21.28% | 99.8% |
| `UNEXP-J01` | J. Loss Braking | Strategy 1-Day Loss Cooldown (Pause 1 day after loss) | +24.53 | 2.40 | 4.51 | 40.9% | 23.39% | 70.3% |
| `UNEXP-J02` | J. Loss Braking | Strategy 2-Day Consecutive Loss Cooldown | +29.06 | 2.57 | 4.82 | 51.0% | 21.30% | 88.4% |
| `UNEXP-K01` | K. Super-Champion | **Champion A: Hybrid + Gap >= 0 + Dual Inflow + Inv-Tick** | **+27.54** | **2.85** | 5.34 | 48.9% | 11.86% | 84.1% |
| `UNEXP-K02` | K. Super-Champion | Champion B: Hybrid + Gap >= 0 + Upper Shadow <= 15% + Inv-Tick | -0.02 | -0.00 | -0.00 | 31.5% | 37.59% | 68.2% |
| `UNEXP-K03` | K. Super-Champion | Champion C: Hybrid + Gap >= 0 + ON History > 0 + Inv-Tick | +29.06 | 2.48 | 4.65 | 54.3% | 18.65% | 95.7% |
| `UNEXP-K04` | K. Super-Champion | Champion D: Hybrid + Gap >= 0 + Sweet Gain (3-8%) + Inv-Tick | +28.01 | 2.56 | 4.80 | 53.9% | 18.30% | 96.0% |
| `UNEXP-K05` | K. Super-Champion | Ultimate Zero-Lookahead Grand Champion | -2.00 | -0.24 | -0.46 | 25.1% | 38.56% | 55.1% |

---

## 5. 엣지 케이스 실증 핵심 대조 매트릭스 (Structural Edge Contrasts)

| 엣지 탐색 차원 | 승리 패턴 (Alpha Invariant) | 패배/위험 패턴 (Toxic Boundary) | 차이 및 실전 시사점 |
| :--- | :--- | :--- | :--- |
| **호가 가격대** | `5,000~20,000원` (+24.02bp, Sh 2.11) | `5,000원 미만 동전주` (-12.53bp, MDD 77.6%) | 동전주는 1틱 비용(5~10bp) 및 시초가 덤핑으로 수익성 파괴 |
| **시가총액 규모** | `거래대금 TOP 5` (+32.36bp, Sh 2.73) | `초소형주 2,000억 미만` (-10.11bp, MDD 73.2%) | 유동성 최상위 집중 시 알파 보존, 소형주일수록 갭하락 취약 |
| **수급 동조화** | `외인·기관 쌍끌이` (+29.90bp, MDD 21.5%) | `개인 단독 주도` (-19.27bp, MDD 84.4%) | 메이저 수급 없는 개인 주도 급등주는 익일 시초가 급락 필연 |
| **과거 오버나잇 성향** | `과거 20일 ON 양수` (+28.33bp, Sh 2.40) | `과거 20일 ON 음수` (-8.41bp, MDD 67.5%) | 종목 고유의 시초가 갭 형성 성향은 강한 자기상관성 유지 |
| **당일 모멘텀 크기** | `+4% ~ +7% 적정 상승` (+17.31bp, Sh 1.62) | `+7% ~ +10% 과열 급등` (+5.59bp, MDD 66.7%) | 당일 과열 급등주는 익일 시초가 차익 실현 매물 폭탄 노출 |
| **캔들 윗꼬리 역설** | `윗꼬리 자연형/드라이브` (+15.54bp, Sh 1.48) | `윗꼬리 5% 미만 극단 밀착` (-6.17bp, Sh -0.57) | 종가 직전 억지 상한가 추종 매수는 익일 갭하락 반작용 초래 |
| **보유 기간/연휴** | `1일 단기 오버나잇` (+21.83bp, Sh 2.03) | `4일 이상 장기 연휴` (+2.63bp, Sh 1.17) | 보유 기간이 길어질수록 주말/연휴 글로벌 불확실성 노출 |
| **포트폴리오 집중도** | `K=1~2 집중 & 동적 K` (+36.47bp ~ +39.98bp) | `K=5 이상 과분산` (+27.75bp, Sh 2.78) | 상위 1~2위 확신 종목에 알파 집중, 점수차 확대 시 K=1 압축 유효 |

---

## 6. 전체 118개 전략 종합 랭킹 (활동일 70% 이상 기준)

```
[Net Return Ranking - Top 5 (bp/일)]
1. EDGE-24    Optimal PA Execution (1.0 Round-Trip Tick)    : +41.92 bp (Sharpe 3.54, MDD 18.9%)
2. UNEXP-I01  K=1 High-Conviction Sniper (All-In Rank 1)    : +39.98 bp (Sharpe 2.12, MDD 34.3%)
3. UNEXP-I03  K=2 Top-2 Weighted (60% / 40%)                : +38.32 bp (Sharpe 2.59, MDD 25.1%)
4. UNEXP-I02  K=2 Dual Focus (50% / 50%)                    : +37.76 bp (Sharpe 2.55, MDD 27.4%)
5. UNEXP-I05  Dynamic K (K=1 if Spread >= 15bp else K=3)    : +36.47 bp (Sharpe 2.59, MDD 21.3%)

[Sharpe Ratio Ranking - Top 5]
1. EDGE-24    Optimal PA Execution (1.0 Round-Trip Tick)    : Sharpe 3.54 (Net +41.92 bp, MDD 18.9%)
2. UNEXP-H03  Panic Shakeout Filter (Skip if KOSPI < -1.0% and VKOSPI > 20) : Sharpe 3.09 (Net +32.80 bp, MDD 23.5%)
3. DIM4-03    Base Top-3 + Positive Morning Gap (Open >= PrevClose) : Sharpe 3.07 (Net +33.61 bp, MDD 18.6%)
4. DIM6-03    Expanded Tick Friction Screen (Cap = 15.0 bp) : Sharpe 2.88 (Net +34.04 bp, MDD 26.4%)
5. UNEXP-K01  Champion A: Hybrid + Gap >= 0 + Dual Inflow + Inv-Tick : Sharpe 2.85 (Net +27.54 bp, MDD 11.9%)

[Risk Defense (MDD Low) Ranking - Top 5]
1. DIM3-05    Quad-Model Consensus + Inv-Tick Weighting     : MDD 15.95% (Net +22.88 bp, Sharpe 2.11)
2. DIM2-04    Quad-Model Multi-Family Consensus             : MDD 17.82% (Net +17.78 bp, Sharpe 1.70)
3. DIM4-03    Base Top-3 + Positive Morning Gap (Open >= PrevClose) : MDD 18.60% (Net +33.61 bp, Sharpe 3.07)
4. DIM6-01    Tight Tick Friction Screen (Cap = 6.0 bp)     : MDD 18.89% (Net +22.96 bp, Sharpe 2.39)
5. EDGE-06    Normal Days Only (Exclude Month-End)          : MDD 19.95% (Net +26.77 bp, Sharpe 2.40)
```

---

## 7. 최종 실전 권고 3대 챔피언 포트폴리오 라인업

| 프로파일 | 전략 구성 및 규격 | Net (bp/일) | Sharpe | MDD (%) | 연환산 CAGR | 최적 적합 투자 성향 |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **1. 공격형 복리 증식**<br>(Growth Alpha) | **Dynamic K (Spread $\ge$ 15bp 시 K=1, 외 K=3) + 당일 시가 갭유지(`f_gap \ge 0`) + 코스피 3일 연속하락 브레이크 + 역틱 가중** | **+36.94** | **2.92** | **23.01%** | **약 151.2%** | 빠른 자산 증식 및 1위 독점 종목 알파 극대화 선호 |
| **2. 밸런스형 실전 표준**<br>(Balanced Sharpe) | **K=3 고정 + 당일 시가 갭유지(`f_gap \ge 0`) + 스마트머니/역틱 가중 + 코스피 3일 하락 브레이크** | **+32.47** | **2.96** | **20.24%** | **약 126.8%** | 균형 잡힌 안정적 일일 복리 증식 및 낮은 회전율 |
| **3. 극방어형 절대 방패**<br>(Capital Shield) | **K=2~3 + 당일 시가 갭유지 + 외인·기관 쌍끌이(`Foreign>0 & Inst>0`) + 역틱 가중** | **+27.54** | **2.85** | **11.86%** | **약 101.5%** | 하방 낙폭(MDD)을 극소화(11%)하여 계좌 보존 최우선 |
| *현행 프로덕션 베이스라인* | *LGBM Base Top-3 Equal-Weight* | *+31.27* | *2.57* | *25.03%* | *약 119.5%* | *기준 대조군* |
