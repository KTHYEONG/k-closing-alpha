# Daily Pipeline Execution Guide

## 실행 시점 및 명령어

### 1. 데이터 수집 및 아카이브 (`collect`)
- **실행 시점**: 장 마감 직전 (15:15 ~ 15:20)
```bash
uv run python -m src.daily.collect
```

### 2. Top-3 종가매매 예측 (`predict`)
- **실행 시점**: 수집 완료 직후 (15:20)
```bash
uv run python -m src.daily.predict
```

### 3. 페이퍼 진입 (`paper_trade --phase entry`)
- **실행 시점**: 연속거래 구간 (15:19)
```bash
uv run python -m src.daily.paper_trade --phase entry
```

### 4. 종가 확정 (`finalize_close`)
- **실행 시점**: 종가단일가 종료 직후 (15:30:30)
```bash
uv run python -m src.daily.finalize_close
```

### 5. 일중 분봉/틱 아카이브 (`archive_intraday`)
- **실행 시점**: NXT 애프터마켓 마감 이후 (20:05)
```bash
uv run python -m src.daily.archive_intraday
```

### 6. 페이퍼 청산 (`paper_trade --phase exit`)
- **실행 시점**: 익일 오전 (09:00)
```bash
uv run python -m src.daily.paper_trade --phase exit
```

## 자동화 시각표 (systemd 유닛 기준, Asia/Seoul)

| 시각 | 유닛 | 명령 |
|---|---|---|
| Mon..Fri 15:15 | `kca-collect` | `uv run python -m src.daily.collect` |
| Mon..Fri 15:20 | `kca-predict` | `uv run python -m src.daily.predict` |
| Mon..Fri 15:19 | `kca-paper-entry` | `uv run python -m src.daily.paper_trade --phase entry` |
| Mon..Fri 15:30:30 | `kca-finalize-close` | `uv run python -m src.daily.finalize_close` |
| Mon..Fri 20:05 | `kca-archive-intraday` | `uv run python -m src.daily.archive_intraday` |
| Mon..Fri 09:00 | `kca-paper-exit` | `uv run python -m src.daily.paper_trade --phase exit` |
| 부팅 시 1회 | `kca-daily-audit` | `uv run python -m src.tools.daily_audit` |

결정창 배치(collect/predict/paper-entry/finalize-close)는 지연 캐치업이 무의미하므로
`Persistent=false`이며, 저녁 아카이브 배치만 `Persistent=true`다.
