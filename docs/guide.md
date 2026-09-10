# Daily Pipeline Execution Guide

## 실행 시점 및 명령어

### 1. 데이터 수집 및 아카이브 (`collect`)
- **실행 시점**: 종가단일가 결정창 시작 (15:20:00). `_validate_decision_window` 게이트가 15:20:00~15:30:00 밖 실행을 RuntimeError로 거부한다(장 마감 직전이 아니라 결정창 안이어야 함).
```bash
uv run python -m src.daily.collect
```

### 2. Top-3 종가매매 예측 (`predict`)
- **실행 시점**: 수집 완료 직후 (15:21)
```bash
uv run python -m src.daily.predict
```

### 3. 종가 확정 (`finalize_close`)
- **실행 시점**: 종가단일가 종료 직후 (15:30:30), 확정까지 최대 15:33:00 재시도
```bash
uv run python -m src.daily.finalize_close
```

### 4. 페이퍼 진입 (`paper_trade --phase entry`)
- **실행 시점**: 고정 시각이 아니라 `finalize_close` 성공 종료 시 `OnSuccess=`로 즉시 체이닝. 체결가는 확정종가(`종가_확정=True`인 행의 `종가`) 그대로이며, 동시호가 구간엔 연속체결이 없으므로 웹소켓을 열지 않는다(미확정 종목은 진입 보류 후 WARNING 로그).
```bash
uv run python -m src.daily.paper_trade --phase entry
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
| Mon..Fri 15:20:00 | `kca-collect` | `uv run python -m src.daily.collect` |
| Mon..Fri 15:21:00 | `kca-predict` | `uv run python -m src.daily.predict` |
| Mon..Fri 15:30:30 | `kca-finalize-close` | `uv run python -m src.daily.finalize_close` |
| finalize-close 성공 직후 (`OnSuccess=`) | `kca-paper-entry` | `uv run python -m src.daily.paper_trade --phase entry` |
| Mon..Fri 20:05 | `kca-archive-intraday` | `uv run python -m src.daily.archive_intraday` |
| Mon..Fri 09:00 | `kca-paper-exit` | `uv run python -m src.daily.paper_trade --phase exit` |
| 부팅 시 1회 | `kca-daily-audit` | `uv run python -m src.tools.daily_audit` |

결정창 배치(collect/predict/finalize-close)는 지연 캐치업이 무의미하므로
`Persistent=false`이며, 저녁 아카이브 배치만 `Persistent=true`다. paper-entry는
고정 타이머 없이 finalize-close의 `OnSuccess=` 이벤트로만 트리거된다(그날 확정이
몇 초 걸렸든 관계없이 확정 직후 실행).
