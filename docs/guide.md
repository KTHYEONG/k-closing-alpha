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

### 3. 아카이브 TSV/CSV 내보내기 (`archive`)
- **실행 시점**: 수집 완료 후 필요 시
```bash
uv run python -m src.utils.export_archive
```

### 4. 일중 분봉/틱 아카이브 (`archive_intraday`)
- **실행 시점**: NXT 애프터마켓 마감 이후 (20:00 이후)
```bash
uv run python -m src.daily.archive_intraday
```
