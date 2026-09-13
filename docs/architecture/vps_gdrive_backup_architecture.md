# VPS 자동화 실사 및 GDrive 백업 구조 개편

> **문서 상태:** 완료 (Completed) — 2026-09-13 실사에서 발견된 결함 전부 수정 및 배포
> **기준 시점:** 2026-09-13
> **관련 문서:** [`docs/architecture/overview.md`](overview.md), [`docs/architecture/design-decisions.md`](design-decisions.md)

---

## 1. 배경

VPS(`or-vps`, hostname `trading-bot`)에 SSH 직접 접속해 k-closing-alpha 관련 systemd 유저 유닛 11개 타이머, 데이터 디렉터리, GDrive 오프사이트 백업(rclone)을 전수 실사했다. 발견된 결함을 6개 계약(S1~S6)으로 분리해 전부 수정·배포했다.

---

## 2. 발견·수정한 자동화 결함 (전부 완료)

| # | 결함 | 근본 원인 | 조치 |
| :--- | :--- | :--- | :--- |
| 1 | VPS git 체크아웃이 origin/main과 34커밋 분기 | 과거 히스토리 재작성을 VPS가 반영 못함 | `git reset --hard origin/main` (데이터는 `.gitignore` 대상이라 무손실) |
| 2 | `kca-paper-entry.timer` 신설 오진단(되돌림) | `kca-finalize-close.service`의 기존 `OnSuccess=kca-paper-entry.service` 이벤트체이닝을 못 보고 중복 타이머 신설 | 신설 타이머·`After=`·관련 테스트 3개 전부 되돌림 |
| 3 | `src/tools/code_sync.py`의 bare `"uv"` 호출 크래시 | systemd 유저 세션 PATH에 `~/.local/bin` 없음 | `_resolve_uv_bin()`(PATH 우선, 고정경로 폴백) |
| 4 | `src/ml/retrain.py` 로깅 누락 | 다른 6개 CLI와 달리 `logging.basicConfig` 누락 | 첫 줄에 추가 |
| S1 | 페이퍼 청산 세션 크래시·무한대기 | 빈 포지션 구독 ValueError, 웹소켓 종료조건 부재 | 전량청산/정규장종료/벽시계 데드라인 중 최초 조건에서 정상 종료 |
| S2 | 일일 감사가 정상일에도, 휴장일에도 오탐 | 결정 로그 경로 오독, KRX 달력 1일 지연게시 | topk_decisions/paper 원장 기반 판정 + KIS 당일오라클 + 평일 1통 다이제스트 |
| S3 | UTC 호스트에 KST 미고정, 알림 사각지대 5개 유닛, code-sync/paper-exit 09:00 충돌 | 설계 당시 미검토 | 전 유닛 TZ 고정, OnFailure 5건 추가, code-sync 07:30/backup 22:15 이동, collect 휴장일 정상종료 |
| S4 | systemd 파일 변경이 code_sync로 자동 반영 안 됨 | git 체크아웃과 `~/.config/systemd/user` 설치본이 별개 사본 | `install_systemd_units`가 매 sync마다 변경분 복사·삭제분 정리·신규 timer만 enable |
| S5 | 무인 주간 재학습이 검증 없이 라이브 모델 교체 | 재학습 성공=발행으로 설계됨 | 예측 순위상관 기반 승격 게이트(임계 0.95, 실측 정상 0.983~0.989 vs 오염 0.808) |
| S6 | GDrive 백업이 VPS 미보유 데이터(altdata 등)를 삭제할 위험 | `rclone sync`가 VPS 로컬을 정답 삼아 미러링 | `rclone copy`+날짜기준 30일 보존으로 전환(아래 4절) |

---

## 3. GDrive 최상위 구조

```
gdrive:quant-lake/
├── projects/   ← 마이그레이션 이전 백업 위치의 잔재(2026-09-11 전후 동결, 정리 대상)
├── live/       ← 운영 백업(아래 4절)
└── _backup/    ← k-closing-alpha와 무관 (타 프로젝트 서버 백업)
```

---

## 4. GDrive 백업 구조 (S6 반영 완료)

### 4.1 설계 원칙

`kca-backup.service`는 `rclone sync`(미러링, VPS 로컬을 정답 삼아 원격을 삭제까지 포함해 맞춤) 대신 **`rclone copy --backup-dir`**를 사용한다. `copy`는 원격에만 있는 파일(WSL이 직접 올린 altdata 10년치 백필 등)을 절대 지우지 않고, VPS가 덮어쓰는 파일은 이전 버전을 `_deleted/<subtree>/<날짜>/`에 보존한다. 즉 **VPS와 WSL 어느 쪽이 만든 데이터든 자동으로 합집합**이 유지되며, 유지보수용 exclude 목록이 필요 없다.

```
ExecStart=rclone copy data     gdrive:.../data     --backup-dir gdrive:.../_deleted/data/<YYYY-MM-DD>
ExecStart=rclone copy artifacts gdrive:.../artifacts --backup-dir gdrive:.../_deleted/artifacts/<YYYY-MM-DD>
```

### 4.2 보존 정책

`kca-backup-prune.timer`(매월 1일)는 `rclone delete --min-age 30d` 대신 **`src/tools/backup_prune.py`**를 실행한다. `--min-age`는 파일 **수정시각** 기준이라 오래전에 수정된 파일이 옮겨지자마자 삭제되는 문제가 있었다. 새 방식은 `_deleted/<subtree>/` 아래 **디렉터리 이름(옮겨진 날짜)** 기준으로 30일 경과분만 정리한다.

### 4.3 실행 항목 (전부 완료)

1. ~~`kca-backup.service`에 `--exclude` 추가~~ → **copy 전환으로 대체, 완료**
2. WSL 쪽 신규 리서치 데이터는 수동 1회성 업로드로 충분(자동화 불필요):
   ```bash
   rclone copy ~/k-closing-alpha/data/history/<신규폴더> gdrive:quant-lake/live/k-closing-alpha/data/history/<신규폴더>
   ```
3. `gdrive:quant-lake/projects/k-closing-alpha` (118MB, 죽은 잔재) 삭제 — **미완료, 사용자 승인 대기**
4. (범위 외) `projects/ETF-Manager`, `projects/mt-etf-king-2026`도 동일 패턴이나 범위 밖.

### 4.4 소유권 노트 (참고, exclude 불필요해졌지만 기록 보존)

`parquet/topk_decisions.parquet`·`parquet/growth_shadow.parquet`는 VPS가 실제 운영 결정을 생성하기 시작하면(2026-09-14~) 그 내용이 권위 있는 감사로그가 된다. WSL은 이 두 파일을 더 이상 gdrive에 재업로드하지 않는 것을 권장한다(copy 방식이라 강제되진 않지만, VPS가 최신으로 계속 덮어쓰므로 충돌 없이 자연히 VPS본이 유지된다).

---

## 5. 미해결 항목

- `gdrive:quant-lake/projects/k-closing-alpha` 삭제 — 사용자 승인 대기 (3.1절)
- `theme.parquet`이 실제로 프로덕션에서 소비되는지 미확인 (`trade_log.parquet`은 테스트 skip 메시지로 미사용 확인됨)
