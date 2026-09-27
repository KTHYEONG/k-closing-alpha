# KCA Unattended-Run Verification Checklist (or-vps)

Purpose: an AI agent verifies, end to end and without human help, that the `kca-*` automation on or-vps ran a full trading day correctly. First target: **Mon 2026-09-28**, the first trading day after the Chuseok closure (09-24/25) and after the ops-hardening deploy (`ADR_20260927_OPS_HARDENING`). The checklist is reusable for any trading day: substitute `D` (KST trading date) and `D+1`.

## Ground rules for the agent

- **Read-only.** Never start/stop/restart units, never `rm`/`mv`/`chown`, never run `rclone` write commands, never `git reset`/`pull` on or-vps. Diagnose and report; ask the user before any remediation.
- **Do not push to `main` during the deploy blackout** (Mon–Fri 08:50–09:40 and 15:10–15:43 KST).
- or-vps clock is **UTC**; every timer is KST (UTC+9). Always pass explicit ranges to `journalctl`.
- Units are **`systemctl --user`** units (never check only the system manager). Logs: `journalctl --user -u <unit>`.
- Wrap every remote command in `timeout 60 ssh or-vps '...'`. Summaries only; write bulky output to `scratch/`.
- Never print secret values from `~/quant-secrets/*.env` (names only).
- A check passes only on **positive evidence** (a log line, a file, a count). "No error seen" is not a pass.

## When to run

| Run at (KST) | Covers |
|---|---|
| D 23:45 or later | Sections 0–5 (full day through backup) |
| D+1 08:00 or later | Section 6 (external watchdog) — optional follow-up |

If run earlier, mark later-stage items `PENDING`, not `FAIL`.

---

## 0. Baseline & deploy integrity

| # | Check | Command (on or-vps unless noted) | PASS criterion |
|---|---|---|---|
| 0.1 | Host checkout = origin/main | `cd ~/k-closing-alpha && git rev-parse --short HEAD` vs local `git rev-parse --short origin/main` (after `git fetch`) | equal |
| 0.2 | Image = checkout | `docker inspect ghcr.io/kthyeong/k-closing-alpha:latest --format '{{range .Config.Env}}{{println .}}{{end}}' \| grep KCA_CODE_COMMIT` | commit prefix equals 0.1 |
| 0.3 | No deploy inside a blackout on D | `TZ=Asia/Seoul git reflog --date=iso-local -10` | no `reset` between 08:50–09:40 or 15:10–15:43 on D |
| 0.4 | All timers enabled & scheduled | `systemctl --user list-timers --all --no-pager \| grep kca-` | 18 kca timers incl. `kca-offsite-verify.timer`; none `n/a` |
| 0.5 | Unit files match repo | `cmp` each `deploy/systemd/kca-*` against `~/.config/systemd/user/<same>` | no differences |
| 0.6 | Runtime bounds deployed | `systemctl --user show kca-collect kca-predict kca-finalize-close kca-archive-intraday kca-daily-audit -p Id -p TimeoutStartUSec` | none `infinity` (collect 9min, predict 7min, finalize 12min, archive 3h, audit 30min) |
| 0.7 | No failed units | `systemctl --user --failed --no-pager` | `0 loaded units listed` |
| 0.8 | No orphan kca containers | `docker ps -a --format '{{.Names}} {{.Status}}' \| grep '^kca-'` | empty after all jobs finished |
| 0.9 | No foreign-owned files | `find ~/k-closing-alpha/data ~/k-closing-alpha/artifacts ~/.cache/kis -not -user ubuntu \| head` | empty |
| 0.10 | Disk headroom | `df -h / ; df -i /` | use < 70%, inodes < 50% |

## 1. Morning chain (D)

| # | Time KST | Unit | Evidence | PASS criterion |
|---|---|---|---|---|
| 1.1 | 07:05 | `kca-kis-token-warmup` | `[SYS] stage=kis_token status=ISSUED` per host data slot | one ISSUED per slot, no `REPEAT_ISSUE` |
| 1.2 | 08:30 | `kca-price-ingest` | `Finished kca-price-ingest.service` | finished; no `Failed to start` |
| 1.3 | 08:39:55 | `kca-auction-open` | `[DATA] stage=auction_capture status=... phase=open date=D` | not FAILED / DEADLINE_EXCEEDED |
| 1.4 | 09:01 | `kca-paper-exit` | `[EXEC] stage=paper_exit ...`; `data/paper/` ledger | open lots from the previous trading day closed with open-auction fills (2026-09-28: the 3 lots entered 2026-09-23, held through the holiday). FAIL on `exit_window_expired`, `calendar_disagreement`, `shifted_session_hold` |

## 2. Decision window (D 15:20–15:45)

| # | Time KST | Unit | Evidence | PASS criterion |
|---|---|---|---|---|
| 2.1 | 15:20:00 | `kca-collect` | journal 06:20–06:30 UTC | finishes < 9 min; no `status=SKIP`; **no** `reason=calendar_disagreement`; no `stage=orderbook_persist status=FAILED` |
| 2.2 | 15:21:00 | `kca-predict` | `[SYS] stage=run_outcome job=predict run_date=D outcome=OK` | OK with `n_picks ≥ 1` or OK `admitted_below_top_k`; `NO_DECISION` = FAIL (report reason); no `history_unreadable` |
| 2.3 | 15:21:05 | `kca-auction-close` | `stage=auction_capture ... phase=close date=D` | completes; not FAILED |
| 2.4 | 15:30:30 | `kca-finalize-close` | `[DATA] stage=close_finalization date=D n_finalized=… n_unconfirmed=…` | `n_finalized ≥ 1`, `n_unconfirmed = 0` (or each explained); run_outcome OK |
| 2.5 | ~15:33 + 15:34 | `kca-paper-entry` | chain run + backstop run | first run books entries for D; second logs `stage=paper_entry status=SKIP reason=already_recorded` |
| 2.6 | window | tokens | `journalctl --user --since "D 06:15 UTC" --until "D 06:40 UTC" \| grep -E "DECISION_WINDOW_ISSUE\|kis_token_cache status=UNREADABLE\|kis_token_lock status="` | zero `DECISION_WINDOW_ISSUE`/`UNREADABLE`/`UNOPENABLE`; `READONLY_FALLBACK` = WARN |
| 2.7 | window | decision artefacts | `data/parquet/topk_decisions.parquet` rows with `decision_date == D`; capture decision manifest for D | present and consistent with 2.2 |

## 3. Evening archive & data (D)

| # | Time KST | Unit | PASS criterion |
|---|---|---|---|
| 3.1 | 15:40 | `kca-archive-intraday-regular` | finished < 3h; no `status=ERROR`; `DEGRADED` = WARN with reason |
| 3.2 | 20:05 | `kca-archive-intraday` | same criteria |
| 3.3 | 21:30 | `kca-price-ingest` | finished; `price_history.parquet` max date == D |
| 3.4 | 21:35 | `kca-altdata-capture` | `[DATA] stage=altdata_capture date=… capture=…`; no `DEGRADED` except DART `quota_exceeded` (known shared-key cause); alphanumeric codes (e.g. `0009K0`) in the universe without a spike of per-symbol failures |

## 4. Audit, alerting & expiry (D 20:15)

| # | Check | PASS criterion |
|---|---|---|
| 4.1 | `kca-daily-audit` finished | started after archive-intraday finished; finished < 30 min |
| 4.2 | Digest verdict | `[DATA] stage=daily_audit day=trading status=OK subject=[kca] 🟢 D 일일점검 완료 (정상)`; if `WARNING`, list each problem |
| 4.3 | Heartbeat | `data/logs/heartbeat/daily_audit.json`: `snapshot_date == D`, `day_kind == trading`, `undelivered_alerts == 0` |
| 4.4 | Alert channel alive | `ls data/logs/alerts/outbox/*.json 2>/dev/null \| wc -l` == 0; digest journaled with `email=True` |
| 4.5 | Expiry notices | digest lines `expiry_notices=` / `expiry_warnings=`. Expected **none** until 2026-11-04 (first: `KIS_DATA_1` expires 2026-12-04; KRX calendar horizon 2026-12-31 → notice from 2026-12-01) |
| 4.6 | Run outcomes | `data/logs/events/<YYYY-MM>/D.jsonl`: every job OK; list any DEGRADED/NO_DECISION with reason |

## 5. Backup & retention (D 21:00 / 22:15)

| # | Unit | PASS criterion |
|---|---|---|
| 5.1 | `kca-backup-prune` (21:00) | `[SYS] stage=backup_prune dry_run=False ...`; no `step=... status=failed`. On 2026-09-28 expect `local_purged` to include `regular/2026-09-22` (≈6.5 GB pre-fix backup) |
| 5.2 | `kca-backup` (22:15) | `data/history/capture/offsite/last_run.json`: `status == "ok"`, `started_at` on D, steps `capture_seal`, `core_panels`, `data`, `artifacts` all ok |
| 5.3 | Core panel latch | no `core_panel:*:missing` (the untracked `sizing_pipeline_bundle.joblib.bak` must not appear) |
| 5.4 | Drive lock contention | no exit 75 (flock wait); if present, name the competing project unit |
| 5.5 | Local sealed retention | from ≈2026-10-12: `sealed_removed` > 0 only with remote MD5 verified; `kept` reasons listed |

## 6. External watchdog (D+1, optional)

| # | Check | PASS criterion |
|---|---|---|
| 6.1 | Watchdog probe (read-only) — on or-vps: `cd ~/k-closing-alpha && ~/.local/bin/uv run --no-sync python -m src.tools.watchdog_probe` | `[SYS] stage=watchdog status=OK problems=none` |
| 6.2 | GitHub "External Dead-Man Watchdog" run at D+1 07:40 KST | succeeded (GitHub UI/API if accessible; otherwise `UNVERIFIED`) |

## 7. Weekly items (when the run spans a weekend)

| # | Unit | PASS criterion |
|---|---|---|
| 7.1 | `kca-retrain` (Sat 22:00) | finished < 2h; registry line appended; no `docker pull` in the unit |
| 7.2 | `kca-core-snapshot` (Sun 10:00) | `[SYS] stage=core_snapshot ... status=ok` |
| 7.3 | `kca-offsite-verify` (Sun 11:00, first run 2026-10-04) | `[SYS] stage=offsite_verify checked=N missing=0 mismatched=0` and `stage=restore_drill ... status=OK` per tier |

---

## Report format (write in Korean)

```
### 🩺 [OPS CHECK] kca 무인운용 점검 (D)
> 🚦 Verdict: ✅ HEALTHY | ⚠️ DEGRADED | ❌ BROKEN

| 섹션 | 결과 | 근거 요약 |
|---|---|---|
| 0 배포·기반 | PASS/WARN/FAIL/PENDING | ... |
| 1 오전 체인 | ... | ... |
| 2 결정창 | ... | ... |
| 3 저녁 적재 | ... | ... |
| 4 감사·알림 | ... | ... |
| 5 백업 | ... | ... |
| 6 워치독 | ... | ... |

- 🔍 발견사항: <FAIL/WARN 항목별 원인 추정과 증거 1~2줄>
- 🎯 권장 조치: <사람 승인이 필요한 조치만, 없으면 "없음">
```

Verdict rules: any FAIL in sections 0, 2 or 4 → ❌ BROKEN; any other FAIL or any WARN → ⚠️ DEGRADED; otherwise ✅ HEALTHY.
