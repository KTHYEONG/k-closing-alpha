# KCA + KRX Daily Routine Checklist (or-vps)

Purpose: an AI agent verifies, read-only and end to end, that one trading day `D` of the `kca-*` automation on
or-vps ran correctly — decision chain, same-day-only data capture, archive, audit, backup and the overnight
backfill — together with the co-located **krx-alpha** collector (`krx-collector` container, §10), which shares KIS
keys, the KIS token cache and the gdrive lock with kca. Substitute `D` (KST trading date) and `D+1` (next calendar
day).

## Ground rules for the agent

- **Read-only.** Never start/stop/restart units, edit env files, `rm`/`mv`/`chown`, run `rclone` write commands, or
  `git reset`/`pull` on or-vps. Diagnose and report; any remediation (including a same-day rerun) needs user approval.
- **Deploy blackout**: never push to `main` Mon–Fri 08:50–09:40 and 15:10–15:43 KST.
- or-vps clock is **UTC**; every timer is KST (UTC+9). Always pass explicit UTC ranges to `journalctl`
  (e.g. D 15:40 KST = D 06:40 UTC; D 23:05 KST = D 14:05 UTC).
- Units are **`systemctl --user`** units. Logs: `journalctl --user -u <unit> --since ... --until ... --no-pager`.
- Wrap remote commands in `timeout 60 ssh or-vps '...'`; write bulky output to `scratch/`, report summaries only.
- Never print secret values from `~/quant-secrets/*.env` (key names only).
- A check passes only on **positive evidence** (a log line, a manifest, a count). "No error seen" is not a pass.
- If a stage has not happened yet, mark it `PENDING`, not `FAIL`.

## When to run

| Run at (KST) | Covers | Why this time |
|---|---|---|
| D 09:10 (optional) | §10.1–10.2 | krx session started; a dead streamer is still recoverable today |
| **D 16:00** (critical pass) | §0, §1, §2, §3.1 start, §3.4 start, §10.1–10.3 | Decision chain is final; kca book and krx after-market shards must be running |
| **D 21:00** (critical pass) | §3 complete, §4, §10.4–10.5 | **Same-day-only data can still be re-collected before 24:00**; krx EOD verdict is out |
| D 23:45 | §5, §6.1–6.2, §10.6–10.7 | Backups done (kca 22:15, krx 23:30); backfill started |
| D+1 07:15 | §6.3–6.6, §7 | Backfill stopped at 06:50; watchdog at 07:40 |

## Same-day deadlines (must not be missed)

These sources serve **the current day only**. A miss after 24:00 KST is permanent data loss.

| Data | Producer | Deadline | Recovery if bad (user approval required) |
|---|---|---|---|
| Regular-session ticks (Kiwoom ka10079) | `kca-archive-intraday-regular` 15:40 | D 24:00 | `systemctl --user start kca-archive-intraday-regular.service` (rerun fills only missing symbols; ~70 min for ~750 symbols) |
| KRX/NXT aftermarket ticks (Kiwoom ka10079) | `kca-archive-intraday` 20:05 | D 24:00 | `systemctl --user start kca-archive-intraday.service` |
| Aftermarket order book (KIS REST snapshots) | `kca-aftermarket-book` 15:40–20:00 | live only | none (snapshots cannot be replayed) — report |
| Decision window inputs / paper entry | 15:20–15:34 chain | 15:30 | none — report |
| krx regular WebSocket (LS, 08:20–15:40) and after-market shards (KIS WS, 15:40/16:00–20:00) | `krx-collector` | live only | none — report the gap window (§10) |
| krx snapshot REST (auction books, investor estimates, rankings, 08:00–15:39) | `krx-collector` | live only | none — report |

Bars (KIS/Kiwoom minute charts) are re-collectable later within vendor retention; ticks and books are not.

---

## 0. Baseline & deploy integrity (any pass)

| # | Check | Command (on or-vps unless noted) | PASS criterion |
|---|---|---|---|
| 0.1 | Host checkout = origin/main | `cd ~/k-closing-alpha && git rev-parse --short HEAD` vs local `git fetch && git rev-parse --short origin/main` | equal |
| 0.2 | Image = checkout | `docker inspect ghcr.io/kthyeong/k-closing-alpha:latest --format '{{range .Config.Env}}{{println .}}{{end}}' \| grep KCA_CODE_COMMIT` | commit prefix equals 0.1 |
| 0.3 | Last CI run green | GitHub API `actions/runs?per_page=1` | `test`, `build-and-push`, `deploy` all success for the deployed sha |
| 0.4 | No deploy in a blackout on D | `TZ=Asia/Seoul git reflog --date=iso-local -10` | no `reset` inside 08:50–09:40 or 15:10–15:43 on D |
| 0.5 | All kca timers scheduled | `systemctl --user list-timers --all --no-pager \| grep -c 'kca-.*\.timer'` | **20** timers, none `n/a` (list in §8) |
| 0.6 | Unit files match repo | `cmp` each `deploy/systemd/kca-*` with `~/.config/systemd/user/<same>` | no differences |
| 0.7 | No failed units | `systemctl --user --failed --no-pager` | `0 loaded units listed` |
| 0.8 | Runtime env sanity (names only) | `grep -E '^(COLLECTION_BACKFILL_SLOTS\|COLLECTION_AFTERMARKET_BOOK_ENABLED\|COLLECTION_AFTERMARKET_BOOK_SLOTS)=' ~/quant-secrets/k-closing-alpha.env`; `grep -c '^COLLECTION_TOSS' ~/quant-secrets/k-closing-alpha.env` | backfill slots `2,3`, book enabled with slots `2,3`; **no** `COLLECTION_TOSS_*` key (Toss overnight phase was removed) |
| 0.9 | Disk headroom | `df -h / ; df -i /` | use < 70 %, inodes < 50 % |
| 0.10 | No foreign-owned files | `find ~/k-closing-alpha/data ~/.cache/kis -not -user ubuntu \| head` | empty |

## 1. Morning chain (D)

| # | Time KST | Unit | Evidence | PASS criterion |
|---|---|---|---|---|
| 1.1 | 06:50–07:00 | `kca-extended-backfill` (previous night) | `docker ps --format '{{.Names}}' \| grep kca-extended-backfill` | container **gone** before 07:05 (stop time 06:50; a running container here is a WARN — it would overlap the morning KIS usage) |
| 1.2 | 07:05 | `kca-kis-token-warmup` | `[SYS] stage=kis_token status=ISSUED` per host data slot | one ISSUED per slot, no `REPEAT_ISSUE` |
| 1.3 | 08:30 | `kca-price-ingest` | `[DATA] stage=price_ingest dates=[...]` line (journal D-1 23:30 UTC onward) | no `Failed to start`; KRX publishes T+1, so this is the run that ingests the previous trading day: `dates=['<previous trading day>'] rows≈2,700` and `price_history.parquet` max date == previous trading day (`dates=[]` is PASS only if that date is already in the panel) |
| 1.4 | 08:39:55 | `kca-auction-open` | `[DATA] stage=auction_capture ... phase=open date=D` | not FAILED / DEADLINE_EXCEEDED |
| 1.5 | 09:01 | `kca-paper-exit` | `[EXEC] stage=paper_exit ...`; `data/paper/` ledger | previous trading day's open lots closed; FAIL on `exit_window_expired`, `calendar_disagreement`, `shifted_session_hold` |
| 1.6 | 11:30 | `kca-price-ingest` (midday) | unit finished | no failure |

## 2. Decision window (D 15:20–15:35) — critical

| # | Time KST | Unit | Evidence | PASS criterion |
|---|---|---|---|---|
| 2.1 | 15:20:00 | `kca-collect` | journal D 06:20–06:30 UTC | finishes < 9 min; no `status=SKIP`; no `reason=calendar_disagreement`; no `stage=orderbook_persist status=FAILED` |
| 2.2 | 15:21:00 | `kca-predict` | `[SYS] stage=run_outcome job=predict run_date=D outcome=OK` | OK with `n_picks ≥ 1` or OK `admitted_below_top_k`; `NO_DECISION` = FAIL (report reason) |
| 2.3 | 15:21:05 | `kca-auction-close` | `stage=auction_capture ... phase=close date=D` | completes; not FAILED |
| 2.4 | 15:30:30 | `kca-finalize-close` | `[DATA] stage=close_finalization date=D n_finalized=… n_unconfirmed=…` | `n_finalized ≥ 1`, `n_unconfirmed = 0` (or each explained) |
| 2.5 | ~15:33 + 15:34 | `kca-paper-entry` | chained run + backstop | first run books entries for D; second logs `status=SKIP reason=already_recorded` |
| 2.6 | window | tokens | `journalctl --user --since "D 06:15 UTC" --until "D 06:40 UTC" \| grep -E "DECISION_WINDOW_ISSUE\|kis_token_cache status=UNREADABLE\|kis_token_lock status="` | zero `DECISION_WINDOW_ISSUE`/`UNREADABLE`/`UNOPENABLE`; `READONLY_FALLBACK` = WARN |
| 2.7 | window | decision artefacts | `data/parquet/topk_decisions.parquet` rows with `decision_date == D` | present and consistent with 2.2 |

## 3. Same-day capture & archive (D 15:40–20:35) — critical

Manifest query used below (read-only, run on or-vps; replace `D`):

```bash
docker run --rm --user $(id -u):$(id -g) -e HOME=/tmp -e UV_CACHE_DIR=/tmp/uv-cache \
  -v ~/k-closing-alpha/data:/app/data:ro ghcr.io/kthyeong/k-closing-alpha:latest uv run python -c "
from src.data.capture_store import CaptureStore, resolve_capture_root
from src.config.collection import CollectionSettings
s = CaptureStore(resolve_capture_root(CollectionSettings()))
for m in s.read_manifests('D'):
    c = m.context
    if c.vendor != 'owner-local':
        print(c.dataset.value, c.session, c.vendor, c.capture_reason, m.status.value, len(m.entries), m.completed_at.isoformat()[:19])
"
```

The **latest** manifest per (dataset, session) decides; `owner-local PENDING` rows are pre-acquisition markers and are ignored.

| # | Time KST | Unit / dataset | PASS criterion |
|---|---|---|---|
| 3.1 | 15:40 → ≤ 18:40 | `kca-archive-intraday-regular` | unit finished (timeout 3h); no `stage=intraday_archive status=ERROR`; `DEGRADED` = WARN with reason |
| 3.2 | 〃 | MINUTE_BARS `regular` (kis) | COMPLETE, entries ≈ cohort size |
| 3.3 | 〃 | TRADE_TICKS `regular` (kis) | **COMPLETE**. `PARTIAL` = **FAIL-until-fixed**: report immediately with the deadline (a same-day rerun before 24:00 fills missing symbols) |
| 3.4 | 15:40 → ~20:00 | `kca-aftermarket-book` | `[DATA] stage=aftermarket_book status=COMPLETE date=D manifests=N`; per-block lines `failed=` small relative to `ok=`; unit finished before 20:05 (timeout 4h40); `cohort=MISSING` / `rank_pool=MISSING` = FAIL |
| 3.5 | 20:05 → ~20:35 | `kca-archive-intraday` | unit finished; `[DATA] stage=krx_aftermarket date=D rows=…` and `[DATA] stage=aftermarket_ticks date=D krx_rows=… nxt_rows=… nxt_symbols=…` present |
| 3.6 | 〃 | MINUTE_BARS `nxt_aftermarket` + `nxt_premarket` (kiwoom), `krx_aftermarket` (kis) | each COMPLETE |
| 3.7 | 〃 | TRADE_TICKS `krx_aftermarket` + `nxt_aftermarket` (kiwoom) | each **COMPLETE** (PARTIAL = FAIL-until-fixed, same-day deadline) |
| 3.8 | 21:30 | `kca-price-ingest` (evening) | finished; `stage=price_ingest` line present; `price_history.parquet` max date == previous trading day. `dates=[] rows=0` is **normal** here: KRX OpenAPI has not published D yet (measured 2026-09-17…09-28: the 21:30 run never ingested D). D lands at D+1 08:30 (§1.3) |
| 3.9 | 21:35 | `kca-altdata-capture` | `[DATA] stage=altdata_capture date=…`; `DEGRADED` only for DART `quota_exceeded` (key shared with another project) |

## 4. Audit & alerting (D 20:15)

| # | Check | PASS criterion |
|---|---|---|
| 4.1 | `kca-daily-audit` ran after both archives and the book job | started after 3.5 finished; finished < 30 min |
| 4.2 | Digest verdict | `[DATA] stage=daily_audit day=trading status=OK ...`; if `WARNING`, list each problem. `step=intraday_complete ... status=FAIL` names the session and reasons — report verbatim |
| 4.3 | Heartbeat | `data/logs/heartbeat/daily_audit.json`: `snapshot_date == D`, `day_kind == trading`, `undelivered_alerts == 0` |
| 4.4 | Alert channel | `ls data/logs/alerts/outbox/*.json 2>/dev/null \| wc -l` == 0; no `alert_drain=FAILED` |
| 4.5 | Run outcomes | `data/logs/events/<YYYY-MM>/D.jsonl`: every job OK; list DEGRADED / NO_DECISION with reason |
| 4.6 | Expiry notices | digest `expiry_notices=` / `expiry_warnings=`; none expected before 2026-11-04 (first: `KIS_DATA_1` 2026-12-04; KRX calendar horizon 2026-12-31) |

## 5. Backup & retention (D 21:00 / 22:15)

| # | Unit | PASS criterion |
|---|---|---|
| 5.1 | `kca-backup-prune` (21:00) | `[SYS] stage=backup_prune dry_run=False ...`; no `step=... status=failed` |
| 5.2 | `kca-backup` (22:15) | `[SYS] stage=offsite_backup status=ok duration_s=…`; `data/history/capture/offsite/last_run.json` `status == "ok"`, `started_at` on D |
| 5.3 | Backup duration | `duration_s` < 7200 = PASS; 7200–14400 = WARN (report segment/member counts); timeout (4h) = FAIL |
| 5.4 | Drive lock contention | no exit 75 (flock wait on `%t/quant-gdrive.lock`). The lock is shared with `krx-host-backup` (23:30), `crypto-pilot-backup` (09:15/21:30) and kca prune/snapshot/verify — if contended, name the competing unit |
| 5.5 | Core panel latch | no `core_panel:*:missing` |

## 6. Overnight extended backfill (D 23:05 → D+1 06:50)

| # | Check | PASS criterion |
|---|---|---|
| 6.1 | Started | `kca-extended-backfill` active from D 14:05 UTC; first `[DATA] stage=extended_backfill date=… session=…` line within 5 min |
| 6.2 | Conflicts at start | `docker ps` shows no other `kca-*` container except the backfill after 23:10; `krx-collector` logs `state=NIGHT_SLEEP` (it does not use KIS at night) |
| 6.3 | Finished | journal around D 21:50 UTC (= D+1 06:50 KST): `[DATA] stage=extended_backfill status=DONE tasks_done=… tasks_remaining=… complete=… no_trades=… not_listed=… failed=… stopped_by_deadline=…`; unit inactive, `Result=success` |
| 6.4 | Progress | `tasks_done > 0`; while catching up, `stopped_by_deadline=True` with `tasks_remaining` shrinking night over night is normal; once caught up `tasks_remaining=0` |
| 6.5 | Price-basis routing | per-task lines carry `repaired= adjusted= basis_failed=`. Expected: `adjusted` ≈ a few % of `complete`; `basis_failed` covers symbols dropped from NXT (Kiwoom has no raw history) and adjusted KRX-aftermarket days — normal, non-terminal. **FAIL** if `basis_failed ≥ complete` on most tasks (raw source broken) or any `price_basis_raw_basis_transport` burst |
| 6.6 | Ledger | `data/history/intraday/backfill_ledger/extended_sessions.parquet`: `price_basis` values only `kis_raw`, `kiwoom_raw` or empty; count of COMPLETE rows with empty `price_basis` on adjusted days shrinks after each night (legacy repair) |

## 7. External watchdog (D+1)

| # | Check | PASS criterion |
|---|---|---|
| 7.1 | Watchdog probe (read-only): `cd ~/k-closing-alpha && ~/.local/bin/uv run --no-sync python -m src.tools.watchdog_probe` | `[SYS] stage=watchdog status=OK problems=none` |
| 7.2 | GitHub "External Dead-Man Watchdog" at D+1 07:40 KST | succeeded (else `UNVERIFIED` if API inaccessible) |

## 8. Reference — schedules on or-vps (KST)

kca (20 timers):

| Time | Unit | Days |
|---|---|---|
| 07:05 | kca-kis-token-warmup | Mon–Fri |
| 08:30 / 11:30 / 21:30 | kca-price-ingest | Mon–Fri |
| 08:39:55 | kca-auction-open | Mon–Fri |
| 09:01 | kca-paper-exit | Mon–Fri |
| 15:20 | kca-collect | Mon–Fri |
| 15:21 | kca-predict | Mon–Fri |
| 15:21:05 | kca-auction-close | Mon–Fri |
| 15:30:30 | kca-finalize-close | Mon–Fri |
| 15:34 | kca-paper-entry (also chained after finalize) | Mon–Fri |
| 15:40 | kca-archive-intraday-regular | Mon–Fri |
| 15:40 | kca-aftermarket-book | Mon–Fri |
| 20:05 | kca-archive-intraday | Mon–Fri |
| 20:15 | kca-daily-audit | Mon–Fri |
| 21:00 | kca-backup-prune | Mon–Fri |
| 21:35 | kca-altdata-capture | Mon–Fri |
| 22:15 | kca-backup | Mon–Fri |
| 23:05 | kca-extended-backfill (stops 06:50) | every day |
| Sat 22:00 | kca-retrain | weekly |
| Sun 10:00 | kca-core-snapshot | weekly |
| Sun 11:00 | kca-offsite-verify | weekly |

Other projects sharing the host (must not be disturbed; report only):

| Time | Unit / container | Interaction with kca |
|---|---|---|
| market hours | `krx-collector` (krx-alpha) | KIS WebSocket on DATA_1–4; `NIGHT_SLEEP` at night. DATA_2 is its exclusive intraday REST slot — kca uses DATA_2/3 only after 15:40 and overnight |
| 22:00 | `krx-deferred-recreate` | recreates krx containers |
| 23:30 | `krx-host-backup` | shares the gdrive flock |
| 08:30 / 23:30 | `mt-etf-daily-refresh` | shares the Toss key (not used by kca overnight anymore) |
| 16:40 / 16:55 / Sat 10:00 | `mt-etf-contest-*` | none known |
| every 5 min / 09:15, 21:30 | `crypto-pilot-liveness` / `crypto-pilot-backup` | backup shares the gdrive flock |

## 9. Weekly items (when the run spans a weekend)

| # | Unit | PASS criterion |
|---|---|---|
| 9.1 | `kca-retrain` (Sat 22:00) | finished < 2h; registry line appended |
| 9.2 | `kca-core-snapshot` (Sun 10:00) | `[SYS] stage=core_snapshot ... status=ok` |
| 9.3 | `kca-offsite-verify` (Sun 11:00) | `[SYS] stage=offsite_verify checked=N missing=0 mismatched=0`; `stage=restore_drill ... status=OK` per tier |
| 9.4 | `kca-extended-backfill` (Sat/Sun 23:05) | runs every night; same criteria as §6 |

## 10. krx-alpha collector (`krx-collector`, same host)

krx-alpha runs as one long-lived container (`python -m src.orchestration.daemon`, TZ=Asia/Seoul) plus two host
timers. It is a separate project: **never restart, recreate or `docker compose` it** — report only.

Evidence sources:
- Persistent event log (survives container recreation; WARNING+ and EVENT-tagged INFO only):
  `~/krx-alpha/data/logs/events-{daemon,cli-collect-stream,cli-collect-snapshots,cli-collect-aftermarket,normalize-worker,toss-program-trades-sync}.jsonl`
  (JSON lines; filter `ts` by D and read `fields.stage` / `fields.status`).
- Container stdout (heartbeats every 600 s; lost on recreate): `docker logs --since <dur> krx-collector`.
- Manifests: `~/krx-alpha/data/manifest/D.json` (regular), `~/krx-alpha/data/manifest/aftermarket/D.{nxt,krx}.shard-NN.json`.
- Host backup status: `~/.local/state/krx-alpha/host_backup_status.json`; log `~/logs/krx-host-backup-<UTC date>.log`.

Daemon states (KST, weekdays): PRE_MARKET_SLEEP (<08:20) → STREAMER_ACTIVE (08:20) → FULL_ACTIVE (08:50) →
AFTER_MARKET_ACTIVE (15:40) → POST_MARKET_EOD (20:00) → NIGHT_SLEEP (20:30); weekends WEEKEND_SLEEP; holidays log
`stage=session status=SKIP reason=market_holiday`.

| # | Time KST | Check | PASS criterion |
|---|---|---|---|
| 10.0 | any | Container health: `docker ps --filter name=krx-collector`; `docker inspect -f '{{.State.StartedAt}} {{.RestartCount}}' krx-collector` | running; no start inside 08:10–22:00 on D (deploys are deferred to the 22:00 recreate); no `[SYS] stage=daemon_restart status=UNCLEAN\|RESTARTED_AFTER_CRASH` or `stage=daemon_crash` on D |
| 10.1 | 08:20 | Session start | `state_change ... to=STREAMER_ACTIVE`; `[SYS] stage=kis_token_preflight ... result=cache` (`issued` = WARN: kca 07:05 warmup did not cover that key, cross-check §1.2; `fail` = FAIL); `[DAEMON] stage=orchestration status=OK ... ready=True`; `[DATA] stage=bars_refresh ... status=OK` (`reason=kis_fallback_used` = WARN); `[DAEMON] stage=streamer status=STARTED` |
| 10.2 | 08:50–15:40 | Regular session | `to=FULL_ACTIVE` at ~08:50; no unrecovered `stage=ingest_watchdog status=STALE` (each STALE must be followed by `RECOVERED`); no `streamer status=FAIL reason=circuit_open`; snapshot child `stage=snapshot_summary ... failed_jobs=` small and no `snapshot_dq ... status=FAIL`; **no `EGW00201`** (per-key TPS overrun) in krx or kca logs 15:15–15:40 |
| 10.3 | 15:31–20:00 | After-market | no `aftermarket_reselection status=FAIL` / `aftermarket_plan status=FAIL`; `to=AFTER_MARKET_ACTIVE` at 15:40; shard manifests exist for **both** `nxt` and `krx` (normally 2 shards each); `stream_disconnect`/`stream_gap` are WARN (report count and times), `stream_outage` or `aftermarket_stream status=FAIL` = FAIL |
| 10.4 | 20:00–20:30 | EOD | `to=POST_MARKET_EOD`; `[DAEMON] stage=eod_maintenance ... status=OK` (DEGRADED = WARN with reason, e.g. `aftermarket_not_ready`; FAIL/`session_data_gap` = FAIL); no `session_reconciliation status=FAIL`, `eod_offload status=FAIL` (hint `rclone_config_reconnect_gdrive` = gdrive auth expired), `eod_disk_guard status=FAIL`; `[SYS] stage=digest status=SENT` |
| 10.5 | after EOD | Program-trades sync | `[DAEMON] stage=program_trades_sync status=OK` (child `status=OK fetched_symbols=… appended_rows=…`); STALE = WARN |
| 10.6 | 20:30 → | Night | `to=NIGHT_SLEEP`; heartbeats `state=NIGHT_SLEEP` (no KIS usage overnight — kca backfill has the keys to itself) |
| 10.7 | 22:00 | `krx-deferred-recreate` | `systemctl --user show krx-deferred-recreate.service -p Result` = success; the container restart at ~22:00 is expected and benign |
| 10.8 | 23:30 | `krx-host-backup` | `host_backup_status.json`: `rc == 0`, `last_ok_at` on D (UTC ~14:3x); `lock_wait_s` small; log has `step=data status=ok rc=0` and `step=prune status=ok rc=0`. `step=lock status=failed rc=75` = FAIL (name `holders`) |

Cross-project invariants (report any violation as FAIL for both projects):
- **KIS DATA_2** is krx's exclusive snapshot REST slot 08:00–15:39; kca may use DATA_2 only from 15:40 (book) and
  overnight (backfill). **DATA_1** is reserved for kca's 15:20 decision window / 15:30 finalize.
- **After-market 15:40–20:00**: krx shards hold KIS WebSocket sessions on the data-slot keys while `kca-aftermarket-book`
  polls REST on DATA_2/3. WS and REST budgets are separate, but any `EGW00201` or WS approval-key error in this window
  must be attributed to one side.
- **Token cache** `~/.cache/kis` is shared: kca's 07:05 warmup is the primary issuer; krx issues only when the cache is
  empty. A same-day reissue refusal or `REPEAT_ISSUE` in either project = FAIL.
- **gdrive lock** `%t/quant-gdrive.lock` is shared by `kca-backup` (22:15), `krx-host-backup` (23:30),
  `crypto-pilot-backup` and kca weekly jobs; a kca backup running past 23:30 delays krx (check `lock_wait_s`).

---

## Report format (write in Korean)

```
### 🩺 [OPS CHECK] kca·krx 하루 루틴 점검 (D, 점검 시각 HH:MM KST)
> 🚦 Verdict: ✅ HEALTHY | ⚠️ DEGRADED | ❌ BROKEN

| 섹션 | 결과 | 근거 요약 |
|---|---|---|
| 0 배포·기반 | PASS/WARN/FAIL/PENDING | ... |
| 1 오전 체인 | ... | ... |
| 2 결정창 | ... | ... |
| 3 당일 수집·아카이브 | ... | ... |
| 4 감사·알림 | ... | ... |
| 5 백업 | ... | ... |
| 6 야간 소급 | ... | ... |
| 7 워치독 | ... | ... |
| 10 krx-alpha | ... | ... |

- ⏰ 당일 마감 항목: <3.3 / 3.7 등 24:00 전에 재수집해야 하는 항목과 남은 시간, 없으면 "없음">
- 🔍 발견사항: <FAIL/WARN 항목별 원인 추정과 증거 1~2줄>
- 🎯 권장 조치: <사람 승인이 필요한 조치만, 없으면 "없음">
```

Verdict rules: any FAIL in §0, §2, §3, §4, §10.2–10.4 or a cross-project invariant → ❌ BROKEN; any other FAIL or any WARN → ⚠️ DEGRADED; otherwise
✅ HEALTHY. A same-day deadline item that is still recoverable must be listed first, before any other finding.
