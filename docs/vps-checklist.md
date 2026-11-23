# kca or-vps Audit Runbook (AI-executed, v4)

> Host `or-vps` (hostname `trading-bot`, user `ubuntu`, 2 vCPU, 11 GiB). Repo `~/k-closing-alpha`; data root `~/k-closing-alpha/data` (capture `data/history/capture`); 32 `kca-*` user timers + `kca-alert@`; container jobs run through `~/.local/lib/kca/kca-run` (image `ghcr.io/kthyeong/k-closing-alpha:latest`, `--rm`), host jobs via `flock … ~/.local/bin/uv run`.
> Days: **D** = audited trading day (M/W: the previous trading day; O/E/N: today), **P** = trading day before D, **N** = next trading day. Times are KST; host clock and `journalctl` are UTC.
> Last verified: 2026-11-02 09:40–09:45 KST against or-vps rev `88d3452` (image `sha-1681172`, code-identical). All five bundles were executed read-only: O for D=2026-11-02, and M/E/N/W for D=2026-10-30 (§8 log)..

## 0. Agent contract

- Loop: pick the window (§2) → run that window's bundled probe (§5.1) in ONE ssh call → every line is `ID STATUS key=val …` → for each non-PASS run the row's Drill (§4/§5.2) → triage (§6) → report per `/home/kth/krx-alpha/docs/vps-agent.md` §8. Shared rules (secrets, freeze, destructive actions, commit/deploy, report format) live only in that brief.
- Verdicts: `PASS | WARN | FAIL | UNKNOWN` (probe exception or missing evidence; never counts as PASS) `| SKIP(reason)` (not applicable by day kind or optional timer off). `CTX INFO` is the first line; `ALL SKIP(non_trading_day)` ends the run on a closed D.
- kca deltas to the shared rules:
  - **Blackout** (`src/tools/deploy_window.py:18-58`): Mon–Fri **08:30–09:40** and **15:10–15:43** (weekday holidays count). Inside: only `systemctl`, `journalctl`, `ls`, `stat`, small-file reads. No bundled probe, no `uv run`, no `rclone`, no parquet reads.
  - **Drive lock** 22:15–00:15 (kca-backup holds `%t/quant-gdrive.lock`; krx-host-backup 23:30 waits on it): no `rclone check`/`size`; `lsf` of a few dirs is fine. Bundles run `nice -n 19 timeout 300`.
  - **Forbidden entry points** (write ledgers, issue tokens, send mail, take locks): never run `src.daily.*`, `src.ml.*`, `src.backfill.*`, `src.tools.{audit_reconcile,daily_audit,alerts,offsite_backup,backup_prune,core_snapshot,capture_offsite,kis_token_warmup,kiwoom_token_rotate,order_resolve}` as `main`. Importing read helpers (`audit_daily_completeness`, `PaperLedger.load_*`, `resolve_session_day`) is allowed. `kca-run --print-argv` prints only.
  - Host tools: host `python3` has no pandas/pyarrow, so use `~/.local/bin/uv run --no-sync python` inside `~/k-closing-alpha`. rclone is `~/.local/bin/rclone` (not on the ssh PATH). Remote root is `gdrive:quant-lake/live/k-closing-alpha`.
  - Containers are `--rm`, so `docker logs` is empty: the journal is the record. Several jobs write no OK run outcome (§3 "evidence"); judge them from the journal or the ledger as the probe does.

## 1. Daily-critical invariants (CORE)

| ID | Domain | Invariant (measurable) | Verified in window | Deadline KST | Sev |
|---|---|---|---|---|---|
| C01 | COLLECT | 15:20 decision input: `stage=realtime_coverage coverage≥0.99 status=COMPLETE` and `manifests/<D>/decision-<D>-*` complete | E (M) | 15:30 | P0 |
| C03 | COLLECT | Aftermarket book (live-only): every `aftermarket-book-<D>-b<n>-<krx\|nxt>` manifest complete, and the journal reports `status=COMPLETE manifests=<same count>` | N (M) | 20:20 | P1 |
| C04 | COLLECT | Evening archive, 7 streams (regular/krx_after/nxt_after bars+ticks, nxt_pre bars): each manifest complete, or a ticks PARTIAL healed by a complete `tape-<D>-<session>-publish` | N (M) | 21:15 | P1 |
| C07 | COLLECT | EOD panel: `price_ingest OK`, `panel_max_date == expected_panel_date == P`, `stale_panel=false` | O | 09:50 | P1 |
| C08 | COLLECT | Tape sweep report covers D, `expiring_needs=0 expired=0 expired_recoverable=0 disk_guard=false` (Kiwoom tape depth 30 d) | N | 21:15 | P1 |
| C17 | COLLECT | Tape sweep runtime: need scan (20:35 → last `stage=tape_universe`) ≤ 30 min, report by 21:10, no `status=DEADLINE` (walk deadline 21:15; an overrun exits 0 and is otherwise invisible); trend 17.6 min 11-12, 19.6 min 11-13 | N | 21:15 | P2 |
| C11 | STORE | All 8 completeness keys True (`archive close_confirmed decision paper_entry paper_exit minute_bars intraday_complete price_history_fresh`) | N (M) | 21:50 | P1 |
| Q08 | STORE | Non-empty partitions for D: 1m `regular nxt_premarket nxt_aftermarket krx_aftermarket`, ticks `regular krx_aftermarket nxt_aftermarket`, `orderbook/aftermarket` | N (M) | 21:15 | P1 |
| Q01 | STORE | `archive.parquet` D: rows>0, unique symbols, 0 nulls, confirmed share ≥0.9, OHLC/등락률 consistent, 0 causality violations | E | 15:45 | P0 |
| B01 | BACKUP | `capture/offsite/last_run.json` `status=ok`, started at or after the last 22:15 slot, all steps ok, `missing_sealed_members=0`, core panels tracked with no issues | M | 01:00 | P1 (shrink P0) |
| B02 | BACKUP | For D, every local ledger segment (`raw normalized manifests`) is listed remotely under `capture_sealed/<tier>/<YYYY-MM>/<D>/`, and no local file older than the run is unsealed | M | 01:00 | P0 |
| B05 | BACKUP | Last weekday prune logged its summary with no failed step; remote `_deleted` has nothing older than 30 d at the last prune; fewer than 7 day dirs due next (D1 cap) | M, N | 21:20 | P2 (cap risk P1) |
| P01 | PAPER | D entry recorded once: entry orders == top-k rows (no duplicate order_id/symbol), or a NO_DECISION row | E (M) | 15:45 | P0 |
| P02 | PAPER | Every P lot has a `<entry>:exit:<D>` sell and a `trades` row; no open lot with decision_date < D | O (M) | 09:30 | P0 |
| P03 | PAPER | Conservation: \|nav − cash − open_market_value\| ≤ 1 KRW for D, one nav row per day, cumulative cost monotone, friction > 0 on every trade, net = gross − cost, fill ids unique | O, E (M) | 09:35 / 15:45 | P0 |
| P04 | PAPER | Causality: `input_available ≤ inference_started ≤ decided < 15:30:00` for every top-k row, and no entry fill before the last `decided_at` | E (M) | 15:45 | P0 |
| P06 | PAPER | Shadow orders (`data/orders/<env>/journal.jsonl` via outcome metrics): `n_unknown=0`, `n_unknown_unresolved_total=0`, `n_missing_at_broker=0`, no shortfall, `n_lots`/`n_picks` fully accounted for | O, E | 09:35 / 15:45 | P0 |

## 2. Check windows (when to look)

| Window | KST (day kind) | Why this moment | Probe | Check IDs |
|---|---|---|---|---|
| **M** next-morning (mandatory) | business day 07:30–08:20 (Monday covers Fri + weekend nights) | 22:15 backup and 23:05 backfill are done; `last_run.json` is not yet overwritten; D's shrink backups still exist; ends before the 08:30 blackout | `kca_probe M` | H01 R01 R02 R03 R05 C01 C03 C04 C05 C11 Q08 P01–P04 B01 B02 B04 B05 A01 A03 A04 X01 |
| **O** post-exit (mandatory) | business day 09:45–11:00 | exit window closed 09:30; price-ingest D-1 landed (a failed 08:30 run is auto-rerun at 09:40) | `kca_probe O` | R01 R02 C05 C07 Q05 P02 P03 P06 A01 A03 X01 |
| **E** post-decision (mandatory) | business day 15:45–16:30 | entry chain ends 15:45; the archive row is still the decision snapshot | `kca_probe E` | R01 R02 C01 C05 C12 C13 C14 C16 Q01 P01 P03 P04 P06 A01 A03 X01 |
| **L** liveness (optional) | 16:30–19:50 | aftermarket-book is the only long live capture (runs to ~19:59) | §5.2 `R06` | R06 |
| **N** evening (mandatory; M re-checks if missed) | business day 21:50–22:10 | archive 20:05, tape 20:35, prune 21:00 and audit 21:20 are done; must finish before the 22:15 backup takes the Drive lock | `kca_probe N` | R01 R03 C03 C04 C06 C08 C17 C13 C15 C11 Q06 Q07 Q08 Q09 Q10 B05 A01 A03 X01 |
| **W** weekly (mandatory) | Monday 07:30–08:20, after M | Sat 22:00 retrain, Sun 10:00 snapshot, Sun 11:00 verify | `kca_probe W` + §5.1 local | H02 H03 H04 R04 P08 B07 B08 Q11 A05 |
| **Ev** event | after a deploy/alert/incident | parity of what changed | H02 H04 + affected rows | — |

On a CLOSED weekday every probe prints `ALL SKIP(non_trading_day)` for D=today. M/W automatically audit the previous trading day.

Evidence volatility:
| Evidence | Lifetime / overwritten when | Must be read in window |
|---|---|---|
| `capture/offsite/last_run.json`, `in_progress.json`, `scan_state.json` | overwritten nightly 22:15 | M |
| `capture/offsite/deferred_history.json` | last 14 runs | M (trend) |
| `logs/heartbeat/daily_audit.json` | rewritten 21:20; `reconciled_at`/open_issues every 10 min 07:00–23:00 | N (D snapshot), M |
| `capture/staging/tape_sweep/last_report.json` | overwritten each 20:35 sweep | N |
| `capture/backups/intraday/<session>/<day>` (shrink evidence; never offsite) | deleted after 3 days by the 21:00 prune (`backup_prune.py:37`) | N, at the latest M+2 |
| local `capture/{raw,normalized,manifests}/<day>` | removed after 30 d once remote md5 verified; extended-backfill re-creates old trade-date dirs at 23:05, pruned at the next weekday 21:00 | any |
| remote `_deleted/<day>` / `snapshots/` | 30 d / last 8 + first of each of 12 months | W |
| journal | latest-only `systemctl show`; LAST resets on reboot for non-persistent timers | any |
| paper + order ledgers | permanent, append-only; offsite | any |
| `logs/remediation/ledger.jsonl` (offsite) | rows older than 60 d dropped by the 21:00 prune under `kca-audit.lock`, except unsettled STARTED and unparseable lines | within 60 d |
| `logs/{audit_digests/<D>.txt,alerts/silent/<D>.jsonl,alerts/digest/<D>.jsonl,events/<YYYY-MM>/<D>.jsonl}` (diagnostics; never offsite) | deleted after 60 days by the 21:00 prune (`OPS_LOG_RETENTION_DAYS`) | within 60 d |
| `logs/alerts/incidents.json` (offsite), `logs/alerts/outbox/` (local only) | incident ledger (one page per kind per KST day); undelivered pages | any |

**Alert policy** (`src/tools/alert_policy.py`). Mail (`[긴급][kca] …`, body exactly `무슨 일 / 자동 조치 / 지금 할 일`) is sent only for CRITICAL incidents: failed `kca-order-{entry,exit,reconcile-entry,reconcile-exit}`, `kca-collect`, `kca-predict`, `kca-finalize-close`; order-session DEGRADED outcomes (except entry-reconcile `partial_or_unfilled`); collect/predict NO_DECISION; finalize_close `picks_unconfirmed/zero_confirmed/empty_archive`; price-ingest `stale_panel` (only recorded from 11:00, i.e. after the 11:30 safety run, including a crashed safety run that left the panel stale); `kca-kis-token-warmup` still failed after auto-remediation; a stale KIS token in the 2 h before 15:20; offsite backup failing ≥ 24 h (two nightly slots); foreign `krx-deferred-recreate` and every `kse-alert@*` relay (kse could not deliver its own CRITICAL page). One incident = one mail per KST day across all paths (reminders at most daily, not before 08:00); a paged measured incident (token, backup) that clears gets one `[해결됨][kca]` line. Everything else (paper, auctions, research roster, postclose snapshot, decision extras, aftermarket book, altdata, archive, extended backfill, tape sweep, retrain, audit/reconcile self-failures, api notice, flow coverage, kiwoom rotate, `krx-host-backup`, `vps-image-gc`, `crypto-pilot-*`, undelivered backlog) is SILENT: `[SYS] stage=alert status=SILENT` + `logs/alerts/silent/<D>.jsonl`, heartbeat, digests. Safe-rerun CRITICAL units are DEFERRED while auto-remediation has tries left (≤ 2 h).

## 3. Routine inventory (code-derived)

Weekday timers also fire on weekday holidays; jobs self-skip through `resolve_session_day`/`trading_session_gate` plus the KIS oracle `is_kis_trading_day`. **P** = `Persistent=true`. Outcome = `data/logs/events/<YYYY-MM>/<D>.jsonl` (`src/tools/run_outcome.py:22-34`; last line per job wins).

| Routine | Trigger | Entry point | Outputs | Success evidence | Deadline | Loss |
|---|---|---|---|---|---|---|
| audit-reconcile (host, `flock -E 75`, exit 75 ok) | /10 min 07:00–23:00 daily | `src/tools/audit_reconcile.py` + `auto_remediation.py` | heartbeat `reconciled_at`, `remediation/ledger.jsonl`, `logs/alerts/digest/<D>.jsonl`; pages measured CRITICAL keys after remediation, `[해결됨]` on clear | `reconciled_at` ≤ 10 min | — | LOW |
| kis-token-warmup (critical, Restart ×4) P | Mon–Fri 07:05 | `src/tools/kis_token_warmup.py` (skips only verified CLOSED, :51-58) | `~/.cache/kis/token_<sha12>.json` | `slot=<…> issued=True` ×7 | 07:10 | MED |
| kiwoom-token-rotate P | Mon–Fri 07:10 | `src/tools/kiwoom_token_rotate.py` | Kiwoom token | `status=ROTATED\|UNCHANGED` | 07:30 | MED |
| api-notice-watch (host) P | 07:40, 17:40 daily | `src/tools/api_notice_watch.py` | `heartbeat/api_notice_watch_state.json` | Result=success | — | LOW |
| price-ingest (standard) P | Mon–Fri 08:30, 11:30 | `src/daily/price_ingest.py:main` :1305 | `history/price_history.parquet` (:1232), security classification | outcome `OK`, `panel_max_date=P` | 09:50 | MED |
| auction-open (optional, batch) | Mon–Fri 08:39:55 | `src/daily/auction_capture.py:main` :995 `--phase open` | raw + `auction-open-<D>-*` manifest | `status=COMPLETE phase=open entries=N` | 09:05 | MED |
| order-exit | Mon–Fri 08:40:10 | `src/daily/order_session.py --phase exit` | `data/orders/<env>/journal.jsonl` (env `KIS_TRADING_ENV` off/mock/live, `src/api/kis/trading.py:104-121`) | outcome `OK exit_complete` / `SKIPPED no_open_positions\|trading_disabled` | 08:59 | HIGH |
| order-reconcile-exit | Mon–Fri 09:00:40 | `order_session --phase reconcile-exit` | journal | outcome `OK reconcile_complete` | 09:33 | HIGH |
| paper-exit (decision) P | Mon–Fri 09:01 | `src/daily/paper_trade.py --phase exit` (window to 09:30, `src/config/market_session.py:97-103`) | `data/paper/{orders,fills,trades,nav}.parquet` | journal `stage=paper_ledger_refresh as_of=D … n_open=0`; **no OK outcome** | 09:30 | HIGH |
| collect (decision) | Mon–Fri 15:20 | `src/daily/collect.py:main` :1100 | `archive.parquet` via `src/daily/archive.py:upsert_archive_snapshot` :165; `capture/decision/<D>/<run>/input.parquet` | `stage=realtime_coverage … status=COMPLETE` (min 0.99, :201); **outcome only on SKIPPED** | 15:29 | HIGH |
| research-roster (standard, Kiwoom ka10032; silent) | Mon–Fri 15:20:30 (inside the blackout) | `src/daily/roster_capture.py:main` | `manifests/<D>/research-roster-<D>-*` + normalized roster frame (members, drop reasons); never carries a cohort | outcome `OK members=…` (`DEGRADED truncated`; `SKIPPED disabled\|non_trading_day`) | 15:20:55 | MED |
| predict (After=collect) | Mon–Fri 15:21 | `src/daily/predict.py:main` :455 | `parquet/{rank_pool_predictions,topk_decisions}.parquet` | outcome `OK` | 15:30 | HIGH |
| auction-close (optional) | Mon–Fri 15:20:25 (not ordered after collect; cohort from `declared/<D>/` until the decision manifest exists, `cohort_source=declared`) | `auction_capture --phase close` | raw + manifest `auction-close-<D>-*` (cohort) + sibling `…-research` (roster extras, dispatched after the cohort each round); early declaration `declared/<D>/<cohort>.json` is not a manifest | `status=COMPLETE phase=close` | 15:36 | MED |
| order-entry | Mon–Fri 15:21:15 | `order_session --phase entry` | journal | outcome `OK entry_complete` | 15:29 | HIGH |
| finalize-close (decision; ExecStopPost starts paper-entry) | Mon–Fri 15:30:30 | `src/daily/finalize_close.py:main` :481 | archive `종가_확정`, raw `close_confirmation` | outcome `OK` (DEGRADED `empty_archive\|picks_unconfirmed\|zero_confirmed`, :148-156) | 15:33 | HIGH |
| order-reconcile-entry | Mon–Fri 15:30:40 | `order_session --phase reconcile-entry` | journal | outcome `OK` | 15:43 | HIGH |
| paper-entry (decision) P | Mon–Fri 15:34 + finalize trigger | `paper_trade --phase entry` | paper orders/fills/decisions/nav | entry fills at the 종가 (`paper_broker.py:220-274`); second run `already_recorded` (:498,:520); outcome only NO_DECISION/DEGRADED/SKIPPED | 15:45 | HIGH |
| postclose-snapshot (standard, batch slot; silent) | Mon–Fri 15:31:30 | `src/daily/postclose_snapshot.py:main` | `manifests/<D>/postclose-<D>-*`, `history/research/{postclose_strength,investor_estimate_postclose}/<YYYY-MM>/<D>.parquet` (write-once) | outcome `OK` (`DEGRADED partial`) | 15:39 (no request after `COLLECTION_POSTCLOSE_SNAPSHOT_DEADLINE_HHMMSS`) | MED |
| archive-intraday-regular P | Mon–Fri 15:40 | `src/daily/archive_intraday.py:main` :1039 `--phase regular` | 1m+ticks `regular`, manifests `archive-<D>-regular-{bars,ticks}-*` | `stage=intraday_archive_stream … status=COMPLETE` | 18:40 | HIGH |
| aftermarket-book (batch) | Mon–Fri 15:40 (to ~19:59 by design) | `src/daily/aftermarket_book.py:main` :632 | `history/orderbook/aftermarket/<YYYY-MM>/<D>.parquet`, 35 primary manifests `aftermarket-book-<D>-b<n>-<venue>` + `…-research` siblings for blocks with sparse rounds | `stage=aftermarket_book status=COMPLETE date=D manifests=N` | 20:20 | HIGH |
| auction-path (bulk, DB증권) | Mon–Fri 15:45 | `src/daily/auction_path.py:main` :304 | `history/auction_path/<YYYY-MM>/<D>.parquet` (cohort + roster extras), sibling `auction-path-<D>-*-research` | outcome from manifest (:270) | 19:00 | MED |
| decision-extras (bulk, no broker) | Mon–Fri 16:30 | `src/daily/extras_extract.py:main` (`--from/--to` retro over local raw) | `history/research/{decision_extras,investor_estimate_decision}/<YYYY-MM>/<D>.parquet` (overwritable, derived from the decision run raw) | outcome `OK` (`SKIPPED no_decision_input`, `DEGRADED invalid_raw_evidence`) | 17:00 | LOW |
| archive-intraday P | Mon–Fri 20:05 | `archive_intraday --phase aftermarket` | 1m `nxt_premarket nxt_aftermarket krx_aftermarket`, ticks `krx_aftermarket nxt_aftermarket` (since 2026-09-29, market_session.py:74) | stream lines COMPLETE; outcome only DEGRADED/SKIPPED | 23:05 | HIGH |
| tape-sweep (bulk, Kiwoom; After=archive-intraday) P | Mon–Fri 20:35 | `src/daily/tick_tape_sweep.py:main` :330 | `capture/staging/tape_backfill/ledger.jsonl`, `staging/tape_sweep/last_report.json`, `tape-<D>-<session>-publish-*` manifests | report day = D; **no outcome** | 21:15 (`TAPE_SWEEP_DEADLINE_HHMMSS`) | HIGH |
| backup-prune (host, Drive lock) P | Mon–Fri 21:00 | `src/tools/backup_prune.py` (+`capture_offsite.py:980-1159`) | deletes remote `_deleted/<day>` > 30 d (cap 7/subtree), local `backups/intraday` > 3 d, local sealed tiers > 30 d after a remote md5 check, local ops-log day files > 60 d, remediation ledger rows > 60 d | `stage=backup_prune dry_run=False purged= … ops_logs_removed= remediation_dropped=` | 21:20 | MED |
| daily-audit (host, `kca-audit.lock`) P | Mon–Fri 21:20 | `src/tools/daily_audit.py` | heartbeat v2, `audit_digests/<D>.txt`, `logs/alerts/digest/<D>.jsonl`; mail only for CRITICAL keys | `stage=daily_audit day=trading status=OK subject=[kca] 🟢 …` | 21:50 | MED |
| altdata-capture (optional) | Mon–Fri 21:35 | `src/daily/altdata_capture.py:main` :137 | `history/altdata/*`, `_manifest.json` | panels `status=ok` | 22:05 | MED |
| backup (host, `flock -w 7200` Drive lock; OnSuccess=audit-reconcile) P | **daily** 22:15 | `src/tools/offsite_backup.py:363-546` | `capture_sealed/<tier>/<YYYY-MM>/<D>/seg-<sha16>.tar.zst` (immutable, md5), `rclone copy data/ artifacts/` (backup-dir `_deleted/<sub>/<day>`), ledger `capture/offsite/ledger/<tier>/<D>.jsonl` | `last_run.json status=ok`; `stage=offsite_backup status=ok`; **no outcome** | 01:00 (`deadline_at`) | HIGH |
| extended-backfill (bulk, batch) | daily 23:05 | `src/backfill/intraday/extended_session_backfill.py:main` :1350 | past-date partitions, `backfill_ledger/extended_sessions.parquet`, `1m_kis_adjusted`, raw for old trade dates | outcome `OK`, `status=DONE tasks_remaining=0 pending_conflicts=0` | 07:05 | MED |
| retrain (6g/1.8 cpu) P | Sat 22:00 | `src/ml/retrain.py --train-core-bundle` (PIT-CORE: extends `data/history/pit_train/rows/` then trains) | `artifacts/models/topk_ranker/retrain_registry.jsonl`, new `pit_train/rows/<D>.parquet` | registry `PROMOTED\|PROMOTED_UNGATED`, `stage=pit_store extend days_written=` | Sun 00:00 | LOW |
| core-snapshot (host, Drive lock) P | Sun 10:00 | `src/tools/core_snapshot.py` | `snapshots/<D>/` (price_history, archive, topk, rank_pool, paper/*, bundle) | `stage=core_snapshot … status=ok` | Sun 12:00 | MED |
| offsite-verify (host, Drive lock) P | Sun 11:00 | `src/tools/capture_offsite.py verify` :1268-1468 | drill staging removed after | `missing=0 mismatched=0`, 3 `restore_drill … status=OK` | Sun 13:00 | HIGH |
| kca-alert@ (host) | OnFailure of every kca unit (+ `krx-deferred-recreate`, `krx-host-backup`, `vps-image-gc`, `crypto-pilot-*`, `kse-alert@*`) | `src/tools/alerts.py` + `alert_policy.py` | CRITICAL: one page per kind/day (`incidents.json`), outbox; SILENT/DEFERRED: `logs/alerts/silent/<D>.jsonl` | `stage=alert unit=<unit> status=SILENT\|DEFERRED\|SENT\|DEDUPED`, `Finished kca-alert@<unit>` | +5 min | MED |
| GitHub `watchdog.yml` | cron `40 22 * * *` UTC = 07:40 KST every day (runs 2–4.5 h late) | `src/tools/watchdog_probe.py`; GHCR PAT expiry ≤ 30 d | — | run success | — | LOW |
| GitHub `deploy.yml` | push to main | `vps-deploy-lib.sh` pull → preflight → `blackout_wait` → promote → converge (`git reset --hard`, `src.tools.code_sync`) | units, wrapper | run success | — | — |

Optional timers (`OPTIONAL_MANUAL_TIMERS`, `src/tools/code_sync.py:35-37`): auction-open, auction-close, altdata-capture. They are never auto-enabled (`install_systemd.sh` enables 29), but all 3 are enabled on the host. Not daily: `1m/regular_consolidated` (Toss backfill, last 2026-08-31) and `1m_kis_adjusted` (extended-backfill output).

## 4. Check catalogue

| ID | Tier | Windows | Assert (machine-checkable PASS) | Fail → Sev | Drill |
|---|---|---|---|---|---|
| H01 | EXT | M | 32 `kca-*` timers listed, no `--failed` kca unit, `Linger=yes` (WARN: a timer not enabled) | P1 | `systemctl --user list-timers 'kca-*' --all` |
| H02 | EXT | W, Ev | host `git status` clean, `kca-run` == repo wrapper, no unit DRIFT / NOT_IN_REPO, `latest` image ID == `sha-<HEAD>`; local: VPS HEAD == `origin/main` (or a deploy waits in `blackout_wait`) | P1 | §5.2 `H02-local` |
| H03 | EXT | W | no exited `kca-*` container, no kca container outside its §3 window; no new `*.bak*/*.pre_*/*.stale*` under `data/` or `artifacts/` | P2 | `docker ps -a`, `find … -maxdepth 4` |
| H04 | EXT | W, Ev | latest `deploy.yml` run on main is `success` (a red test job freezes `latest`); after a history rewrite, pinned SHAs in `tests/` fixtures are ancestors of `origin/main`; the 22 wrapper units print `rc=0` via `--print-argv` | P1 | §5.2 `H04` |
| R01 | EXT | M O E N | each unit due in the window has ≥1 `Finished` since its slot (optional timers only if enabled); a `Failed with result` followed by a Finished rerun is WARN | P1 (decision chain P0) | `journalctl --user -u kca-<u> --since <UTC>` |
| R02 | EXT | M O E | outcome per job ∈ allowed set: price_ingest OK; order_exit OK/SKIPPED(no_open_positions\|trading_disabled); order_reconcile_exit OK/SKIPPED(no_orders); predict OK; order_entry OK; finalize_close OK; order_reconcile_entry OK; extended_backfill OK | P1 (decision P0) | `events/<YYYY-MM>/<D>.jsonl` |
| R03 | EXT | M N | lists DEGRADED/NO_DECISION outcomes of D (WARN; each must map to an alert/known condition). `auction_path DEGRADED` is research (P2). A DEGRADED `extended_backfill` is actionable (transient outage, conflict); deterministic gaps show only as `deterministic_failed/exhausted` counts | P2 | outcome `reason`/metrics |
| R04 | EXT | W | last registry row `PROMOTED\|PROMOTED_UNGATED\|MANUAL_HOTFIX` ≤ 8 d old and the live `model_version` is in the registry. A `strategy or screen changed vs live bundle` REJECTED fails the unit by design: the operator certifies and reruns with `--skip-promotion-gate`; auto-remediation never reruns retrain. PIT-CORE freshness: `strategy_id == KCA-TOPK-COSTAWARE-002`; `pit_train/rows` newest day and the registry `training_cutoff` both ≥ the trading day 6 sessions before today (FAIL: a silent `days_written=0 reason=no_usable_days` still trains on the stale store); the last Sat run has a `stage=pit_store extend` line (WARN absent); WARN on `oom\|exit code 137\|Killed` in the retrain journal (6 GB container; measured peak 3.1 GB on 2026-11-09) | P1 | `journalctl -u kca-retrain` |
| R05 | EXT | M | today 07:05: ≥7 distinct `slot=… issued=True` (DATA_n, PRIMARY, TRADING_MOCK) and the Kiwoom rotate status line; CLOSED day → `non_trading_day` skip | P1 | warmup journal |
| R06 | EXT | L | `kca-aftermarket-book` active with container `kca-aftermarket-book` up between 15:40 and 19:59; ~92 % of its 4 h 40 min timeout is normal | P1 | §5.2 `R06` |
| C01 | CORE | E M | §1 | P0 | collect journal `stage=realtime_quote/requote` |
| C03 | CORE | N M | §1; PARTIAL (WARN) entry reasons `deadline\|vendor_failure\|missed_start`; `nxt_not_listed` is normal | P1 | §5.2 `C-manifest` |
| C04 | CORE | N M | §1; output `streams={…: complete\|tape_healed\|partial(n failed)\|missing}`. Partial bars = WARN; missing = FAIL | P1 | §5.2 `C-manifest`, journal `stage=intraday_archive_stream` |
| C05 | EXT | O E M | auction open/close primary manifests complete, `-research` siblings excluded (SKIP if timer off) | P2 | auction journal `status=LATE/DEGRADED` |
| C06 | EXT | N | `auction-path-<D>-*` manifest complete (else WARN; 10-30 PARTIAL) | P2 | outcome metrics |
| C12 | EXT | E | research roster manifest COMPLETE and completed ≤ 15:20:55, members 150–400 (outcome metrics); SKIP when the outcome is SKIPPED or the timer is off | P2 | roster journal `stage=research_roster` |
| C13 | EXT | E N | `-research` sibling manifests present for auction-close (E) and auction-path/aftermarket-book (N); deadline/missed-start share of their entries ≤ 10 %. Primary manifests are judged by C03/C05/C06 without them | P2 | sibling `manifest-partial.json` reasons |
| C14 | EXT | E | postclose manifest present, completed ≤ 15:39, good-entry share ≥ 0.95, `research/postclose_strength/<D>` exists | P2 | postclose journal `stage=postclose_snapshot` |
| C15 | EXT | N | `research/decision_extras/<D>` rows == decision cohort rows, `kiwoom_cntr_str` filled ≥ 0.9 (Toss-union names have none), investor series partition exists | P2 | `uv run python -m src.daily.extras_extract --date <D>` (idempotent) |
| C16 | EXT | E | `auction_close` outcome metrics for D: PASS when no close capture entry hit the deadline (`n_deadline == 0`, `rounds_with_deadline == 0`); otherwise WARN with `cause` `overcommit` (plan OVERCOMMIT), `late_start` (`start_lag_s > 5`) or `vendor`; no outcome record = WARN `outcome=absent`; SKIP(timer_off) when the timer is off | P2 | `events/<YYYY-MM>/<D>.jsonl` `auction_close` metrics |
| C07 | CORE | O | §1; WARN on `flow_shortfall`/`program_flow_shortfall`. A 08:30 `stale_panel_pending_retry` is healthy when 11:30 is OK; an 11:30 DEGRADED `stale_panel` (or a failed 11:30 run with a stale panel) pages `[긴급][kca] 가격 이력 미갱신` | P1 | price-ingest journal |
| C08 | CORE | N | §1 (report `days_checked[-1] == D`) | P1 | `last_report.json` `unresolved`, `stage=tape_sweep status=WARNING` |
| C11 | CORE | N M | §1: `audit_daily_completeness(D)` (7 keys) + `intraday_complete` = the daily-audit run for D logged no `step=intraday_complete status=FAIL date=D` (mirrors `daily_audit.py:2308`) | P1 | daily-audit journal for D |
| Q01 | CORE | E | §1 thresholds: confirmed share <0.9 FAIL, <0.99 WARN; `수급_실패` share >0.1 WARN; other flags >0 FAIL; `등락률` mismatch >1 % rows FAIL (computed on 종가, not 결정_종가); 종가 vs 결정_종가 gap p99 >3 % WARN. Causality (`execution_timestamp` ≥ 15:30) is judged on confirmed rows only: an unconfirmed row keeps its 15:20 snapshot ts by design (`finalize_close` never promotes frozen prices; `paper_broker` rejects a fill whose ts precedes `placed_at`). Rows follow the market screen (93–1,247/day); `snap_in_1520` drops at large `n_raw` (0.694 at 1,247 rows) and is info | P0 | §5.2 `Q-spot` |
| Q05 | EXT | O | price_history day P: rows/prev5 median ∈ [0.97, 1.03], no dup (day or total), no invalid OHLCV, one `kospi_pct` | P1 | — |
| Q06 | EXT | N | 1m regular D: no dup `(symbol, ts_hms)`, no invalid bar, ts covers 09:01–15:29 | P1 | — |
| Q07 | EXT | N | day P: 1m volume / EOD volume median ≤1.01 (≈0.96 for vendor `ls`; judge drift vs baseline ±0.02, share <0.9 ±0.05); last-bar close matches `close_raw` share ≥0.98 (FAIL <0.9) | P2 | — |
| Q08 | CORE | N M | §1 (`auction_path` missing = WARN) | P1 | `ls history/intraday/*/<session>/<YYYY-MM>/` |
| Q09 | EXT | N | every `stage=intraday_replace … SHRINK_ALLOWED` since P 23:00 has `backups/intraday/<session>/<D>/*-pre-*.parquet`; WARN if `quarantine/shrink/<D>` has files. **Evidence expires D+3** | P0 | journal grep |
| Q10 | EXT | N | `tick_source_diff` metrics for D plus heartbeat for D without `intraday:regular_ticks:*:{source_diff_systemic,certified_gap}`. A tick vs bar gap with `tape_complete` proof is a vendor difference (info) | P2 | heartbeat `info_notes` |
| Q11 | EXT | W | altdata panels `ok` with trading-day lag ≤ 2 (`credit_balance` ≤ 4); backfill ledgers (`extended_sessions toss_regular nxt_calibration`) have no FAILED with ≥3 attempts; NXT table sha == fit report; quarantine entries listed (each needs a `moves-*.json`) | P2 | — |
| Q12 | EXT | M (manual) | spot check: 2 top-k + 1 random symbol agree across archive 종가, `price_history.close_raw` (next day), last 1m bar, paper entry `fill_price` | P1 | §5.2 `Q-spot` |
| B01 | CORE | M | §1; WARN: start delay >15 min (Drive lock contention), `deferred_dates>0`, `in_progress.json` present | P1 | `journalctl -u kca-backup`, `steps.*.stderr` |
| B02 | CORE | M | §1 (`segs>0`, `remote_missing0`, `unsealed_pre_run0` per tier) | P0 | §5.2 `B-parity` |
| B03 | EXT | W | remote `data/` and `artifacts/` vs local: `rclone check --one-way --size-only` shows `+`/`*` only for files modified after `last_run.started_at` (heartbeat, ledgers, backfill partitions) | P1 | §5.2 `B03` |
| B04 | EXT | M | deferred trend not stalled 3 runs and oldest deferred ≤ 21 d; WARN if `capture_seal.status != ok` (then `deferred_dates` reads 0 falsely, D6) | P2 | `deferred_history.json` |
| B05 | CORE | M N | §1; WARN `sealed_kept>0` or a local tier date < prune cutoff with mtime before the prune (named, reason in journal) | P2/P1 | prune journal `stage=capture_prune` |
| B07 | EXT | W | Sun verify `missing=0 mismatched=0` and 3 restore drills (raw, normalized, manifests) `OK` | P1 | verify journal `finding=` |
| B08 | EXT | W | remote `snapshots/` newest == last Sunday; core-snapshot `status=ok` | P1 | core-snapshot journal |
| P01 | CORE | E M | §1; WARN when fills < orders (statuses `ZERO_QTY NO_SNAPSHOT_ROW UNCONFIRMED INSUFFICIENT_CASH`) | P0 | `paper/orders.parquet` |
| P02 | CORE | O M | §1 | P0 | paper-exit journal (`exit_window_expired`) |
| P03 | CORE | O E M | §1 | P0 | §5.2 `P-ledger` |
| P04 | CORE | E M | §1 | P0 | `topk_decisions.parquet` |
| P06 | CORE | O E | §1. Entry reconcile `n_intents` counts exit + entry intents for the day (e.g. 6 for 3 picks). A stuck UNKNOWN intent is operator-only (`src.daily.order_resolve`, dry-run first). In mock, a connection fault on an order POST is recovered automatically: `kis_order ... transport_error=<class>`, `resend=entry|exit` send/result lines are normal traces; `duplicate_suspect` or `resend_conflict` WARNING is a finding (possible duplicate BUY) | P0 | `data/orders/<env>/journal.jsonl` |
| P08 | EXT | W | Live edge vs backtest: per-trade mean `net_ret` from `paper/trades.parquet` over the last 20 and 60 exit days (`exit_filled_at`), against the backtest expectation ≈ +36 bp/trade (PIT-CORE K3, 2-tick cost). WARN when the 60-day mean ≤ 0 or the 20-day mean < −50 bp (with ≥ 30 trades); never FAIL (single-trade sd is several %). Also reports win rate and mean cost | P2 | `paper/trades.parquet` |
| P07 | EXT | E (manual) | paper entry symbols == order-journal entry intent symbols for D (same top-k) | P1 | journal `kind=intent` payload |
| A01 | EXT | all | heartbeat `schema_version 2`, `undelivered_alerts 0`, `reconciled_at` ≤ 15 min (07:00–23:05), `snapshot_date == D` after 21:30 (N) and in M; WARN on open issues | P1 | `cat logs/heartbeat/*.json`, `tail remediation/ledger.jsonl` |
| A02 | EXT | M | digest `logs/audit_digests/<D>.txt` line 1 `[kca] 🟢 <D> 일일점검 완료`; open issue keys each map to a finding; remediation rows `started` without `resolved/unresolved` are in flight; `unresolved/skipped_budget` need the operator; no forbidden unit (collect, predict, finalize, paper-*, aftermarket-book, auction-*, retrain, daily-audit, kiwoom-rotate) in the ledger | P1 | `head -1 audit_digests/<D>.txt` |
| A03 | EXT | all | every unit with `Failed with result` since the window start has a `kca-alert@<unit>` run | P1 | alert journal |
| A04 | EXT | M | dashboard kca card all OK (non-OK must map to a finding; its `[[schedule]]` list lives in `quant-dashboard/config.toml`) | P2 | `public/status.json` |
| A05 | EXT | W | `watchdog.yml` last 5 scheduled runs `success` | P1 | `gh run list --workflow watchdog.yml` |
| X01 | EXT | all | kca journal since the window start: no `HostBrokerDirNotSharedError AdmissionDirNotSharedError SCHEMA_INVALID` (FAIL); WARN on `IGW00201 EGW00201 EGW00133 EGW00121 AUTH_REJECTED ADOPTED_ROTATED TokenIssueThrottled TokenStoreLockTimeout TOKEN_REPLACED http=429`; any hit on decision units 15:20–15:31 = P1 | P2/P1 | grep with unit names |
| X02 | EXT | M | Drive lock: kca-backup start delay ≤ 15 min (B01 `start_delay_min`); other holders: `kse-backup` 02:30, `krx-host-backup` 23:30 (`krx-alpha/deploy/host/krx-host-backup.sh:7,131`) | P2 | `journalctl -u kca-backup -u krx-host-backup -u kse-backup` |
| X03 | EXT | — | shared, probed once by the ops brief: clock drift, disk/inodes < 80 %, memory, `~/.cache/kis` mode 700 + `.host-admission` marker + contract file names/owners, `~/quant-secrets` 700/600 | — | ops brief |
| X04 | EXT | — | shared, probed once by the ops brief: `deploy/vps-deploy-lib.sh` == krx `deploy/host/vps-deploy-lib.sh`; `src/api/host_broker_contract.py` == krx `deploy/shared/host_broker_contract.py`; `deploy/shared/kis_slot_assignments.toml` == krx canonical | — | `sha256sum` (local) |
| X05 | EXT | — | shared, probed once by the ops brief: `vps-image-gc` (05:30) keeps `latest` + 5 `sha-*`; krx-collector root files in `~/.cache/kis` re-owned by the reconcile `sudo chown` | — | ops brief |

## 5. Probes

### 5.1 Bundled window probes

Run from the local repo root, outside the blackouts. The probe code lives in `ops/probes/kca_probe.py` (single source). `D` is optional; it defaults to the previous trading day for M/W and to today for O/E/N. Window letters: M, O, E, N, W; the script exits 2 without output for any other argument.

```bash
kca_probe() { ssh -o ConnectTimeout=15 or-vps "cd ~/k-closing-alpha && TZ=Asia/Seoul nice -n 19 timeout 300 ~/.local/bin/uv run --no-sync python ops/probes/kca_probe.py $1 ${2:-}"; }
kca_probe M            # O | E | N | W ; e.g. kca_probe N 2026-10-30 re-audits a past day
# W adds local evidence (no ssh):
git -C /home/kth/k-closing-alpha fetch -q && git -C /home/kth/k-closing-alpha log -1 --format='%h %ci' origin/main
gh run list -R kthyeong/k-closing-alpha --workflow deploy.yml --limit 1; gh run list -R kthyeong/k-closing-alpha --workflow watchdog.yml --limit 5
```

### 5.2 Drill-down probes

- `H02-local`: `ssh or-vps 'git -C ~/k-closing-alpha log -1 --format=%h'` vs `git ls-remote https://github.com/kthyeong/k-closing-alpha refs/heads/main`.
- `H04`: `ssh or-vps 'for f in ~/.config/systemd/user/kca-*.service; do l=$(grep -m1 "^ExecStart=%h/.local/lib/kca/kca-run " "$f") || continue; eval "~/.local/lib/kca/kca-run --print-argv ${l#ExecStart=%h/.local/lib/kca/kca-run }" >/dev/null 2>&1; echo "$(basename $f) rc=$?"; done | grep -v "rc=0$"'` (empty = PASS; a `USAGE_ERROR` means the unit cannot start). Fixture SHAs: `git merge-base --is-ancestor <sha> origin/main`.
- `R06`: `ssh or-vps 'systemctl --user is-active kca-aftermarket-book.service; docker ps --filter name=kca-aftermarket-book --format "{{.Status}}"'`.
- `C-manifest <D> <prefix>`: `ssh or-vps 'cd ~/k-closing-alpha/data/history/capture/manifests/<D> && for d in <prefix>*; do echo "$d $(ls $d | tr "\n" " ")"; done'`. For PARTIAL entries: `python3 -c` reading `manifest-partial.json` `entries[].{symbol,status,reason}` with a Counter.
- `B-parity <D> <tier>`: `~/.local/bin/rclone lsf gdrive:quant-lake/live/k-closing-alpha/capture_sealed/<tier>/<YYYY-MM>/<D>/` vs `python3 -c` listing `segment_name` from `capture/offsite/ledger/<tier>/<D>.jsonl` (or `<D>.jsonl.gz` via `gzip.open(..., "rt")` when only the compacted copy exists).
- `B03` (W only, outside 22:15–00:15): `cd ~/k-closing-alpha && nice -n 19 timeout 600 ~/.local/bin/rclone check data gdrive:quant-lake/live/k-closing-alpha/data --one-way --size-only --exclude "/history/capture/**" --combined - 2>/dev/null | awk '$1=="+"||$1=="*"' | tail -20`.
- `P-ledger`: `uv run --no-sync python -c` using `PaperLedger(root=Path("data/paper")).load_open_positions()` and `nav.parquet` tail 5 (`as_of_date nav cash open_market_value cumulative_cost`).
- `Q-spot <D> <sym>`: archive row (`종가 결정_종가 종가_확정`), last 1m bar, `price_history.close_raw` (next day), paper fill `fill_price` for the symbol.
- Journal window for any unit: `journalctl --user -u kca-<u>.service --since "<UTC>" --until "<UTC>" -o short-iso | grep -vE "pam_unix|COMMAND=" | grep -E "Starting|Finished|Failed|stage=|outcome=" | cut -c1-250 | tail -30`. Convert with `k(){ date -u -d "TZ=\"Asia/Seoul\" $1" +"%F %T"; }`.

## 6. Triage map

| Symptom (verdict line) | Likely cause | Where in code / logs | Fix direction (root cause) |
|---|---|---|---|
| `C01 FAIL coverage<0.99` | KIS decision slots throttled (`EGW00201`) or requote deadline `152800` hit | `collect.py:201,954,1309`; slots 1/5 (`kis_slot_assignments.toml`) | slot contention vs krx; requote budget |
| `C03 WARN partial` / `INCOMPLETE held_roster_unavailable` | aftermarket roster missing or `slot_token_failed` (slots 2/3) | `aftermarket_book.py:589-673` | token/slot health at 15:40 |
| `C04 partial(n failed)` on ticks, no tape_healed | vendor_failure in the evening archive; tape sweep not yet published | `archive_intraday.py:292-372`; `tick_tape_sweep.py` | read `open` (holes not yet filled by a tape manifest), not `failed`, and check C08; the cause lives in the raw pages' `return_code` (see `kiwoom_tick_vendor_failure_retry_spec.md`); the tape horizon is 30 d |
| `C07 FAIL stale` / `R01 price-ingest failed` | KRX `data-dbg.krx.co.kr` read timeout (10-30, 11-03); the strict fetch now retries ~10 min per endpoint (`PRICE_INGEST_KRX_RETRY`), then the unit fails SILENTLY (no OnFailure mail) and is auto-rerun after the 09:40 blackout | remediation ledger `failed_unit:kca-price-ingest`; `logs/alerts/silent/<D>.jsonl` | expected once; the only page is `stale_panel` after the 11:30 safety run; repeat = KRX outage |
| `C08 WARN expiring>0` | Kiwoom tape needs ≤3 d from the 30-d depth | `config/collection.py:125-130`; `market_session.py:85-87` | auto `intraday:tape_expiring:` rerun of tape-sweep; else manual |
| `C11 missing=[intraday_complete]` | blocking `intraday:*` issues in the D audit | daily-audit journal `step=intraday_complete status=FAIL` | the issue keys name the session/symbol |
| `Q09 FAIL` | shrink replace without its pre-backup | `intraday_store.py:347,400-413` | P0: restore from `_deleted`/sealed raw before D+3 |
| `B01 FAIL status!=ok` | rclone exit (quota/auth), seal deadline 90 min, core panel shrink | `offsite_backup.py:465-537`; `last_run.steps.*.stderr` | `core_panels` issue = a data problem upstream (never accept a shrink) |
| `B01 WARN start_delay` | Drive lock held (kse/krx backup, prune overrun) | X02 journals | stagger schedules |
| `B02 FAIL remote_missing` | segment upload lost or ledger ahead of the remote | `capture_offsite.py:459-494` | rerun backup (safe-rerun list); then B07 drill |
| `B02 FAIL unsealed_pre_run` | seal skipped the date (deadline/newest-first order) | `capture_offsite.py:783-822` | check `dates_scanned`, deferred |
| `B05 FAIL cap_risk` | ≥7 `_deleted` day dirs due (prune missed days; backup is daily, prune weekdays) | `backup_prune.py:34,112-115` (D1) | operator run with `--max-purge-per-subtree` |
| `P02 FAIL unexited` | paper-exit `exit_window_expired`/no open quote | `paper_trade.py:~820`, `market_session.py:97-103` | never re-run paper-exit after 09:30 (forbidden); `paper_ledger_correct` by operator |
| `P03 FAIL nav_minus_cash_mv>1` | corrections or fill-id collision | `paper_broker.py:125-135,362-374` | `src/tools/paper_ledger_correct.py` (operator) |
| `P01 FAIL dup` | double entry (paper-entry runs twice; idempotency compares `decision_date` without dtype normalization, D4) | `paper_trade.py:412-428` | P0 incident |
| `X01 WARN EGW00121/ADOPTED_ROTATED` | a client sent a request without a token (fixed: `_handle_request` primes it; the daily 09:01 `kca-paper-exit` hit was this) or another process rotated the key | `token_<key_id>.json` mtime in `~/.cache/kis` (names/stat only) vs the hit time; `kis_token status=ISSUED` lines of all projects | mtime unchanged since warmup = a client bypassed `ensure_token` (code defect); mtime at the hit = attribute the issuer |
| `A01 FAIL reconciled_age` | reconcile lock busy (exit 75) or the unit is failing (SILENT: no mail) | `audit_reconcile.py` | journal of `kca-audit-reconcile` |
| Expected mail missing / unexpected mail | policy classification or same-day dedup | `alert_policy.py` tables; `logs/alerts/incidents.json`; `stage=alert status=` lines | change the policy table, never silence by editing the ledger |
| `UNKNOWN err=…` | probe drift (schema/path change) | this file | fix the probe in the same change (§8) |
| Restore needed: `price_history` (since `ed5c912`, a `data/` restore has no monolith) | offsite form is base+patch | `src/tools/price_history_offsite.py` | `rclone copy gdrive:quant-lake/live/k-closing-alpha/data/history/price_history_offsite <tmp>` then `uv run python -m src.tools.price_history_offsite restore --from-dir <tmp> --target <new path>` (refuses an existing target; checks md5 + rows). From a snapshot: first `rclone copy …/snapshots/_blobs/price_history <tmp>`, then `…/snapshots/<D>/data/history/price_history_offsite <tmp>`, same restore. Snapshots ≤ 2026-11-01 hold the monolith |
| `steps.offsite_ledger` FAIL `ledger_not_append_only` | a seal ledger was rewritten, not appended | `offsite_backup.py` `classify_ledger_files`; `capture_offsite.py:664` | the rewritten file was uploaded with a `_deleted` version; find the writer that rewrote it (root cause), never reset `ledger_upload_state.json` to silence it |
| `R02 DEGRADED entry_incomplete` + urgent mail `[긴급][kca] 종가 매수 주문 실패`, `P06 WARN not_sent=1` | order POST failed before any HTTP status (`kis_order verdict=ambiguous http=`); 11-11 `179880`: 23 ms, no status, likely a closed keep-alive connection | `kca-order-entry` journal; `src/api/kis/trading.py` `transport_error`; `order_gateway.py` `_entry_resend_allowed` | read `transport_error`; if `ServerDisconnectedError`/`ClientOSError`/`ClientConnectionResetError` in mock the gateway resends once after a 5 s settle re-read, so a remaining `not_found` means the resend was refused (halted, past 15:28, send cap) or the class was a timeout; live never resends (`LIVE_RESEND_ENABLED=False`) |
| `resend_conflict` / `duplicate_suspect` WARNING | final re-read found 2+ matching orders, or an unbound order matches an intent sent twice | `order_gateway.py` `_entry_final_check`, `reconcile` | do not resend; compare broker orders with the journal and cancel the surplus by hand in the broker UI if a duplicate BUY exists |

## 7. Baselines and known conditions

Baselines (trailing rule: compare with the prev-5 trading-day median, never with a fixed level):
| Metric | Seed (date) |
|---|---|
| archive rows/day | 93–1,247 (10-12..16); 156–172 (10-29/30) |
| price_history rows/day | ~2,762–2,767 (10-30) |
| 1m regular | 485 k bars / 1,273 symbols, vendor `ls`, vol ratio median 0.959 (10-16) |
| aftermarket-book manifests | 35 primary (18 × krx/nxt batches; 10-30); `-research` siblings: 35 primary + 21 research dirs (56 total; 11-05), auction-path 1 primary + 1 research, cut_share 0.0 |
| research roster | 261 members (11-05 live, finished 15:20:40, 10 s; run 15:20:30); 254 of 400 ranking rows in the 11-04 dry run; drops 138 `not_listed_in_panel` (135 ETF/ETN), 7 `non_screenable_class` (11-04 21:10 dry run) |
| tape sweep | needs 771, recovered 604, unresolved 167, expired 0 (10-30) |
| backup | 3–13 min; `segments` 2–170 per night (11-01: 2) |
| runtimes | postclose-snapshot 44 s (15:31:30–15:32:14, 591 entries: 432 TRADE_TICKS + 159 INVESTOR_ESTIMATE), research-roster 10 s, collect 40 s (11-05), predict done 15:21:13, collect 25–95 s, predict ~12 s, finalize 10 s–2.5 min, archive-regular 28–57 min, archive 9–33 min, tape-sweep 24–31 min, daily-audit 1.5–4.5 min (incl. ~80 s chown), extended-backfill ~2.5 min (until 11-04; direction-free superset drain runs to the 06:50 stop) |
| paper NAV | 0 open lots after exit (11-02) |

Known conditions:
| Since | Condition | Expected verdict while open | Re-check / expiry | Close when |
|---|---|---|---|---|
| 2026-11-14 | CSAT shifted session 2026-11-19 (10:00–16:30, `KRX_CALENDAR.shifted_sessions`). kca does not trade or collect on a SHIFTED day: predict `NO_DECISION session_shifted` and exit `DEGRADED shifted_session_hold` (11-18 lots) are SILENT; entry/reconcile `SKIPPED session_shifted`; collect `SKIP` (no cohort); archive-intraday (both phases), auction-open/close, auction-path, research-roster, postclose-snapshot `SKIPPED session_shifted`; aftermarket-book SKIP; tape sweep never selects 11-19 (no needs, nothing settled). 11-20: exit sells the 11-18 lots, auction-open roster may be PARTIAL (no 11-19 cohort), archive logs `missing_previous` | 11-19: O/E/N show those SKIPPED/SILENT outcomes, no `[긴급]` mail; A01/C11 list the SILENT audit gaps `collection:session:0:shifted`, `collection:cohort:0:missing_cohort`, auction/aftermarket-book missing and missing steps archive/close_confirmed/decision/minute_bars (expected, no page); 11-20 O: order exit OK with 11-18 lots sold | 2026-11-20 N | 11-20 N clean; reuse this row for 2027-01-04 once it is registered as SHIFTED |
| 2026-11-09 | PIT-CORE switch (`KCA-TOPK-COSTAWARE-002`, bundle `9dea1ec206ff`, cutoff 11-05, store 905 days; rollback copy `artifacts/models/_rollback/topk_ranker_A_20261109/`). First live decision 11-10 15:21: P04 shows the new `model_version`; first CORE retrain Sat 11-14 22:00 | R04 PASS from W 11-16 with `store_max`/`cutoff` ≥ 11-13; P08 n small until ~12-08 | 2026-11-16 | W 11-16 PASS |
| 2026-11-05 | CLOSED: extended-backfill reached `status=DONE tasks_remaining=0 complete=47611` on the 11-04 23:05 run (00:51) | — | — | — |
| 2026-11-04 | (closed 11-05, see above) Direction-free PIT superset (`d6abd78`): extended-backfill drains ~40 k new KIS symbol-days over several nights and stops at 06:50 | R02/R03 WARN `tasks_remaining>0` on the 23:05 run | 2026-11-12 | `status=DONE tasks_remaining=0` again |
| 2026-11-05 | Toss manual backfill finished `status=DONE tasks_done=927 tasks_remaining=0`, but `failed=6771` rows remain: one rerun for FAILED rows is still open (needs the Toss token: not run by the ops audit) | Q11 lists `toss_regular` FAILED rows | 2026-11-06 | rerun done and Q11 clean |
| 2026-11-04 | (unit part finished 11-05) Manual Toss pre-retention backfill `kca-toss-backfill-manual.service` (transient, 927 dates, started 21:16) | X01 WARN `TOKEN_REPLACED` (Toss) possible; Q11 lists `toss_regular` FAILED rows; H03 sees container `kca-toss-backfill` | 2026-11-06 | unit inactive with `status=DONE`; rerun once for FAILED rows |
| 2026-11-05 | A01 WARN `collection:ticks:273:session_violation` + `terminal_proof_missing` on N: the `postclose-*` manifest (`postclose-snapshot`, session `regular` TRADE_TICKS, last_event after close) is counted by `_audit_regular_bars` (`daily_audit.py:326-381`) as regular-tick coverage; audit false positive from the postclose snapshot, not a tick loss (C03/C04/C05 PASS) | A01 WARN each N until fixed | 2026-11-12 | the audit skips `reason=postclose-snapshot` entries (fix pending, spec first) |
| 2026-11-05 | C15 WARN `cntr_str_share=0.857` (39/273 rows NaN incl. 005930, 000660, 005490): rule needs >= 0.9; 11-04 was 0.945 (530 rows) | C15 WARN | 2026-11-12 | cause found (Kiwoom ranking list coverage vs cohort) or rule recalibrated |
| 2026-11-04 | `decision_extras` absent for 2026-09-18..10-02 (local raw already pruned after sealing) | none (C15 is N-only, D-scoped) | — | restore sealed raw if those days are needed |
| 2026-11-04 | `pit1520_panel` rebuilt before the superset drain finishes shows many `NOT_FETCHED` exclusions | none (manual build only) | 2026-11-12 | rebuild after the drain |
| 2026-11-13 | Local capture backlog after the 11-04 manual Toss pre-retention backfill (909 old `raw/` dates 2023-01..2026-10, 7.7 GB; plus the 10-30 sweep). Root cause of the slow drain: one `rclone md5sum <segment>` per ledger entry costs ~6.9 s (path walk), so the 40-min budget removed only 64-199 dirs/run (77 on 11-12). Fixed: prune reuses one `rclone md5sum <tier>/<YYYY-MM>` listing per month (~3.6 s) | B05 WARN `kept_local_pre_prune_n` > 0 until the first run after deploy | 2026-11-16 N (first weekday prune after deploy) | N shows B05 PASS (`kept_local_pre_prune_n=0`) and `stage=capture_prune` without `status=DEADLINE`; else recheck `status=LISTING_FAILED` in the prune journal |
| 2026-11-02 | D1 prune purge cap 7 per subtree vs daily backup | B05 PASS; FAIL `cap_risk` after a ≥1-week prune outage | 2026-12-01 | cap sized from the backup cadence (fix pending) |
| 2026-11-02 | D2 `audit_daily_completeness` omits `intraday_complete` (added in `main()` :2308) | C11 derives it from the journal | 2026-12-01 | helper returns 8 keys (fix pending) |
| 2026-11-02 | D3 heartbeat severity fallback parses the subject (`daily_audit.py:1910-1913`) | none (latent) | 2026-12-01 | fix pending |
| 2026-11-02 | D4 paper-entry idempotency compares `decision_date` unnormalized (`paper_trade.py:425`) | P01 would FAIL `dup` if it ever triggers | 2026-12-01 | fix pending |
| 2026-11-02 | D5 `kca-audit-reconcile` in its own safe-rerun list | none (busy check blocks it) | 2026-12-01 | fix pending |
| 2026-11-02 | D6 seal exception records `deferred_dates=0` | B04 WARN `seal_status_ok=False` | 2026-12-01 | fix pending |
| 2026-11-02 | D7 loose-copy excludes hardcoded to `/history/capture/**` | none unless `COLLECTION_ROOT`/`HISTORY_DIR` is overridden | 2026-12-01 | fix pending |
| 2026-11-02 | D8 core panel shrink tolerance 0 rows, no override for a legitimate dedupe | B01/B08 FAIL on a legitimate dedupe | 2026-12-01 | fix pending |
| 2026-11-02 | D9 no OK outcome for collect/paper/tape-sweep/backup/prune/verify/snapshot | probes use the journal/ledger instead | 2026-12-01 | fix pending |
| 2026-11-12 | D10 daily 09:01 `kca-paper-exit` `EGW00121` → `ADOPTED_ROTATED` (keys `134e3b771841`, then `c996650a5b46` = DATA_1) was a no-token first request (`Bearer None`), not a cross-project re-issue; fixed in `KisApiClient._handle_request` | X01 WARN on O until the fix deploys | 2026-11-16 | O shows no 09:01 hit |
| 2026-11-02 | D11 no in-code assert nav = cash + open MV | P03 is the only guard | 2026-12-01 | fix pending |
| 2026-10-29 | History rewrite of the README commit (`1681172` → `88d3452`): `deploy.yml` run 40891357295 test job failed, so `latest` stays on `sha-1681172` (code-identical tree). Fixture SHA fix committed locally in `e18a930` (not pushed) | H02 WARN `code_identical=True`; H04 FAIL | 2026-11-09 | the fixture fix is committed and deploy is green |
| 2026-11-02 | Offsite churn reduction `ed5c912`/`a2523e6` (pending deploy): `price_history` goes offsite as base+patch form `data/history/price_history_offsite/` (monolith excluded from the loose copy; first successful run moves the remote monolith to `_deleted/data/<day>/`); seal ledgers upload append-only without `_deleted` versions (`capture/offsite/ledger_upload_state.json`); snapshots store the base as shared blobs `snapshots/_blobs/price_history/` | B01: `steps.price_history_form` ok (`rebased` ≈ monthly) and `steps.offsite_ledger` ok with `rewritten=0` (else FAIL `ledger_not_append_only`); B03 lists remote `data/history/price_history.parquet` missing after the first run (expected); kca-backup peak RSS +2–3 GiB for ~15 s | 2026-11-16 | 3 clean nightly runs after deploy |
| 2026-11-02 | Dashboard kca card `kca.api_actions` WARN | A04 WARN | 2026-11-09 | mapped to an API notice/action finding, or cleared |
| 2026-11-10 | Seal ledger compaction: `<D>.jsonl` older than 14 days becomes `<D>.jsonl.gz` after the remote copy is proven identical (`backup_prune` step `ledger_compact`); append rehydrates first; B02 and B-parity read the gz fallback | B02 `segs>0` for compacted dates (no false FAIL); backup_prune log `ledger_compacted`/`ledger_bytes_saved` | after deploy: 3 clean nightly runs | NOT VERIFIED on VPS until the next M window |
| 2026-10-30 | `auction_path` DEGRADED / manifest PARTIAL (DB증권 research capture) | R03 WARN, C06 WARN | 2026-11-16 | 3 consecutive COMPLETE days |
| 2026-10-30 | 종가 vs 결정_종가 gap p99 3.09 % (normal ≲ 3 %) | Q01 WARN when p99 > 0.03 | 2026-11-16 | recalibrate the threshold if it is seen ≥ 3 of 10 days |
| 2026-10-30 | Extended backfill `pending_manifest_conflict` fixed and deployed (`pending_conflicts=0` 2026-10-30..11-01) | R02/R03 PASS; any recurrence from the 23:05 run is a regression (P2) | 2026-11-30 | — |
| 2026-10-31 | Host reboot 13:30 KST: non-persistent timers show LAST `-` until their next fire; judge R01 from the journal (the probe already does) | none | next reboot | — |
| 2026-10-22 | `price_basis_raw_basis_unavailable`: 393 keys, a stable vendor gap growing only with new corporate actions | heartbeat info | 2026-11-30 | — |
| 2026-10-16 | Daily audit `missing_bars` is judged on D only; extended backfill can rewrite past dates with `vendor=kis` trade-only symbols (10-16: 213) that a re-audit would flag | re-audits of past D may show it | 2026-11-30 | — |
| 2026-10-09 | `stale_kis_token:*` on CLOSED days (warmup skips; budget exhausted) | regression if seen again (P2) | 2026-11-30 | — |
| standing | Dashboard `[[schedule]]` list is separate (`quant-dashboard/config.toml`); a removed slot (price-ingest 21:30, `627cd1c`) raises `예정 실행이 지연` there | A04 WARN | on dashboard change | fix in the dashboard config |
| standing | ops-sentinel `job_late` vs `After=`: a unit held behind a slow predecessor stays PENDING (a slow archive-intraday raised a false `job_late:kca-tape-sweep`) | R01 PASS after it finishes | — | — |
| standing | Failed state persists until the unit's next run (e.g. retrain until next Saturday unless `reset-failed`) | H01 FAIL with the cause named | — | — |
| standing | `TOKEN_REPLACED` (Toss token shared by PC and VPS), Drive lock and KIS slots are cross-project; attribute precisely | X01/X02 WARN | — | — |
| 2026-11-06 | `P02 FAIL unexited=1` for `105740` (KOSDAQ): paper-exit (09:01:58) recorded UNFILLED because no dated open quote appeared within 6 x 10 s; the 09:00 open was delayed for many symbols (2-minute opening extension, first prints 09:02:09-09:02:43 on a gap-down day). Whether 105740 itself opened late or was halted is NOT established: krx does not collect it (not in the live 200) and the 15:20 archive has no 11-06 row for it. `load_open_positions` carries the lot into the next exit run (Mon 11-09 09:00). Do NOT re-run paper-exit and do NOT correct the ledger. Retry budget fix (6 -> 24 attempts) is committed locally (47268d4), pending push/deploy | P02 FAIL on this lot until it exits; confirm `105740:exit:2026-11-09` fills on 11-09 O, else drill (halt?) | 2026-11-09 | — |
| 2026-11-06 (N 21:55) | `kca_probe N` rev 02730bcd7: R01 R03 C03 C04 C05 C06 C13 C08 Q06-Q10 A03 X01 PASS; C11 FAIL missing `paper_exit` (A01 open issue `missing:paper_exit`; probably the 105740 UNFILLED lot, not verified); C15 WARN cntr_str_share 0.651 (152 rows); B05 WARN prune summary (5 kept pre-prune raw dirs, known); A01 WARN `collection:ticks:152:session_violation`/`terminal_proof_missing` (known postclose false positive) + `offsite_backup:deferred`. B01 NOT VERIFIED until the 22:15 backup completes: check at Monday M |

## 8. Maintenance

- A new routine needs: a §3 row, a §4 presence check and a content check, a window placement (§2), and a §7 baseline. A routine without a content check is incomplete.
- Every incident ends as a check, a triage row or a known condition with an expiry date. When a timer, unit, wrapper flag, data path or contract file changes, update §3, §4 and §5 in the same change.
- Retired IDs (v3 → v4; never reuse): `DEP-01/02/03` → H02; `DEP-04/07` → H04; `SCH-01` → H01; `SCH-02`, `RUN-01/02/05` → R01; `SCH-03` dropped (blackout tuple; deploy.yml enforces it); `RUN-03` → R02/R03; `RUN-04` → P06; `RUN-06` → A03/R03; `BRK-01/02/04/05/06` → X03/X05/R05; `BRK-03` → X01; `HOST-01/02/04` → X03/X05; `HOST-03/05` → H03; `DI-ARCH-01..09` → Q01; `DI-PH-*` → Q05; `DI-1M-01..03` → Q06; `DI-1M-04/05` → Q07; `DI-1M-06` → Q10; `DI-1M-07` → Q09; `DI-TOPK-*` → P04; `DI-PAPER-*` → P03; `DI-BAK-01` → B04; `PAP-01/03` → P02; `PAP-02` → P01; `PAP-04` → P03; `PAP-05` → P06; `ML-01/02` → R04; `ML-03`, `RS-01..06` → Q11; `RS-07` → Q08; `RS-08` → C08; `RS-09` → C03/C04; `BAK-01/02` → B01; `BAK-03` → B04; `BAK-04` → B07; `BAK-05` → B05/B08; `BAK-06` → B03; `BAK-07` → B05; `ALR-01` → A01; `ALR-02` → A02; `ALR-03` → A03; `ALR-04/05` → A04/A05.
- Last probe execution log:
| Date (KST) | Rev | Window / D | Result |
|---|---|---|---|
| 2026-11-05 21:50 | a8a01fc09 | E 15:48, N 21:50 / 11-05 (first research day) | E all PASS (C12 261 members 15:20:36, C13 auction-close 1749 entries cut 0.0, C14 15:32:14 good_share 1.0, P01-P06, X01 clean); N: R01 R03 C03 C04 C05 C06 C13(auction-path+aftermarket-book 3498 entries) C08 C11 Q06-Q10 A03 X01 PASS; WARN C15 (0.857), B05 (known prune), A01 (postclose false positive, see §7) |
| 2026-11-04 21:37 | 3dfe41f | E, N / 11-04 (research rows added) | C03 C04 C06 C15 PASS (C15 rows 530 = cohort, cntr_str 0.945); R01 FAIL / C12 C14 FAIL / C13 WARN only because research-roster, postclose-snapshot and decision-extras first fire on 11-05 (expected for any D ≤ 11-04); C05 WARN close partial (one KIS vendor_failure 15:25) |
| 2026-11-02 09:40 | 88d3452 | O / 11-02 | all PASS except X01 WARN (`EGW00121`/`AUTH_REJECTED`/`ADOPTED_ROTATED` ×1, D10); 5.6 s |
| 2026-11-02 09:41 | 88d3452 | M / 10-30 (re-audit) | CORE all PASS; WARN R03 (`auction_path DEGRADED`), X01 (D10), A04 (`kca.api_actions`); 23 s |
| 2026-11-02 09:41 | 88d3452 | E / 10-30 (retro) | CORE all PASS; WARN Q01 (`close_gap_p99=0.0309` > 0.03), X01; 5 s |
| 2026-11-02 09:41 | 88d3452 | N / 10-30 (retro) | CORE all PASS (C04 `regular-ticks: tape_healed`); WARN R03, C06 (`auction-path partial`), X01; 14 s |
| 2026-11-02 09:42 | 88d3452 | W / 10-30 | B07 B08 R04 Q11 PASS; WARN H02 (`latest`=sha-1681172, `code_identical=True`), H03 (5 known stray `.bak/.pre_*`); 4 s |
| 2026-11-02 09:43 | local | W local + drills | H04 FAIL: latest `deploy.yml` run 40891357295 (88d3452) test job failed; `--print-argv` 22/22 rc=0; fixture SHAs ancestors (working copy). A05 PASS (5/5 watchdog runs). R06 drill executed: inactive (outside 15:40–19:59; expected) |
| — | — | live E/N for 2026-11-02 | not executed live: their evidence does not exist before 15:45 / 21:50. Executed retroactively on 10-30 instead; extracted bodies pass `ast.parse` |
