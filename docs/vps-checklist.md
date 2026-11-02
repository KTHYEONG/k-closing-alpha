# kca or-vps Audit Runbook (AI-executed, v4)

> Host `or-vps` (hostname `trading-bot`, user `ubuntu`, 2 vCPU, 11 GiB). Repo `~/k-closing-alpha`; data root `~/k-closing-alpha/data` (capture `data/history/capture`); 29 `kca-*` user timers + `kca-alert@`; container jobs run through `~/.local/lib/kca/kca-run` (image `ghcr.io/kthyeong/k-closing-alpha:latest`, `--rm`), host jobs via `flock … ~/.local/bin/uv run`.
> Days: **D** = audited trading day (M/W: the previous trading day; O/E/N: today), **P** = trading day before D, **N** = next trading day. Times are KST; host clock and `journalctl` are UTC.
> Last verified: 2026-11-02 09:40–09:45 KST against or-vps rev `88d3452` (image `sha-1681172`, code-identical). All five bundles were executed read-only: O for D=2026-11-02, and M/E/N/W for D=2026-10-30 (§8 log)..

## 0. Agent contract

- Loop: pick the window (§2) → run that window's bundled probe (§5.1) in ONE ssh call → every line is `ID STATUS key=val …` → for each non-PASS run the row's Drill (§4/§5.2) → triage (§6) → report per `/home/kth/krx-alpha/docs/vps-ops-agent.md` §8. Shared rules (secrets, freeze, destructive actions, commit/deploy, report format) live only in that brief.
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
| **E** post-decision (mandatory) | business day 15:45–16:30 | entry chain ends 15:45; the archive row is still the decision snapshot | `kca_probe E` | R01 R02 C01 C05 Q01 P01 P03 P04 P06 A01 A03 X01 |
| **L** liveness (optional) | 16:30–19:50 | aftermarket-book is the only long live capture (runs to ~19:59) | §5.2 `R06` | R06 |
| **N** evening (mandatory; M re-checks if missed) | business day 21:50–22:10 | archive 20:05, tape 20:35, prune 21:00 and audit 21:20 are done; must finish before the 22:15 backup takes the Drive lock | `kca_probe N` | R01 R03 C03 C04 C06 C08 C11 Q06 Q07 Q08 Q09 Q10 B05 A01 A03 X01 |
| **W** weekly (mandatory) | Monday 07:30–08:20, after M | Sat 22:00 retrain, Sun 10:00 snapshot, Sun 11:00 verify | `kca_probe W` + §5.1 local | H02 H03 H04 R04 B07 B08 Q11 A05 |
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
| `logs/events/*.jsonl`, `remediation/ledger.jsonl`, `audit_digests/<D>.txt`, paper + order ledgers | permanent, append-only | any |

## 3. Routine inventory (code-derived)

Weekday timers also fire on weekday holidays; jobs self-skip through `resolve_session_day`/`trading_session_gate` plus the KIS oracle `is_kis_trading_day`. **P** = `Persistent=true`. Outcome = `data/logs/events/<YYYY-MM>/<D>.jsonl` (`src/tools/run_outcome.py:22-34`; last line per job wins).

| Routine | Trigger | Entry point | Outputs | Success evidence | Deadline | Loss |
|---|---|---|---|---|---|---|
| audit-reconcile (host, `flock -E 75`, exit 75 ok) | /10 min 07:00–23:00 daily | `src/tools/audit_reconcile.py` + `auto_remediation.py` | heartbeat `reconciled_at`, `remediation/ledger.jsonl` | `reconciled_at` ≤ 10 min | — | LOW |
| kis-token-warmup (critical, Restart ×4) P | Mon–Fri 07:05 | `src/tools/kis_token_warmup.py` (skips only verified CLOSED, :51-58) | `~/.cache/kis/token_<sha12>.json` | `slot=<…> issued=True` ×7 | 07:10 | MED |
| kiwoom-token-rotate P | Mon–Fri 07:10 | `src/tools/kiwoom_token_rotate.py` | Kiwoom token | `status=ROTATED\|UNCHANGED` | 07:30 | MED |
| api-notice-watch (host) P | 07:40, 17:40 daily | `src/tools/api_notice_watch.py` | `heartbeat/api_notice_watch_state.json` | Result=success | — | LOW |
| price-ingest (standard) P | Mon–Fri 08:30, 11:30 | `src/daily/price_ingest.py:main` :1305 | `history/price_history.parquet` (:1232), security classification | outcome `OK`, `panel_max_date=P` | 09:50 | MED |
| auction-open (optional, batch) | Mon–Fri 08:39:55 | `src/daily/auction_capture.py:main` :995 `--phase open` | raw + `auction-open-<D>-*` manifest | `status=COMPLETE phase=open entries=N` | 09:05 | MED |
| order-exit | Mon–Fri 08:40:10 | `src/daily/order_session.py --phase exit` | `data/orders/<env>/journal.jsonl` (env `KIS_TRADING_ENV` off/mock/live, `src/api/kis/trading.py:104-121`) | outcome `OK exit_complete` / `SKIPPED no_open_positions\|trading_disabled` | 08:59 | HIGH |
| order-reconcile-exit | Mon–Fri 09:00:40 | `order_session --phase reconcile-exit` | journal | outcome `OK reconcile_complete` | 09:33 | HIGH |
| paper-exit (decision) P | Mon–Fri 09:01 | `src/daily/paper_trade.py --phase exit` (window to 09:30, `src/config/market_session.py:97-103`) | `data/paper/{orders,fills,trades,nav}.parquet` | journal `stage=paper_ledger_refresh as_of=D … n_open=0`; **no OK outcome** | 09:30 | HIGH |
| collect (decision) | Mon–Fri 15:20 | `src/daily/collect.py:main` :1100 | `archive.parquet` via `src/daily/archive.py:upsert_archive_snapshot` :165; `capture/decision/<D>/<run>/input.parquet` | `stage=realtime_coverage … status=COMPLETE` (min 0.99, :201); **outcome only on SKIPPED** | 15:29 | HIGH |
| predict (After=collect) | Mon–Fri 15:21 | `src/daily/predict.py:main` :455 | `parquet/{rank_pool_predictions,topk_decisions}.parquet` | outcome `OK` | 15:30 | HIGH |
| auction-close (optional) | Mon–Fri 15:21:05 | `auction_capture --phase close` | raw + manifest | `status=COMPLETE phase=close` | 15:36 | MED |
| order-entry | Mon–Fri 15:21:15 | `order_session --phase entry` | journal | outcome `OK entry_complete` | 15:29 | HIGH |
| finalize-close (decision; ExecStopPost starts paper-entry) | Mon–Fri 15:30:30 | `src/daily/finalize_close.py:main` :481 | archive `종가_확정`, raw `close_confirmation` | outcome `OK` (DEGRADED `empty_archive\|picks_unconfirmed\|zero_confirmed`, :148-156) | 15:33 | HIGH |
| order-reconcile-entry | Mon–Fri 15:30:40 | `order_session --phase reconcile-entry` | journal | outcome `OK` | 15:43 | HIGH |
| paper-entry (decision) P | Mon–Fri 15:34 + finalize trigger | `paper_trade --phase entry` | paper orders/fills/decisions/nav | entry fills at the 종가 (`paper_broker.py:220-274`); second run `already_recorded` (:498,:520); outcome only NO_DECISION/DEGRADED/SKIPPED | 15:45 | HIGH |
| archive-intraday-regular P | Mon–Fri 15:40 | `src/daily/archive_intraday.py:main` :1039 `--phase regular` | 1m+ticks `regular`, manifests `archive-<D>-regular-{bars,ticks}-*` | `stage=intraday_archive_stream … status=COMPLETE` | 18:40 | HIGH |
| aftermarket-book (batch) | Mon–Fri 15:40 (to ~19:59 by design) | `src/daily/aftermarket_book.py:main` :632 | `history/orderbook/aftermarket/<YYYY-MM>/<D>.parquet`, 35 manifests `aftermarket-book-<D>-b<n>-<venue>` | `stage=aftermarket_book status=COMPLETE date=D manifests=N` | 20:20 | HIGH |
| auction-path (bulk, DB증권) | Mon–Fri 15:45 | `src/daily/auction_path.py:main` :304 | `history/auction_path/<YYYY-MM>/<D>.parquet` | outcome from manifest (:270) | 19:00 | MED |
| archive-intraday P | Mon–Fri 20:05 | `archive_intraday --phase aftermarket` | 1m `nxt_premarket nxt_aftermarket krx_aftermarket`, ticks `krx_aftermarket nxt_aftermarket` (since 2026-09-29, market_session.py:74) | stream lines COMPLETE; outcome only DEGRADED/SKIPPED | 23:05 | HIGH |
| tape-sweep (bulk, Kiwoom; After=archive-intraday) P | Mon–Fri 20:35 | `src/daily/tick_tape_sweep.py:main` :330 | `capture/staging/tape_backfill/ledger.jsonl`, `staging/tape_sweep/last_report.json`, `tape-<D>-<session>-publish-*` manifests | report day = D; **no outcome** | 21:15 (`TAPE_SWEEP_DEADLINE_HHMMSS`) | HIGH |
| backup-prune (host, Drive lock) P | Mon–Fri 21:00 | `src/tools/backup_prune.py` (+`capture_offsite.py:980-1159`) | deletes remote `_deleted/<day>` > 30 d (cap 7/subtree), local `backups/intraday` > 3 d, local sealed tiers > 30 d after a remote md5 check | `stage=backup_prune dry_run=False purged= … sealed_kept=` | 21:20 | MED |
| daily-audit (host, `kca-audit.lock`) P | Mon–Fri 21:20 | `src/tools/daily_audit.py` | heartbeat v2, `audit_digests/<D>.txt`, mail | `stage=daily_audit day=trading status=OK subject=[kca] 🟢 …` | 21:50 | MED |
| altdata-capture (optional) | Mon–Fri 21:35 | `src/daily/altdata_capture.py:main` :137 | `history/altdata/*`, `_manifest.json` | panels `status=ok` | 22:05 | MED |
| backup (host, `flock -w 7200` Drive lock; OnSuccess=audit-reconcile) P | **daily** 22:15 | `src/tools/offsite_backup.py:363-546` | `capture_sealed/<tier>/<YYYY-MM>/<D>/seg-<sha16>.tar.zst` (immutable, md5), `rclone copy data/ artifacts/` (backup-dir `_deleted/<sub>/<day>`), ledger `capture/offsite/ledger/<tier>/<D>.jsonl` | `last_run.json status=ok`; `stage=offsite_backup status=ok`; **no outcome** | 01:00 (`deadline_at`) | HIGH |
| extended-backfill (bulk, batch) | daily 23:05 | `src/backfill/intraday/extended_session_backfill.py:main` :1350 | past-date partitions, `backfill_ledger/extended_sessions.parquet`, `1m_kis_adjusted`, raw for old trade dates | outcome `OK`, `status=DONE tasks_remaining=0 pending_conflicts=0` | 07:05 | MED |
| retrain (6g/1.8 cpu) P | Sat 22:00 | `src/ml/retrain.py --train-ranker-bundle` | `artifacts/models/topk_ranker/retrain_registry.jsonl` | registry `PROMOTED\|PROMOTED_UNGATED` | Sun 00:00 | LOW |
| core-snapshot (host, Drive lock) P | Sun 10:00 | `src/tools/core_snapshot.py` | `snapshots/<D>/` (price_history, archive, topk, rank_pool, paper/*, bundle) | `stage=core_snapshot … status=ok` | Sun 12:00 | MED |
| offsite-verify (host, Drive lock) P | Sun 11:00 | `src/tools/capture_offsite.py verify` :1268-1468 | drill staging removed after | `missing=0 mismatched=0`, 3 `restore_drill … status=OK` | Sun 13:00 | HIGH |
| kca-alert@ (host) | OnFailure of every kca unit (+ `krx-deferred-recreate`, `krx-host-backup`) | `src/tools/alerts.py` | mail + webhook, outbox | `Finished kca-alert@<unit>` | +5 min | MED |
| GitHub `watchdog.yml` | cron `40 22 * * 1-5` UTC = 07:40 KST Tue–Sat (runs 2–4.5 h late) | `src/tools/watchdog_probe.py`; GHCR PAT expiry ≤ 30 d | — | run success | — | LOW |
| GitHub `deploy.yml` | push to main | `vps-deploy-lib.sh` pull → preflight → `blackout_wait` → promote → converge (`git reset --hard`, `src.tools.code_sync`) | units, wrapper | run success | — | — |

Optional timers (`OPTIONAL_MANUAL_TIMERS`, `src/tools/code_sync.py:35-37`): auction-open, auction-close, altdata-capture. They are never auto-enabled (`install_systemd.sh` enables 26), but all 3 are enabled on the host. Not daily: `1m/regular_consolidated` (Toss backfill, last 2026-08-31) and `1m_kis_adjusted` (extended-backfill output).

## 4. Check catalogue

| ID | Tier | Windows | Assert (machine-checkable PASS) | Fail → Sev | Drill |
|---|---|---|---|---|---|
| H01 | EXT | M | 29 `kca-*` timers listed, no `--failed` kca unit, `Linger=yes` (WARN: a timer not enabled) | P1 | `systemctl --user list-timers 'kca-*' --all` |
| H02 | EXT | W, Ev | host `git status` clean, `kca-run` == repo wrapper, no unit DRIFT / NOT_IN_REPO, `latest` image ID == `sha-<HEAD>`; local: VPS HEAD == `origin/main` (or a deploy waits in `blackout_wait`) | P1 | §5.2 `H02-local` |
| H03 | EXT | W | no exited `kca-*` container, no kca container outside its §3 window; no new `*.bak*/*.pre_*/*.stale*` under `data/` or `artifacts/` | P2 | `docker ps -a`, `find … -maxdepth 4` |
| H04 | EXT | W, Ev | latest `deploy.yml` run on main is `success` (a red test job freezes `latest`); after a history rewrite, pinned SHAs in `tests/` fixtures are ancestors of `origin/main`; the 22 wrapper units print `rc=0` via `--print-argv` | P1 | §5.2 `H04` |
| R01 | EXT | M O E N | each unit due in the window has ≥1 `Finished` since its slot (optional timers only if enabled); a `Failed with result` followed by a Finished rerun is WARN | P1 (decision chain P0) | `journalctl --user -u kca-<u> --since <UTC>` |
| R02 | EXT | M O E | outcome per job ∈ allowed set: price_ingest OK; order_exit OK/SKIPPED(no_open_positions\|trading_disabled); order_reconcile_exit OK/SKIPPED(no_orders); predict OK; order_entry OK; finalize_close OK; order_reconcile_entry OK; extended_backfill OK | P1 (decision P0) | `events/<YYYY-MM>/<D>.jsonl` |
| R03 | EXT | M N | lists DEGRADED/NO_DECISION outcomes of D (WARN; each must map to an alert/known condition). `auction_path DEGRADED` is research (P2). A DEGRADED `extended_backfill` is actionable (transient outage, conflict); deterministic gaps show only as `deterministic_failed/exhausted` counts | P2 | outcome `reason`/metrics |
| R04 | EXT | W | last registry row `PROMOTED\|PROMOTED_UNGATED\|MANUAL_HOTFIX` ≤ 8 d old and the live `model_version` is in the registry. A `strategy or screen changed vs live bundle` REJECTED fails the unit by design: the operator certifies and reruns with `--skip-promotion-gate`; auto-remediation never reruns retrain | P1 | `journalctl -u kca-retrain` |
| R05 | EXT | M | today 07:05: ≥7 distinct `slot=… issued=True` (DATA_n, PRIMARY, TRADING_MOCK) and the Kiwoom rotate status line; CLOSED day → `non_trading_day` skip | P1 | warmup journal |
| R06 | EXT | L | `kca-aftermarket-book` active with container `kca-aftermarket-book` up between 15:40 and 19:59; ~92 % of its 4 h 40 min timeout is normal | P1 | §5.2 `R06` |
| C01 | CORE | E M | §1 | P0 | collect journal `stage=realtime_quote/requote` |
| C03 | CORE | N M | §1; PARTIAL (WARN) entry reasons `deadline\|vendor_failure\|missed_start`; `nxt_not_listed` is normal | P1 | §5.2 `C-manifest` |
| C04 | CORE | N M | §1; output `streams={…: complete\|tape_healed\|partial(n failed)\|missing}`. Partial bars = WARN; missing = FAIL | P1 | §5.2 `C-manifest`, journal `stage=intraday_archive_stream` |
| C05 | EXT | O E M | auction open/close manifests complete (SKIP if timer off) | P2 | auction journal `status=LATE/DEGRADED` |
| C06 | EXT | N | `auction-path-<D>-*` manifest complete (else WARN; 10-30 PARTIAL) | P2 | outcome metrics |
| C07 | CORE | O | §1; WARN on `flow_shortfall`/`program_flow_shortfall`. A 08:30 `stale_panel_pending_retry` is healthy when 11:30 is OK; an 11:30 DEGRADED `stale_panel` needs an alert | P1 | price-ingest journal |
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
| P06 | CORE | O E | §1. Entry reconcile `n_intents` counts exit + entry intents for the day (e.g. 6 for 3 picks). A stuck UNKNOWN intent is operator-only (`src.daily.order_resolve`, dry-run first) | P0 | `data/orders/<env>/journal.jsonl` |
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

Run from the local repo root, outside the blackouts. Each call extracts the `lib` block plus one window block and streams them over one ssh (nothing is written on either side). `D` is optional; it defaults to the previous trading day for M/W and to today for O/E/N.

```bash
kca_probe() { awk -v w="probe=$1" '/^```python probe=/{f=($2=="probe=lib"||$2==w);next} /^```/{f=0} f' docs/vps-checklist.md \
  | ssh -o ConnectTimeout=15 or-vps "cd ~/k-closing-alpha && TZ=Asia/Seoul nice -n 19 timeout 300 ~/.local/bin/uv run --no-sync python - $1 ${2:-}"; }
kca_probe M            # O | E | N | W ; e.g. kca_probe N 2026-10-30 re-audits a past day
# W adds local evidence (no ssh):
git -C /home/kth/k-closing-alpha fetch -q && git -C /home/kth/k-closing-alpha log -1 --format='%h %ci' origin/main
gh run list -R kthyeong/k-closing-alpha --workflow deploy.yml --limit 1; gh run list -R kthyeong/k-closing-alpha --workflow watchdog.yml --limit 5
```

Shared helpers (always prepended):

```python probe=lib
import collections, json, os, re, subprocess, sys
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
from src.data.session_calendar import resolve_session_day
KST, UTC = ZoneInfo("Asia/Seoul"), ZoneInfo("UTC")
NOW = datetime.now(KST)
TODAY = NOW.date()
W = sys.argv[1]
ROOT, CAP = Path("data"), Path("data/history/capture")
RCLONE, REMOTE = os.path.expanduser("~/.local/bin/rclone"), "gdrive:quant-lake/live/k-closing-alpha"
def out(cid, st, **kv):
    print(cid, st, *(f"{k}={str(v).replace(' ', '')}" for k, v in kv.items()), flush=True)
def lv(fail, warn=False):
    return "FAIL" if fail else ("WARN" if warn else "PASS")
def check(cid):
    def deco(fn):
        try:
            fn()
        except Exception as exc:  # a probe crash is UNKNOWN, never PASS
            out(cid, "UNKNOWN", err=f"{type(exc).__name__}:{str(exc)[:140]}")
        return fn
    return deco
def kind(d):
    return resolve_session_day(d).kind.value
def is_td(d):
    return kind(d) in ("STANDARD", "SHIFTED")
def prev_td(d):
    x = d - timedelta(days=1)
    while not is_td(x):
        x -= timedelta(days=1)
    return x
def at(d, hms):
    return datetime.combine(d, datetime.strptime(hms, "%H:%M:%S").time(), KST)
def sh(cmd, timeout=120):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout
def jr(units, since, until=None, pat=None):
    cmd = ["journalctl", "--user", "--no-pager", "-o", "cat", "--since", since.astimezone(UTC).strftime("%F %T UTC")]
    for u in units if isinstance(units, (list, tuple)) else [units]:
        cmd += ["-u", u if u.endswith(".service") or "*" in u else f"kca-{u}.service"]
    if until:
        cmd += ["--until", until.astimezone(UTC).strftime("%F %T UTC")]
    lines = sh(cmd).splitlines()
    return [l for l in lines if pat is None or re.search(pat, l)]
def outcomes(d):
    p, last = ROOT / f"logs/events/{d:%Y-%m}/{d}.jsonl", {}
    if p.exists():
        for line in p.read_text().splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            last[r.get("job")] = r
    return last
def expect(cid, d, spec):
    o, bad = outcomes(d), []
    for job, allowed in spec.items():
        r = o.get(job)
        got = "absent" if r is None else f"{r['outcome']}:{r.get('reason', '')}"
        if r is None or not (r["outcome"] in allowed or got in allowed):
            bad.append(f"{job}={got}")
    out(cid, lv(bool(bad)), date=d, jobs=len(spec), bad=",".join(bad) or "-")
    return o
def runs(cid, specs):
    miss, rec = [], []
    for unit, d, hms, need in specs:
        if unit in OPTIONAL and sh(["systemctl", "--user", "is-enabled", f"kca-{unit}.timer"]).strip() != "enabled":
            continue
        lines = jr(unit, at(d, hms) - timedelta(seconds=90))
        fin = sum(l.startswith("Finished ") for l in lines)
        bad = sum("Failed with result" in l for l in lines)
        if fin < need:
            miss.append(f"{unit}:fin{fin}/{need}:failed{bad}")
        elif bad:
            rec.append(f"{unit}:failed{bad}_rerun_ok")
    out(cid, lv(bool(miss), bool(rec)), units=len(specs), missing=",".join(miss) or "-", recovered=",".join(rec) or "-")
OPTIONAL = {"auction-open", "auction-close", "altdata-capture"}
def manifest_status(d, prefix):
    res = {}
    for run in sorted((CAP / f"manifests/{d}").glob(prefix + "*")):
        st = {f.name[9:-5] for f in run.glob("manifest-*.json")}
        res[run.name] = "complete" if "complete" in st else ("partial" if "partial" in st else ("pending" if st else "none"))
    return res
def failed_entries(d, run):
    m = json.loads((CAP / f"manifests/{d}/{run}/manifest-partial.json").read_text())
    return sum(e.get("status") != "COMPLETE" for e in m.get("entries", []))
def heartbeat():
    return json.loads((ROOT / "logs/heartbeat/daily_audit.json").read_text())
def timer_active(name):
    return sh(["systemctl", "--user", "is-active", f"kca-{name}.timer"]).strip() == "active"
D = date.fromisoformat(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] else (prev_td(TODAY) if W in ("M", "W") else TODAY)
P = prev_td(D)
REV = sh(["git", "log", "-1", "--format=%h"]).strip()
out("CTX", "INFO", now=NOW.strftime("%F_%T"), window=W, today=f"{TODAY}:{kind(TODAY)}", D=f"{D}:{kind(D)}", P=P, rev=REV)
if not is_td(D):
    out("ALL", "SKIP(non_trading_day)", D=D)
    sys.exit(0)
# ---------- shared check bodies (called by the window blocks) ----------
def c_manifests_live(d):
    @check("C03")
    def _():
        st = manifest_status(d, f"aftermarket-book-{d}-")
        c = collections.Counter(st.values())
        line = jr("aftermarket-book", at(d, "15:39:00"), at(d, "23:59:00"), rf"stage=aftermarket_book status=\w+ date={d}")
        ok_line = bool(line) and "status=COMPLETE" in line[-1] and f"manifests={len(st)}" in line[-1]
        out("C03", lv(not st or c["pending"] + c["none"] > 0, c["partial"] > 0 or not ok_line), batches=len(st), status=dict(c), journal=(line[-1].split("status=")[1][:40] if line else "absent"))
    @check("C04")
    def _():
        streams = ["regular-bars", "regular-ticks", "nxt-premarket-bars", "nxt-aftermarket-bars", "krx-aftermarket-bars", "krx-aftermarket-ticks", "nxt-aftermarket-ticks"]
        tape = manifest_status(d, f"tape-{d}-")
        res, fail, warn = {}, [], []
        for s in streams:
            st = manifest_status(d, f"archive-{d}-{s}-")
            best = "complete" if "complete" in st.values() else ("partial" if "partial" in st.values() else "missing")
            if best == "partial" and s.endswith("ticks"):
                sess = s[: -len("-ticks")].replace("-", "_")
                if any(k.startswith(f"tape-{d}-{sess}-") and v == "complete" for k, v in tape.items()):
                    best = "tape_healed"
            if best == "partial":
                run = [k for k, v in st.items() if v == "partial"][-1]
                best = f"partial({failed_entries(d, run)}failed)"
                warn.append(s)
            if best == "missing":
                fail.append(s)
            res[s] = best
        out("C04", lv(bool(fail), bool(warn)), streams=res)
    @check("C05")
    def _():
        res = {}
        for ph, unit in (("open", "auction-open"), ("close", "auction-close")):
            if not timer_active(unit):
                res[ph] = "SKIP(timer_off)"
                continue
            st = manifest_status(d, f"auction-{ph}-{d}-")
            res[ph] = ",".join(sorted(set(st.values()))) or "missing"
        out("C05", lv(any(v == "missing" for v in res.values()), any(v not in ("complete", "SKIP(timer_off)") for v in res.values())), **res)
def c_decision(d):
    @check("C01")
    def _():
        st = manifest_status(d, f"decision-{d}-")
        cov = jr("collect", at(d, "15:19:00"), at(d, "15:35:00"), r"stage=realtime_coverage")
        m = re.search(r"coverage=([\d.]+) status=(\w+)", cov[-1]) if cov else None
        covv = float(m.group(1)) if m else -1.0
        out("C01", lv("complete" not in st.values() or covv < 0.99, m is not None and m.group(2) != "COMPLETE"), manifest=",".join(st.values()) or "missing", coverage=covv, status=m.group(2) if m else "absent")
def q_partitions(d, ticks_after=True):
    @check("Q08")
    def _():
        paths = {f"1m/{s}": ROOT / f"history/intraday/1m/{s}/{d:%Y-%m}/{d}.parquet" for s in ("regular", "nxt_premarket", "nxt_aftermarket", "krx_aftermarket")}
        paths.update({f"ticks/{s}": ROOT / f"history/intraday/ticks/{s}/{d:%Y-%m}/{d}.parquet" for s in ("regular", "krx_aftermarket", "nxt_aftermarket")})
        paths["orderbook/aftermarket"] = ROOT / f"history/orderbook/aftermarket/{d:%Y-%m}/{d}.parquet"
        paths["auction_path"] = ROOT / f"history/auction_path/{d:%Y-%m}/{d}.parquet"
        size = {k: (p.stat().st_size if p.exists() else 0) for k, p in paths.items()}
        miss = [k for k, v in size.items() if v == 0 and k != "auction_path"]
        out("Q08", lv(bool(miss), size["auction_path"] == 0), missing=",".join(miss) or "-", kib={k: v // 1024 for k, v in size.items()})
def p_ledger(d, p, morning):
    import pandas as pd
    from src.execution.paper_broker import PaperLedger
    led = PaperLedger(root=ROOT / "paper")
    fills = led.load_effective_fills()
    fills["dd"] = fills["decision_date"].astype(str).str[:10]
    @check("P02")
    def _():
        opn = led.load_open_positions()
        stale = sorted(set(opn.loc[opn["decision_date"].astype(str).str[:10] < str(d), "decision_date"].astype(str).str[:10]))
        buys = fills[(fills["side"].astype(str) == "buy") & (fills["dd"] == str(p))]
        sells = fills[fills["side"].astype(str) == "sell"]
        exited = set(sells["entry_order_id"].astype(str))
        unexited = sorted(set(buys["order_id"].astype(str)) - exited)
        tr = pd.read_parquet(ROOT / "paper/trades.parquet")
        no_trade = sorted(set(sells[sells["entry_order_id"].astype(str).isin(buys["order_id"].astype(str))]["order_id"].astype(str)) - set(tr["exit_order_id"].astype(str)))
        out("P02", lv(bool(stale) or bool(unexited) or bool(no_trade)), prev_lots=len(buys), unexited=len(unexited), sells_without_trade=len(no_trade), stale_open_decisions=stale or "-")
    @check("P03")
    def _():
        nav = pd.read_parquet(ROOT / "paper/nav.parquet")
        nav["day"] = nav["as_of_date"].astype(str).str[:10]
        row = nav[nav["day"] == str(d)]
        gap = abs(float(row.iloc[-1]["nav"]) - float(row.iloc[-1]["cash"]) - float(row.iloc[-1]["open_market_value"])) if len(row) else float("nan")
        dup = int(nav.duplicated(["day"]).sum())
        cc = nav.sort_values("day")["cumulative_cost"].astype(float).diff().dropna()
        tr = pd.read_parquet(ROOT / "paper/trades.parquet")
        fric = int(((tr["buy_fee"] <= 0) | (tr["sell_fee"] <= 0) | (tr["sell_tax"] <= 0)).sum())
        net = float((tr["gross_pnl"].astype(float) - tr["cost"].astype(float) - tr["net_pnl"].astype(float)).abs().max())
        dupf = int(pd.read_parquet(ROOT / "paper/fills.parquet")["order_id"].duplicated().sum())
        out("P03", lv(not len(row) or gap > 1.0 or dup > 0 or bool((cc < -1e-9).any()) or fric > 0 or net > 1.0 or dupf > 0),
            nav_minus_cash_mv=round(gap, 2), dup_nav_days=dup, cost_monotone=not bool((cc < -1e-9).any()), zero_friction=fric, max_net_err=round(net, 2), dup_fill_ids=dupf, nav=int(row.iloc[-1]["nav"]) if len(row) else "absent")
    if morning:
        return
    @check("P01")
    def _():
        orders = pd.read_parquet(ROOT / "paper/orders.parquet")
        eo = orders[(orders["decision_date"].astype(str).str[:10] == str(d)) & (orders["reason"].astype(str) == "entry")]
        tk = pd.read_parquet(ROOT / "parquet/topk_decisions.parquet")
        k = int((tk["decision_date"].astype(str).str[:10] == str(d)).sum())
        nod = pd.read_parquet(ROOT / "paper/decisions.parquet")
        nodec = int((nod["decision_date"].astype(str).str[:10] == str(d)).sum())
        eb = fills[(fills["side"].astype(str) == "buy") & (fills["dd"] == str(d))]
        dup = int(eo["order_id"].duplicated().sum()) + int(eo["symbol"].duplicated().sum())
        st = dict(collections.Counter(eo["status"].astype(str)))
        ok = (len(eo) == k and dup == 0) or (k == 0 and nodec > 0)
        out("P01", lv(not ok, len(eb) < len(eo)), topk=k, entry_orders=len(eo), entry_fills=len(eb), dup=dup, status=st, no_decision_rows=nodec)
    @check("P04")
    def _():
        tk = pd.read_parquet(ROOT / "parquet/topk_decisions.parquet")
        t = tk[tk["decision_date"].astype(str).str[:10] == str(d)]
        ts = {c: pd.to_datetime(t[c], utc=True).dt.tz_convert(KST) for c in ("input_available_at", "inference_started_at", "decided_at")}
        viol = int((ts["input_available_at"] > ts["inference_started_at"]).sum() + (ts["inference_started_at"] > ts["decided_at"]).sum() + (ts["decided_at"].dt.strftime("%H%M%S") >= "153000").sum())
        eb = fills[(fills["side"].astype(str) == "buy") & (fills["dd"] == str(d))]
        fa = pd.to_datetime(eb["filled_at"], utc=True).dt.tz_convert(KST)
        early = int((fa < ts["decided_at"].max()).sum()) if len(t) and len(eb) else 0
        out("P04", lv(viol > 0 or early > 0 or len(t) != t["symbol"].nunique()), rows=len(t), causality_violations=viol, fills_before_decision=early, decided_max=ts["decided_at"].max().strftime("%T") if len(t) else "-", first_fill=fa.min().strftime("%T") if len(eb) else "-")
def b_offsite(d):
    lr = json.loads((CAP / "offsite/last_run.json").read_text())
    @check("B01")
    def _():
        slot = at(TODAY, "22:15:00") if NOW >= at(TODAY, "22:15:00") else at(TODAY - timedelta(days=1), "22:15:00")
        started = datetime.fromisoformat(lr["started_at"]).astimezone(KST)
        seal = lr["steps"].get("capture_seal", {})
        steps_bad = [k for k, v in lr["steps"].items() if v.get("status") not in ("ok", "deferred")]
        inprog = (CAP / "offsite/in_progress.json").exists()
        late = (started - slot).total_seconds() / 60
        out("B01", lv(lr["status"] != "ok" or bool(steps_bad) or started < slot - timedelta(minutes=1) or seal.get("missing_sealed_members", 1) != 0 or not lr["core_panels"],
                       late > 15 or seal.get("deferred_dates", 0) > 0 or inprog),
            status=lr["status"], started=started.strftime("%m-%d_%T"), start_delay_min=round(late, 1), steps_bad=steps_bad or "-", segments=seal.get("segments_committed"), deferred=seal.get("deferred_dates"), missing_sealed=seal.get("missing_sealed_members"), in_progress=inprog, core_issues=lr["steps"].get("core_panels", {}).get("issues"))
    @check("B02")
    def _():
        started = datetime.fromisoformat(lr["started_at"]).timestamp()
        res, bad = {}, []
        for tier in ("raw", "normalized", "manifests"):
            lp = CAP / f"offsite/ledger/{tier}/{d}.jsonl"
            local = CAP / tier / str(d)
            if not local.exists():
                res[tier] = "no_local"
                continue
            segs = [json.loads(l) for l in lp.read_text().splitlines() if l.strip()] if lp.exists() else []
            members = {m["path"] for s in segs for m in s["members"]}
            unsealed = 0
            for r, _, fs in os.walk(local):
                for f in fs:
                    full = Path(r) / f
                    if f.endswith(".lock") or f.startswith("stage-") or full.is_symlink():
                        continue
                    if full.stat().st_mtime < started and full.relative_to(CAP).as_posix() not in members:
                        unsealed += 1
            dirs = {s["remote_path"].rsplit("/", 1)[0] for s in segs}
            remote = set()
            for dr in dirs:
                remote |= {x.strip() for x in sh([RCLONE, "lsf", "--files-only", dr], 120).splitlines()}
            missing = sum(s["remote_path"].rsplit("/", 1)[1] not in remote for s in segs)
            res[tier] = f"segs{len(segs)}:remote_missing{missing}:unsealed_pre_run{unsealed}"
            if not segs or missing or unsealed:
                bad.append(tier)
        out("B02", lv(bool(bad)), date=d, **res)
    @check("B04")
    def _():
        hist = json.loads((CAP / "offsite/deferred_history.json").read_text())[-5:]
        tr = [int(e.get("deferred_dates", -1)) for e in hist]
        stall = len(tr) >= 3 and tr[-1] > 0 and tr[-1] >= tr[-2] >= tr[-3]
        old = hist[-1].get("oldest_deferred_date") if hist else ""
        aged = bool(old) and (TODAY - date.fromisoformat(old)).days > 21
        seal_ok = lr["steps"].get("capture_seal", {}).get("status") == "ok"
        out("B04", lv(stall or aged, not seal_ok), trend=tr, oldest=old or "-", seal_status_ok=seal_ok)
def b_prune():
    @check("B05")
    def _():
        last = TODAY if NOW >= at(TODAY, "21:05:00") and TODAY.weekday() < 5 else TODAY - timedelta(days=1)
        while last.weekday() >= 5:
            last -= timedelta(days=1)
        summ = jr("backup-prune", at(last, "20:59:00"), at(last, "23:59:00"), r"stage=backup_prune (dry_run|step=)")
        failed = [l for l in summ if "status=failed" in l]
        kept = re.search(r"sealed_kept=(\d+)", summ[-1]).group(1) if summ and "sealed_kept" in summ[-1] else "?"
        ages = {}
        for sub in ("data", "artifacts"):
            ds = [x.strip("/") for x in sh([RCLONE, "lsf", "--dirs-only", f"{REMOTE}/_deleted/{sub}"], 120).splitlines()]
            ds = sorted(x for x in ds if re.fullmatch(r"\d{4}-\d{2}-\d{2}", x))
            over_now = sum(date.fromisoformat(x) < last - timedelta(days=30) for x in ds)
            next_batch = sum(date.fromisoformat(x) < TODAY + timedelta(days=1) - timedelta(days=30) for x in ds)
            ages[sub] = f"n{len(ds)}:oldest{ds[0] if ds else '-'}:over30_at_last_prune{over_now}:due_next{next_batch}"
            if over_now or next_batch >= 7:
                failed.append(f"{sub}:cap_risk" if next_batch >= 7 else f"{sub}:not_purged")
        cutoff = last - timedelta(days=30)
        kept_local = []
        for tier in ("raw", "normalized", "manifests"):
            for p in (CAP / tier).iterdir():
                if re.fullmatch(r"\d{4}-\d{2}-\d{2}", p.name) and date.fromisoformat(p.name) < cutoff and datetime.fromtimestamp(p.stat().st_mtime, KST) < at(last, "21:00:00"):
                    kept_local.append(f"{tier}/{p.name}")
        out("B05", lv(not summ or bool(failed), kept != "0" or bool(kept_local)), last_prune=last, summary=bool(summ), failed=failed or "-", sealed_kept=kept, kept_local_pre_prune=kept_local[:5] or "-", remote_deleted=ages)
def a_heartbeat(expect_date=None):
    @check("A01")
    def _():
        h = heartbeat()
        rec = datetime.fromisoformat(h["reconciled_at"]).astimezone(KST)
        in_hours = at(TODAY, "07:00:00") <= NOW <= at(TODAY, "23:05:00")
        age = (NOW - rec).total_seconds() / 60
        keys = [i.get("key") for i in h.get("open_issues", [])]
        out("A01", lv(h.get("schema_version") != 2 or h.get("undelivered_alerts", 1) != 0 or (in_hours and age > 15) or (expect_date is not None and h.get("snapshot_date") != str(expect_date)), bool(keys) or h.get("severity") != "OK"),
            snapshot=h.get("snapshot_date"), severity=h.get("severity"), reconciled_age_min=round(age, 1), undelivered=h.get("undelivered_alerts"), open_issues=keys[:8] or "-", provisional=h.get("provisional_reasons") or "-")
def x_tokens(since, units="kca-*"):
    @check("X01")
    def _():
        pat = r"IGW00201|EGW00201|EGW00133|EGW00121|AUTH_REJECTED|ADOPTED_ROTATED|TokenIssueThrottled|HostBrokerDirNotSharedError|AdmissionDirNotSharedError|SCHEMA_INVALID|TokenStoreLockTimeout|TOKEN_REPLACED|http=429"
        hits = collections.Counter(m for l in jr(units, since, pat=pat) for m in re.findall(pat, l))
        hard = {k: v for k, v in hits.items() if k in ("HostBrokerDirNotSharedError", "AdmissionDirNotSharedError", "SCHEMA_INVALID")}
        out("X01", lv(bool(hard), bool(hits)), since=since.strftime("%m-%d_%H:%M"), hits=dict(hits) or "-")
def a_alerts(since):
    @check("A03")
    def _():
        failed = sorted({m for l in jr("kca-*", since, pat=r"Failed with result") for m in re.findall(r"(kca-[a-z0-9-]+\.service)", l)})
        alerts = " ".join(jr("kca-alert@*", since, pat=r"Finished|Failed"))
        unalerted = [u for u in failed if u not in alerts]
        out("A03", lv(bool(unalerted), bool(failed)), failed_units=failed or "-", unalerted=unalerted or "-")
def c_complete(d):
    @check("C11")
    def _():
        import contextlib, io
        from src.tools.daily_audit import audit_daily_completeness
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            r = audit_daily_completeness(str(d))
        ran = jr("daily-audit", at(d, "21:19:00"), at(d, "23:59:00"), r"stage=daily_audit day=")
        fail = jr("daily-audit", at(d, "21:19:00"), at(d, "23:59:00"), rf"step=intraday_complete status=FAIL date={d}")
        r["intraday_complete"] = bool(ran) and not fail
        miss = [k for k, v in r.items() if not v]
        out("C11", lv(bool(miss)), keys=len(r), missing=miss or "-", audit_ran=bool(ran))
def r_degraded(d):
    @check("R03")
    def _():
        o = outcomes(d)
        bad = {j: f"{r['outcome']}:{r.get('reason', '')[:60]}" for j, r in o.items() if r["outcome"] in ("DEGRADED", "NO_DECISION")}
        out("R03", lv(False, bool(bad)), date=d, jobs=len(o), degraded=bad or "-")
```

Window M (next-morning):

```python probe=M
import pandas as pd
YDAY = TODAY - timedelta(days=1)
@check("H01")
def _():
    n = sum("kca-" in l for l in sh(["systemctl", "--user", "list-timers", "kca-*", "--all", "--no-legend"]).splitlines())
    off = [l.split()[0] for l in sh(["systemctl", "--user", "list-unit-files", "kca-*.timer", "--no-legend"]).splitlines() if l.split()[1:2] != ["enabled"]]
    failed = [l.split()[0] for l in sh(["systemctl", "--user", "list-units", "--failed", "--no-legend", "--plain", "kca-*"]).splitlines() if l.strip()]
    linger = sh(["loginctl", "show-user", os.environ.get("USER", "ubuntu"), "-p", "Linger", "--value"]).strip()
    out("H01", lv(n != 29 or bool(failed) or linger != "yes", bool(off)), timers=n, not_enabled=off or "-", failed=failed or "-", linger=linger)
wk = TODAY.weekday() < 5
runs("R01", [("archive-intraday", D, "20:05:00", 1), ("tape-sweep", D, "20:35:00", 1), ("backup-prune", D, "21:00:00", 1), ("daily-audit", D, "21:20:00", 1),
             ("altdata-capture", D, "21:35:00", 1), ("backup", YDAY, "22:15:00", 1), ("extended-backfill", YDAY, "23:05:00", 1)]
     + ([("kis-token-warmup", TODAY, "07:05:00", 1), ("kiwoom-token-rotate", TODAY, "07:10:00", 1)] if wk and NOW > at(TODAY, "07:12:00") else []))
expect("R02", YDAY, {"extended_backfill": {"OK"}})
r_degraded(D)
@check("R05")
def _():
    if not wk or NOW < at(TODAY, "07:12:00"):
        out("R05", "SKIP(not_due)")
        return
    w = " ".join(jr("kis-token-warmup", at(TODAY, "07:04:00"), pat=r"stage=kis_token_warmup|non_trading_day"))
    slots = sorted(set(re.findall(r"slot=(\w+) key_id=\w+ issued=True", w)))
    rot = jr("kiwoom-token-rotate", at(TODAY, "07:09:00"), pat=r"status=(ROTATED|UNCHANGED)")
    skipped = "non_trading_day" in w
    out("R05", lv(not skipped and (len(slots) < 7 or not rot)), kis_slots=len(slots), slots=slots or "-", kiwoom=bool(rot), closed_skip=skipped)
c_decision(D)
c_manifests_live(D)
c_complete(D)
q_partitions(D)
p_ledger(D, P, morning=False)
b_offsite(D)
b_prune()
a_heartbeat(expect_date=D)
a_alerts(at(D, "15:00:00"))
x_tokens(at(D, "15:00:00"))
@check("A04")
def _():
    s = json.loads(Path("/home/ubuntu/quant-dashboard/public/status.json").read_text())
    proj = [p for p in s["projects"] if "closing" in p["name"] or p["name"] in ("kca",)]
    bad = [(c["id"], c["level"]) for p in proj for c in p["checks"] if c["level"] != "OK"]
    out("A04", lv(False, bool(bad) or not proj), generated=s["generated_at"][:16], card=[p["name"] for p in proj], non_ok=bad or "-")
```

Window O (post-exit):

```python probe=O
import pandas as pd
runs("R01", [("price-ingest", D, "08:30:00", 1), ("auction-open", D, "08:39:55", 1), ("order-exit", D, "08:40:10", 1),
             ("order-reconcile-exit", D, "09:00:40", 1), ("paper-exit", D, "09:01:00", 1)])
o = expect("R02", D, {"price_ingest": {"OK"}, "order_exit": {"OK", "SKIPPED:no_open_positions", "SKIPPED:trading_disabled"},
                      "order_reconcile_exit": {"OK", "SKIPPED:no_orders", "SKIPPED:trading_disabled"}})
@check("C07")
def _():
    m = o["price_ingest"]["metrics"]
    out("C07", lv(m["panel_max_date"] != str(P) or m["stale_panel"] or m["panel_max_date"] != m["expected_panel_date"], bool(m.get("flow_shortfall")) or bool(m.get("program_flow_shortfall"))),
        panel_max=m["panel_max_date"], expected=P, new_rows=m["n_new_rows"], stale=m["stale_panel"], flow_shortfall=m.get("flow_shortfall") or "-")
@check("Q05")
def _():
    ph = pd.read_parquet(ROOT / "history/price_history.parquet", columns=["date", "symbol", "high", "low", "close_raw", "volume", "kospi_pct"])
    ph["day"] = ph["date"].astype(str).str[:10]
    cnt = ph.groupby("day").size()
    last, base = int(cnt.get(str(P), 0)), float(cnt[cnt.index < str(P)].tail(5).median())
    dd = ph[ph["day"] == str(P)]
    bad = int(((dd["high"] < dd["low"]) | (dd["close_raw"].astype(float) <= 0) | (dd["volume"] < 0)).sum())
    dup, tdup, kp = int(dd["symbol"].duplicated().sum()), int(ph.duplicated(["date", "symbol"]).sum()), int(dd["kospi_pct"].nunique())
    r = last / base if base else float("nan")
    out("Q05", lv(last == 0 or bad > 0 or dup > 0 or tdup > 0 or kp > 1, not 0.97 <= r <= 1.03), day=P, rows=last, prev5_median=int(base), ratio=round(r, 3), invalid=bad, dup=dup, dup_total=tdup, kospi_pct_distinct=kp)
@check("C05")
def _():
    if not timer_active("auction-open"):
        out("C05", "SKIP(timer_off)")
        return
    st = manifest_status(D, f"auction-open-{D}-")
    j = jr("auction-open", at(D, "08:39:00"), at(D, "09:40:00"), r"stage=auction_capture status=\w+ phase=open")
    out("C05", lv(not st, "complete" not in st.values()), open=",".join(st.values()) or "missing", journal=j[-1].split("status=")[1].split()[0] if j else "absent")
p_ledger(D, P, morning=True)
@check("P06")
def _():
    ex, rc = o["order_exit"].get("metrics", {}), o["order_reconcile_exit"].get("metrics", {})
    if o["order_exit"]["outcome"] == "SKIPPED":
        out("P06", "SKIP(" + o["order_exit"].get("reason", "") + ")")
        return
    acc = ex.get("n_acked", 0) + ex.get("n_rejected", 0) + ex.get("n_skipped", 0)
    bad = ex.get("n_unknown", 1) or ex.get("n_unknown_unresolved_total", 1) or rc.get("n_unknown", 1) or rc.get("n_missing_at_broker", 1) or rc.get("position_shortfalls") or rc.get("n_unknown_unresolved_total", 1)
    out("P06", lv(bool(bad) or acc != ex.get("n_lots"), rc.get("n_filled") != rc.get("n_intents") or ex.get("n_rejected", 0) > 0),
        env=ex.get("env"), lots=ex.get("n_lots"), acked=ex.get("n_acked"), rejected=ex.get("n_rejected"), unknown=ex.get("n_unknown"), rec_filled=f"{rc.get('n_filled')}/{rc.get('n_intents')}", missing_at_broker=rc.get("n_missing_at_broker"), shortfalls=rc.get("position_shortfalls") or "-")
a_heartbeat()
a_alerts(at(D, "07:00:00"))
x_tokens(at(D, "07:00:00"))
```

Window E (post-decision):

```python probe=E
import numpy as np
import pandas as pd
runs("R01", [("collect", D, "15:20:00", 1), ("predict", D, "15:21:00", 1), ("auction-close", D, "15:21:05", 1), ("order-entry", D, "15:21:15", 1),
             ("finalize-close", D, "15:30:30", 1), ("order-reconcile-entry", D, "15:30:40", 1), ("paper-entry", D, "15:30:30", 1)])
o = expect("R02", D, {"predict": {"OK"}, "order_entry": {"OK", "SKIPPED:trading_disabled"}, "finalize_close": {"OK"},
                      "order_reconcile_entry": {"OK", "SKIPPED:trading_disabled"}})
c_decision(D)
@check("Q01")
def _():
    a = pd.read_parquet(ROOT / "history/archive.parquet")
    a["day"] = a["스냅샷_날짜"].astype(str).str[:10]
    d = a[a["day"] == str(D)]
    med = float(a[a["day"] < str(D)].groupby("day").size().tail(5).median())
    nulls = int(d[["종가", "결정_종가", "전일종가", "등락률", "거래대금"]].isna().sum().sum())
    conf = float(d["종가_확정"].fillna(0).astype(float).mean()) if len(d) else 0.0
    flags = {c: int((d[c].fillna(0).astype(float) > 0).sum()) for c in ("가격_비정상", "수급_실패", "지수_실패", "현재가_실패")}
    o_, h, l, c = (d[k].astype(float) for k in ("시가", "고가", "저가", "종가"))
    ohlc = int(((h < np.maximum(o_, c) - 1e-9) | (l > np.minimum(o_, c) + 1e-9) | (h < l)).sum())
    chg = int((((c / d["전일종가"].astype(float) - 1) * 100 - d["등락률"].astype(float)).abs() > 0.01).sum())
    gap99 = float(((c - d["결정_종가"].astype(float)).abs() / d["결정_종가"].astype(float)).quantile(0.99)) if len(d) else 0.0
    snap = pd.to_datetime(d["snapshot_timestamp"], utc=True).dt.tz_convert(KST)
    feat = pd.to_datetime(d["feature_available_timestamp"], utc=True).dt.tz_convert(KST)
    exe = pd.to_datetime(d["execution_timestamp"], utc=True).dt.tz_convert(KST)
    confirmed_rows = d["종가_확정"].fillna(0).astype(float).to_numpy() > 0  # unconfirmed rows keep the 15:20 snapshot ts by design (finalize_close never promotes frozen prices); their share is judged by `confirmed`
    causal = int((feat < snap).sum() + ((exe.dt.strftime("%H%M%S") < "153000").to_numpy() & confirmed_rows).sum())
    in1520 = float(((snap.dt.strftime("%H%M%S") >= "152000") & (snap.dt.strftime("%H%M%S") <= "152059")).mean()) if len(d) else 0.0
    fail = len(d) == 0 or int(d["종목코드"].duplicated().sum()) > 0 or nulls > 0 or conf < 0.9 or ohlc > 0 or causal > 0 or flags["가격_비정상"] + flags["지수_실패"] + flags["현재가_실패"] > 0 or chg > max(1, len(d) * 0.01)
    out("Q01", lv(fail, conf < 0.99 or flags["수급_실패"] > 0.1 * max(len(d), 1) or gap99 > 0.03), rows=len(d), prev5_median=int(med), confirmed=round(conf, 3), nulls=nulls, flags=flags, ohlc_bad=ohlc, chg_mismatch=chg, close_gap_p99=round(gap99, 4), causality_violations=causal, snap_in_1520=round(in1520, 3))
@check("C05")
def _():
    if not timer_active("auction-close"):
        out("C05", "SKIP(timer_off)")
        return
    st = manifest_status(D, f"auction-close-{D}-")
    out("C05", lv(not st, "complete" not in st.values()), close=",".join(st.values()) or "missing")
p_ledger(D, P, morning=False)
@check("P06")
def _():
    en, rc = o["order_entry"].get("metrics", {}), o["order_reconcile_entry"].get("metrics", {})
    if o["order_entry"]["outcome"] == "SKIPPED":
        out("P06", "SKIP(" + o["order_entry"].get("reason", "") + ")")
        return
    picks = en.get("n_picks", en.get("n_lots"))
    acc = en.get("n_acked", 0) + en.get("n_rejected", 0) + en.get("n_not_sent", 0) + en.get("n_skipped", 0)
    bad = en.get("n_unknown", 1) or rc.get("n_unknown", 1) or rc.get("n_missing_at_broker", 1) or rc.get("n_unknown_unresolved_total", 1)
    out("P06", lv(bool(bad) or acc != picks, rc.get("n_filled") != rc.get("n_intents")), env=en.get("env"), picks=picks, acked=en.get("n_acked"), rejected=en.get("n_rejected"),
        not_sent=en.get("n_not_sent"), rec_filled=f"{rc.get('n_filled')}/{rc.get('n_intents')}", unknown=rc.get("n_unknown"), shortfalls=rc.get("position_shortfalls") or "-")
a_heartbeat()
a_alerts(at(D, "15:00:00"))
x_tokens(at(D, "15:15:00"))
```

Window N (evening):

```python probe=N
import pandas as pd
runs("R01", [("archive-intraday-regular", D, "15:40:00", 1), ("aftermarket-book", D, "15:40:00", 1), ("auction-path", D, "15:45:00", 1), ("archive-intraday", D, "20:05:00", 1),
             ("tape-sweep", D, "20:35:00", 1), ("backup-prune", D, "21:00:00", 1), ("daily-audit", D, "21:20:00", 1)]
     + ([("altdata-capture", D, "21:35:00", 1)] if NOW > at(D, "21:45:00") else []))
r_degraded(D)
c_manifests_live(D)
@check("C06")
def _():
    st = manifest_status(D, f"auction-path-{D}-")
    out("C06", lv(False, "complete" not in st.values()), status=",".join(st.values()) or "missing")
@check("C08")
def _():
    r = json.loads((CAP / "staging/tape_sweep/last_report.json").read_text())
    n = lambda x: len(x) if isinstance(x, (list, tuple, dict)) else int(x or 0)
    out("C08", lv(r.get("days_checked", [""])[-1] != str(D) or bool(r.get("disk_guard")), n(r.get("expiring_needs")) > 0 or n(r.get("expired_recoverable")) > 0 or n(r.get("expired")) > 0),
        last_day=r.get("days_checked", ["-"])[-1], needs=r.get("needs"), expiring=n(r.get("expiring_needs")), expired=n(r.get("expired")), expired_recoverable=n(r.get("expired_recoverable")), unresolved=n(r.get("unresolved")), disk_guard=r.get("disk_guard"))
c_complete(D)
q_partitions(D)
@check("Q06")
def _():
    m = pd.read_parquet(ROOT / f"history/intraday/1m/regular/{D:%Y-%m}/{D}.parquet")
    dup = int(m.duplicated(["symbol", "ts_hms"]).sum())
    bad = int(((m["high"] < m["low"]) | (m["volume"] < 0) | (m["close"] <= 0)).sum())
    ts = pd.to_numeric(m["ts_hms"], errors="coerce")
    out("Q06", lv(dup > 0 or bad > 0 or ts.min() > 90100 or ts.max() < 152900), rows=len(m), symbols=m["symbol"].nunique(), vendors={k: int(v) for k, v in m["vendor"].astype(str).value_counts().items()}, dup=dup, invalid=bad, ts=f"{int(ts.min())}..{int(ts.max())}")
@check("Q07")
def _():
    ph = pd.read_parquet(ROOT / "history/price_history.parquet", columns=["date", "symbol", "volume", "close_raw"])
    ph["day"] = ph["date"].astype(str).str[:10]
    days = sorted(ph["day"].unique())
    days = [x for x in days if x <= str(P)][-5:]
    def ratio(day):
        mm = pd.read_parquet(ROOT / f"history/intraday/1m/regular/{day[:7]}/{day}.parquet", columns=["symbol", "ts_hms", "volume", "close"])
        pp = ph[ph["day"] == day].set_index(ph[ph["day"] == day]["symbol"].astype(str))
        v = mm.groupby(mm["symbol"].astype(str))["volume"].sum()
        rr = (v / pp["volume"][pp["volume"] > 0]).dropna()
        lb = mm.sort_values(["symbol", "ts_hms"]).groupby(mm["symbol"].astype(str)).tail(1).set_index("symbol")["close"]
        k = pd.concat([lb.rename("b"), pp["close_raw"].astype(float).rename("c")], axis=1).dropna()
        return float(rr.median()), float((rr < 0.9).mean()), float(((k["b"] - k["c"]).abs() / k["c"] < 0.001).mean())
    cur, base = ratio(days[-1]), [ratio(x) for x in days[:-1]]
    bm, bs = float(pd.Series([b[0] for b in base]).median()), float(pd.Series([b[1] for b in base]).median())
    out("Q07", lv(cur[0] > 1.01 or cur[2] < 0.9, abs(cur[0] - bm) > 0.02 or abs(cur[1] - bs) > 0.05 or cur[2] < 0.98), day=days[-1], vol_ratio=round(cur[0], 4), base=round(bm, 4), share_below_0_9=round(cur[1], 3), base_share=round(bs, 3), last_bar_match=round(cur[2], 3))
@check("Q09")
def _():
    sh_lines = jr(["kca-archive-intraday.service", "kca-archive-intraday-regular.service", "kca-tape-sweep.service", "kca-extended-backfill.service"], at(P, "23:00:00"), at(D + timedelta(days=1), "07:00:00"), r"stage=intraday_replace .*SHRINK_ALLOWED")
    bdirs = [p for p in (CAP / "backups/intraday").glob(f"*/{D}") if p.is_dir()]
    nb = sum(len(list(p.glob("*-pre-*.parquet"))) for p in bdirs)
    q = CAP / f"quarantine/shrink/{D}"
    nq = sum(1 for _ in q.rglob("*") if _.is_file()) if q.exists() else 0
    out("Q09", lv(len(sh_lines) > 0 and nb == 0, nq > 0), shrink_allowed=len(sh_lines), pre_backups_today=nb, quarantine_files=nq, expires=D + timedelta(days=3))
@check("Q10")
def _():
    rec = [r for r in (json.loads(l) for l in (ROOT / f"logs/events/{D:%Y-%m}/{D}.jsonl").read_text().splitlines() if l.strip()) if r.get("job") == "tick_source_diff" and r.get("run_date") == str(D)]
    keys = [i["key"] for i in heartbeat().get("open_issues", [])] if heartbeat().get("snapshot_date") == str(D) else None
    gap = keys is not None and any(k.startswith("intraday:regular_ticks:") and k.endswith((":source_diff_systemic", ":certified_gap")) for k in keys)
    mt = rec[-1]["metrics"] if rec else {}
    out("Q10", lv(False, gap or keys is None), n=mt.get("n", 0), max_shortfall=mt.get("max_relative_shortfall", 0), symbols=mt.get("symbols", [])[:5] or "-", heartbeat_for_D=keys is not None, gap_warning=gap)
b_prune()
a_heartbeat(expect_date=D if NOW > at(D, "21:30:00") else None)
a_alerts(at(D, "15:30:00"))
x_tokens(at(D, "15:35:00"))
```

Window W (weekly, run after M):

```python probe=W
import hashlib
import pandas as pd
sun = TODAY - timedelta(days=(TODAY.weekday() + 1) % 7)
sat = sun - timedelta(days=1)
@check("H02")
def _():
    dirty = [l for l in sh(["git", "status", "--porcelain"]).splitlines() if l.strip()]
    wrap = Path(os.path.expanduser("~/.local/lib/kca/kca-run")).read_bytes() == Path("deploy/host/kca-run").read_bytes()
    inst = Path(os.path.expanduser("~/.config/systemd/user"))
    drift = [p.name for p in Path("deploy/systemd").glob("kca-*") if not (inst / p.name).exists() or (inst / p.name).read_bytes() != p.read_bytes()]
    extra = [p.name for p in inst.glob("kca-*") if not (Path("deploy/systemd") / p.name).exists()]
    imgs = dict(l.split() for l in sh(["docker", "images", "ghcr.io/kthyeong/k-closing-alpha", "--format", "{{.Tag}} {{.ID}}"]).splitlines() if l.strip())
    head = sh(["git", "rev-parse", "HEAD"]).strip()
    same = imgs.get("latest") == imgs.get(f"sha-{head}")
    img_sha = next((k[4:] for k, v in imgs.items() if k.startswith("sha-") and v == imgs.get("latest")), "")
    code_same = same or (bool(img_sha) and subprocess.run(["git", "diff", "--quiet", img_sha, head, "--", "src", "deploy", "pyproject.toml", "uv.lock"]).returncode == 0)
    out("H02", lv(bool(dirty) or not wrap or bool(drift) or bool(extra) or not code_same, not same), head=head[:9], dirty=len(dirty), wrapper_match=wrap, drift=drift or "-", not_in_repo=extra or "-", latest_image=img_sha[:9] or "?", latest_is_head=same, code_identical=code_same, sha_tags=sum(k.startswith("sha-") for k in imgs))
@check("H03")
def _():
    ps = [l for l in sh(["docker", "ps", "-a", "--format", "{{.Names}} {{.Status}}"]).splitlines() if l.startswith("kca-")]
    stray = sh(["find", "data", "artifacts", "-maxdepth", "4", "(", "-name", "*.bak*", "-o", "-name", "*.pre_*", "-o", "-name", "*.stale*", ")", "-printf", "%p\n"]).split()[:8]
    out("H03", lv(any("Exited" in l for l in ps), bool(stray)), kca_containers=ps or "-", stray=stray or "-")
@check("R04")
def _():
    reg = [json.loads(x) for x in Path("artifacts/models/topk_ranker/retrain_registry.jsonl").read_text().splitlines() if x.strip()]
    last = reg[-1]
    age = (TODAY - date.fromisoformat(str(last["attempted_at"])[:10])).days
    t = pd.read_parquet(ROOT / "parquet/topk_decisions.parquet", columns=["decided_at", "model_version"])
    mv = str(t.sort_values("decided_at")["model_version"].iloc[-1])
    known = any(str(r.get("trained_at", "")) and str(r["trained_at"]) in mv for r in reg)
    out("R04", lv(last["outcome"] not in ("PROMOTED", "PROMOTED_UNGATED", "MANUAL_HOTFIX") or not known, age > 8 or last["outcome"] == "MANUAL_HOTFIX"), last=last["outcome"], age_days=age, agreement=last.get("agreement"), live_model_in_registry=known)
@check("B07")
def _():
    v = jr("offsite-verify", at(sun, "10:59:00"), at(sun, "23:59:00"), r"stage=(offsite_verify|restore_drill)|finding=")
    s = next((l for l in v if "stage=offsite_verify" in l and "missing=" in l), "")
    drills = [re.search(r"tier=(\w+) .*status=(\w+)", l).groups() for l in v if "stage=restore_drill" in l]
    ok = "missing=0" in s and "mismatched=0" in s and len(drills) >= 3 and all(st == "OK" for _, st in drills)
    out("B07", lv(not ok), sunday=sun, summary=s.split("stage=offsite_verify")[-1][:80] or "absent", drills=drills or "-")
@check("B08")
def _():
    snaps = sorted(x.strip("/") for x in sh([RCLONE, "lsf", "--dirs-only", f"{REMOTE}/snapshots"]).splitlines())
    cs = jr("core-snapshot", at(sun, "09:59:00"), at(sun, "23:59:00"), r"stage=core_snapshot .*status=")
    out("B08", lv(not snaps or snaps[-1] != str(sun) or not cs or "status=ok" not in cs[-1]), newest=snaps[-1] if snaps else "-", count=len(snaps), core_snapshot=cs[-1].split("status=")[1][:20] if cs else "absent")
@check("Q11")
def _():
    man = json.loads((ROOT / "history/altdata/_manifest.json").read_text())
    notok = {k: v["status"] for k, v in man["panels"].items() if v["status"] != "ok"}
    def tdlag(x):
        n, y = 0, D
        while y > date.fromisoformat(x):
            n, y = n + is_td(y), y - timedelta(days=1)
        return n
    lag = {k: v["last_date"] for k, v in man["panels"].items() if tdlag(v["last_date"]) > (4 if k == "credit_balance" else 2)}
    stuck = {}
    for name in ("extended_sessions", "toss_regular", "nxt_calibration"):
        f = pd.read_parquet(ROOT / f"history/intraday/backfill_ledger/{name}.parquet").drop_duplicates(["snapshot_date", "session", "symbol"], keep="last")
        stuck[name] = int(((f["status"] == "FAILED") & (pd.to_numeric(f["attempts"], errors="coerce") >= 3)).sum())
    rep = json.loads((ROOT / "history/nxt_decomposition_fit_report.json").read_text())
    sha_ok = hashlib.sha256((ROOT / "history/nxt_calibration_table.parquet").read_bytes()).hexdigest() == rep["table_sha256"]
    q = [p.name for p in (ROOT / "quarantine").iterdir()] if (ROOT / "quarantine").exists() else []
    out("Q11", lv(bool(notok) or any(stuck.values()) or not sha_ok, bool(lag)), altdata_not_ok=notok or "-", altdata_lagging=lag or "-", ledger_failed_3plus=stuck, nxt_table_sha_ok=sha_ok, quarantine=q or "-")
```

### 5.2 Drill-down probes

- `H02-local`: `ssh or-vps 'git -C ~/k-closing-alpha log -1 --format=%h'` vs `git ls-remote https://github.com/kthyeong/k-closing-alpha refs/heads/main`.
- `H04`: `ssh or-vps 'for f in ~/.config/systemd/user/kca-*.service; do l=$(grep -m1 "^ExecStart=%h/.local/lib/kca/kca-run " "$f") || continue; eval "~/.local/lib/kca/kca-run --print-argv ${l#ExecStart=%h/.local/lib/kca/kca-run }" >/dev/null 2>&1; echo "$(basename $f) rc=$?"; done | grep -v "rc=0$"'` (empty = PASS; a `USAGE_ERROR` means the unit cannot start). Fixture SHAs: `git merge-base --is-ancestor <sha> origin/main`.
- `R06`: `ssh or-vps 'systemctl --user is-active kca-aftermarket-book.service; docker ps --filter name=kca-aftermarket-book --format "{{.Status}}"'`.
- `C-manifest <D> <prefix>`: `ssh or-vps 'cd ~/k-closing-alpha/data/history/capture/manifests/<D> && for d in <prefix>*; do echo "$d $(ls $d | tr "\n" " ")"; done'`. For PARTIAL entries: `python3 -c` reading `manifest-partial.json` `entries[].{symbol,status,reason}` with a Counter.
- `B-parity <D> <tier>`: `~/.local/bin/rclone lsf gdrive:quant-lake/live/k-closing-alpha/capture_sealed/<tier>/<YYYY-MM>/<D>/` vs `python3 -c` listing `segment_name` from `capture/offsite/ledger/<tier>/<D>.jsonl`.
- `B03` (W only, outside 22:15–00:15): `cd ~/k-closing-alpha && nice -n 19 timeout 600 ~/.local/bin/rclone check data gdrive:quant-lake/live/k-closing-alpha/data --one-way --size-only --exclude "/history/capture/**" --combined - 2>/dev/null | awk '$1=="+"||$1=="*"' | tail -20`.
- `P-ledger`: `uv run --no-sync python -c` using `PaperLedger(root=Path("data/paper")).load_open_positions()` and `nav.parquet` tail 5 (`as_of_date nav cash open_market_value cumulative_cost`).
- `Q-spot <D> <sym>`: archive row (`종가 결정_종가 종가_확정`), last 1m bar, `price_history.close_raw` (next day), paper fill `fill_price` for the symbol.
- Journal window for any unit: `journalctl --user -u kca-<u>.service --since "<UTC>" --until "<UTC>" -o short-iso | grep -vE "pam_unix|COMMAND=" | grep -E "Starting|Finished|Failed|stage=|outcome=" | cut -c1-250 | tail -30`. Convert with `k(){ date -u -d "TZ=\"Asia/Seoul\" $1" +"%F %T"; }`.

## 6. Triage map

| Symptom (verdict line) | Likely cause | Where in code / logs | Fix direction (root cause) |
|---|---|---|---|
| `C01 FAIL coverage<0.99` | KIS decision slots throttled (`EGW00201`) or requote deadline `152800` hit | `collect.py:201,954,1309`; slots 1/5 (`kis_slot_assignments.toml`) | slot contention vs krx; requote budget |
| `C03 WARN partial` / `INCOMPLETE held_roster_unavailable` | aftermarket roster missing or `slot_token_failed` (slots 2/3) | `aftermarket_book.py:589-673` | token/slot health at 15:40 |
| `C04 partial(n failed)` on ticks, no tape_healed | vendor_failure in the evening archive; tape sweep not yet published | `archive_intraday.py:292-372`; `tick_tape_sweep.py` | check C08; the tape horizon is 30 d |
| `C07 FAIL stale` / `R01 price-ingest failed` | KRX `data-dbg.krx.co.kr` timeout (10-30); auto rerun after the 09:40 blackout | remediation ledger `failed_unit:kca-price-ingest` | expected once; repeat = KRX outage |
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
| `X01 WARN EGW00121/ADOPTED_ROTATED` | another project re-issued a shared KIS key after 07:05 | token store `~/.cache/kis` | attribute (krx/kse); harmless if adopted |
| `A01 FAIL reconciled_age` | reconcile lock busy (exit 75) or the unit is failing | `audit_reconcile.py` | journal of `kca-audit-reconcile` |
| `UNKNOWN err=…` | probe drift (schema/path change) | this file | fix the probe in the same change (§8) |
| Restore needed: `price_history` (since `ed5c912`, a `data/` restore has no monolith) | offsite form is base+patch | `src/tools/price_history_offsite.py` | `rclone copy gdrive:quant-lake/live/k-closing-alpha/data/history/price_history_offsite <tmp>` then `uv run python -m src.tools.price_history_offsite restore --from-dir <tmp> --target <new path>` (refuses an existing target; checks md5 + rows). From a snapshot: first `rclone copy …/snapshots/_blobs/price_history <tmp>`, then `…/snapshots/<D>/data/history/price_history_offsite <tmp>`, same restore. Snapshots ≤ 2026-11-01 hold the monolith |
| `steps.offsite_ledger` FAIL `ledger_not_append_only` | a seal ledger was rewritten, not appended | `offsite_backup.py` `classify_ledger_files`; `capture_offsite.py:664` | the rewritten file was uploaded with a `_deleted` version; find the writer that rewrote it (root cause), never reset `ledger_upload_state.json` to silence it |

## 7. Baselines and known conditions

Baselines (trailing rule: compare with the prev-5 trading-day median, never with a fixed level):
| Metric | Seed (date) |
|---|---|
| archive rows/day | 93–1,247 (10-12..16); 156–172 (10-29/30) |
| price_history rows/day | ~2,762–2,767 (10-30) |
| 1m regular | 485 k bars / 1,273 symbols, vendor `ls`, vol ratio median 0.959 (10-16) |
| aftermarket-book manifests | 35 (18 × krx/nxt batches; 10-30) |
| tape sweep | needs 771, recovered 604, unresolved 167, expired 0 (10-30) |
| backup | 3–13 min; `segments` 2–170 per night (11-01: 2) |
| runtimes | collect 25–95 s, predict ~12 s, finalize 10 s–2.5 min, archive-regular 28–57 min, archive 9–33 min, tape-sweep 24–31 min, daily-audit 1.5–4.5 min (incl. ~80 s chown), extended-backfill ~2.5 min |
[redacted]

Known conditions:
| Since | Condition | Expected verdict while open | Re-check / expiry | Close when |
|---|---|---|---|---|
| 2026-11-03 | Local capture backlog: on 10-30 a sweep rewrote 85 old-date `raw/` dirs (2026-01..10); `capture_prune` drains it under its time budget (`status=DEADLINE`, removed 82 on 11-02, one `rclone md5sum` per sealed entry), disk 35% | B05 WARN `kept_local_pre_prune` shrinking each run | 2026-11-05 M | still listing 2026-0x dirs at 11-05 M → prune throughput defect (batch md5 via one `lsjson --hash` instead of per-entry `md5sum`) |
| 2026-11-02 | D1 prune purge cap 7 per subtree vs daily backup | B05 PASS; FAIL `cap_risk` after a ≥1-week prune outage | 2026-12-01 | cap sized from the backup cadence (fix pending) |
| 2026-11-02 | D2 `audit_daily_completeness` omits `intraday_complete` (added in `main()` :2308) | C11 derives it from the journal | 2026-12-01 | helper returns 8 keys (fix pending) |
| 2026-11-02 | D3 heartbeat severity fallback parses the subject (`daily_audit.py:1910-1913`) | none (latent) | 2026-12-01 | fix pending |
| 2026-11-02 | D4 paper-entry idempotency compares `decision_date` unnormalized (`paper_trade.py:425`) | P01 would FAIL `dup` if it ever triggers | 2026-12-01 | fix pending |
| 2026-11-02 | D5 `kca-audit-reconcile` in its own safe-rerun list | none (busy check blocks it) | 2026-12-01 | fix pending |
| 2026-11-02 | D6 seal exception records `deferred_dates=0` | B04 WARN `seal_status_ok=False` | 2026-12-01 | fix pending |
| 2026-11-02 | D7 loose-copy excludes hardcoded to `/history/capture/**` | none unless `COLLECTION_ROOT`/`HISTORY_DIR` is overridden | 2026-12-01 | fix pending |
| 2026-11-02 | D8 core panel shrink tolerance 0 rows, no override for a legitimate dedupe | B01/B08 FAIL on a legitimate dedupe | 2026-12-01 | fix pending |
| 2026-11-02 | D9 no OK outcome for collect/paper/tape-sweep/backup/prune/verify/snapshot | probes use the journal/ledger instead | 2026-12-01 | fix pending |
| 2026-11-02 | D10 shared KIS key `134e3b771841` rejected (`EGW00121`) at 09:01 and `ADOPTED_ROTATED` | X01 WARN on O | 2026-11-16 | attributed (krx/kse re-issue) |
| 2026-11-02 | D11 no in-code assert nav = cash + open MV | P03 is the only guard | 2026-12-01 | fix pending |
| 2026-10-29 | History rewrite of the README commit (`1681172` → `88d3452`): `deploy.yml` run 40891357295 test job failed, so `latest` stays on `sha-1681172` (code-identical tree). Fixture SHA fix committed locally in `e18a930` (not pushed) | H02 WARN `code_identical=True`; H04 FAIL | 2026-11-09 | the fixture fix is committed and deploy is green |
| 2026-11-02 | Offsite churn reduction `ed5c912`/`a2523e6` (pending deploy): `price_history` goes offsite as base+patch form `data/history/price_history_offsite/` (monolith excluded from the loose copy; first successful run moves the remote monolith to `_deleted/data/<day>/`); seal ledgers upload append-only without `_deleted` versions (`capture/offsite/ledger_upload_state.json`); snapshots store the base as shared blobs `snapshots/_blobs/price_history/` | B01: `steps.price_history_form` ok (`rebased` ≈ monthly) and `steps.offsite_ledger` ok with `rewritten=0` (else FAIL `ledger_not_append_only`); B03 lists remote `data/history/price_history.parquet` missing after the first run (expected); kca-backup peak RSS +2–3 GiB for ~15 s | 2026-11-16 | 3 clean nightly runs after deploy |
| 2026-11-02 | Dashboard kca card `kca.api_actions` WARN | A04 WARN | 2026-11-09 | mapped to an API notice/action finding, or cleared |
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

## 8. Maintenance

- A new routine needs: a §3 row, a §4 presence check and a content check, a window placement (§2), and a §7 baseline. A routine without a content check is incomplete.
- Every incident ends as a check, a triage row or a known condition with an expiry date. When a timer, unit, wrapper flag, data path or contract file changes, update §3, §4 and §5 in the same change.
- Retired IDs (v3 → v4; never reuse): `DEP-01/02/03` → H02; `DEP-04/07` → H04; `SCH-01` → H01; `SCH-02`, `RUN-01/02/05` → R01; `SCH-03` dropped (blackout tuple; deploy.yml enforces it); `RUN-03` → R02/R03; `RUN-04` → P06; `RUN-06` → A03/R03; `BRK-01/02/04/05/06` → X03/X05/R05; `BRK-03` → X01; `HOST-01/02/04` → X03/X05; `HOST-03/05` → H03; `DI-ARCH-01..09` → Q01; `DI-PH-*` → Q05; `DI-1M-01..03` → Q06; `DI-1M-04/05` → Q07; `DI-1M-06` → Q10; `DI-1M-07` → Q09; `DI-TOPK-*` → P04; `DI-PAPER-*` → P03; `DI-BAK-01` → B04; `PAP-01/03` → P02; `PAP-02` → P01; `PAP-04` → P03; `PAP-05` → P06; `ML-01/02` → R04; `ML-03`, `RS-01..06` → Q11; `RS-07` → Q08; `RS-08` → C08; `RS-09` → C03/C04; `BAK-01/02` → B01; `BAK-03` → B04; `BAK-04` → B07; `BAK-05` → B05/B08; `BAK-06` → B03; `BAK-07` → B05; `ALR-01` → A01; `ALR-02` → A02; `ALR-03` → A03; `ALR-04/05` → A04/A05.
- Last probe execution log:
| Date (KST) | Rev | Window / D | Result |
|---|---|---|---|
| 2026-11-02 09:40 | 88d3452 | O / 11-02 | all PASS except X01 WARN (`EGW00121`/`AUTH_REJECTED`/`ADOPTED_ROTATED` ×1, D10); 5.6 s |
| 2026-11-02 09:41 | 88d3452 | M / 10-30 (re-audit) | CORE all PASS; WARN R03 (`auction_path DEGRADED`), X01 (D10), A04 (`kca.api_actions`); 23 s |
| 2026-11-02 09:41 | 88d3452 | E / 10-30 (retro) | CORE all PASS; WARN Q01 (`close_gap_p99=0.0309` > 0.03), X01; 5 s |
| 2026-11-02 09:41 | 88d3452 | N / 10-30 (retro) | CORE all PASS (C04 `regular-ticks: tape_healed`); WARN R03, C06 (`auction-path partial`), X01; 14 s |
| 2026-11-02 09:42 | 88d3452 | W / 10-30 | B07 B08 R04 Q11 PASS; WARN H02 (`latest`=sha-1681172, `code_identical=True`), H03 (5 known stray `.bak/.pre_*`); 4 s |
| 2026-11-02 09:43 | local | W local + drills | H04 FAIL: latest `deploy.yml` run 40891357295 (88d3452) test job failed; `--print-argv` 22/22 rc=0; fixture SHAs ancestors (working copy). A05 PASS (5/5 watchdog runs). R06 drill executed: inactive (outside 15:40–19:59; expected) |
| — | — | live E/N for 2026-11-02 | not executed live: their evidence does not exist before 15:45 / 21:50. Executed retroactively on 10-30 instead; extracted bodies pass `ast.parse` |
