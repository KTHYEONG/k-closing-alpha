# VPS Automation Health Checklist (or-vps) — v2

> **Audience:** an AI agent that audits and periodically reports on the production automation of `k-closing-alpha`
> on host `or-vps`. **Goal:** prove with evidence — not assumptions — that every scheduled job ran on time, that the
> data it produced is *correct* (not merely present), that paper trading is conserved and causal, that offsite
> backup can restore it, and that the alert paths would have told a human. A green `systemctl` is not proof.
>
> **Design contract of this file:** generic and durable. It describes *how to verify*, derives expectations from the
> repo (timers, calendar, run-outcome log) instead of hard-coding them, and ships executable probes that print one
> machine-parseable line per check (`CHECK <ID> <PASS|WARN|FAIL> <detail>`). Last verified against the live host on
> 2026-10-05; if a command here disagrees with the host, the host is truth — report the drift (§14).

## 0. Autonomous loop

The host now runs a closed loop by itself; the AI check verifies the loop's evidence rather than re-deriving it.

- **10-minute tick** (`kca-audit-reconcile`, 07:00–23:00 KST): measure (failed units, stale tokens, backup, ops sentinel)
  → remediate (allowlisted safe re-runs only) → hold the first mail while a fix is in flight → notify only on
  persistence past the grace → publish the heartbeat. A busy lock skips the tick silently (exit 75 = success).
- **Hold-then-notify**: a fresh issue with an active remediation attempt produces no mail; it notifies through the
  normal path only if still open after the grace. A key that self-resolves while held never mailed and needs no
  retraction (ledger `RESOLVED`, heartbeat `info_notes` carries `auto_remediated=<key>:<unit>`).
- **Advisory tier**: research-degraded findings (e.g. uncertified tick gaps) never mail; they ride the heartbeat info
  notes and escalate to warnings only after consecutive audits.
- **Evidence first**: read `logs/heartbeat/daily_audit.json` (`open_issues`, `info_notes`, `reconciled_at`),
  `logs/remediation/ledger.jsonl` (STARTED without a matching RESOLVED/UNRESOLVED = still in flight), and
  `logs/heartbeat/advisory_streaks.json` before running any probe. Run probes second. Intervene only for blocking
  issues or `UNRESOLVED` ledger rows.

## 0b. Run Modes and Cadence

Pick the mode from the clock and the request; do not run heavier modes than the situation needs.

| Mode | When (KST) | Scope | Weight |
|---|---|---|---|
| **W — Window watch** | inside a critical window (§4); one pass ~2 min after the slot | only the slot's units + outcome + the one data check it feeds | light (journal/systemctl only inside blackouts) |
| **D — Daily audit** | trading-day evening after 23:00 (backup running until ~00:00 is normal) **or** next morning 07:30–08:20 | §1–§12 for audit date `D` | medium (parquet probes) |
| **WK — Weekly** | Monday 07:30–08:20 (covers Sat retrain, Sun snapshot/verify) | D plus §13 | medium |
| **M — Monthly** | first Monday | WK plus local restore sample (§10.5), docker/journal growth, expiries, checklist drift (§14) | medium |
| **E — Event** | after a deploy, an alert, an incident, or an operator question | affected areas + §2 context + D probes | light–medium |

Periodic chat reporting is optional. The tick and the watchdog are independent of any session: absence of a chat
report never implies the host stopped. Each run saves its report (§15) and compares with the previous one.

## 1. Rules of Engagement

- **Read-only on the host.** Never `start/stop/restart/enable/disable/reset-failed` units, never `docker run/rm/pull`, never
  edit files under `~/k-closing-alpha`, `~/quant-secrets`, `~/.cache/kis`, never run `rclone` write verbs
  (`copy/sync/move/delete/purge/dedupe/cleanup`). Allowed rclone: `lsf lsjson lsd size about cat check --one-way`.
  Exception: only when the operator explicitly authorizes a named action in the current conversation.
- **Never run job entry points by hand** (`src.daily.*`, `src.tools.daily_audit`, `audit_reconcile`, `offsite_backup`,
  `backup_prune`, `capture_offsite verify`, `alerts`, `backfill_*`, `toss_*`, `nxt_*`): they write ledgers, consume broker
  quota, send mail, or take the Drive lock. Pure read helpers and the probes in this file are allowed.
- **Trading blackout 08:50–09:40 and 15:10–15:50 KST on weekdays** (`src/tools/deploy_window.py`): only `systemctl`,
  `journalctl`, `stat`, `ls`, `cat` of small files. No parquet scans, `du`, `find` over `data/`, `rclone check/size`.
  Heavy probes use `nice -n 10 timeout 300` and run outside blackouts and outside 22:15–00:15 when avoidable (backup,
  extended backfill 23:05, Drive lock).
- **Secrets:** never print `~/quant-secrets/*.env`, `env`, token caches, `rclone config show`. Report key *names* and modes.
- **Timezones:** host clock and `journalctl --since/--until` are **UTC**; convert KST with
  `k(){ date -u -d "TZ=\"Asia/Seoul\" $1" +"%F %T"; }` then `k "2026-10-02 15:00"` -> `2026-10-02 06:00:00`
  (do **not** use `TZ=Asia/Seoul date -d ... -u`: `-u` re-reads the input as UTC and returns it unchanged); every timer `OnCalendar` is **Asia/Seoul**
  (KST = UTC+9). Convert before judging "late/missing". Pass `TZ=Asia/Seoul` to Python. Date columns inside parquet are KST dates.
- **Evidence or it did not happen:** every PASS cites the command/line it rests on. Ambiguous → `WARN`, never silent PASS.
  Things you could not check go in the report's `NOT VERIFIED` list with the reason.
- **Diagnose, do not fix.** Report to the operator in Korean (CLAUDE.md §4); scratch work goes under local `scratch/`.
- **rclone is not on the non-interactive ssh PATH:** always call `~/.local/bin/rclone` (shown as `$R` below).
- **Do not blame this repo for neighbours.** The host also runs `krx-alpha`, `crypto-pilot`, `mt-etf-king-2026`,
  `quant-dashboard`. Shared resources are CPU/RAM (12 GiB), disk, the KIS quota, and `flock %t/quant-gdrive.lock`;
  report contention, do not audit the neighbours' logic.

## 2. Context Derivation (always first)

```bash
ssh -o ConnectTimeout=15 or-vps 'hostname; date -u; TZ=Asia/Seoul date; uptime -p; whoami; loginctl show-user $USER -p Linger'
ssh or-vps 'cd ~/k-closing-alpha && TZ=Asia/Seoul ~/.local/bin/uv run --no-sync python - <<"EOF"
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo
from src.data.session_calendar import resolve_session_day
now = datetime.now(ZoneInfo("Asia/Seoul"))
for k in range(0, 8):
    d = now.date() - timedelta(days=k)
    print(d.isoformat(), d.strftime("%a"), resolve_session_day(d).kind.value)
EOF'
```

- **Day kind** per date: `STANDARD` (trading), `SHIFTED` (trading, shifted clock), `CLOSED` (weekend/holiday), `UNKNOWN`
  (calendar not extended — itself a finding). Weekday timers (`Mon..Fri`) still fire on weekday holidays; the *job* must
  then record `SKIPPED` (§5).
- **Audit date `D`** = most recent `STANDARD/SHIFTED` date whose evening chain has finished (≥ 23:00 KST) — normally yesterday
  or today after 23:00. Also note `D-1` (paper exit of `D-1` lots happens on `D`).
- SSH failure is a P0 finding (retry once after 30 s, then report; the watchdog may also be firing).
- Uptime shorter than the last check interval = a reboot; `Persistent=true` timers catch up, others may have been skipped.
- Load the previous report (§15). If absent, say "no baseline" and treat trends as unknown, not as stable.

## 3. Check Catalog Conventions

- IDs: `<AREA>-<NN>`. Status: `PASS | WARN | FAIL | SKIP` (SKIP = not applicable today, with the reason, e.g. holiday).
- Severity if not PASS: **P0** data loss/corruption, ledger inconsistency, decision chain down, backup shrink, silent alert path
  failure; **P1** a scheduled job silently not running or failing, causality violation, quota hit on the decision chain,
  restore impossible; **P2** research capture gaps, trends, hygiene.
- **Stateful signals — know what is fresh and what is stale** (most false alarms and most missed problems live here):
  | Signal | Updated when | Pitfall |
  |---|---|---|
  | `systemctl --failed` / unit `Result` | the unit's next run | stays failed after the cause is fixed until the next successful run |
  | `daily_audit.json` heartbeat | each audit (Mon–Fri 21:20) and each 10-minute reconcile tick (07:00–23:00) | v2 fields: `severity`, `open_issues`, `provisional_reasons`, `audit_kind`, `reconciled_at`; the dashboard card is derived from it |
  | `audit_alert_state.json` | audit/reconcile | opened issues notify once; resolution and 24 h reminders are the only repeats |
  | `offsite/last_run.json` | **end** of a backup | during a run it still shows the previous night; use `offsite/in_progress.json` (exists only while running; stale after its `deadline_at`) |
 | `offsite/deferred_history.json` | **end** of every backup (rolling 14 runs) | `offsite_backup:draining` (informational) while the deferred date count falls vs `deferred` (warning) after 3 stalled runs or an oldest deferred date older than 21 days; a missing/corrupt history reads as first observation (draining), never as stalled |
  | `logs/events/<YYYY-MM>/<D>.jsonl` run outcomes | each job | `OK`, `DEGRADED`, `NO_DECISION`, `SKIPPED` (no-op on closed days) — `SKIPPED` is not success |
  | tape sweep `last_report.json` | each sweep | absent = never ran; the audit does not warn about absence |
  | dashboard `~/quant-dashboard/public/status.json` | every minute | mirrors the above; a card that disagrees with ground truth is itself a finding |

## 4. Critical Moments (do not miss these)

For each window run mode **W** about 2 minutes after the slot. Inside blackouts use light commands only.

| KST | Unit(s) | What must be true | Fail meaning |
|---|---|---|---|
| 07:05 / 07:10 | `kca-kis-token-warmup`, `kca-kiwoom-token-rotate` | `Result=success`; every declared KIS host slot has a fresh token (no stale-token line in the next digest); Kiwoom expiry moved out of the 15:20 window. Warmup skips only on a verified `CLOSED` day | P1: every KIS call of the day degrades |
| 08:30 | `kca-price-ingest` (also 11:30, 21:30) | outcome `OK` with `n_new_rows ≥ 0`; on `CLOSED` days `SKIPPED` | P1 features stale |
| 08:39:55 (optional) | `kca-auction-open` | research only | P2 |
| **09:01** | `kca-paper-exit` | every older open lot has a sell fill + `trigger` + trades row before 09:30 | P0 ledger (missed exit) |
| **15:20 → 15:35** | `collect` 15:20 → `predict` 15:21 → `auction-close` 15:21:05 → `finalize-close` 15:30:30 → `paper-entry` 15:34 | each finished inside its budget (collect 9, predict 7, finalize 12, entry 10 min); **`predict` decided before 15:30:00**; `finalize-close` before `paper-entry`; no `EGW00201`/rate-limit lines; outcomes `OK` (or `SKIPPED` on closed days) | **P0** decision chain; P1 quota |
| 15:40 | `archive-intraday-regular`, `aftermarket-book` | `Finished`; regular partition for `D` appears | P1 audit/backup input |
| 20:05 | `archive-intraday` | NXT/KRX aftermarket partitions for `D` | P1 |
| 20:35 | `tape-sweep` | `Finished` (deadline 21:15); report refreshed; a holiday run must not crash | P1 |
| 21:00 / 21:20 | `backup-prune`, `daily-audit` | audit heartbeat `finished_at` ≈ 21:20–21:45, digest delivered | P1 alert path |
| 22:15 → ~00:00 | `kca-backup` | `in_progress.json` appears at start, disappears at end; `last_run.json` updated; duration follows §10 baselines | P0 if shrink, P1 if failed |
| 23:05 | `kca-extended-backfill` | runs up to 8 h; ledger counts move (§12) | P2 |
| Sat 22:00 | `kca-retrain` | `PROMOTED` (or a reasoned `REJECTED`); next Monday `predict` loads it | P1 |
| Sun 10:00 / 11:00 | `kca-core-snapshot`, `kca-offsite-verify` | both succeed; verify `missing=0 mismatched=0`, restore drill OK | P1 DR |
| 07:00–23:00 every 10 min | `kca-audit-reconcile` | runs every day including weekends; heartbeat `reconciled_at` advances ~10 min; a busy lock skips silently | P2 |
| 07:40 Tue–Sat (GHA) | `watchdog.yml` | succeeded | P1 dead-man |

Normal evidence (observed on trading day 2026-10-02) so you can recognise health: `collect` finished 15:20:46 (`realtime_coverage ... coverage=1.0000 status=COMPLETE`);
`predict` outcome `OK n_picks=3` at 15:21:16 (before 15:30:00); `finalize-close` `close_finalization n_finalized == archive rows, n_unconfirmed=0` at ~15:31:04;
`paper-entry` at 15:34 logs `SKIP reason=already_recorded` — healthy, because `finalize-close` already recorded the entry and the 15:34 unit is the idempotent
backstop (a missing entry is only a finding if no fill/NO_DECISION row exists, §8). Token units: warmup `status=ISSUED` on trading days and
`SKIP reason=non_trading_day` on `CLOSED`; Kiwoom rotate `ROTATED` or `UNCHANGED` (both healthy); neither is a data load.

Derive the live schedule instead of trusting this table (it drifts when timers change):

```bash
ssh or-vps 'cd ~/k-closing-alpha && for t in deploy/systemd/kca-*.timer; do n=$(basename $t); \
  printf "%-38s %s | last=%s | next=%s\n" "$n" "$(grep -E "^OnCalendar" $t | cut -d= -f2- | tr "\n" ";")" \
  "$(systemctl --user show $n -p LastTriggerUSec --value)" "$(systemctl --user show $n -p NextElapseUSecRealtime --value)"; done'
```

## 5. Deploy, Scheduler and Unit Evidence

```bash
git -C /home/kth/k-closing-alpha fetch -q; git -C /home/kth/k-closing-alpha log -1 --format="%h %ci %s" origin/main
gh run list --workflow deploy.yml --limit 5
ssh or-vps 'cd ~/k-closing-alpha && git log -1 --format="%h %ci %s" && git status --porcelain | head && \
  docker images ghcr.io/kthyeong/k-closing-alpha --format "{{.Tag}} {{.ID}} {{.CreatedSince}}" | head -3 && \
  docker image inspect ghcr.io/kthyeong/k-closing-alpha:latest --format "{{.Id}}"'
ssh or-vps 'cd ~/k-closing-alpha && for t in deploy/systemd/*.timer; do n=$(basename $t); \
    echo "$n $(systemctl --user is-enabled $n 2>&1)"; done | grep -v " enabled$"; \
  for f in deploy/systemd/kca-*; do cmp -s "$f" ~/.config/systemd/user/$(basename $f) || echo "DRIFT $(basename $f)"; done; \
  systemctl --user list-units --failed --no-legend; systemctl --user list-timers "kca-*" --all --no-pager | tail -5'
```

- [ ] **DEP-01** VPS HEAD == `origin/main`. A gap is explained only if `gh run list` shows that commit's deploy `in_progress`
      (wait/re-check, not drift), or a deploy deferred by a blackout. A gap with a failed or absent run is a finding.
- [ ] **DEP-02** VPS `git status --porcelain` empty; `latest` image ID == `sha-<HEAD>` image ID; last 5 deploys green or explained.
- [ ] **SCH-01** Every repo timer is `enabled` except `OPTIONAL_MANUAL_TIMERS` (`src/tools/code_sync.py`: auction-open,
      auction-close, altdata-capture — report their state). A disabled non-optional timer is **P1**.
- [ ] **SCH-02** Every enabled timer fired at its most recent slot (convert UTC→KST; `-` on an old enabled timer = finding).
- [ ] **SCH-03** No `DRIFT` (installed unit == repo unit); `Linger=yes`.
- [ ] **SCH-04** Each `--failed` unit: get the cause (`journalctl --user -u <unit> --since "<D-1 UTC> 12:00" --no-pager | tail -60`),
      classify transient/precondition/code bug, and say whether a newer deploy already fixed it (failed state persists until the next run).

Per-unit run evidence for every unit due on `D` (build `--since/--until` with the `k` helper of §1;
`systemctl show ... ExecMain*Timestamp` and `Result` describe only the **latest** run, so for `D` use the journal window and the
run-outcome log, never `show`; oneshot containers are `--rm`, so `docker logs` is empty
afterwards — the journal is the only record):

```bash
ssh or-vps 'journalctl --user -u kca-<name>.service --since "<D-1 15:00 KST as UTC>" --until "<D+1 15:00 KST as UTC>" --no-pager -o short-iso \
  | grep -E "Starting|Finished|Failed|status=|outcome=|\[(SYS|DATA|ALGO|PORTFOLIO|RISK|EXEC)\]" | tail -25'
ssh or-vps 'systemctl --user show kca-<name>.service -p ExecMainStartTimestamp -p ExecMainExitTimestamp -p ExecMainStatus -p Result'
```

- [ ] **RUN-01** exactly one `Finished` per due unit on `D` (documented retries allowed), `Result=success`.
- [ ] **RUN-02** runtime ≤ 70 % of `TimeoutStartSec` (else WARN: approaching a hang).
- [ ] **RUN-03** grep journals for `DEGRADED NO_DECISION QUOTA EGW00 rate Traceback timeout DISK_GUARD LEDGER_INVALID TOKEN_REPLACED
      AdmissionDirNotSharedError`. Exit 0 with `DEGRADED` is still a finding.
- [ ] **RUN-04** closed day (§2): weekday timers must log `SKIP reason=non_trading_day` and record `SKIPPED`; a job that
      "ran normally" on a closed day with rows written is a calendar-disagreement finding.
- [ ] **RUN-05** run outcomes `cat data/logs/events/<YYYY-MM>/<D>.jsonl`: every recording job `OK`; `DEGRADED/NO_DECISION` need
      a matching alert (§11); a trading-day `predict` of `SKIPPED` is a calendar disagreement (P1).

## 6. Host Resources

```bash
ssh or-vps 'df -h / ; df -i / | tail -1; free -h | sed -n 2p; docker system df; docker ps -a --format "{{.Names}} {{.Status}}"; \
  journalctl --user --disk-usage; du -sh ~/k-closing-alpha/data/history/capture 2>/dev/null'   # du only outside blackout
```

- [ ] **HOST-01** disk < 80 %, inodes < 80 %; record free GB and the change since the last report (capture grows daily; project
      the days until 80 %).
- [ ] **HOST-02** memory `available` > 2 GiB at rest (retrain uses `--memory=6g`; the host is shared).
- [ ] **HOST-03** `Up` `kca-*` containers only inside a job window (else a hung unit). An `Exited` `kca-*` container is a leftover
      of a hand-run `docker run` without `--rm` (P2 hygiene: name it, say who likely created it, ask before removal). Neighbours' containers are listed, not judged.
- [ ] **HOST-04** `ghcr.io/kthyeong/k-closing-alpha` images: `latest` plus at most the 2 previous `sha-*` tags (rollback set); older tags are
      cleanup candidates (report; remove only with operator authorization, never touch neighbours' images). Journal size reported with trend
      (no auto-prune by this repo).

## 7. Data Integrity — four levels (presence is not correctness)

The 09-22 collection-integrity incident (a test fixture polluted the live store, presence checks were green) is the reason
this section exists. Run the probes of Appendix A on every **D**; all emit `CHECK` lines.

| Level | Question | Examples (probe IDs) |
|---|---|---|
| L1 presence | does the artifact exist with plausible size/rows vs the trailing 5 days? | `DI-ARCH-01`, `DI-PH-02`, `DI-1M-01`, `RS-07`, `RS-09` |
| L2 structure | schema, nulls, duplicates, ranges, monotone keys | `DI-ARCH-02/03/06`, `DI-PH-03/04/06`, `DI-1M-02/03` |
| L3 cross-source semantics | do independent sources agree? | `DI-ARCH-07/08` (change ratio vs final close), `DI-1M-04/05` (1m volume vs EOD volume; last bar vs EOD close), top-k vs archive |
| L4 causality / PIT | is time ordering respected? | `DI-ARCH-09`, `DI-TOPK-01` (snapshot ≤ feature_available ≤ inference ≤ decided < 15:30 ≤ execution) |

Interpretation rules (learned from live data):

- Thresholds that depend on vendor mix must be **compared with the trailing baseline**, not with an absolute number: stored 1m
  volume is ~0.96 of EOD volume (median) with ~10 % of symbols below 0.9 *every* day for the `ls` vendor; a **drift** from that
  baseline is the signal, not the level. Baseline fields are printed so you can judge.
- `archive.parquet` `등락률` is computed on the **final** close (`종가`), not on `결정_종가` (the 15:20 print) — compare accordingly;
  `결정_종가` vs `종가` gap p99 ≲ 3 % is normal.
- tick vs bar gap with a `tape_complete` proof = vendor difference, informational: the daily audit reports it as
  `regular_ticks_source_diff` (an info line plus a `tick_source_diff` run-outcome record, never `intraday_complete`);
  only `certified_gap` or `source_diff_systemic` warns (see `DI-1M-06`).
- Row counts of `archive` vary with the market (observed 257–771 per day); only a ratio outside 0.5–2× of the trailing median warns.
- Never call a data problem "fixed" because a file exists; quote the probe line.

Additional manual checks (sample, then escalate if wrong):

- [ ] **DI-MAN-01** pick 2 symbols from `D`'s top-k and 1 random symbol: archive close, price_history `close_raw`, 1m last bar and
      `paper/fills` entry price all agree (entry == confirmed close; exit of `D-1` lots == `D` open-auction price).
- [ ] **DI-MAN-02** `quarantine/` entries new since the last report (each is a rejected anomaly); no synthetic/test symbols or
      cohort IDs in live stores (`2026-09-22-test-fixture-cohort` is a known historical entry — report as hygiene, not new).
- [ ] **DI-MAN-03** capture manifests for `D` (`data/history/capture/manifests/<D>`): list collectors with `missing_entries`,
      `quota_exceeded`, `empty`.
- [ ] **DI-MAN-04** the audit's own helpers agree with the probes (no email, no writes):

```bash
ssh or-vps 'cd ~/k-closing-alpha && TZ=Asia/Seoul timeout 180 ~/.local/bin/uv run --no-sync python - <<"EOF"
from src.tools.daily_audit import audit_daily_completeness, list_failed_kca_units
from src.tools.run_outcome import load_run_outcomes
D = "<D>"
print("completeness", audit_daily_completeness(D)); print("failed_units", list_failed_kca_units()); print("outcomes", load_run_outcomes(D))
EOF'
```

All completeness keys (`archive close_confirmed decision paper_entry paper_exit minute_bars price_history_fresh`) must be `True`.

## 8. Paper Trading Ledger (`data/paper/`)

Run probe IDs `DI-PAPER-*` (nav conservation, duplicates, cumulative cost monotone, friction on every closed trade, net = gross − cost)
and verify manually:

- [ ] **PAP-01** open lots belong only to decision date `D` after `D`'s entry; an older `decision_date` = missed exit (P0).
- [ ] **PAP-02** for `D`: entry fills for the persisted top-k, or a `decisions.parquet` NO_DECISION row with a reason — never neither.
- [ ] **PAP-03** every `D-1` lot has a sell fill with `trigger` and a `trades` row; orders without fills are explained by `reason`.
- [ ] **PAP-04** `nav` one row per trading day since paper start (no gaps), `n_open_positions` equals the open-lot count,
      `cash` delta equals Σ(sell − buy − fees − tax) of the day's fills.
- [ ] **PAP-05** which model priced the day: `topk_decisions.model_version` matches a `PROMOTED`/`MANUAL_HOTFIX` registry row (`RS-04`).

## 9. Models and Research Data

Probes `RS-01` (altdata panels `status`/lag), `RS-02-*` (backfill ledgers), `RS-03` (NXT decomposition artifacts bound by digest),
`RS-04` (retrain registry vs live model), `RS-05/06` (stray files, quarantine), `RS-07..09` (session partitions, tape sweep, manifests).

- [ ] **ML-01** latest registry row: `PROMOTED` within the last 8 days; `agreement` with the live bundle ≳ 0.9 (observed 0.989); a
      `REJECTED` or `MANUAL_HOTFIX` row is read in full (`reasons`), not just counted.
- [ ] **ML-02** retrain journal (`kca-retrain`): `pit_status`/`pit_reasons` lines; PIT gate mode is `advisory` unless the operator
      says otherwise; a missing reconstruction certification means arm-1 scoring (expected while ADOPT is not granted).
- [ ] **ML-03** research artifacts consistent: `nxt_decomposition_fit_report.table_sha256` == sha256 of the table (`RS-03`);
      ledger `FAILED` rows with ≥ 3 attempts = stuck (P2); `EXHAUSTED` growth reported.
- [ ] **ML-04** a retrain bundle trained on EOD features is scored on the 15:20 panel; `pit_haircut` report age ≤ 14 days
      when the gate is advisory/enforce (see `retrain_gate`).

## 10. Offsite Backup (gdrive) — can we restore the research data?

```bash
ssh or-vps 'cd ~/k-closing-alpha && ls -la data/history/capture/offsite/; cat data/history/capture/offsite/in_progress.json 2>/dev/null; echo; \
  python3 -m json.tool data/history/capture/offsite/last_run.json | head -60; \
  journalctl --user -u kca-backup.service --since "-8d" --no-pager -o short-iso | grep -E "Starting kca-backup|offsite_backup status|Failed" | cut -c1-30,70-210'
ssh or-vps 'journalctl --user -u kca-offsite-verify.service --since "-8d" --no-pager -o cat | grep -E "offsite_verify|restore_drill|mismatch" | tail -6'
ssh or-vps 'R=~/.local/bin/rclone; $R about gdrive: 2>&1 | head -4; $R lsd gdrive:quant-lake/live/k-closing-alpha; $R lsf gdrive:quant-lake/live/k-closing-alpha/snapshots --max-depth 1 | tail -5; $R lsf gdrive:quant-lake/live/k-closing-alpha/_deleted/data --max-depth 1 | sort | sed -n "1p;\$p"'
```

- [ ] **BAK-01** last run `status=ok`, every `steps.*` ok (`deferred` is a WARN only when stalled/aged — track whether it drains),
      started at/after the last Mon–Fri 22:15 KST slot. `offsite_backup:draining` is informational (backlog shrinking:
      digest info line `backup backlog draining: <previous> -> <current> dates`); `offsite_backup:deferred` warns only when
      the deferred date count stalls (3 consecutive non-decreasing runs, new captures can briefly outpace sealing so 1–2 are grace)
      or the oldest deferred date is older than 21 days (approaching local sealed-retention). The marker's `started_at` is written after the unit's pre-steps, normally 0–2 min after the slot. While a run is active `in_progress.json` exists and `now < deadline_at`
      (else interrupted = P1). Outside the run window no marker may exist. `DI-BAK-01` prints the deferred-history trend.
- [ ] **BAK-02** `core_panels` row counts never shrink (`CORE_ROW_SHRINK_TOLERANCE = 0`); a shrink is **P0**.
- [ ] **BAK-03** duration baseline: steady state is minutes (50 s – 6 min observed with ≤ 6 segments); hours are expected only while
      draining a backlog (≈ 20–26 s per sealed segment regardless of size — Drive per-file latency). Report `duration_s`,
      `segments`, `archive_bytes` per night and flag a sustained > 60 min with few segments as a regression.
- [ ] **BAK-04** weekly `kca-offsite-verify`: `missing=0 mismatched=0` and all restore drills `OK` (raw, normalized, manifests).
- [ ] **BAK-05** `_deleted/<subtree>/<date>` oldest ≤ 30 days (prune works); `snapshots/` latest is last Sunday (≤ 8 weekly + ≤ 12 monthly);
      gdrive free space ≥ 2× weekly growth.
- [ ] **BAK-06** local-vs-remote parity of loose data (outside blackout and outside the backup window; read-only; ~1–3 min):

```bash
ssh or-vps 'cd ~/k-closing-alpha && ~/.local/bin/rclone check data gdrive:quant-lake/live/k-closing-alpha/data --one-way --size-only \
  --exclude "/history/capture/**" --combined - 2>/dev/null | awk "{c[\$1]++} \$1==\"+\"||\$1==\"*\"{print} END{for(k in c) print \"count\",k,c[k]}" | tail -40'
```

  Lines `+ path` (present locally, missing on gdrive) and `* path` (size differs) must all be files modified after `last_run.json.started_at`
  (changes since the last night). Anything older is a backup gap (P1). Also detect remote duplicates:
  `~/.local/bin/rclone lsf -R --files-only gdrive:quant-lake/live/k-closing-alpha/data --exclude "/history/capture/**" | sort | uniq -d`
  (identical-content duplicates are P2 hygiene; differing sizes are P1).
- [ ] **BAK-07** flock contention: a backup started much later than 22:15 → find who held `quant-gdrive.lock`; not a bug.

### 10.5 Restore rehearsal for local ML research (monthly, from the operator's PC — rclone is `drive.readonly`)

1. `rclone copy gdrive:quant-lake/live/k-closing-alpha/data <scratch>/data --exclude "/history/capture/**"` (≈ 1.6 GiB).
2. With `DATA_DIR=<scratch>/data` run the Appendix-A probes against the **latest backed-up** date; they must give the same
   `CHECK` results as on the VPS.
3. Compare `sha256` of `price_history.parquet`, `archive.parquet`, `paper/*.parquet` with the VPS (`sha256sum` over ssh) for files
   not modified since the last backup.
4. Capture evidence (raw tick pages) restores per date with `python -m src.tools.capture_offsite restore --tier <tier> --date <D> --dest <path>`;
   do this only for one rotating date and delete the scratch copy afterwards.
5. Delete the scratch copy; record time and size in the report.

## 11. Alerting, Heartbeat and Dead-Man Paths

```bash
ssh or-vps 'cd ~/k-closing-alpha/data/logs/heartbeat && cat daily_audit.json; echo; cat audit_alert_state.json'
ssh or-vps 'journalctl --user -u "kca-alert@*" -u kca-audit-reconcile.service --since "-3d" --no-pager -o short-iso | grep -E "Starting|Finished|Failed|outcome|channel|reconcile" | tail -30 | cut -c1-220'
gh run list --workflow watchdog.yml --limit 5
ssh or-vps 'python3 - <<"EOF"
import json
s=json.load(open("/home/ubuntu/quant-dashboard/public/status.json")); print(s["generated_at"], s["level"], s["market_day"])
for p in s["projects"]:
    for c in p["checks"]:
        if c["level"]!="OK": print(p["name"], c["id"], c["level"], c["detail"][:160])
EOF'
```

- [ ] **ALR-01** heartbeat `snapshot_date` is the latest **weekday** audited (it can be later than `D`, e.g. a weekday holiday has `day_kind=holiday`);
      its `finished_at` is ≈ 21:20–21:45 KST of that weekday or a later `reconciled_at`; `undelivered_alerts == 0`; `schema_version == 2`.
      `info_notes` carries the digest info lines for trend (tape residual, draining backlog, source diffs); `draining` never sets `provisional_reasons`.
- [ ] **ALR-02** every `open_issues[*]` maps to a finding in this report; `provisional_reasons` non-empty only while the backup runs
      (`offsite_backup:running` — `draining` is settled-slow, not provisional).
      A WARN subject with `open_issues == []`, or `open_issues` that no longer reproduce, means reconcile did not clear it (P2 — name the stale source).
      A `deferred`-open state that now reads `draining` resolves with `offsite_backup:deferred -> draining` (no new issue, no notify).
- [ ] **ALR-03** each failed unit in the window produced a `kca-alert@<unit>` instance that **delivered**; a failed alert = P1 blind spot.
- [ ] **ALR-04** `watchdog.yml` last 5 scheduled runs succeeded; GHCR PAT expiry > 30 days.
- [ ] **ALR-05** dashboard cards vs ground truth: any non-OK card must correspond to a current finding; any ground-truth finding with an
      OK card is a monitoring gap (P1). Ask the operator (do not assume) whether the digest email for `D` arrived.
- [ ] **ALR-06** closed-day behaviour: weekday holiday digest is `HOLIDAY_SKIP` (no mail) unless a real problem exists.

## 12. Credentials, Tokens, Expiries and Quotas

```bash
ssh or-vps 'stat -c "%a %U %n" ~/quant-secrets/*.env ~/.cache/kis; ls ~/.cache/kis | sed "s/_[0-9a-f]\{8,\}.*//" | sort | uniq -c; \
  journalctl --user -u kca-kis-token-warmup.service -u kca-kiwoom-token-rotate.service --since "-2d" --no-pager | grep -E "Finished|Failed|status=|slot|expire" | tail -12'
```

- [ ] **CRD-01** env files mode `600`, `~/.cache/kis` mode `700`, owner `ubuntu`.
- [ ] **CRD-02** token warmup/rotate succeeded on `D`; the KIS admission directory is shared with the host mount (neighbours use it).
- [ ] **CRD-03** no `EGW00201`/429 on the decision chain (15:20–15:31) — P1; the KIS quota is shared with `krx-alpha` etc.
- [ ] **CRD-04** Toss token is shared between PC and VPS: `TOKEN_REPLACED` in logs means two hosts used it at once (coordinate backfills).
- [ ] **CRD-05** expiry notices (`src/tools/expiry_notices.py`): credentials and KRX calendar horizon — report days left; the calendar
      must contain next year's holidays before year-end.
- [ ] **CRD-06** DART status `020` may come from the neighbour sharing the key; check before blaming this repo.

## 13. Weekly and Monthly Items

- [ ] Sat retrain result and Monday `predict` bundle load (`NO_DECISION`/parity errors absent) — §9.
- [ ] Sun core snapshot and offsite verify — §10.
- [ ] Trends for the week: disk/gdrive growth, backup durations, archive/price rows, ledger `FAILED/EXHAUSTED` counts, quarantine, docker/journal size.
- [ ] Hygiene: stray `*.bak*`, `*.pre_*`, scratch/leftover files on the host; unused old images; known entries are listed once and tracked, not re-reported as new.
- [ ] Monthly: §10.5 restore rehearsal; calendar horizon; PAT/credential expiries; this file's drift check (§14).

## 14. Known Pitfalls (each was a real incident or false alarm)

- **Stale-looking warnings.** A failed unit stays failed until its next run; the heartbeat WARN persists until the next audit/reconcile.
  Report the *cause*, whether it is fixed in the deployed commit, and when the state will clear.
- **Audit/backup race (fixed 2026-10-05).** `last_run.json` is written at the end; an audit during the 22:15–00:00 run used to read
  `offsite_backup:stale`. Now `running` is informational and `interrupted` is the warning.
- **Holiday handling.** Weekday timers fire on holidays; units must `SKIP`/record `SKIPPED`. The tape sweep used to treat days after the newest
  `price_history` date as trading days and crash on holidays (fixed 2026-10-05) — a repeat means the calendar helper regressed.
- **UTC vs KST.** `journalctl --since "2026-10-05 20:53"` on the host means 20:53 UTC. Convert.
- **`docker logs` is empty** for `--rm` oneshot units; use the journal.
- **Value correctness vs presence.** A file with plausible size can hold wrong values (09-22). Always run the probes.
- **Shared resources.** `TOKEN_REPLACED` (Toss token used from PC and VPS), `AdmissionDirNotSharedError` (admission dir not mounted), the
  Drive lock, and KIS quota are cross-project; attribute precisely.
- **Calendar `UNKNOWN`.** Outside the verified range the resolver cannot say; jobs fall back conservatively and the expiry notice should already have fired.
- **Checklist drift.** When a timer/unit/data path is added, removed or renamed, update §4/§5 and Appendix A in the same change; the live
  schedule command in §4 and the probes are the authority, the tables are a convenience.

## 15. Report Format and Persistence

Write in Korean (keys/badges English). Save the full report as `scratch/vps_reports/<YYYY-MM-DD>-<mode>.md` plus a machine block
`scratch/vps_reports/<YYYY-MM-DD>-<mode>.json`; the next run loads the newest JSON for deltas.

```
## VPS Audit — D=<D> (<mode>, checked <KST timestamp>, previous: <date or none>)
Overall: PASS | WARN | FAIL            Next check due: <KST>

| Area | Status | Evidence (1 line, with delta vs previous) |
|---|---|---|
| Context (day kind, uptime) | | |
| Deploy / scheduler parity | | |
| Critical windows (07:05 · 09:01 · 15:20–15:35 · 20:35 · 22:15) | | |
| Run evidence / outcomes | | |
| Host resources | | disk 21% (150G free, −0.3 pt/day) |
| Data integrity L1–L4 (probe CHECK lines) | | n PASS / n WARN / n FAIL |
| Paper ledger | | |
| Models / research data | | |
| Offsite backup + restore | | last ok <time>, <segments> segs, <duration_s> s |
| Alerts / heartbeat / dashboard | | |
| Credentials / quotas / expiries | | |

### Findings (severity-ordered)
- [P0|P1|P2] <ID> <title> — 근거: <command + key output>, 영향: <what breaks / since when>, 상태: <new | continuing since <date> | resolved>, 제안: <owner/next diagnostic>

### Resolved since last report / Known and unchanged
### NOT VERIFIED (what, why)
```

Machine block keys (stable): `date`, `mode`, `overall`, `checks` (`id → status`), `metrics` (`disk_pct`, `mem_avail_gib`, `archive_rows_D`,
`ph_rows_D`, `m1_rows_D`, `backup_duration_s`, `backup_segments`, `failed_units`, `open_issues`, `ledger_failed`, `tape_unresolved`,
`nav_D`, `gdrive_free_gib`). A new finding that was `continuing` for > 3 reports is escalated one severity.

---

## Appendix A — Executable Probes

Extract the fenced blocks into scratch files and pipe them over ssh (read-only; outside blackout; `D` is the audit date):

```bash
mkdir -p scratch && for p in data research; do awk "/^\`\`\`python probe=$p/{f=1;next}/^\`\`\`/{f=0}f" docs/vps-checklist.md > scratch/probe_$p.py; done
for p in data research; do ssh or-vps 'cd ~/k-closing-alpha && TZ=Asia/Seoul nice -n 10 timeout 300 ~/.local/bin/uv run --no-sync python - <D>' < scratch/probe_$p.py; done
```

Output is `CHECK <ID> <PASS|WARN|FAIL> <detail>`. Baselines observed on 2026-10-02: archive 455 rows,
price_history 2,766 rows/day, 1m `ls` 377 k bars/990 symbols, volume ratio median 0.960, 3 top-k rows, nav conservation 0.00.
Probe behavior (track deltas, do not re-open as new): `RS-01` measures panel lag in trading days against a per-panel
allowance table (`credit_balance` observed T+3 trading days → allow 4); `RS-05` WARNs only for `*.pre_*` files older
than 7 days (message: delete after offsite verify); `RS-06` quarantine entries are INFO (PASS with manifest detail) when
each carries a `moves-*.json` manifest, WARN otherwise; `RS-08` reads report fields `unrecoverable`,
`expired_recoverable`, `expiring_needs` and WARNs only for `expiring_needs > 0`, `expired_recoverable > 0`, or a stale
`run_date` (`unresolved` alone is tracked, not warned).

```python probe=data
import sys
from pathlib import Path

import numpy as np
import pandas as pd

D = sys.argv[1]
ROOT = Path("data")
out = []


def emit(cid, status, detail):
    print(f"CHECK {cid} {status} {detail}", flush=True)


def level(cond_fail, cond_warn=False):
    return "FAIL" if cond_fail else ("WARN" if cond_warn else "PASS")


# ---- archive.parquet (15:20 snapshot) ----
a = pd.read_parquet(ROOT / "history/archive.parquet")
a["day"] = a["스냅샷_날짜"].astype(str).str[:10]
cnt = a.groupby("day").size()
prev = cnt[cnt.index < D].tail(5)
d = a[a["day"] == D].copy()
if d.empty:
    emit("DI-ARCH-01", "FAIL", f"no archive rows for {D}")
else:
    med = float(prev.median()) if len(prev) else float("nan")
    ratio = len(d) / med if med == med and med > 0 else float("nan")
    emit("DI-ARCH-01", level(False, not (0.5 <= ratio <= 2.0)), f"rows={len(d)} prev5_median={med:.0f} ratio={ratio:.2f}")
    dup = int(d.duplicated(["종목코드"]).sum())
    emit("DI-ARCH-02", level(dup > 0), f"duplicate_symbols={dup}")
    nulls = {c: int(d[c].isna().sum()) for c in ("종가", "결정_종가", "전일종가", "등락률", "거래대금")}
    emit("DI-ARCH-03", level(any(nulls.values())), f"nulls={nulls}")
    confirmed = float(d["종가_확정"].fillna(0).astype(float).mean())
    emit("DI-ARCH-04", level(confirmed < 0.9, confirmed < 0.99), f"close_confirmed_share={confirmed:.3f}")
    flags = {c: int((d[c].fillna(0).astype(float) > 0).sum()) for c in ("가격_비정상", "수급_실패", "지수_실패", "현재가_실패")}
    flow_share = flags["수급_실패"] / len(d)
    emit("DI-ARCH-05", level(flags["가격_비정상"] > 0 or flags["지수_실패"] > 0 or flags["현재가_실패"] > 0, flow_share > 0.1), f"flags={flags} flow_fail_share={flow_share:.3f}")
    o, h, l, c = (d[k].astype(float) for k in ("시가", "고가", "저가", "종가"))
    bad_ohlc = int(((h < np.maximum(o, c) - 1e-9) | (l > np.minimum(o, c) + 1e-9) | (h < l)).sum())
    emit("DI-ARCH-06", level(bad_ohlc > 0), f"ohlc_inconsistent={bad_ohlc}")
    chg = (d["종가"].astype(float) / d["전일종가"].astype(float) - 1.0) * 100.0
    bad_chg = int(((chg - d["등락률"].astype(float)).abs() > 0.01).sum())
    emit("DI-ARCH-07", level(bad_chg > max(1, len(d) * 0.01)), f"chg_ratio(final close)_mismatch={bad_chg}")
    gap = ((d["종가"].astype(float) - d["결정_종가"].astype(float)).abs() / d["결정_종가"].astype(float))
    emit("DI-ARCH-08", level(False, float(gap.quantile(0.99)) > 0.03), f"decision_vs_final_close_gap p50={gap.median():.4f} p99={gap.quantile(0.99):.4f}")
    snap = pd.to_datetime(d["snapshot_timestamp"], utc=True).dt.tz_convert("Asia/Seoul")
    feat = pd.to_datetime(d["feature_available_timestamp"], utc=True).dt.tz_convert("Asia/Seoul")
    execu = pd.to_datetime(d["execution_timestamp"], utc=True).dt.tz_convert("Asia/Seoul")
    s_ok = (snap.dt.strftime("%H%M%S") >= "152000").mean(), (snap.dt.strftime("%H%M%S") <= "152059").mean()
    causal = int((feat < snap).sum() + (execu.dt.strftime("%H%M%S") < "153000").sum())
    emit("DI-ARCH-09", level(causal > 0, min(s_ok) < 0.95), f"causality_violations={causal} snapshot_in_15:20 share={min(s_ok):.3f}")

# ---- price_history ----
ph = pd.read_parquet(ROOT / "history/price_history.parquet", columns=["date", "symbol", "high", "low", "close", "close_raw", "prev_close", "volume", "daily_change_pct", "kospi_pct"])
ph["day"] = ph["date"].astype(str).str[:10]
mx = ph["day"].max()
arch_days = sorted(a["day"].unique())
prev_arch = max([x for x in arch_days if x < D], default=None)
PHD = D if mx >= D else mx  # price_history for D arrives with the next 08:30 ingest; compare D-1 on the evening of D
lag_ok = mx >= D or (prev_arch is not None and mx >= prev_arch)
emit("DI-PH-01", level(not lag_ok), f"max_date={mx} audit_date={D} compared_day={PHD} (D itself lands at the next 08:30 ingest)")
pcnt = ph.groupby("day").size()
last, before = pcnt.get(PHD, 0), pcnt[pcnt.index < PHD].tail(5)
r = last / float(before.median()) if len(before) else float("nan")
emit("DI-PH-02", level(last == 0, not (0.97 <= r <= 1.03)), f"rows_{PHD}={last} prev5_median={before.median():.0f} ratio={r:.3f}")
dd = ph[ph["day"] == PHD]
emit("DI-PH-03", level(int(dd.duplicated(["symbol"]).sum()) > 0), f"dup_symbols_D={int(dd.duplicated(['symbol']).sum())}")
bad = int(((dd["high"] < dd["low"]) | (dd["close_raw"].astype(float) <= 0) | (dd["volume"] < 0)).sum())
emit("DI-PH-04", level(bad > 0), f"ohlcv_invalid_D={bad}")
emit("DI-PH-05", level(int(dd["kospi_pct"].nunique()) > 1), f"kospi_pct_distinct_D={int(dd['kospi_pct'].nunique())}")
total_dup = int(ph.duplicated(["date", "symbol"]).sum())
emit("DI-PH-06", level(total_dup > 0), f"duplicate_date_symbol_total={total_dup}")

# ---- 1m regular partition ----
pth = ROOT / f"history/intraday/1m/regular/{D[:7]}/{D}.parquet"
if not pth.exists():
    emit("DI-1M-01", "FAIL", f"partition missing {pth}")
else:
    m = pd.read_parquet(pth)
    vend = m["vendor"].astype(str).value_counts().to_dict()
    emit("DI-1M-01", "PASS", f"rows={len(m)} symbols={m['symbol'].nunique()} vendors={vend}")
    dupm = int(m.duplicated(["symbol", "ts_hms"]).sum())
    badm = int(((m["high"] < m["low"]) | (m["volume"] < 0) | (m["close"] <= 0)).sum())
    emit("DI-1M-02", level(dupm > 0 or badm > 0), f"dup_bars={dupm} invalid_bars={badm}")
    ts = pd.to_numeric(m["ts_hms"], errors="coerce")
    emit("DI-1M-03", level(ts.min() > 90100 or ts.max() < 152900), f"ts_range={int(ts.min())}..{int(ts.max())}")
    def vol_ratio(day):
        mm = m if day == D else pd.read_parquet(ROOT / f"history/intraday/1m/regular/{day[:7]}/{day}.parquet", columns=["symbol", "volume"])
        pp = ph[ph["day"] == day]
        v = mm.groupby(mm["symbol"].astype(str))["volume"].sum()
        p = pp.set_index(pp["symbol"].astype(str))["volume"]
        rr = (v / p[p > 0]).dropna()
        return float(rr.median()), float((rr < 0.9).mean()), len(rr)
    hist_days = sorted(pcnt[pcnt.index < PHD].tail(4).index)
    base = [vol_ratio(x) for x in hist_days if (ROOT / f"history/intraday/1m/regular/{x[:7]}/{x}.parquet").exists()]
    cur = vol_ratio(PHD)
    bmed = float(np.median([b[0] for b in base])) if base else float("nan")
    bsh = float(np.median([b[1] for b in base])) if base else float("nan")
    drift = abs(cur[0] - bmed) > 0.02 or abs(cur[1] - bsh) > 0.05
    emit("DI-1M-04", level(cur[0] > 1.01, drift), f"vol_ratio_vs_eod median={cur[0]:.4f} share_below_0.9={cur[1]:.3f} n={cur[2]} baseline_median={bmed:.4f} baseline_share={bsh:.3f}")
    mp = m if PHD == D else pd.read_parquet(ROOT / f"history/intraday/1m/regular/{PHD[:7]}/{PHD}.parquet")
    ordered = mp.sort_values(["symbol", "ts_hms"])
    lastbar = ordered.groupby(ordered["symbol"].astype(str)).tail(1).set_index(ordered.groupby(ordered["symbol"].astype(str)).tail(1)["symbol"].astype(str))["close"]
    cr = dd.set_index(dd["symbol"].astype(str))["close_raw"].astype(float)
    k = pd.concat([lastbar.rename("b"), cr.rename("c")], axis=1).dropna()
    match = float(((k["b"] - k["c"]).abs() / k["c"] < 0.001).mean())
    emit("DI-1M-05", level(match < 0.9, match < 0.98), f"last_bar_close_matches_eod share={match:.3f} n={len(k)}")
    # ---- regular tick vs bar vendor disagreement (DI-1M-06) ----
    import json as _json
    evp = ROOT / f"logs/events/{D[:7]}/{D}.jsonl"
    src_n, src_max, src_syms = 0, 0.0, []
    evidence_error = False
    if evp.exists():
        for _line in evp.read_text().splitlines():
            try:
                _rec = _json.loads(_line)
            except ValueError:
                evidence_error = True
                continue
            if isinstance(_rec, dict) and _rec.get("job") == "tick_source_diff" and _rec.get("run_date") == D:
                try:
                    _m = _rec["metrics"]
                    src_n = int(_m["n"])
                    src_max = float(_m["max_relative_shortfall"])
                    src_syms = _m["symbols"][:10]
                    if src_n < 0 or not np.isfinite(src_max) or not 0 <= src_max <= 1:
                        raise ValueError("invalid source-diff metrics")
                    if not isinstance(src_syms, list) or not all(isinstance(s, str) for s in src_syms):
                        raise ValueError("invalid source-diff symbols")
                except (KeyError, TypeError, ValueError):
                    evidence_error = True
    hbp = ROOT / "logs/heartbeat/daily_audit.json"
    gap_warning = False
    heartbeat_verified = False
    if hbp.exists():
        try:
            _hb = _json.loads(hbp.read_text())
            if isinstance(_hb, dict) and _hb.get("snapshot_date") == D:
                _open = _hb["open_issues"]
                if not isinstance(_open, list):
                    raise ValueError("invalid heartbeat issues")
                _keys = [_e["key"] for _e in _open]
                if not all(isinstance(_k, str) for _k in _keys):
                    raise ValueError("invalid heartbeat issue key")
                gap_warning = any(
                    _k.startswith("intraday:regular_ticks:") and _k.endswith((":source_diff_systemic", ":certified_gap"))
                    for _k in _keys
                )
                heartbeat_verified = True
        except (KeyError, TypeError, ValueError):
            evidence_error = True
    probe_warning = gap_warning or evidence_error or not heartbeat_verified
    symbols_text = ','.join(src_syms) if isinstance(src_syms, list) and all(isinstance(s, str) for s in src_syms) else 'unreadable'
    emit("DI-1M-06", level(False, probe_warning), f"source_diff n={src_n} max={src_max * 100:.1f}% symbols={symbols_text or 'none'} heartbeat_verified={heartbeat_verified} gap_warning={gap_warning} evidence_error={evidence_error}")

# ---- top-k decision + causality ----
t = pd.read_parquet(ROOT / "parquet/topk_decisions.parquet")
t["day"] = t["decision_date"].astype(str).str[:10]
td = t[t["day"] == D]
if td.empty:
    emit("DI-TOPK-01", "WARN", f"no decision rows for {D} (must be a NO_DECISION row in paper/decisions.parquet)")
else:
    dec = pd.to_datetime(td["decided_at"], utc=True).dt.tz_convert("Asia/Seoul")
    inp = pd.to_datetime(td["input_available_at"], utc=True).dt.tz_convert("Asia/Seoul")
    inf = pd.to_datetime(td["inference_started_at"], utc=True).dt.tz_convert("Asia/Seoul")
    viol = int((inp > inf).sum() + (inf > dec).sum() + (dec.dt.strftime("%H%M%S") >= "153000").sum())
    emit("DI-TOPK-01", level(viol > 0 or len(td) != td["symbol"].nunique()), f"rows={len(td)} causality_violations={viol} models={td['model_version'].nunique()}")
    emit("DI-TOPK-02", level(not np.isfinite(td["pred"].astype(float)).all()), f"pred_finite={bool(np.isfinite(td['pred'].astype(float)).all())} model={td['model_version'].iloc[0]}")

# ---- paper ledger conservation ----
P = ROOT / "paper"
nav = pd.read_parquet(P / "nav.parquet")
nav["day"] = nav["as_of_date"].astype(str).str[:10]
row = nav[nav["day"] == D]
if row.empty:
    emit("DI-PAPER-01", "FAIL", f"no nav row for {D}")
else:
    r0 = row.iloc[-1]
    gap = abs(float(r0["nav"]) - float(r0["cash"]) - float(r0["open_market_value"]))
    emit("DI-PAPER-01", level(gap > 1.0), f"nav_minus_cash_minus_mv={gap:.2f}")
    dup_nav = int(nav.duplicated(["day"]).sum())
    emit("DI-PAPER-02", level(dup_nav > 0), f"duplicate_nav_days={dup_nav} rows={len(nav)}")
    cc = nav.sort_values("day")["cumulative_cost"].astype(float).diff().dropna()
    emit("DI-PAPER-03", level(bool((cc < -1e-9).any())), f"cumulative_cost_monotone={not bool((cc < -1e-9).any())}")
tr = pd.read_parquet(P / "trades.parquet")
fric = int(((tr["buy_fee"] <= 0) | (tr["sell_fee"] <= 0) | (tr["sell_tax"] <= 0)).sum())
emit("DI-PAPER-04", level(fric > 0), f"closed_trades={len(tr)} zero_friction_trades={fric}")
fl = pd.read_parquet(P / "fills.parquet")
emit("DI-PAPER-05", level(int(fl.duplicated(["order_id"]).sum()) > 0), f"duplicate_order_fills={int(fl.duplicated(['order_id']).sum())}")
net = (tr["gross_pnl"].astype(float) - tr["cost"].astype(float) - tr["net_pnl"].astype(float)).abs().max()
emit("DI-PAPER-06", level(float(net) > 1.0), f"max|gross-cost-net|={float(net):.2f}")

# ---- offsite backup deferred-history trend (DI-BAK-01) ----
import json as _bjson
_hp = ROOT / "history/capture/offsite/deferred_history.json"
_hist = []
_herr = None
if _hp.exists():
    try:
        _raw = _bjson.loads(_hp.read_text())
        _hist = _raw[-5:] if isinstance(_raw, list) else []
        if not isinstance(_raw, list):
            _herr = "malformed"
    except ValueError:
        _herr = "malformed"
_trend = [(str(_e.get("run_started_at", ""))[:10], int(_e.get("deferred_dates", -1))) for _e in _hist if isinstance(_e, dict)]
_stall = len(_trend) >= 3 and _trend[-1][1] > 0 and all(_trend[-k][1] >= _trend[-k - 1][1] for k in (1, 2))
_aged = False
if _hist and isinstance(_hist[-1], dict) and _hist[-1].get("oldest_deferred_date"):
    try:
        from datetime import date as _bdate
        _aged = (_bdate.fromisoformat(D) - _bdate.fromisoformat(str(_hist[-1]["oldest_deferred_date"]))).days > 21
    except ValueError:
        pass
emit("DI-BAK-01", level(_stall or _aged), f"trend={_trend or 'empty(first observation drains)'} stalled={_stall} aged={_aged} err={_herr}")
```

```python probe=research
import hashlib, json, sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

D = sys.argv[1]
ROOT = Path("data")


def emit(cid, status, detail):
    print(f"CHECK {cid} {status} {detail}", flush=True)


def lvl(fail, warn=False):
    return "FAIL" if fail else ("WARN" if warn else "PASS")


def days_between(a, b):
    return (date.fromisoformat(b) - date.fromisoformat(a)).days


# RS-01 altdata panels (lag in trading days vs per-panel allowance)
man = json.loads((ROOT / "history/altdata/_manifest.json").read_text())
ALLOW_TD = {"credit_balance": 4}
def trading_lag(a, b):
    d, end, n = date.fromisoformat(a) + timedelta(days=1), date.fromisoformat(b), 0
    while d <= end:
        if d.weekday() < 5:
            n += 1
        d += timedelta(days=1)
    return n
bad = {k: (v["status"], v["last_date"]) for k, v in man["panels"].items() if v["status"] != "ok"}
lag = {k: trading_lag(v["last_date"], D) for k, v in man["panels"].items()}
over = {k: lag[k] for k in lag if lag[k] > ALLOW_TD.get(k, 2)}
emit("RS-01", lvl(bool(bad), bool(over)), f"not_ok={bad} trading_lag_vs_D={lag} over_allowance={over}")

# RS-02 backfill ledgers (latest record per key)
for name in ("extended_sessions", "toss_regular", "nxt_calibration"):
    f = pd.read_parquet(ROOT / f"history/intraday/backfill_ledger/{name}.parquet")
    last = f.drop_duplicates(["snapshot_date", "session", "symbol"], keep="last")
    vc = last["status"].value_counts().to_dict()
    failed = last[last["status"] == "FAILED"]
    stuck = int((pd.to_numeric(failed["attempts"], errors="coerce") >= 3).sum())
    emit(f"RS-02-{name}", lvl(stuck > 0, vc.get("FAILED", 0) > 0.05 * len(last)), f"keys={len(last)} status={vc} failed_with_3plus_attempts={stuck}")

# RS-03 decomposition config binding
cfgp, repp, tabp = (ROOT / "history" / n for n in ("nxt_decomposition_config.json", "nxt_decomposition_fit_report.json", "nxt_calibration_table.parquet"))
if cfgp.exists() and repp.exists() and tabp.exists():
    rep = json.loads(repp.read_text())
    sha = hashlib.sha256(tabp.read_bytes()).hexdigest()
    tab = pd.read_parquet(tabp, columns=["date"])
    tmax = str(tab["date"].astype(str).max())[:10]
    emit("RS-03", lvl(sha != rep["table_sha256"], days_between(tmax, D) > 30), f"table_sha_matches_fit={sha == rep['table_sha256']} table_max_date={tmax} fit_holdout_p90={rep['holdout_rel_err_p90']:.3f}")
else:
    emit("RS-03", "WARN", "decomposition artifacts absent")

# RS-04 retrain registry vs live decision model
reg = [json.loads(x) for x in Path("artifacts/models/topk_ranker/retrain_registry.jsonl").read_text().splitlines() if x.strip()]
lastr = reg[-1]
age = abs(days_between(str(lastr["attempted_at"])[:10], D))
t = pd.read_parquet(ROOT / "parquet/topk_decisions.parquet")
mv = str(t.sort_values("decided_at")["model_version"].iloc[-1])
known = any(str(r.get("trained_at", "")) and str(r["trained_at"]) in mv for r in reg)
emit("RS-04", lvl(lastr["outcome"] not in ("PROMOTED", "PROMOTED_UNGATED", "MANUAL_HOTFIX") or not known, age > 8 or lastr["outcome"] == "MANUAL_HOTFIX"), f"last_outcome={lastr['outcome']} age_days={age} agreement={lastr.get('agreement')} live_model_in_registry={known}")

# RS-05 classification + stray pre-restore files (WARN only past 7 days old)
import time as _time
cls = ROOT / "history/altdata/security_classification.parquet"
c = pd.read_parquet(cls)
aged = sorted(p.name for p in (ROOT / "history/altdata").glob("*.pre_*") if _time.time() - p.stat().st_mtime > 7 * 86400)
emit("RS-05", lvl(False, bool(aged)), f"classification_rows={len(c)} pre_restore_older_than_7d={aged} action=delete after offsite verify")

# RS-06 quarantine (INFO when each entry carries a moves-*.json manifest)
qd = ROOT / "quarantine"
entries = sorted(qd.iterdir()) if qd.exists() else []
undoc = []
for _p in entries:
    _ok = any(_p.glob("moves-*.json")) if _p.is_dir() else any(_p.parent.glob(f"{_p.stem}.moves-*.json"))
    if not _ok:
        undoc.append(_p.name)
emit("RS-06", lvl(False, bool(undoc)), f"quarantine_entries={[p.name for p in entries]} documented_info={len(entries) - len(undoc)} undocumented={undoc}")

# RS-07 session partitions for D
miss = []
sizes = {}
for sess in ("regular", "nxt_premarket", "nxt_aftermarket", "krx_aftermarket"):
    p = ROOT / f"history/intraday/1m/{sess}/{D[:7]}/{D}.parquet"
    if not p.exists():
        miss.append(sess)
    else:
        sizes[sess] = p.stat().st_size
for sess in ("regular",):
    p = ROOT / f"history/intraday/ticks/{sess}/{D[:7]}/{D}.parquet"
    if not p.exists():
        miss.append(f"ticks/{sess}")
emit("RS-07", lvl(bool(miss)), f"missing={miss} bytes={sizes}")

# RS-08 tape sweep report (WARN only on expiring/expired-recoverable needs or stale run)
rp = ROOT / "history/capture/staging/tape_sweep/last_report.json"
if not rp.exists():
    emit("RS-08", "FAIL", "tape sweep report absent (sweep never ran; the daily audit does not warn about this)")
else:
    r = json.loads(rp.read_text())

    def n(x):
        return len(x) if isinstance(x, (list, tuple, dict)) else int(x or 0)

    stale = days_between(r["run_date"], D) > 4
    emit("RS-08", lvl(False, n(r.get("expiring_needs", 0)) > 0 or n(r.get("expired_recoverable", ())) > 0 or stale), f"run_date={r['run_date']} stale={stale} unrecoverable={n(r.get('unrecoverable', ()))} expired_recoverable={n(r.get('expired_recoverable', ()))} expiring_needs={n(r.get('expiring_needs', 0))} unresolved_tracked={n(r.get('unresolved', ()))}")

# RS-09 capture manifests for D
mp = ROOT / f"history/capture/manifests/{D}"
emit("RS-09", lvl(not mp.exists()), f"manifest_dir_entries={len(list(mp.iterdir())) if mp.exists() else 0}")
```
