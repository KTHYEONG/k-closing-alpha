# VPS Automation Health Checklist (or-vps)

> **Audience:** an AI agent asked to audit the production automation on `or-vps`.
> **Goal:** prove — with evidence, not assumptions — that collection, storage, paper trading, alerting, and
> offsite (gdrive) backup all ran correctly, and surface every silent gap. A green `systemctl` alone is not proof.

## 0. Rules of Engagement (read first)

- **Read-only.** Never `start`/`stop`/`restart`/`enable`/`disable` units, never `docker run/rm/pull`, never edit
  files under `~/k-closing-alpha`, `~/quant-secrets`, or `~/.cache/kis`, never run `rclone` write verbs
  (`copy`/`sync`/`move`/`delete`/`purge`). Only `rclone lsf/lsjson/size/about/check --one-way` style reads.
- **Never run job entry points by hand** (`src.daily.*`, `src.tools.daily_audit` main, `src.tools.offsite_backup`,
  `src.tools.alerts`, …): they write ledgers, consume broker quota, or send emails. Importing pure read helpers in a
  Python snippet (section 4/6) is allowed.
- **Trading blackout:** avoid any heavy command (large `find`, `du` over `data/`, parquet scans) during
  **08:50–09:40 KST** and **15:10–15:50 KST** on weekdays (`src/tools/deploy_window.py`). The host shares CPU and the
  KIS quota with the live decision chain.
- **Secrets:** never `cat`/print `~/quant-secrets/*.env`, `env`, token caches, or `rclone config show`. Report only
  key *names* and file modes.
- **Timezones:** the VPS system clock and `journalctl` are **UTC**; every timer `OnCalendar` is **Asia/Seoul**.
  KST = UTC+9. Always convert before judging "late"/"missing". Pass `TZ=Asia/Seoul` to Python snippets.
- **Evidence or it did not happen:** every PASS line cites the command output it rests on. Use `WARN` when the
  evidence is ambiguous, never silent PASS.
- If something is broken, **diagnose, do not fix**. Scratch analysis goes under local `scratch/`; report to the
  operator in Korean (CLAUDE.md §4).

## 1. Access & Baseline

```bash
ssh -o ConnectTimeout=15 or-vps 'hostname; date -u; TZ=Asia/Seoul date; uptime; whoami'
```

- [ ] SSH works over Tailscale (`or-vps` → 100.x address). Failure here is itself a P0 finding.
- [ ] Note current KST time and whether today is a KRX trading day (holiday/weekend changes expectations below;
      `src/config` calendar via `KRX_CALENDAR`).
- [ ] Uptime: a reboot since the last check means `Persistent=false` timers may have been skipped — note it.

Choose the **audit date `D`** = most recent completed KRX trading day whose evening chain (≥ 23:00 KST) has finished.

## 2. Host Resources

```bash
ssh or-vps 'df -h / ~; df -i /; free -h; docker system df; \
  du -sh ~/k-closing-alpha/data/history/capture/{raw,normalized,backups,staging} 2>/dev/null; \
  docker ps -a --format "{{.Names}} {{.Status}}" | grep kca- ; journalctl --user --disk-usage'
```

- [ ] Root disk usage < 80 %, inodes < 80 %. Record free GB and compare with previous report (trend matters:
      capture grows daily; tape sweep has its own disk guard, see §8).
- [ ] Memory: `available` comfortably > 2 GiB (retrain runs with `--memory=6g`).
- [ ] **No lingering `kca-*` containers** outside a currently running job window. A stuck container means a
      hung unit (`ExecStopPost` should `docker rm -f`).
- [ ] Docker reclaimable images not unbounded (old `sha-*` tags accumulate; just report size).

## 3. Deploy & Code Parity

```bash
git -C ~/k-closing-alpha fetch -q origin 2>/dev/null; \
ssh or-vps 'cd ~/k-closing-alpha && git log -1 --format="%h %ci %s" && git status --porcelain | head && \
  docker images ghcr.io/kthyeong/k-closing-alpha --format "{{.Tag}} {{.ID}} {{.CreatedAt}}" | head -4 && \
  docker image inspect ghcr.io/kthyeong/k-closing-alpha:latest --format "{{.Id}}"'
git -C /home/kth/k-closing-alpha log -1 --format="%h %ci %s" origin/main
gh run list --workflow deploy.yml --limit 5
```

- [ ] VPS checkout HEAD == `origin/main` HEAD (or the newest commit whose deploy run succeeded). A gap means a
      deploy is failing or deferred by the blackout — check `gh run view` for the last deploy run.
- [ ] `git status --porcelain` on the VPS is **empty** (no hand edits on the host).
- [ ] `latest` image ID == `sha-<HEAD>` image ID (code and image move together).
- [ ] Last 5 `deploy.yml` runs: all success, or failures explained.

## 4. Scheduler Parity (the most common silent gap)

```bash
ssh or-vps 'systemctl --user list-timers "kca-*" --all --no-pager; \
  for t in ~/.config/systemd/user/kca-*.timer; do n=$(basename $t); \
    printf "%-40s %s\n" "$n" "$(systemctl --user is-enabled $n)"; done; \
  loginctl show-user $USER -p Linger; systemctl --user list-units --failed "kca-*" --no-legend'
ssh or-vps 'cd ~/k-closing-alpha && for f in deploy/systemd/kca-*; do \
  cmp -s "$f" ~/.config/systemd/user/$(basename $f) || echo "DRIFT $(basename $f)"; done'
```

- [ ] **Every repo timer in `deploy/systemd/*.timer` is `enabled`**, except the optional manual set
      `OPTIONAL_MANUAL_TIMERS` in `src/tools/code_sync.py` (`kca-auction-open`, `kca-auction-close`,
      `kca-altdata-capture`) whose state is an operator choice — report their state either way.
      Rationale: `code_sync` only auto-enables timers *newly installed* in that deploy; a timer installed while
      disabled stays disabled forever, and its job never runs. A `disabled` non-optional timer is a P1 finding.
- [ ] `list-timers` LAST column: every enabled timer fired at its most recent scheduled slot (convert UTC→KST).
      `-` (never ran) on an enabled timer older than its first slot is a finding.
- [ ] No `DRIFT` lines (installed unit == repo unit).
- [ ] `Linger=yes` (user timers survive logout/reboot).
- [ ] `--failed` list: for each failed unit, get the cause:
      `journalctl --user -u <unit> --since "<D> 00:00" --no-pager | tail -40`. Classify: transient broker/API,
      data precondition, code bug. Remember failed state persists until the next successful run.

### Expected weekday schedule (KST) and dependency chain

| KST | Unit | Role | Hard gate for |
|---|---|---|---|
| 07:05 | kca-kis-token-warmup | KIS token issuance for host data slots (retries 4×/2h) | all KIS calls |
| 07:10 | kca-kiwoom-token-rotate | moves Kiwoom 24h token expiry out of 15:20 window | Kiwoom tapes |
| 08:30 / 11:30 / 21:30 | kca-price-ingest | `price_history.parquet` (KRX bulk + KIS flows/index) | features |
| 08:39:55 | kca-auction-open (optional) | open-auction capture, research | — |
| 09:01 | kca-paper-exit | D+1 open-auction paper exit of open lots | paper ledger |
| 15:20 | kca-collect | read-only snapshot → `archive.parquet` | predict |
| 15:21 | kca-predict | top-k decision → `data/parquet/topk_decisions.parquet` | paper entry |
| 15:21:05 | kca-auction-close (optional) | close-auction capture, research | — |
| 15:30:30 | kca-finalize-close | confirmed close (`close_confirmed`) ; triggers paper-entry | paper entry |
| 15:34 | kca-paper-entry | paper entry at confirmed close | paper ledger |
| 15:40 | kca-archive-intraday-regular / kca-aftermarket-book | regular bars+ticks / aftermarket book | audit, backup |
| 20:05 | kca-archive-intraday | NXT/KRX aftermarket bars | audit, backup |
| 20:35 | kca-tape-sweep | recover tick gaps from Kiwoom tapes (default no-new-walk deadline 21:15) | intraday completeness |
| 21:20 | kca-daily-audit | completeness audit (After= tape-sweep, so it sees the sweep result) + **one digest email per weekday** + heartbeat | watchdog |
| 21:00 | kca-backup-prune | purge `_deleted` snapshots > 30 d on gdrive | — |
| 21:35 | kca-altdata-capture (optional) | slow altdata panels | — |
| 22:15 | kca-backup | sealed capture segments + loose `data/`,`artifacts/` → gdrive | offsite |
| 23:05 daily | kca-extended-backfill | extended-session 1m backfill (bulk class, up to 8h) | — |
| Sat 22:00 | kca-retrain | weekly ranker bundle retrain + re-certification | predict bundle |
| Sun 10:00 | kca-core-snapshot | immutable weekly/monthly core-panel snapshots | DR |
| Sun 11:00 | kca-offsite-verify | remote re-verification + restore drill | DR |
| 07:40 Tue–Sat (GHA) | watchdog.yml | external dead-man probe of audit heartbeat + GHCR PAT expiry | — |

Drive writers (`backup`, `backup-prune`, `core-snapshot`, `offsite-verify`) are serialized by
`flock %t/quant-gdrive.lock` shared with other projects on the host; a long wait there is contention, not a hang,
unless it exceeds the 7200 s `-w` budget.

## 5. Per-Unit Run Evidence for Date D

For each unit in the table that was due on `D`:

```bash
ssh or-vps 'journalctl --user -u kca-<name>.service --since "<D-1> 15:00" --until "<D+1> 15:00" --no-pager \
  -o short-iso | grep -E "Started|Finished|Failed|status=|\[(SYS|DATA|ALGO|PORTFOLIO|RISK|EXEC)\]" | tail -25'
ssh or-vps 'systemctl --user show kca-<name>.service -p ExecMainStartTimestamp -p ExecMainExitTimestamp \
  -p ExecMainStatus -p Result'
```

- [ ] Each due unit has exactly one `Finished` (or a documented retry) on `D`; `Result=success`.
- [ ] Runtime well under its `TimeoutStartSec` (a run at > 70 % of timeout is a WARN — approaching a hang).
      Critical-path budgets: collect 9 min, predict 7 min, finalize 12 min, paper-entry 10 min, paper-exit 15 min.
- [ ] `predict` finished **before** 15:30 KST and `finalize-close` before `paper-entry` (causality of the chain).
- [ ] Grep each journal for `DEGRADED`, `NO_DECISION`, `QUOTA`, `EGW00`, `rate`, `Traceback`, `timeout`,
      `DISK_GUARD`, `LEDGER_INVALID`. Exit 0 with `DEGRADED` is still a finding.
- [ ] Run-outcome event log (`data/logs/events/<YYYY-MM>/<D>.jsonl`, written by `src/tools/run_outcome.py`):

```bash
ssh or-vps 'cd ~/k-closing-alpha && cat data/logs/events/$(echo <D> | cut -c1-7)/<D>.jsonl'
```

  Every job that records outcomes (`price_ingest`, `predict`, `finalize_close`, …) shows `OK`; any `DEGRADED` /
  `NO_DECISION` must have a matching alert email (§9).

## 6. Pipeline Completeness & Data Integrity (D)

Run the audit's own pure read helpers (no email, no writes):

```bash
ssh or-vps 'cd ~/k-closing-alpha && TZ=Asia/Seoul timeout 180 ~/.local/bin/uv run --no-sync python - <<"EOF"
from src.tools.daily_audit import audit_daily_completeness, list_failed_kca_units
from src.tools.run_outcome import load_run_outcomes
D = "<D>"
print("completeness", audit_daily_completeness(D))
print("failed_units", list_failed_kca_units())
print("outcomes", load_run_outcomes(D))
EOF'
```

- [ ] All completeness keys `True`: `archive`, `close_confirmed`, `decision`, `paper_entry`, `paper_exit`,
      `minute_bars`, `price_history_fresh`.
- [ ] `archive.parquet` rows for `D`: count is plausible vs. previous days (sudden drop > 20 % = WARN), and
      `close_confirmed` true for the decision rows.
- [ ] `price_history.parquet` max date ≥ `D` after the 21:30 ingest; no duplicate `(date, symbol)` rows.
- [ ] Intraday partitions exist and are non-trivial in size:
      `data/history/intraday/1m/{regular,...}/<YYYY-MM>/<D>.parquet` (`src/data/intraday_store.py`). Compare file
      size/rows with the previous 5 trading days.
- [ ] Value-level sanity (presence is not correctness — see the 09-22 collection-integrity incident): for a few
      symbols in `D`'s top-k, the 15:30 1m bar close / `archive` close / `price_history` close agree, and
      `daily_change_pct` is consistent with previous close. Any mismatch → WARN with the rows.
- [ ] `data/quarantine/` — list new entries since last check; each one is a data anomaly the pipeline rejected.
- [ ] No synthetic/test symbols or cohort IDs in production stores (a test fixture once polluted the live store).
- [ ] `data/history/capture/manifests/` for `D`: per-collector status; any `missing_entries`, `quota_exceeded`,
      `empty` is listed with its collector name.
- [ ] Tape sweep report `data/history/capture/staging/tape_sweep/last_report.json`: exists, `run_date` within the
      last trading day, `disk_guard` false, `expiring_needs` = 0 (or reported). **Absence of this file means the
      sweep has never run** — the daily audit treats a missing report as "no issues", so it will not warn you.

## 7. Paper Trading Ledger (`data/paper/`)

```bash
ssh or-vps 'cd ~/k-closing-alpha && TZ=Asia/Seoul timeout 120 ~/.local/bin/uv run --no-sync python - <<"EOF"
from pathlib import Path
import pandas as pd
from src.config import settings
from src.daily.paper_trade import PaperLedger
P = Path(settings.PAPER_DIR)
L = PaperLedger(root=P)
op = L.load_open_positions()
print("open_lots", len(op), sorted(op["decision_date"].astype(str).unique()) if len(op) else [])
for k in ("orders", "fills", "decisions", "trades", "nav"):
    f = pd.read_parquet(P / f"{k}.parquet"); print(k, f.shape)
print(pd.read_parquet(P / "nav.parquet").tail(5).to_string())
print(pd.read_parquet(P / "trades.parquet").tail(5).to_string())
EOF'
```

- [ ] Open lots belong **only** to decision date `D` (after D's entry) — any older `decision_date` means a missed
      paper exit.
- [ ] For `D`: either entry fills exist for the persisted top-k symbols, or a `decisions.parquet` NO_DECISION row
      with a reason. Never both missing.
- [ ] For `D` (as exit day of `D-1`): every `D-1` lot has a sell fill with a `trigger`, and a matching `trades` row.
- [ ] Orders without fills (`status` other than filled) are explained by `reason` (e.g. missed, ceiling).
- [ ] Fill prices are causal: entry price == `D`'s confirmed close (not an intraday print); exit price == `D+1`
      open-auction price (`stck_oprc`). Spot-check 1–2 symbols against `archive` / 1m bars.
- [ ] Conservation: `nav = cash + open_market_value` within float tolerance; `cash` delta day-over-day equals
      Σ(sell proceeds − buy cost − fees − tax) of that day's fills; `cumulative_cost` is non-decreasing;
      `n_open_positions` equals open-lot count.
- [ ] Every closed trade deducts fees and tax (`buy_fee`, sell fee, tax columns > 0). Zero friction = bug.
- [ ] `nav` has one row per trading day with no gaps since paper start.
- [ ] No duplicate `order_id` / `(entry_order_id, side)` rows (idempotency after restarts / catch-up).

## 8. Offsite Backup (gdrive)

```bash
ssh or-vps 'cd ~/k-closing-alpha && python3 -m json.tool data/history/capture/offsite/last_run.json | head -80; \
  journalctl --user -u kca-backup.service -u kca-backup-prune.service --since "<D> 20:00" --no-pager \
  | grep -E "Started|Finished|Failed|status=|ERROR|budget|lock" | tail -30'
ssh or-vps 'rclone about gdrive: 2>&1 | head; \
  rclone lsf gdrive:quant-lake/live/k-closing-alpha --max-depth 1; \
  rclone lsf gdrive:quant-lake/live/k-closing-alpha/snapshots --max-depth 1 | tail -5; \
  rclone lsf gdrive:quant-lake/live/k-closing-alpha/_deleted/data --max-depth 1 | sort | head -3'
```

- [ ] `last_run.json`: `status == "ok"`, `finished_at` is the night of `D` (22:15 KST slot, UTC in file), every
      `steps.*` ok, finished within the 90-minute seal budget (`BACKUP_SEAL_BUDGET`) for the seal step.
- [ ] `core_panels` list: each core panel present with row counts **not shrinking** vs. previous run
      (`CORE_ROW_SHRINK_TOLERANCE = 0`). A shrink is P0 (data loss propagating offsite).
- [ ] Sealed capture segments for `D` exist remotely for each tier (tar.zst; count/size vs. local manifest).
      Optionally: `uv run --no-sync python -m src.tools.capture_offsite --help` to see read-only verify options;
      do not run `verify` manually during the week (it is the Sunday drill and holds the Drive lock).
- [ ] `_deleted/<subtree>/<YYYY-MM-DD>` oldest dir ≤ 30 days old (prune working) and not growing unbounded.
- [ ] `snapshots/`: latest weekly snapshot is from the last Sunday; ≤ 8 weekly + ≤ 12 monthly kept.
- [ ] Last Sunday's `kca-offsite-verify` result: `journalctl --user -u kca-offsite-verify.service --since "-8d"`
      — restore drill passed, zero hash mismatches.
- [ ] gdrive quota: free space ≥ 2× the weekly growth.
- [ ] Flock contention: if backup started much later than 22:15 KST, check which other project held
      `quant-gdrive.lock` (`journalctl --user` of other projects' backup units) — coordination issue, not a bug.

## 9. Alerting & Dead-Man Paths

```bash
ssh or-vps 'cd ~/k-closing-alpha && cat data/logs/heartbeat/daily_audit.json; echo; \
  journalctl --user -u "kca-alert@*" --since "-3d" --no-pager | grep -E "Started|Finished|Failed|outcome|channel" | tail -30'
gh run list --workflow watchdog.yml --limit 5
```

- [ ] Heartbeat `snapshot_date == D`, `undelivered_alerts == 0`, `finished_at` ≈ 21:20–21:45 KST on `D`.
- [ ] Heartbeat `subject` read and every token in it explained: `누락 <step>` (missing step),
      `실패유닛 <unit>` (failed unit), `수집이상 <collector:...>` (collection anomaly). Each must map to a finding in
      §4–§8; unexplained digest warnings are findings themselves.
- [ ] For every unit that failed in the window, a `kca-alert@<unit>` instance ran and **delivered** (webhook
      and/or email channel outcome not failed). An `OnFailure` alert that itself failed = P1 (blind spot).
- [ ] `watchdog.yml` last 5 scheduled runs (07:40 KST Tue–Sat) all succeeded; a failed probe means either the
      heartbeat is stale or Tailscale/SSH path is broken.
- [ ] GHCR PAT expiry step in the watchdog: > 30 days left.
- [ ] Ask the operator (do not assume) whether the digest email for `D` actually arrived — the email channel is
      the only human-facing signal.

## 10. Credentials, Tokens & Expiries

```bash
ssh or-vps 'stat -c "%a %U %n" ~/quant-secrets/*.env ~/.cache/kis; ls -la ~/.cache/kis | head -20; \
  journalctl --user -u kca-kis-token-warmup.service -u kca-kiwoom-token-rotate.service --since "<D> 00:00" \
  --no-pager | grep -E "Finished|Failed|status=|slot|expire" | tail -20'
```

- [ ] Secret env files mode `600`, `~/.cache/kis` mode `700`, owner `ubuntu` (`code_sync` contract).
- [ ] KIS token warmup succeeded on `D` for every declared host data slot (no stale-token line in the audit
      digest). Kiwoom rotate succeeded (token expiry kept out of the 15:20 decision window).
- [ ] KIS data quota is shared with other host projects (`krx-alpha`, …): look for rate-limit / `EGW00201`
      errors in `collect`/`predict`/`finalize` journals at 15:20–15:31 — any hit on the decision chain is P1.
- [ ] DART quota: `status=020` failures may originate from the other project sharing the key; check before blaming
      this repo.
- [ ] Expiry notices (`src/tools/expiry_notices.py`): credentials and KRX calendar horizon — any item within its
      warning window is reported with days left (calendar must contain next year's holidays before year-end).

## 11. Weekly Items (check on Mon or after a weekend)

- [ ] `kca-retrain` (Sat 22:00 KST): succeeded, new bundle certified; `predict` on the following Monday loaded it
      with no `NO_DECISION` / bundle-parity error (`src/tools/deploy_preflight.py` contract).
- [ ] `kca-core-snapshot` (Sun 10:00) and `kca-offsite-verify` (Sun 11:00): both succeeded (§8).
- [ ] Disk / gdrive growth trend over the week.
- [ ] Docker image and journal disk usage trend.

## 12. Report Format

Report to the operator in Korean, as a card. Keep keys/badges in English:

```
## VPS Audit — <D> (checked <KST timestamp>)
Overall: PASS | WARN | FAIL

| Area | Status | Evidence (1 line) |
|---|---|---|
| Host resources | PASS | disk 23% (147G free), mem avail 9.7Gi |
| Deploy parity | ... | ... |
| Scheduler parity | ... | ... |
| Decision chain (collect→predict→finalize→entry) | ... | ... |
| Paper exit | ... | ... |
| Evening archive / tape sweep | ... | ... |
| Data integrity | ... | ... |
| Paper ledger conservation | ... | ... |
| Offsite backup / prune / snapshots | ... | ... |
| Alerts / heartbeat / watchdog | ... | ... |
| Credentials / expiries | ... | ... |

### Findings (severity-ordered)
- [P0|P1|P2] <title> — 근거: <command + key output>, 영향: <what breaks / since when>, 제안: <next diagnostic or fix owner>
```

Severity guide: **P0** data loss, ledger corruption, decision chain down, backup shrink; **P1** a job silently not
running, alert path broken, quota hits on the decision chain; **P2** optional research capture failures, trends,
hygiene.
