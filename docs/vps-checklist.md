# VPS Automation Audit Checklist (or-vps) — v3

> **Audience:** an AI agent auditing the production automation of `k-closing-alpha` (kca) on host `or-vps`
> (hostname `trading-bot`, user `ubuntu`, 2 vCPU, 11 GiB RAM). **Goal:** prove with evidence that every scheduled
> job ran on time, produced correct data, respected the host broker contract, and is backed up — a green
> `systemctl` is not proof. Last verified against the live host on 2026-10-17 (HEAD `cd00451`). Every command in
> this file was run read-only against production on that date. If the host disagrees with this file, the host is
> truth: report the drift.

## 0. Rules of Engagement

| Rule | Detail |
|---|---|
| Read-only | Never `systemctl start/stop/restart/enable/disable/reset-failed/daemon-reload`, `docker run/rm/pull/exec`, `rclone` write verbs (`copy sync move delete purge dedupe cleanup`), `git` writes, or edits under `~/k-closing-alpha`, `~/quant-secrets`, `~/.cache/kis`, `~/.config/systemd`. Allowed rclone: `about lsd lsf lsjson size check --one-way`. `kca-run --print-argv` is allowed (prints argv, runs nothing). Exceptions only for a named action the operator authorizes in the current conversation. |
| No job entry points | Never run `src.daily.*`, `src.ml.*`, `src.backfill.*`, `src.tools.{audit_reconcile,daily_audit,alerts,offsite_backup,backup_prune,core_snapshot,capture_offsite,kis_token_warmup,kiwoom_token_rotate}` as `main`: they write ledgers, issue tokens, consume broker quota, send mail or take the Drive lock. Importing the pure read helpers of §2 and the Appendix A probes is allowed. |
| No broker traffic | Never send a request to any broker/vendor API (KIS, LS, Toss, DB금융투자, Kiwoom, KRX). |
| Secrets | Never print `~/quant-secrets/*.env`, `env`, token/state file contents in `~/.cache/kis`, `rclone config show`, or `docker inspect … Config.Env`. Names, modes, owners, sizes, mtimes only. |
| Leave nothing behind | Do not create files anywhere (host or local): no report files, no evidence dumps, no scratch scripts. Pipe probes over ssh stdin (heredoc or `awk … docs/vps-checklist.md \| ssh …`). The report is the chat reply only. |
| Deploy blackout | `src/tools/deploy_window.py`: **Mon–Fri 08:30–09:40 and 15:10–15:43 KST** (weekends none; weekday holidays still count). Inside: only `systemctl`, `journalctl`, `stat`, `ls`, small-file reads. No parquet probes, `du`, `find` over `data/`, `rclone`, `uv run`. Heavy probes use `nice -n 19 timeout 300` and avoid 22:15–00:15 (backup, Drive lock). |
| Timezones | Host clock and `journalctl --since/--until` are **UTC**; every timer `OnCalendar` and every parquet date is **KST** (UTC+9). Convert with `k(){ date -u -d "TZ=\"Asia/Seoul\" $1" +"%F %T"; }; k "2026-10-16 15:20"` → `2026-10-16 06:20:00`. Pass `TZ=Asia/Seoul` to Python. |
| Evidence | Every PASS cites the command and the line it rests on. Ambiguous → WARN. Unchecked items go to `NOT VERIFIED` with the reason. Diagnose, do not fix. Report in Korean (keys/badges English). |
| Neighbours | The host also runs `krx-alpha` (container `krx-collector`, `krx-*` timers, `vps-image-gc`), `k-stock-engine` (`kse-*`), `crypto-pilot`, `mt-etf-*`, `quant-dashboard`. Report contention with them (§12); do not audit their logic. |
| rclone path | Not on the non-interactive ssh PATH: use `~/.local/bin/rclone`. Host Python: `~/.local/bin/uv run --no-sync python` inside `~/k-closing-alpha`. |

## 1. Run Modes

| Mode | When (KST) | Scope |
|---|---|---|
| **W** window watch | ~2 min after a §3 slot | that slot's units + run outcome (light commands only inside blackouts) |
| **D** daily | trading day ≥ 23:00, or next morning 07:30–08:20 | §2–§11 for audit date `D` |
| **WK** weekly | Monday 07:30–08:20 | D + Sat retrain, Sun snapshot/verify, trends |
| **E** event | after a deploy, alert, incident or operator question | affected areas + §4 |

`D` = most recent `STANDARD`/`SHIFTED` day whose evening chain finished. On a `CLOSED` weekday, run data probes with
the previous trading day and expect `SKIPPED reason=non_trading_day` outcomes for that weekday.

## 2. Context (always first)

```bash
ssh -o ConnectTimeout=15 or-vps 'hostname; date -u; TZ=Asia/Seoul date; uptime; loginctl show-user $USER -p Linger; free -h | sed -n 2,3p; df -h / | tail -1; df -i / | tail -1'
ssh or-vps 'cd ~/k-closing-alpha && TZ=Asia/Seoul timeout 180 ~/.local/bin/uv run --no-sync python - <D>' <<'EOF'
import sys
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from src.data.session_calendar import resolve_session_day
from src.tools.daily_audit import audit_daily_completeness, list_failed_kca_units
from src.tools.run_outcome import load_run_outcomes
now = datetime.now(ZoneInfo("Asia/Seoul"))
print("days", [(str(now.date() - timedelta(days=k)), resolve_session_day(now.date() - timedelta(days=k)).kind.value) for k in range(8)])
D = sys.argv[1]
print("completeness", audit_daily_completeness(D))
print("failed_units", list_failed_kca_units())
print("outcomes", load_run_outcomes(D))
EOF
```

- Day kinds: `STANDARD`/`SHIFTED` trading, `CLOSED` weekend/holiday, `UNKNOWN` = calendar not extended (finding).
- All completeness keys (`archive close_confirmed decision paper_entry paper_exit minute_bars price_history_fresh`) must be `True`.
- SSH failure = P0 (retry once after 30 s). Uptime shorter than the last check = reboot: `Persistent=true` timers catch up, others are skipped.

## 3. Schedule and Expected Evidence (28 timers, all KST)

Containerized units run `ExecStart=%h/.local/lib/kca/kca-run <job> [--kis-data] [--admission critical|standard|bulk] [--role decision|batch] [--resource …] -- uv run python -m <module>`
(installed from `deploy/host/kca-run`; image `ghcr.io/kthyeong/k-closing-alpha:latest`; `--user 1001:1001`; `--kis-data` adds
`kis-data.env`; `--admission` mounts `~/.cache/kis` and sets `BROKER_ADMISSION_CLASS`; `--role` sets `KIS_DATA_ROLE`). Host units
(`flock … uv run`) are marked *host*. Normal runtimes observed 2026-10-12..16.

| KST | Unit | Flags | Timeout | Normal runtime / evidence | If not |
|---|---|---|---|---|---|
| 07:00–23:00 /10 min, daily | `kca-audit-reconcile` (host) | `flock -E 75 kca-audit.lock` | 9 min | ~1–2 s; heartbeat `reconciled_at` advances; busy lock exits 75 = success | P2 |
| Mon–Fri 07:05 | `kca-kis-token-warmup` | kis-data, critical | 5 min | ~6 s; `stage=kis_token_warmup slot=<DATA_1..5,PRIMARY,TRADING_MOCK> issued=True`; `CLOSED` → `SKIP reason=non_trading_day` | P1 |
| Mon–Fri 07:10 | `kca-kiwoom-token-rotate` | critical | 5 min | ~5 s; `status=ROTATED` or `UNCHANGED` | P1 |
| Mon–Fri 08:30 / 11:30 | `kca-price-ingest` | kis-data, standard | 45 min | 08:30 ~8 min, ingests `D-1` (~2,762 rows); 11:30 ~1 min, `n_new_rows=0` when 08:30 succeeded; 08:30 may log INFO `stale_panel_pending_retry` (healthy if 11:30 is OK); 11:30 DEGRADED `stale_panel` needs an alert | P1 |
| Mon–Fri 08:39:55 | `kca-auction-open` (optional) | kis-data, critical, batch | 40 min | runs to ~09:03; `auction_capture status=COMPLETE phase=open` | P2 |
| Mon–Fri 08:40:10 | `kca-order-exit` | critical | 18 min | ~20 s; outcome `OK`, or `SKIPPED no_open_positions`/`trading_disabled` | P0 |
| Mon–Fri 09:00:40 | `kca-order-reconcile-exit` | critical | 32 min | ~5 s; outcome `OK`/`SKIPPED no_orders` | P1 |
| Mon–Fri 09:01 | `kca-paper-exit` | kis-data, critical, decision | 15 min | ~5 s; every older open lot sold before 09:30 | P0 |
| Mon–Fri 15:20:00 | `kca-collect` | kis-data, critical, decision | 9 min | 25–95 s (scales with `n_raw` 93–1,247); `realtime_coverage … coverage=1.0000 status=COMPLETE` | P0 |
| Mon–Fri 15:21 | `kca-predict` | kis-data, critical | 7 min | starts when collect ends (`After=`); ~12 s; outcome `OK` **before 15:30:00** | P0 |
| Mon–Fri 15:21:05 | `kca-auction-close` (optional) | kis-data, critical, batch | 15 min | starts after collect; ~8 min; research | P2 |
| Mon–Fri 15:21:15 | `kca-order-entry` | critical | 8 min | ~45 s; waits for predict; `[EXEC] stage=order_session env=<mock\|live> phase=entry … outcome=OK` | P0 |
| Mon–Fri 15:30:30 | `kca-finalize-close` | kis-data, critical, decision | 12 min | 10 s–2.5 min; ExecStopPost starts `kca-paper-entry` | P0 |
| Mon–Fri 15:30:40 | `kca-order-reconcile-entry` | critical | 12 min | ~5 s; outcome `OK` (`DEGRADED broker_unreadable` needs an alert) | P1 |
| Mon–Fri 15:34 | `kca-paper-entry` | kis-data, critical, decision | 10 min | runs twice (finalize trigger + timer); `SKIP reason=already_recorded` on the second is healthy | P0 |
| Mon–Fri 15:40 | `kca-archive-intraday-regular` | kis-data, standard | 3 h | 28–57 min; regular 1m + ticks partitions for `D` | P1 |
| Mon–Fri 15:40 | `kca-aftermarket-book` | kis-data, standard, batch | 4 h 40 min | runs to ~19:59 by design (≈92 % of timeout is normal) | P2 |
| Mon–Fri 15:45 | `kca-auction-path` | kis-data, bulk | 3 h 20 min | ~9 min; outcome `auction_path OK` | P2 |
| Mon–Fri 20:05 | `kca-archive-intraday` | kis-data, standard | 3 h | 9–33 min; nxt/krx aftermarket + nxt premarket partitions | P1 |
| Mon–Fri 20:35 | `kca-tape-sweep` | bulk | 60 min | starts after archive-intraday ends; 24–31 min; `last_report.json` `run_date=D` | P1 |
| Mon–Fri 21:00 | `kca-backup-prune` (host) | `flock quant-gdrive.lock` | 3 h | ~2 min; `stage=backup_prune … purged=… sealed_removed=…` | P2 |
| Mon–Fri 21:20 | `kca-daily-audit` (host) | `flock kca-audit.lock` | 30 min | 1.5–4.5 min incl. ~80 s `sudo chown -R` pre-step; subject `🟢 … 일일점검 완료` | P1 |
| Mon–Fri 21:35 | `kca-altdata-capture` (optional) | kis-data, standard, batch | 30 min | 4–6 min | P2 |
| Mon–Fri 22:15 | `kca-backup` (host) | `flock quant-gdrive.lock` | 3 h | 3–10 min incl. ~95 s chown pre-step; `offsite_backup status=ok` | P0 shrink / P1 fail |
| daily 23:05 | `kca-extended-backfill` | kis-data, bulk, batch | 8 h | ~2.5 min (weekends too); `extended_backfill status=DONE`; run_outcome job=extended_backfill outcome=OK | P2 |
| Sat 22:00 | `kca-retrain` | `--memory=6g --cpus=1.8 --memory-swap=8g` | 2 h | registry row `PROMOTED`; a strategy change yields `REJECTED` + unit failure by design (§10) | P1 |
| Sun 10:00 | `kca-core-snapshot` (host) | `flock quant-gdrive.lock` | 2 h | ~1 min | P1 |
| Sun 11:00 | `kca-offsite-verify` (host) | `flock quant-gdrive.lock` | 2 h | ~6 min; `missing=0 mismatched=0`, 3 restore drills `OK` | P1 |
| on failure | `kca-alert@<unit>` (host) | — | 5 min | delivers webhook + mail; also used by `krx-deferred-recreate`, `krx-host-backup` | P1 |

Optional timers (`OPTIONAL_MANUAL_TIMERS` in `src/tools/code_sync.py`): auction-open, auction-close, altdata-capture —
report their state; all three were enabled on 2026-10-17. Weekday timers fire on weekday holidays; jobs must record `SKIPPED`.
GitHub `watchdog.yml` (cron `40 22 * * 1-5` UTC = 07:40 KST Tue–Sat) is the external dead-man; GitHub delays it 2–4.5 h.

## 4. Deploy and Scheduler Parity

```bash
git -C /home/kth/k-closing-alpha fetch -q; git -C /home/kth/k-closing-alpha log -1 --format="%h %ci %s" origin/main
gh run list --workflow deploy.yml --limit 3 -R kthyeong/k-closing-alpha
ssh or-vps 'cd ~/k-closing-alpha && git log -1 --format="%h %ci %s" && git status --porcelain | head; \
  docker images ghcr.io/kthyeong/k-closing-alpha --format "{{.Tag}} {{.ID}} {{.CreatedSince}}"; \
  cmp -s deploy/host/kca-run ~/.local/lib/kca/kca-run && echo WRAPPER_MATCH || echo WRAPPER_DRIFT; \
  for f in deploy/systemd/kca-*; do cmp -s "$f" ~/.config/systemd/user/$(basename $f) || echo "DRIFT $(basename $f)"; done; \
  ls ~/.config/systemd/user/ | grep "^kca-" | while read u; do [ -e deploy/systemd/$u ] || echo "NOT_IN_REPO $u"; done; \
  for t in deploy/systemd/*.timer; do n=$(basename $t); echo "$n $(systemctl --user is-enabled $n 2>&1) $(systemctl --user is-active $n 2>&1)"; done | grep -v " enabled active$"; \
  systemctl --user list-units --failed --no-legend; systemctl --user list-timers "kca-*" --all --no-pager'
ssh or-vps bash -s <<'EOF'
set -o pipefail
for f in ~/.config/systemd/user/kca-*.service; do
  line=$(grep -m1 '^ExecStart=%h/.local/lib/kca/kca-run ' "$f") || continue
  out=$(eval "~/.local/lib/kca/kca-run --print-argv ${line#ExecStart=%h/.local/lib/kca/kca-run }" 2>&1 | tr '\0' ' '); rc=$?
  printf '%-36s rc=%s user=%s kis-data=%s admission=%s role=%s\n' "$(basename "$f")" "$rc" \
    "$(grep -oE -- '--user [0-9:]+' <<<"$out" | cut -d' ' -f2)" "$(grep -c 'kis-data.env' <<<"$out")" \
    "$(grep -oE 'BROKER_ADMISSION_CLASS=[a-z]+' <<<"$out" | cut -d= -f2)" "$(grep -oE 'KIS_DATA_ROLE=[a-z]+' <<<"$out" | cut -d= -f2)"
done
EOF
ssh or-vps 'cd ~/k-closing-alpha && TZ=Asia/Seoul timeout 60 ~/.local/bin/uv run --no-sync python -c "
from datetime import datetime; from zoneinfo import ZoneInfo
from src.tools.deploy_window import DEPLOY_BLACKOUT_WINDOWS, blackout_remaining
print([(a.isoformat(), b.isoformat()) for a, b in DEPLOY_BLACKOUT_WINDOWS], blackout_remaining(datetime.now(ZoneInfo(\"Asia/Seoul\"))))"'
```

| ID | Check | Severity if not |
|---|---|---|
| DEP-01 | VPS HEAD == `origin/main`, or the newer commit's deploy is in progress / waiting on the blackout (`deploy.yml` step `blackout_wait`) | P1 |
| DEP-02 | `git status --porcelain` empty; `latest` image ID == `sha-<HEAD>` image ID | P1 |
| DEP-03 | `WRAPPER_MATCH`; no `DRIFT`; no `NOT_IN_REPO` (the former `kca-kis-token-warmup.service.d` drop-in is gone since the warmup unit carries kis-data itself; 2026-10-20) | P1 |
| DEP-04 | 22 wrapper units print `rc=0 user=1001:1001` with the flags of §3 (a `USAGE_ERROR` line = unit cannot start) | P1 |
| SCH-01 | all 28 timers `enabled active`; `Linger=yes`; no `--failed` units | P1 |
| SCH-02 | each timer's `LAST` equals its most recent slot (UTC→KST). Units started by `After=` ordering or by remediation do not move `LAST` | P1 |
| SCH-03 | blackout windows print `[('08:30:00','09:40:00'), ('15:10:00','15:43:00')]` | P2 |

## 5. Run Evidence

```bash
ssh or-vps 'for s in $(systemctl --user list-unit-files "kca-*.service" --no-legend | awk "{print \$1}" | grep -v "@"); do \
  r=$(systemctl --user show $s -p Result -p ExecMainStatus -p ExecMainStartTimestamp -p ExecMainExitTimestamp --value | tr "\n" "|"); \
  j=$(journalctl --user -u $s --since "-8d" --no-pager -o cat | grep -cE "^Finished "); \
  f=$(journalctl --user -u $s --since "-8d" --no-pager -o cat | grep -cE "Failed with result|Main process exited, code=exited, status=[1-9]"); \
  printf "%-38s fin8d=%-4s fail8d=%-3s %s\n" $s $j $f "$r"; done'
ssh or-vps 'journalctl --user -u kca-<name>.service --since "<UTC from k>" --until "<UTC from k>" --no-pager -o short-iso \
  | grep -vE "pam_unix|COMMAND=" | grep -E "Starting|Finished|Failed|Consumed|stage=|outcome=" | cut -c1-250 | tail -30'
ssh or-vps 'cd ~/k-closing-alpha/data/logs/events/<YYYY-MM> && python3 - <D>' <<'EOF'
import collections, json, sys
D = sys.argv[1]
rows = [json.loads(l) for l in open(f"{D}.jsonl") if l.strip()]
print(D, dict(collections.Counter(r.get("outcome") for r in rows)))
for r in rows:
    print(f"  {r['ts'][11:19]} {r['job']:<24} {r['outcome']:<11} {r.get('reason', '')}")
EOF
```

| ID | Check | Severity if not |
|---|---|---|
| RUN-01 | one `Finished` per due unit per day (paper-entry twice, price-ingest twice); `ExecMainStatus=0`. `show` describes only the latest run; use the journal window for `D` | P1 |
| RUN-02 | runtime inside §3 "normal"; > 70 % of `TimeoutStartSec` is WARN except aftermarket-book (by design) | P2 |
| RUN-03 | run outcomes for `D`: decision jobs `OK`; `DEGRADED`/`NO_DECISION` need a matching alert (§11); a trading-day `predict` `SKIPPED` is a calendar disagreement | P1 |
| RUN-04 | order units: journal `data/orders/<env>/journal.jsonl` holds the day's intents; current env from `[EXEC] stage=order_session env=…` (mock on 2026-10-16). A stuck `UNKNOWN` intent is operator-only (`src.daily.order_resolve`, dry-run first) | P0 |
| RUN-05 | `CLOSED` day: weekday units log `SKIP reason=non_trading_day` and record `SKIPPED`; rows written on a closed day = calendar bug | P1 |
| RUN-06 | `Traceback` in a job that exited 0, or `status=DEGRADED` without an outcome/heartbeat entry, is a silent failure | P2 |

Containers are `--rm`: after a run `docker logs` is empty and the journal is the only record.

## 6. Shared Host Broker Directory (host broker contract v1)

Contract: `src/api/host_broker_contract.py`, byte-identical to krx-alpha `deploy/shared/host_broker_contract.py`
(sha256 `0bfab4cc…`). Shared by kca, krx-collector and kse. KIS slot ownership: `deploy/shared/kis_slot_assignments.toml`
(canonical in krx-alpha). Valid names in `~/.cache/kis` (each may have a `<name>.lock` sibling):

| Kind | Name pattern | Writer |
|---|---|---|
| marker | `.host-admission` | `deploy/install_systemd.sh` (touched every deploy) |
| token (LS, Toss, DB) | `token_<ls\|toss\|dbsec>_<sha12>.json` | shared token store (Kiwoom has no token file; pacing only) |
| DB issue guard | `token_issue_dbsec_<sha12>.state` | 61 s issue spacing |
| pacing | `admission_<vendor>_<sha12>[_<scope>].state` (scope case-preserving, e.g. `_RANKING`, `_ka10079`, `_chart`) | host pacer |
| legacy KIS | `token_<sha12>.json`, `tps_<sha12>.state` | KIS key pool (one per slot + PRIMARY + TRADING_MOCK) |

```bash
ssh or-vps python3 - <<'EOF'
import os, re, time
d = os.path.expanduser("~/.cache/kis")
V, H = "(ls|toss|dbsec|kiwoom)", "[0-9a-f]{12}"
ok = [r"\.host-admission", rf"token_{V}_{H}\.json", rf"token_issue_{V}_{H}\.state",
      rf"admission_{V}_{H}(_[A-Za-z0-9_-]+)?\.state", rf"token_{H}\.json", rf"tps_{H}\.state"]
st = os.stat(d)
print(f"dir mode={oct(st.st_mode & 0o777)} marker={os.path.isfile(os.path.join(d, '.host-admission'))}")
names, now = sorted(os.listdir(d)), time.time()
for n in names:
    base = n[:-5] if n.endswith(".lock") else n
    s = os.stat(os.path.join(d, n))
    if not any(re.fullmatch(r, base) for r in ok):
        print(f"NONCONTRACT {n} size={s.st_size} age_d={(now - s.st_mtime) / 86400:.1f}")
    elif n.endswith(".lock") and base not in names:
        print(f"ORPHAN_LOCK {n} age_d={(now - s.st_mtime) / 86400:.1f}")
    if s.st_uid != os.getuid():
        print(f"FOREIGN_OWNER {n} uid={s.st_uid}")
print(f"total={len(names)}")
EOF
ssh or-vps 'stat -c "%a %U:%G %n" ~/.cache/kis ~/quant-secrets ~/quant-secrets/*.env; ls -la ~/.cache/kis | awk "{print \$1, \$3, \$5, \$6, \$7, \$8, \$9}"'
ssh or-vps 'journalctl --user -u "kca-*" -u "kse-*" --since "-8d" --no-pager -o cat | grep -oE "IGW00201|EGW00201|EGW00133|TokenIssueThrottled|HostBrokerDirNotSharedError|AdmissionDirNotSharedError|SCHEMA_INVALID|TokenStoreLockTimeout|TOKEN_REPLACED|http=429" | sort | uniq -c; \
  timeout 60 docker logs --since 168h krx-collector 2>&1 | grep -oE "IGW00201|EGW00201|TokenIssueThrottled|HostBrokerDirNotSharedError|SCHEMA_INVALID" | sort | uniq -c; \
  journalctl --user -u kca-kis-token-warmup.service -u kca-kiwoom-token-rotate.service --since "-2d" --no-pager -o cat | grep -E "stage=" | cut -c1-200 | tail -15'
```

| ID | Check | Severity if not |
|---|---|---|
| BRK-01 | dir mode `700`, owner `ubuntu`, `marker=True` (missing marker → every container job raises `HostBrokerDirNotSharedError`) | P0 |
| BRK-02 | no `NONCONTRACT`, `ORPHAN_LOCK` or `FOREIGN_OWNER` lines (root-owned files come from the root `krx-collector` container and are re-owned by the reconcile `sudo chown`, every 10 min 07:00–23:00) | P2 |
| BRK-03 | zero `IGW00201` (LS/DB throttle), `EGW00201` (KIS TPS), `EGW00133` (KIS token issue limit), `TokenIssueThrottled`, `HostBrokerDirNotSharedError`/`AdmissionDirNotSharedError`, `SCHEMA_INVALID` (corrupt shared token record). Any hit 15:20–15:31 KST | P1 (else P2) |
| BRK-04 | warmup on the last trading day: one `issued=True` per slot (`DATA_1..5`, `PRIMARY`, `TRADING_MOCK`); legacy `token_<sha12>.json` mtimes = that 07:05 run | P1 |
| BRK-05 | env files `600`, `~/quant-secrets` `700` | P1 |
| BRK-06 | `docker logs --since` covers only the current krx-collector container lifetime; say so if it restarted | info |

## 7. Host Resources and Storage

```bash
ssh or-vps 'df -h /; df -i / | tail -1; free -h | sed -n 2,3p; docker system df; docker ps --format "{{.Names}} {{.Image}} {{.Status}}"; journalctl --user --disk-usage; \
  cd ~/k-closing-alpha && nice -n 19 du -sh data artifacts; for d in data/history data/history/capture; do nice -n 19 du -sh $d/* | sort -h | tail -6; done'
ssh or-vps 'cd ~/k-closing-alpha && nice -n 19 find data artifacts -maxdepth 4 \( -name "*.bak*" -o -name "*.pre_*" -o -name "*.stale*" \) -printf "%TY-%Tm-%Td %s %p\n"; ls ~/ops-scratch ~/k-closing-alpha/scratch 2>&1 | head'
```

| ID | Check | Severity if not |
|---|---|---|
| HOST-01 | disk < 80 %, inodes < 80 % (2026-10-17: 24 %, 143 GiB free; inodes 7 %). kca data 8.9 GiB, of which `capture/raw` ~250–460 MB per trading day, held 30 days locally | P1 |
| HOST-02 | memory available > 2 GiB at rest; `kca-backup` peak 1–3.5 GiB, `kca-daily-audit` 1.4 GiB, retrain capped at 6 GiB | P2 |
| HOST-03 | no `kca-*` container outside its §3 window (hung unit) and no exited `kca-*` leftovers | P1 |
| HOST-04 | kca images: `latest` + newest 5 `sha-*` tags (pruned daily 05:30 KST by krx-alpha `vps-image-gc.timer`, `--keep 5`) | P2 |
| HOST-05 | stray files: known on 2026-10-17 — `security_classification.parquet.pre_backfill_20261004`, `sizing_pipeline_bundle.joblib.{bak,pre_002_20261011}`, `~/ops-scratch/` (old audit scripts). Report new ones; list known ones once | P2 |

## 8. Data Integrity (presence is not correctness)

Run Appendix A on every `D` (outside blackout). Levels: L1 presence (`DI-ARCH-01`, `DI-PH-02`, `DI-1M-01`, `RS-07`, `RS-09`),
L2 structure (`DI-ARCH-02/03/06`, `DI-PH-03/04/06`, `DI-1M-02/03`), L3 cross-source (`DI-ARCH-07/08`, `DI-1M-04/05/06`),
L4 causality (`DI-ARCH-09`, `DI-TOPK-01`: snapshot ≤ feature_available ≤ inference ≤ decided < 15:30 ≤ execution).

- Archive rows follow the market screen (93–1,247 per day observed); `DI-ARCH-01` ratio WARN is informational unless `collect` coverage < 1.
- `DI-ARCH-09` share of snapshots inside 15:20:xx drops when `n_raw` is large (0.694 at 1,247 rows on 10-16); causality violations must stay 0.
- `price_history` for `D` lands at the next 08:30 ingest; the probe compares `D-1` on the evening of `D`.
- Stored 1m volume ≈ 0.96 of EOD volume for vendor `ls`; judge drift vs the printed baseline, not the level.
- `등락률` is computed on the final close (`종가`), not `결정_종가`; their gap p99 ≲ 3 % is normal.
- Tick vs bar gaps with a `tape_complete` proof are vendor differences (`tick_source_diff` outcome, info note); only `certified_gap`/`source_diff_systemic` warn.
- Manual spot check: 2 top-k symbols + 1 random symbol agree across archive close, `price_history.close_raw`, last 1m bar, paper entry fill.

## 9. Paper Ledger and Orders

| ID | Check | Severity if not |
|---|---|---|
| PAP-01 | open lots belong only to decision date `D` after `D`'s entry; an older `decision_date` = missed exit | P0 |
| PAP-02 | `D` has entry fills for the persisted top-k or a `paper/decisions.parquet` NO_DECISION row | P0 |
| PAP-03 | every `D-1` lot has a sell fill with `trigger` and a `trades` row | P0 |
| PAP-04 | `DI-PAPER-01..06` PASS (nav = cash + open MV, no duplicate nav days/fills, cost monotone, friction on every trade, net = gross − cost) | P0 |
| PAP-05 | order journal intents for `D` match the top-k (`n_picks == n_acked + n_rejected + n_not_sent`), no lingering `UNKNOWN` | P0 |

## 10. Models

| ID | Check | Severity if not |
|---|---|---|
| ML-01 | `RS-04`: last registry row `PROMOTED`/`PROMOTED_UNGATED`/`MANUAL_HOTFIX` within 8 days and the live `model_version` is in the registry | P1 |
| ML-02 | Sat retrain journal: a `strategy or screen changed vs live bundle` rejection fails the unit by design; the operator certifies and reruns with `--skip-promotion-gate` (registry `PROMOTED_UNGATED`), then may `reset-failed`. Auto-remediation never reruns retrain (`skipped_forbidden`) | P1 |
| ML-03 | `RS-03` table digest matches the fit report; `RS-02-*` no FAILED with ≥ 3 attempts | P2 |

## 11. Offsite Backup, Alerts and Heartbeat

```bash
ssh or-vps 'cd ~/k-closing-alpha/data/history/capture/offsite && ls in_progress.json 2>&1 | tail -1; python3 -c "
import json
r = json.load(open(\"last_run.json\")); print({k: r[k] for k in (\"status\", \"started_at\", \"finished_at\")})
for k, v in r[\"steps\"].items(): print(\" step\", k, json.dumps(v)[:200])
for p in r[\"core_panels\"]: print(\" core\", p[\"relpath\"], p[\"rows\"], p[\"max_date\"])
print(\"deferred_tail\", json.load(open(\"deferred_history.json\"))[-3:])"'
ssh or-vps 'journalctl --user -u kca-backup.service -u kca-backup-prune.service -u kca-offsite-verify.service --since "-8d" --no-pager -o cat \
  | grep -E "offsite_backup status|stage=backup_prune|offsite_verify|restore_drill" | cut -c1-600 | tail -12'
ssh or-vps 'R=~/.local/bin/rclone; timeout 120 $R about gdrive: | head -4; timeout 60 $R lsd gdrive:quant-lake/live/k-closing-alpha; \
  timeout 60 $R lsf gdrive:quant-lake/live/k-closing-alpha/snapshots --max-depth 1 | tail -4; \
  timeout 60 $R lsf gdrive:quant-lake/live/k-closing-alpha/_deleted/data --max-depth 1 | sort | sed -n "1p;\$p"'
ssh or-vps 'cd ~/k-closing-alpha && nice -n 19 timeout 600 ~/.local/bin/rclone check data gdrive:quant-lake/live/k-closing-alpha/data --one-way --size-only \
  --exclude "/history/capture/**" --combined - 2>/dev/null | awk "{c[\$1]++} \$1==\"+\"||\$1==\"*\"{print} END{for(k in c) print \"count\",k,c[k]}" | tail -20'
ssh or-vps 'cd ~/k-closing-alpha/data/logs && python3 -c "
import json; h = json.load(open(\"heartbeat/daily_audit.json\"))
print({k: h.get(k) for k in (\"schema_version\", \"snapshot_date\", \"severity\", \"audit_kind\", \"day_kind\", \"finished_at\", \"reconciled_at\", \"undelivered_alerts\", \"provisional_reasons\")})
print(\"open\", [i.get(\"key\") for i in h.get(\"open_issues\", [])]); print(\"info\", h.get(\"info_notes\"))"; \
  cat heartbeat/audit_alert_state.json heartbeat/advisory_streaks.json; tail -n 5 remediation/ledger.jsonl | cut -c1-220'
ssh or-vps 'journalctl --user -u "kca-alert@*" --since "-8d" --no-pager -o short-iso | grep -E "Starting|Finished|Failed" | cut -c1-200 | tail -8'
ssh or-vps python3 - <<'EOF'
import json
s = json.load(open("/home/ubuntu/quant-dashboard/public/status.json"))
print(s["generated_at"], s["level"], s.get("market_day"))
for p in s["projects"]:
    print(p["name"], len(p["checks"]), [(c["id"], c["level"], c["detail"][:120]) for c in p["checks"] if c["level"] != "OK"])
EOF
gh run list --workflow watchdog.yml --limit 5 -R kthyeong/k-closing-alpha
```

| ID | Check | Severity if not |
|---|---|---|
| BAK-01 | `last_run.json` `status=ok`, every step ok, `started_at` after the last Mon–Fri 22:15 slot (+0–2 min chown pre-step); `in_progress.json` exists only while a run is active and before its `deadline_at` | P1 |
| BAK-02 | `core_panels` rows never shrink between nights | P0 |
| BAK-03 | `deferred_dates` 0, or draining; stalled 3 runs or oldest deferred > 21 days = WARN | P2 |
| BAK-04 | Sun verify `missing=0 mismatched=0`, restore drills `OK` for raw, normalized, manifests | P1 |
| BAK-05 | `snapshots/` newest = last Sunday; `_deleted/data` oldest ≤ 30 days at the last weekday prune (weekends add up to 2 days) | P2 |
| BAK-06 | `rclone check` `+`/`*` entries only for files modified after `last_run.started_at` (heartbeat, extended-backfill partitions, ledgers) | P1 |
| BAK-07 | local sealed capture older than 30 days is removed by backup-prune; dates kept past that (`sealed_kept>0`) are named in the report | P2 |
| ALR-01 | heartbeat `schema_version 2`, `snapshot_date` = latest weekday audited, `undelivered_alerts 0`, `reconciled_at` within 10 min during 07:00–23:00 | P1 |
| ALR-02 | every `open_issues` key maps to a finding; ledger rows `started` without a later `resolved`/`unresolved` are in flight; `unresolved`/`skipped_budget` need the operator | P1 |
| ALR-03 | every failed unit produced a `kca-alert@<unit>` that `Finished` | P1 |
| ALR-04 | dashboard kca card non-OK ⇔ a current finding | P1 |
| ALR-05 | `watchdog.yml` last 5 scheduled runs `success` | P1 |

## 12. Cross-Project Contention Map

| KST | kca | Neighbour | Shared resource | Watch for |
|---|---|---|---|---|
| 07:05 | token warmup (7 KIS keys) | krx-collector relies on it (krx deploy enables the timer and adds the drop-in) | KIS tokens, issue limit | `EGW00133` |
| 07:58–08:50 | — | krx premarket WS slot 5; `kse-morning` 08:32 (kse repo `config/providers.toml` `pool_slot = 5`, bulk) | KIS data slot 5 (kse not in the slot table) | `EGW00201` |
| 08:30–09:03 | price-ingest, auction-open (slot 3), order-exit, paper-exit | `kse-auction-open` 08:39:30 (kis-data) | KIS data keys | `EGW00201`, late paper-exit |
| 15:20–15:30 | collect/predict/finalize (slots 1, 5 decision) | krx snapshot REST slot 2 until 15:39 | KIS slots per slot table | `EGW00201` on the decision chain = P1 |
| 15:40–20:00 | aftermarket-book, archive-regular (slots 2, 3), auction-path | krx aftermarket WS slots 1–4; krx-collector (same LS app key) | KIS keys (REST vs WS), LS key | `IGW00201` |
| 20:35–21:10 | tape-sweep (Kiwoom) | `kse-collect-dbsec` 20:40 (DB) | different vendors; shared DB token | `TokenIssueThrottled` |
| 21:00–02:30 | backup-prune, backup, core-snapshot, offsite-verify | `kse-backup` 02:30 | `flock %t/quant-gdrive.lock` | late backup start |
| 23:05 | extended-backfill (slots 2, 3) | — | KIS slots | `EGW00201` |
| every 10 min | audit-reconcile `sudo chown -R ~/.cache/kis` | krx-collector runs as root with `~/.cache/kis` mounted rw | file ownership | `FOREIGN_OWNER`, `PermissionError` |
| on deploy | — | krx deploy writes `~/.config/systemd/user/kca-kis-token-warmup.service.d/10-kis-data-env.conf`; `krx-*` units use `OnFailure=kca-alert@` | kca unit namespace | drift invisible to kca `code_sync` |

## 13. Known Standing Conditions and Pitfalls

- **Failed state persists** until the unit's next run (e.g. retrain until next Saturday unless the operator runs `reset-failed`). Report cause, whether the deployed commit fixed it, and when it clears.
- **Extended backfill `pending_manifest_conflict`**: fixed to publish one task-level PENDING checkpoint before fan-out; after this fix is deployed, any `pending_manifest_conflict` WARNING from the 23:05 run (now logged as `stage=extended_backfill`) is a regression (P2).
- **Stale KIS token on closed days**: 2026-10-09..11 reconcile flagged `stale_kis_token:*` on holidays/weekends and reran warmup, which skips on `CLOSED` days (budget exhausted). Not seen on 2026-10-17; a recurrence is a calendar-awareness regression (P2).
- **Backup prune keeps old dates**: `raw/` and `manifests/2026-09-14` survived past 30 days (`sealed_kept=2`, reasons not logged at INFO).
- **Dashboard schedule list is separate**: `/home/ubuntu/quant-dashboard/config.toml` carries its own `[[schedule]]` blocks. A removed timer slot (kca-price-ingest 21:30, dropped in `627cd1c`) left there raises a recurring `예정 실행이 지연` FAIL on the dashboard kca card; fix it in the dashboard config, not in kca.
- **Daily audit `missing_bars` is judged on D only**: the dense-partition rule exempts symbols stored solely as `vendor=dbsec` trade-minute fallback bars (KIS TPS limit fallback). Extended backfill can rewrite a past date with `vendor=kis` trade-only symbols (10-16: 213), which a re-audit of that date would flag; not an issue for the current audit.
- **ops-sentinel `job_late` vs `After=`**: a unit held back by a running `After=` predecessor stays PENDING; a slow `archive-intraday` (32–39 min on 2026-10-16/19) used to raise a false `job_late:kca-tape-sweep`.
- **Holidays**: weekday timers fire; jobs must `SKIP`; the daily audit digest is `HOLIDAY_SKIP`. Run data probes against the previous trading day.
- **`journalctl` times are UTC**; `docker logs` is empty for `--rm` units; the journal is the record.
- **Shared resources** (`TOKEN_REPLACED` for the Toss token used from PC and VPS, Drive lock, KIS slots) are cross-project; attribute precisely.
- **Checklist drift**: when a timer, unit, wrapper flag, data path or contract file name changes, update §3, §6 and Appendix A in the same change.

## 14. Report Format (chat reply only — no files)

```
## kca VPS Audit — D=<D> (<mode>, checked <KST>)
Overall: PASS | WARN | FAIL

| Area | Status | Evidence (1 line) |
|---|---|---|
| Context / day kind | | |
| Deploy / scheduler parity (DEP, SCH) | | |
| Run evidence / outcomes (RUN) | | |
| Shared broker dir / contract (BRK) | | |
| Host resources / storage (HOST) | | |
| Data integrity (Appendix A) | | n PASS / n WARN / n FAIL |
| Paper ledger / orders (PAP) | | |
| Models (ML) | | |
| Backup / alerts / heartbeat (BAK, ALR) | | |
| Cross-project contention | | |

### Findings (severity-ordered)
- [P0|P1|P2] <ID> <title> — 근거: <command + key output>; 영향: <what breaks>; 상태: new | continuing since <date> | resolved; 제안: <next step>
### Known and unchanged
### NOT VERIFIED (what, why)
```

Severity: **P0** data loss/corruption, ledger inconsistency, decision chain down, backup shrink, broker dir unshared;
**P1** a scheduled job silently not running or failing, causality violation, throttling on the decision chain,
restore impossible, alert path broken; **P2** research gaps, hygiene, trends, cross-project coupling.

---

## Appendix A — Executable Probes

Run from the local repo root, outside blackouts; nothing is written on either side (`D` = audit date):

```bash
cd /home/kth/k-closing-alpha
for p in data research; do awk "/^\`\`\`python probe=$p/{f=1;next}/^\`\`\`/{f=0}f" docs/vps-checklist.md \
  | ssh or-vps "cd ~/k-closing-alpha && TZ=Asia/Seoul nice -n 19 timeout 300 ~/.local/bin/uv run --no-sync python - <D>"; done
```

Output is `CHECK <ID> <PASS|WARN|FAIL> <detail>`. Baselines for `D=2026-10-16`: archive 1,247 rows (prev-5 median 197),
`price_history` 2,762 rows/day, 1m regular 485 k bars / 1,273 symbols (`ls`), volume ratio median 0.959, 3 top-k rows,
nav conservation 0.00; only `DI-ARCH-01`, `DI-ARCH-09` and `RS-05` WARN. Standing probe behaviour: `RS-01` allows
`credit_balance` 4 trading days of lag; `RS-05` WARNs for `*.pre_*` older than 7 days; `RS-06` is PASS when each
quarantine entry has a `moves-*.json` manifest (`2026-09-22-test-fixture-cohort` is known); `RS-08` WARNs only for
`expiring_needs > 0`, `expired_recoverable > 0` or a stale `run_date`.

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
