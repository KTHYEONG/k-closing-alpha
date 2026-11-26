#!/usr/bin/env python3
# mypy: ignore-errors
"""KCA window probes (run on the VPS from the repo root).

Usage: python ops/probes/kca_probe.py <M|O|E|N|W> [YYYY-MM-DD]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
if len(sys.argv) < 2 or sys.argv[1] not in {"M", "O", "E", "N", "W"}:
    sys.stderr.write("usage: kca_probe.py <M|O|E|N|W> [YYYY-MM-DD]\n")
    sys.exit(2)

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
def shifted(d):
    return kind(d) == "SHIFTED"
SHIFT_HOLD = "SKIPPED:session_shifted"
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
def manifest_status(d, prefix, research=False):
    # research=False: primary manifests only; `<run>-research` siblings (roster extras) share the prefix and are judged by C13
    res = {}
    for run in sorted((CAP / f"manifests/{d}").glob(prefix + "*")):
        if run.name.endswith("-research") != research:
            continue
        st = {f.name[9:-5] for f in run.glob("manifest-*.json")}
        res[run.name] = "complete" if "complete" in st else ("partial" if "partial" in st else ("pending" if st else "none"))
    return res
OPEN_STATUSES = {"FAILED", "PARTIAL", "UNKNOWN"}
def failed_entries(d, run):
    m = json.loads((CAP / f"manifests/{d}/{run}/manifest-partial.json").read_text())
    return sum(e.get("status") in OPEN_STATUSES for e in m.get("entries", []))
def failed_symbols(d, run):
    m = json.loads((CAP / f"manifests/{d}/{run}/manifest-partial.json").read_text())
    return {e.get("symbol") for e in m.get("entries", []) if e.get("status") in OPEN_STATUSES}
def open_symbols(d, run, session):
    healed = {e.get("symbol") for f in (CAP / f"manifests/{d}").glob(f"tape-{d}-{session}-*/manifest-*.json") for e in json.loads(f.read_text()).get("entries", []) if e.get("status") in ("COMPLETE", "NO_TRADES")}
    return len(failed_symbols(d, run) - healed)
def manifest_entries(d, run):
    for f in sorted((CAP / f"manifests/{d}/{run}").glob("manifest-*.json")):
        return json.loads(f.read_text())
    return {}
def research_p(d, ds):
    return ROOT / f"history/research/{ds}/{d:%Y-%m}/{d}.parquet"
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
        n_all = len(st) + len(manifest_status(d, f"aftermarket-book-{d}-", research=True))
        ok_line = bool(line) and "status=COMPLETE" in line[-1] and f"manifests={n_all}" in line[-1]
        out("C03", lv(not st or c["pending"] + c["none"] > 0, c["partial"] > 0 or not ok_line), batches=len(st), status=dict(c), journal=(line[-1].split("status=")[1][:40] if line else "absent"))
    @check("C04")
    def _():
        streams = ["regular-bars", "regular-ticks", "nxt-premarket-bars", "nxt-aftermarket-bars", "krx-aftermarket-bars", "krx-aftermarket-ticks", "nxt-aftermarket-ticks"]
        res, fail, warn = {}, [], []
        for s in streams:
            st = manifest_status(d, f"archive-{d}-{s}-")
            best = "complete" if "complete" in st.values() else ("partial" if "partial" in st.values() else "missing")
            if best == "partial":
                run = [k for k, v in st.items() if v == "partial"][-1]
                tick = s.endswith("ticks")
                opened = open_symbols(d, run, s[: -len("-ticks")].replace("-", "_")) if tick else len(failed_symbols(d, run))
                best = "tape_healed" if tick and opened == 0 else f"partial({failed_entries(d, run)}failed,{opened}open)"
                if best != "tape_healed":
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
def _research_share(d, prefixes):
    tot = cut = 0
    found = {}
    for pre in prefixes:
        st = manifest_status(d, pre, research=True)
        found[pre.split(f"-{d}-")[0]] = ",".join(sorted(set(st.values()))) or "missing"
        for run in st:
            es = manifest_entries(d, run).get("entries", [])
            tot += len(es)
            cut += sum(e.get("reason") in ("deadline", "missed_start") for e in es)
    return found, tot, cut
def c_research_e(d):
    @check("C12")
    def _():
        if not timer_active("research-roster"):
            out("C12", "SKIP(timer_off)")
            return
        o = outcomes(d).get("research_roster")
        if o and o["outcome"] == "SKIPPED":
            out("C12", f"SKIP({o.get('reason', '')})")
            return
        runs_ = [p for p in (CAP / f"manifests/{d}").glob(f"research-roster-{d}-*")]
        m = manifest_entries(d, sorted(runs_)[-1].name) if runs_ else {}
        done = datetime.fromisoformat(m["completed_at"]).astimezone(KST) if m else None
        n = (o or {}).get("metrics", {}).get("members")
        out("C12", lv(not m, m.get("status") != "COMPLETE" or done > at(d, "15:20:55") or not n or not 150 <= n <= 400),
            status=m.get("status", "missing"), completed=done.strftime("%T") if done else "-", members=n, outcome=(o or {}).get("outcome", "absent"))
    @check("C13")
    def _():
        if kind(d) == "SHIFTED":
            out("C13", "SKIP(session_shifted)")
            return
        found, tot, cut = _research_share(d, [f"auction-close-{d}-"])
        share = cut / tot if tot else 0.0
        out("C13", lv(False, "missing" in found.values() or share > 0.10), **found, entries=tot, cut_share=round(share, 3))
    @check("C14")
    def _():
        if not timer_active("postclose-snapshot"):
            out("C14", "SKIP(timer_off)")
            return
        if kind(d) == "SHIFTED":
            out("C14", "SKIP(session_shifted)")
            return
        st = manifest_status(d, f"postclose-{d}-")
        runs_ = sorted(st)
        m = manifest_entries(d, runs_[-1]) if runs_ else {}
        es = m.get("entries", [])
        good = sum(e.get("status") in ("COMPLETE", "NO_TRADES", "NOT_APPLICABLE") for e in es) / len(es) if es else 0.0
        done = datetime.fromisoformat(m["completed_at"]).astimezone(KST) if m else None
        part = research_p(d, "postclose_strength").exists()
        out("C14", lv(not m or not part, good < 0.95 or done > at(d, "15:39:00")), status=m.get("status", "missing"),
            completed=done.strftime("%T") if done else "-", good_share=round(good, 3), partition=part)
def c_research_n(d):
    @check("C13")
    def _():
        found, tot, cut = _research_share(d, [f"auction-path-{d}-", f"aftermarket-book-{d}-"])
        share = cut / tot if tot else 0.0
        out("C13", lv(False, "missing" in found.values() or share > 0.10), **found, entries=tot, cut_share=round(share, 3))
    @check("C15")
    def _():
        import pandas as pd
        p = research_p(d, "decision_extras")
        if not p.exists():
            out("C15", "FAIL", partition="missing")
            return
        x = pd.read_parquet(p)
        dec = sorted((CAP / f"decision/{d}").glob("*/input.parquet"))
        n_cohort = len(pd.read_parquet(dec[-1], columns=["종목코드"])) if dec else -1
        cntr = float(x["kiwoom_cntr_str"].notna().mean()) if len(x) else 0.0
        out("C15", lv(False, len(x) != n_cohort or cntr < 0.9 or not research_p(d, "investor_estimate_decision").exists()),
            rows=len(x), cohort=n_cohort, cntr_str_share=round(cntr, 3))
def c_decision(d):
    @check("C01")
    def _():
        if kind(d) == "SHIFTED":
            out("C01", "SKIP(session_shifted)")
            return
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
        if shifted(d):  # exit is held on a SHIFTED day: the previous lots must stay open, with no sell yet
            out("P02", lv(len(unexited) != len(buys) or no_trade or stale not in ([], [str(p)])), hold="shifted", prev_lots=len(buys), unexited=len(unexited), sells_without_trade=len(no_trade), stale_open_decisions=stale or "-")
            return
        out("P02", lv(bool(stale) or bool(unexited) or bool(no_trade)), prev_lots=len(buys), unexited=len(unexited), sells_without_trade=len(no_trade), stale_open_decisions=stale or "-")
    @check("P03")
    def _():
        nav = pd.read_parquet(ROOT / "paper/nav.parquet")
        nav["day"] = nav["as_of_date"].astype(str).str[:10]
        row = nav[nav["day"] == str(p if shifted(d) else d)]  # no nav row is written on a SHIFTED day
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
        import gzip
        started = datetime.fromisoformat(lr["started_at"]).timestamp()
        res, bad = {}, []
        for tier in ("raw", "normalized", "manifests"):
            lp = CAP / f"offsite/ledger/{tier}/{d}.jsonl"
            lgz = lp.with_name(lp.name + ".gz")
            local = CAP / tier / str(d)
            if not local.exists():
                res[tier] = "no_local"
                continue
            text = lp.read_text() if lp.exists() else gzip.open(lgz, "rt").read() if lgz.exists() else ""
            segs = [json.loads(l) for l in text.splitlines() if l.strip()]
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
            for p in sorted((CAP / tier).iterdir()):
                if re.fullmatch(r"\d{4}-\d{2}-\d{2}", p.name) and date.fromisoformat(p.name) < cutoff and datetime.fromtimestamp(p.stat().st_mtime, KST) < at(last, "21:00:00"):
                    kept_local.append(f"{tier}/{p.name}")
        out("B05", lv(not summ or bool(failed), kept != "0" or bool(kept_local)), last_prune=last, summary=bool(summ), failed=failed or "-", sealed_kept=kept, kept_local_pre_prune_n=len(kept_local), kept_local_pre_prune=kept_local[:5] or "-", remote_deleted=ages)
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

if W == "M":
    import pandas as pd
    YDAY = TODAY - timedelta(days=1)
    @check("H01")
    def _():
        n = sum("kca-" in l for l in sh(["systemctl", "--user", "list-timers", "kca-*", "--all", "--no-legend"]).splitlines())
        off = [l.split()[0] for l in sh(["systemctl", "--user", "list-unit-files", "kca-*.timer", "--no-legend"]).splitlines() if l.split()[1:2] != ["enabled"]]
        failed = [l.split()[0] for l in sh(["systemctl", "--user", "list-units", "--failed", "--no-legend", "--plain", "kca-*"]).splitlines() if l.strip()]
        linger = sh(["loginctl", "show-user", os.environ.get("USER", "ubuntu"), "-p", "Linger", "--value"]).strip()
        expected = sum(1 for p in (Path(__file__).resolve().parents[2] / "deploy" / "systemd").glob("kca-*.timer") if not p.name.startswith("kca-alert@"))
        out("H01", lv(n != expected or bool(failed) or linger != "yes", bool(off)), timers=n, expected=expected, not_enabled=off or "-", failed=failed or "-", linger=linger)
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
    if shifted(D):  # a KRX-shifted D has no cohort, archive or 1m/tick partitions by design (collect/archive_intraday skip it)
        for _cid in ("C03", "C04", "C05", "C11", "Q08"):
            out(_cid, "SKIP(session_shifted)")
    else:
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
        s = json.loads(Path("/home/ubuntu/quant-ops/public/status.json").read_text())
        proj = [p for p in s["projects"] if "closing" in p["name"] or p["name"] in ("kca",)]
        bad = [(c["id"], c["level"]) for p in proj for c in p["checks"] if c["level"] != "OK"]
        out("A04", lv(False, bool(bad) or not proj), generated=s["generated_at"][:16], card=[p["name"] for p in proj], non_ok=bad or "-")

if W == "O":
    import pandas as pd
    runs("R01", [("price-ingest", D, "08:30:00", 1), ("auction-open", D, "08:39:55", 1), ("order-exit", D, "08:40:10", 1),
                 ("order-reconcile-exit", D, "09:00:40", 1), ("paper-exit", D, "09:01:00", 1)])
    o = expect("R02", D, {"price_ingest": {"OK"},
                          "order_exit": {"DEGRADED:shifted_session_hold"} if shifted(D) else {"OK", "SKIPPED:no_open_positions", "SKIPPED:trading_disabled"},
                          "order_reconcile_exit": {SHIFT_HOLD} if shifted(D) else {"OK", "SKIPPED:no_orders", "SKIPPED:trading_disabled"}})
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
        if shifted(D):
            out("C05", "SKIP(session_shifted)")
            return
        st = manifest_status(D, f"auction-open-{D}-")
        j = jr("auction-open", at(D, "08:39:00"), at(D, "09:40:00"), r"stage=auction_capture status=\w+ phase=open")
        out("C05", lv(not st, "complete" not in st.values()), open=",".join(st.values()) or "missing", journal=j[-1].split("status=")[1].split()[0] if j else "absent")
    p_ledger(D, P, morning=True)
    @check("P06")
    def _():
        ex, rc = o["order_exit"].get("metrics", {}), o["order_reconcile_exit"].get("metrics", {})
        if shifted(D) or o["order_exit"]["outcome"] == "SKIPPED":
            out("P06", "SKIP(" + (o["order_exit"].get("reason", "") or "session_shifted") + ")")
            return
        acc = ex.get("n_acked", 0) + ex.get("n_rejected", 0) + ex.get("n_skipped", 0)
        bad = ex.get("n_unknown", 1) or ex.get("n_unknown_unresolved_total", 1) or rc.get("n_unknown", 1) or rc.get("n_missing_at_broker", 1) or rc.get("position_shortfalls") or rc.get("n_unknown_unresolved_total", 1)
        out("P06", lv(bool(bad) or acc != ex.get("n_lots"), rc.get("n_filled") != rc.get("n_intents") or ex.get("n_rejected", 0) > 0),
            env=ex.get("env"), lots=ex.get("n_lots"), acked=ex.get("n_acked"), rejected=ex.get("n_rejected"), unknown=ex.get("n_unknown"), rec_filled=f"{rc.get('n_filled')}/{rc.get('n_intents')}", missing_at_broker=rc.get("n_missing_at_broker"), shortfalls=rc.get("position_shortfalls") or "-")
    a_heartbeat()
    a_alerts(at(D, "07:00:00"))
    x_tokens(at(D, "07:00:00"))

if W == "E":
    import numpy as np
    import pandas as pd
    runs("R01", [("collect", D, "15:20:00", 1), ("research-roster", D, "15:20:30", 1), ("predict", D, "15:21:00", 1), ("auction-close", D, "15:20:25", 1), ("order-entry", D, "15:21:15", 1),
                 ("finalize-close", D, "15:30:30", 1), ("order-reconcile-entry", D, "15:30:40", 1), ("paper-entry", D, "15:30:30", 1), ("postclose-snapshot", D, "15:31:30", 1)])
    o = expect("R02", D, {"predict": {"NO_DECISION:session_shifted", SHIFT_HOLD} if shifted(D) else {"OK"},
                          "order_entry": {SHIFT_HOLD} if shifted(D) else {"OK", "SKIPPED:trading_disabled"},
                          "finalize_close": {SHIFT_HOLD, "OK:session_shifted"} if shifted(D) else {"OK"},
                          "order_reconcile_entry": {SHIFT_HOLD} if shifted(D) else {"OK", "SKIPPED:trading_disabled"}})
    c_decision(D)
    @check("Q01")
    def _():
        if shifted(D):
            out("Q01", "SKIP(session_shifted)")
            return
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
        if shifted(D):
            out("C05", "SKIP(session_shifted)")
            return
        st = manifest_status(D, f"auction-close-{D}-")
        out("C05", lv(not st, "complete" not in st.values()), close=",".join(st.values()) or "missing")
    c_research_e(D)
    @check("C16")
    def _():
        if not timer_active("auction-close"):
            out("C16", "SKIP(timer_off)")
            return
        rec = o.get("auction_close")
        if rec is None and shifted(D):
            out("C16", "SKIP(session_shifted_no_outcome)")
            return
        if rec is None:
            out("C16", "WARN", outcome="absent")
            return
        if rec["outcome"] == "SKIPPED":
            out("C16", f"SKIP({rec.get('reason', '')})")
            return
        m = rec.get("metrics", {})
        fields = {k: m.get(k, "-") for k in ("roster", "capacity_per_round", "demand_per_round", "status", "start_lag_s", "rounds_with_deadline", "max_missed_rounds_per_symbol")}
        if not (m.get("rounds_with_deadline", 0) or m.get("n_deadline", 0)):
            out("C16", "PASS", **fields)
            return
        cause = "overcommit" if m.get("status") == "OVERCOMMIT" else "late_start" if m.get("start_lag_s", 0) > 5 else "vendor"
        out("C16", "WARN", cause=cause, **fields)
    p_ledger(D, P, morning=False)
    @check("P06")
    def _():
        en, rc = o["order_entry"].get("metrics", {}), o["order_reconcile_entry"].get("metrics", {})
        if shifted(D) or o["order_entry"]["outcome"] == "SKIPPED":
            out("P06", "SKIP(" + (o["order_entry"].get("reason", "") or "session_shifted") + ")")
            return
        picks = en.get("n_picks", en.get("n_lots"))
        acc = en.get("n_acked", 0) + en.get("n_rejected", 0) + en.get("n_not_sent", 0) + en.get("n_skipped", 0)
        bad = en.get("n_unknown", 1) or rc.get("n_unknown", 1) or rc.get("n_missing_at_broker", 1) or rc.get("n_unknown_unresolved_total", 1)
        out("P06", lv(bool(bad) or acc != picks, rc.get("n_filled") != rc.get("n_intents")), env=en.get("env"), picks=picks, acked=en.get("n_acked"), rejected=en.get("n_rejected"),
            not_sent=en.get("n_not_sent"), rec_filled=f"{rc.get('n_filled')}/{rc.get('n_intents')}", unknown=rc.get("n_unknown"), shortfalls=rc.get("position_shortfalls") or "-")
    a_heartbeat()
    a_alerts(at(D, "15:00:00"))
    x_tokens(at(D, "15:15:00"))

if W == "N":
    import pandas as pd
    runs("R01", [("archive-intraday-regular", D, "15:40:00", 1), ("aftermarket-book", D, "15:40:00", 1), ("auction-path", D, "15:45:00", 1), ("decision-extras", D, "16:30:00", 1), ("archive-intraday", D, "20:05:00", 1),
                 ("tape-sweep", D, "20:35:00", 1), ("backup-prune", D, "21:00:00", 1), ("daily-audit", D, "21:20:00", 1)]
         + ([("altdata-capture", D, "21:35:00", 1)] if NOW > at(D, "21:45:00") else []))
    r_degraded(D)
    if shifted(D):  # archive-intraday, auction-path, cohort completeness and the 1m/tick partitions are skipped by design on a KRX-shifted day
        for _cid in ("C03", "C04", "C05", "C06", "C13", "C15", "C11", "Q06", "Q08"):
            out(_cid, "SKIP(session_shifted)")
    else:
        c_manifests_live(D)
        @check("C06")
        def _():
            st = manifest_status(D, f"auction-path-{D}-")
            out("C06", lv(False, "complete" not in st.values()), status=",".join(st.values()) or "missing")
        c_research_n(D)
    @check("C08")
    def _():
        r = json.loads((CAP / "staging/tape_sweep/last_report.json").read_text())
        n = lambda x: len(x) if isinstance(x, (list, tuple, dict)) else int(x or 0)
        out("C08", lv(r.get("days_checked", [""])[-1] != str(D) or bool(r.get("disk_guard")), n(r.get("expiring_needs")) > 0 or n(r.get("expired_recoverable")) > 0 or n(r.get("expired")) > 0),
            last_day=r.get("days_checked", ["-"])[-1], needs=r.get("needs"), expiring=n(r.get("expiring_needs")), expired=n(r.get("expired")), expired_recoverable=n(r.get("expired_recoverable")), unresolved=n(r.get("unresolved")), disk_guard=r.get("disk_guard"))
    @check("C17")
    def _():
        # Scan cost grows with the 30-day need window (17.6 min 11-12, 19.6 min 11-13); the walk deadline is 21:15 and an overrun exits 0.
        L = jr("tape-sweep", at(D, "20:30:00"), at(D, "23:59:59"), pat=r"stage=(tape_universe|tape_sweep_report)|stage=tape_sweep status=NOOP|status=DEADLINE")
        ts = lambda l: datetime.strptime(l[:19], "%Y-%m-%d %H:%M:%S").replace(tzinfo=KST)
        uni = [l for l in L if "stage=tape_universe" in l]; rep = [l for l in L if "stage=tape_sweep_report" in l or "status=NOOP" in l]
        scan = round((ts(uni[-1]) - at(D, "20:35:00")).total_seconds() / 60, 1) if uni else None
        end = ts(rep[-1]) if rep else None; dl = sum("status=DEADLINE" in l for l in L)
        out("C17", lv(end is None, scan is None or scan > 30 or dl > 0 or end > at(D, "21:10:00")), scan_min=scan, finished=f"{end:%H:%M}" if end else "-", deadline=dl)
    if not shifted(D):
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
        days = [x for x in days if x <= str(P) and not shifted(date.fromisoformat(x))][-5:]  # a SHIFTED day (e.g. CSAT 10:00-16:30) has a truncated 1m partition by design
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

if W == "W":
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
        floor = TODAY
        for _ in range(6):
            floor = prev_td(floor)
        rows = sorted(p.stem for p in (ROOT / "history/pit_train/rows").glob("*.parquet"))
        store_max = rows[-1] if rows else "-"
        cutoff = str(last.get("training_cutoff", ""))[:10]
        ext = jr("retrain", at(sat, "21:59:00"), at(sun, "06:00:00"), r"stage=pit_store extend")
        oom = jr("retrain", at(sat, "21:59:00"), at(sun, "06:00:00"), r"(?i)oom|exit code 137|killed")
        stale = store_max < str(floor) or cutoff < str(floor) or last.get("strategy_id") != "KCA-TOPK-COSTAWARE-002"
        out("R04", lv(last["outcome"] not in ("PROMOTED", "PROMOTED_UNGATED", "MANUAL_HOTFIX") or not known or stale, age > 8 or last["outcome"] == "MANUAL_HOTFIX" or not ext or bool(oom)),
            last=last["outcome"], age_days=age, agreement=last.get("agreement"), live_model_in_registry=known, strategy=last.get("strategy_id"),
            store_max=store_max, cutoff=cutoff, floor=floor, extend=(ext[-1].split("extend")[-1][:40] if ext else "absent"), oom_lines=len(oom))
    @check("P08")
    def _():
        tr = pd.read_parquet(ROOT / "paper/trades.parquet", columns=["exit_filled_at", "net_ret", "cost"])
        tr["d"] = pd.to_datetime(tr["exit_filled_at"], utc=True).dt.tz_convert(KST).dt.date
        days = sorted(tr["d"].unique())
        def win(n):
            t = tr[tr["d"].isin(days[-n:])]
            return len(t), (float(t["net_ret"].mean()) * 1e4 if len(t) else float("nan")), (float((t["net_ret"] > 0).mean()) if len(t) else float("nan"))
        n20, m20, w20 = win(20)
        n60, m60, w60 = win(60)
        out("P08", lv(False, (n60 >= 30 and m60 <= 0) or (n20 >= 30 and m20 < -50)), expect_bp=36,
            n20=n20, mean20_bp=round(m20, 1), win20=round(w20, 2), n60=n60, mean60_bp=round(m60, 1), win60=round(w60, 2), mean_cost=round(float(tr["cost"].tail(60).mean()), 0))
    @check("B07")
    def _():
        v = jr("offsite-verify", at(sun, "10:59:00"), at(sun, "23:59:00"), r"stage=(offsite_verify|restore_drill)|finding=")
        s = next((l for l in v if "stage=offsite_verify" in l and "missing=" in l), "")
        drills = [re.search(r"tier=(\w+) .*status=(\w+)", l).groups() for l in v if "stage=restore_drill" in l]
        ok = "missing=0" in s and "mismatched=0" in s and len(drills) >= 3 and all(st == "OK" for _, st in drills)
        out("B07", lv(not ok), sunday=sun, summary=s.split("stage=offsite_verify")[-1][:80] or "absent", drills=drills or "-")
    @check("B08")
    def _():
        snaps = sorted(x.strip("/") for x in sh([RCLONE, "lsf", "--dirs-only", f"{REMOTE}/snapshots"]).splitlines() if re.fullmatch(r"\d{4}-\d{2}-\d{2}/?", x.strip()))
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
