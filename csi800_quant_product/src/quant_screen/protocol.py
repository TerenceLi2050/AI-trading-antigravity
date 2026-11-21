import json
from datetime import datetime, time
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from .data import benchmark_for
from .io import file_lock, read_table, sha256_file, utc_now, write_csv, write_json, write_parquet
from .labels import executable_labels
from .market import trade_flags
from .sessions import CST


import os


def portable_path(path: Path, base: Path) -> str:
    """Return a clean POSIX-style relative path under base when possible, else absolute string."""
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return str(path)


def make_pointer(root: Path, run: Path, state: str, signal_date: str | None, **extra) -> dict:
    return {
        "run": portable_path(run, root),
        "report": portable_path(run / "report.html", root),
        "signal_date": signal_date,
        "state": state,
        "completed_utc": utc_now(),
        **extra,
    }


def publish_report_pointer(root: Path, pointer: dict, replay: bool) -> None:
    with file_lock(root / ".report_publish.lock"):
        recent = root / ("latest_replay.json" if replay else "latest_run.json")
        write_json(recent, pointer)
        if pointer["state"] == "FUNDAMENTAL_SCREEN" and not replay:
            latest = root / "latest.json"
            if latest.is_file():
                prior = json.loads(latest.read_text(encoding="utf-8"))
                if prior.get("signal_date") and pointer.get("signal_date") and prior["signal_date"] > pointer["signal_date"]:
                    return
            write_json(latest, pointer)


def evaluate_frozen(root, run, current_code):
    protocol = root / "lockbox_protocol.json"
    record = json.loads(protocol.read_text(encoding="utf-8"))
    archived = root / "runs" / record["development_run_id"]
    frozen = archived / "frozen_evaluation"
    if current_code != record["inputs"]["code_sha256"]:
        return {"state": "PROTOCOL_MISMATCH", "reason": "frozen program changed"}
    for name, digest in record["frozen_files"].items():
        if Path(name).name != name or not (frozen / name).is_file() or sha256_file(frozen / name) != digest:
            return {"state": "PROTOCOL_MISMATCH", "reason": f"frozen input changed: {name}"}
    model = archived / "model.joblib"
    if not model.is_file() or sha256_file(model) != record["model_sha256"]:
        return {"state": "PROTOCOL_MISMATCH", "reason": "frozen model changed"}
    import joblib
    from .study import evaluate_locked

    feature_set = SimpleNamespace(panel=pd.read_parquet(frozen / "panel.parquet"),
                                  market=pd.read_parquet(frozen / "market.parquet").set_index("date"))
    policy = json.loads((frozen / "config.json").read_text(encoding="utf-8"))
    state = evaluate_locked(pd.read_parquet(frozen / "quotes.parquet"), feature_set, policy, joblib.load(model),
                            pd.read_parquet(frozen / "coverage.parquet").set_index("date"), root, run)
    return {"state": state, "protocol": str(protocol), "end": record["inputs"]["end"]}


def lockbox_request(action, research, audit, bundle, quotes, features, config, coverage, root: Path, run: Path) -> dict:
    protocol = root / "lockbox_protocol.json"
    if action == "evaluate" and protocol.is_file() and "frozen_files" in json.loads(protocol.read_text(encoding="utf-8")):
        return evaluate_frozen(root, run, audit["code_sha256"])
    if research["lockbox_state"] != "READY" or bundle is None:
        return {"state": "SEALED", "reasons": research["research_gate_reasons"]}
    evidence = {"config_sha256": audit["config_sha256"], "code_sha256": audit["code_sha256"],
                "inputs": audit["inputs"], "legacy_sha256": (audit.get("legacy") or {}).get("sha256"),
                "release_sha256": {path: sha256_file(Path(path)) for path in audit.get("releases", []) if Path(path).is_file()},
                "start": config["research"]["lockbox_start"], "end": str(features.market.index[-1].date()),
                "strategy": "frozen_model_ensemble_equal_weight"}
    if action == "freeze":
        if protocol.exists():
            return {"state": "ALREADY_FROZEN", "protocol": str(protocol)}
        model = run / "model.joblib"
        start = pd.Timestamp(config["research"]["lockbox_start"])
        frozen = run / "frozen_evaluation"
        write_parquet(frozen / "quotes.parquet", quotes.loc[quotes["date"].ge(start)])
        write_parquet(frozen / "panel.parquet", features.panel.loc[features.panel["date"].ge(start)])
        write_parquet(frozen / "market.parquet", features.market.loc[start:].rename_axis("date").reset_index())
        write_parquet(frozen / "coverage.parquet", coverage.loc[start:].rename_axis("date").reset_index())
        write_json(frozen / "config.json", config)
        record = {"state": "FROZEN", "created_utc": utc_now(), "development_run": str(run), "inputs": evidence,
                  "development_run_id": run.name, "frozen_files": {p.name: sha256_file(p) for p in frozen.iterdir()},
                  "model_sha256": sha256_file(model), "model_fit": bundle.train_stats,
                  "acceptance": {"minimum_sessions": 126, "positive_normal_and_2x_excess": True,
                                 "maximum_unpriced_weight": 0.05}}
        with file_lock(root / ".freeze.lock"):
            if protocol.exists():
                return {"state": "ALREADY_FROZEN", "protocol": str(protocol)}
            write_json(protocol, record)
        return {"state": "FROZEN", "protocol": str(protocol)}
    if not protocol.is_file():
        return {"state": "PROTOCOL_REQUIRED"}
    archived = json.loads(protocol.read_text(encoding="utf-8"))
    original = Path(archived["development_run"]) / "model.joblib"
    if archived["inputs"] != evidence or archived["model_fit"] != bundle.train_stats or sha256_file(original) != archived["model_sha256"]:
        return {"state": "PROTOCOL_MISMATCH", "protocol": str(protocol)}
    # Load the locally frozen artifact after checking its digest, then use the exclusive evaluation record.
    import joblib
    from .study import evaluate_locked
    state = evaluate_locked(quotes, features, config, joblib.load(original), coverage, root, run)
    return {"state": state, "protocol": str(protocol)}


def _horizon_from_run(root, info):
    run = Path(info["run"])
    for path in (root / "runs" / run.name / "config.json", run / "config.json"):
        if path.is_file():
            policy = json.loads(path.read_text(encoding="utf-8"))
            return int(policy["research"]["horizon"])
    return None


def record_prospective(root: Path, run: Path, screen: dict, picks: pd.DataFrame, audit: dict, *,
                       horizon=None, sessions=None, variants=None) -> dict:
    date = screen["signal_date"]
    source = audit.get("source", "legacy_pipeline")
    code = audit.get("code_sha256", {})
    modules = ("features.py", "current.py", "screen.py", "market.py", "labels.py")
    identity = ["raw-hfq-common-period-v1", source, audit["config_sha256"], {m: code.get(m) for m in modules}, horizon]
    strategy = sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    path = root / "prospective" / strategy / date
    fingerprint = path / "signal.json"
    prospective_file = path / "picks.csv"
    entry, exit_ = pd.NaT, pd.NaT
    if sessions is not None:
        days = sessions.index[sessions["is_session"].eq(1)]
        entry, exit_ = _trade_dates(days, pd.Timestamp(date), horizon)
    with file_lock(path / ".record.lock"):
        if fingerprint.exists():
            return {"status": "already_recorded", "path": str(fingerprint)}
        saved = picks.assign(variant="quality_mix") if variants is None else variants
        write_csv(prospective_file, saved)
        write_json(fingerprint, {"signal_date": date, "observed_utc": utc_now(), "run": str(run),
                   "strategy_version": strategy, "horizon_sessions": horizon,
                   "planned_entry": str(entry.date()) if pd.notna(entry) else None,
                   "planned_exit": str(exit_.date()) if pd.notna(exit_) else None,
                   "scope": "published_quality_observation", "picks_sha256": sha256_file(prospective_file),
                   "config_sha256": audit["config_sha256"], "manifest_sha256": audit.get("manifest_sha256"),
                   "verified_codes": screen["session"]["verified_codes"]})
    return {"status": "recorded", "path": str(fingerprint)}


def _trade_dates(days, date, horizon):
    pos = days.get_indexer([date])[0]
    entry = days[pos + 1] if 0 <= pos < len(days) - 1 else pd.NaT
    exit_ = days[pos + 1 + horizon] if horizon is not None and 0 <= pos < len(days) - 1 - horizon else pd.NaT
    return entry, exit_


def _records(root):
    return sorted((root / "prospective").rglob("signal.json"))


def _settled(path, picks_hash):
    if not path.is_file():
        return pd.DataFrame(), False
    record = json.loads(path.read_text(encoding="utf-8"))
    raw = json.dumps(record["rows"], sort_keys=True, ensure_ascii=False).encode()
    if record["picks_sha256"] != picks_hash or sha256(raw).hexdigest() != record["rows_sha256"]:
        raise ValueError("Prospective settlement checksum mismatch")
    return pd.DataFrame(record["rows"]), record["complete"]


def tracking_codes(root):
    codes = set()
    for path in _records(root):
        info = json.loads(path.read_text(encoding="utf-8"))
        saved, complete = _settled(path.parent / "settlement.json", info["picks_sha256"])
        if not complete:
            picks = path.parent / "picks.csv"
            if sha256_file(picks) != info["picks_sha256"]:
                raise ValueError("Prospective picks failed verification")
            pending = read_table(picks)
            if not saved.empty:
                pending = pending.loc[~pending["code"].isin(saved["code"])]
            codes.update(pending["code"])
    return sorted(codes)


def prospective_outcomes(root: Path, run: Path, quotes, market, sessions, horizon: int) -> dict:
    records = _records(root)
    if not records or sessions is None:
        return {"records": len(records), "matured": 0}
    days = sessions.index[sessions["is_session"].eq(1)]
    rows = []
    for path in records:
        info = json.loads(path.read_text(encoding="utf-8"))
        picks = path.parent / "picks.csv"
        if sha256_file(picks) != info["picks_sha256"]:
            raise ValueError("Prospective shortlist checksum mismatch")
        saved = read_table(picks)
        if "variant" not in saved:
            saved["variant"] = "quality_mix"
        codes = saved[["code", "rank", "variant"]]
        date = pd.Timestamp(info["signal_date"])
        held = info.get("horizon_sessions") or _horizon_from_run(root, info)
        entry, exit_ = _trade_dates(days, date, held)
        entry = pd.Timestamp(info["planned_entry"]) if info.get("planned_entry") else entry
        exit_ = pd.Timestamp(info["planned_exit"]) if info.get("planned_exit") else exit_
        settlement = path.parent / "settlement.json"
        previous, complete = _settled(settlement, info["picks_sha256"])
        if complete:
            rows.append(previous)
            continue
        result = codes.assign(date=date, planned_entry=entry, planned_exit=exit_, label_status="PENDING",
                              strategy_version=info.get("strategy_version", "legacy"), horizon_sessions=held,
                              forward_return=float("nan"), excess_return=float("nan"))
        observed = datetime.fromisoformat(info["observed_utc"])
        if pd.notna(entry) and observed >= datetime.combine(entry.date(), time(9, 15), CST):
            result["label_status"] = "OBSERVED_TOO_LATE"
        elif held is None:
            result["label_status"] = "HORIZON_UNKNOWN"
        elif pd.notna(exit_) and exit_ <= market.index[-1]:
            daily = days[(days >= date) & (days <= exit_)]
            if len(daily) == held + 2 and daily[1] == entry and daily.isin(market.index).all():
                focus = quotes.loc[quotes["code"].isin(codes["code"]) & quotes["date"].isin(daily)]
                labels = executable_labels(focus, benchmark_for(market.loc[daily]), held)
                columns = ["label_status", "forward_return", "excess_return"]
                at_signal = labels.loc[labels["date"].eq(date)].set_index("code")
                result = result.set_index("code")
                result[columns] = at_signal[columns].reindex(result.index)
                result["label_status"] = result["label_status"].fillna("SIGNAL_QUOTE_MISSING")
                result = result.reset_index()
            else:
                result["label_status"] = "BENCHMARK_INCOMPLETE"
        terminal = result["label_status"].isin(["OBSERVED_TOO_LATE", "realized"])
        if pd.notna(exit_) and exit_ <= market.index[-1]:
            # A blocked open is terminal only when the bar and its trading state are actually archived.
            for flag, day in [("entry_blocked_or_missing", entry), ("exit_blocked_or_missing", exit_)]:
                bars = quotes.loc[quotes["date"].eq(day) & quotes["trade_status"].notna()]
                flags = trade_flags(bars, market.index)
                known = bars["trade_status"].eq(0) | (flags["open_ok"] & ~flags["missing_open_limit"] & flags["is_st_known"])
                terminal |= result["label_status"].eq(flag) & result["code"].isin(bars.loc[known, "code"])
        pairs = ["variant", "code"]
        if terminal.any():
            with file_lock(path.parent / ".record.lock"):
                previous, _ = _settled(settlement, info["picks_sha256"])
                additions = result.loc[terminal]
                if not previous.empty:
                    additions = additions.loc[~pd.MultiIndex.from_frame(additions[pairs]).isin(pd.MultiIndex.from_frame(previous[pairs]))]
                if not additions.empty:
                    previous = pd.concat([previous, additions], ignore_index=True)
                    settled_rows = json.loads(previous.to_json(orient="records", date_format="iso", double_precision=15))
                    for item in settled_rows:
                        for key in ("date", "planned_entry", "planned_exit"):
                            if item.get(key):
                                item[key] = item[key][:10]
                    digest = sha256(json.dumps(settled_rows, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
                    write_json(settlement, {"picks_sha256": info["picks_sha256"], "rows_sha256": digest, "rows": settled_rows,
                               "complete": len(previous) == len(codes), "evaluated_utc": utc_now()})
                    previous = pd.DataFrame(settled_rows)
        if not previous.empty:
            unresolved = ~pd.MultiIndex.from_frame(result[pairs]).isin(pd.MultiIndex.from_frame(previous[pairs]))
            result = pd.concat([previous, result.loc[unresolved]], ignore_index=True)
        rows.append(result)
    combined = pd.concat(rows, ignore_index=True)
    for column in ("date", "planned_entry", "planned_exit"):
        combined[column] = pd.to_datetime(combined[column], format="mixed")
    write_csv(run / "prospective_outcomes.csv", combined)
    summary = combined.groupby(["strategy_version", "variant"]).agg(
        signals=("date", "nunique"), observations=("code", "size"), realized=("forward_return", "count"),
        pending=("label_status", lambda x: x.eq("PENDING").sum()), mean_realized_return=("forward_return", "mean"),
        mean_realized_excess=("excess_return", "mean")).reset_index()
    write_csv(run / "prospective_summary.csv", summary)
    return {"records": len(records), "unique_signal_dates": int(combined["date"].nunique()),
            "matured": int(combined["label_status"].eq("realized").sum()),
            "pending": int(combined["label_status"].eq("PENDING").sum())}
