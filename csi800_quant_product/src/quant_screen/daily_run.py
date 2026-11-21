from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path

import pandas as pd

from .current import apply_current_membership, apply_current_quality, load_current
from .daily import available_releases, collect_daily, verify_release
from .data import quote_audit
from .features import build_features
from .io import dates, file_lock, read_table, sha256_file, utc_now, write_csv, write_json
from .market import trade_flags
from .protocol import make_pointer, portable_path, prospective_outcomes, publish_report_pointer, record_prospective, tracking_codes
from .report import report_html
from .screen import diversified_screen, generate, prospective_variants
from .sessions import CST, load_sessions, quote_completion, readiness


def _failed(output, run, replay, message, root=None, **details):
    run_ref = portable_path(run, root) if root is not None else str(run)
    result = {"state": "DATA_UNAVAILABLE", "reasons": [message], "signal_date": None,
              "completed_utc": utc_now(), "run": run_ref, **details}
    write_json(run / "screen.json", result)
    publish_report_pointer(output, result, replay)
    print(f"{message}\n{run / 'screen.json'}", flush=True)
    return 2


def _selection_digest(frame):
    text = frame[["rank", "code", "industry", "final_score"]].to_csv(index=False, float_format="%.12g", lineterminator="\n")
    return sha256(text.encode("utf-8")).hexdigest()


def run_daily(root, output, run, config, config_hash, code_hashes, schedule, requested, expected, phase, today,
              refresh, replay, force=False, model_path=None):
    folder = root / config["data"]["daily_updates"]
    cutoff = expected if expected is not None else requested
    known_by = (requested.tz_localize(CST) + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)) if replay else None
    selected, rejected = available_releases(folder, cutoff, known_by)
    status = {"status": "using_archive", "rejected_releases": rejected}
    if refresh and expected is not None:
        prior = json.loads(selected.read_text(encoding="utf-8")) if selected else {}
        pointer = output / "latest_run.json"
        previous_run = json.loads(pointer.read_text(encoding="utf-8")) if pointer.is_file() else {}
        retry = previous_run.get("state") == "RESEARCH_ONLY" and previous_run.get("signal_date") == str(expected.date())
        if (force or retry or prior.get("last_session") != str(expected.date()) or prior.get("current_financials", {}).get("status") != "ok"
                or prior.get("current_financials", {}).get("rows", 0) < 800 * config["data"]["minimum_fundamental_coverage"]
                or prior.get("minimum_coverage", 0) < config["data"]["minimum_member_coverage"]
                or prior.get("history_sessions") != config["data"]["warmup_sessions"]):
            try:
                with file_lock(folder / ".collection.lock", wait_for_collection=True):
                    current, _ = available_releases(folder, cutoff)
                    if current == selected or current is None or current.parent.name[:8] != expected.strftime("%Y%m%d"):
                        new = collect_daily(folder, schedule, expected, config, current, tracking_codes(output))
                        status.update(status="published", collected_release=new.parent.name)
                    else:
                        status["status"] = "up_to_date"
                    selected, ignored = available_releases(folder, cutoff)
                    status["rejected_releases"] = ignored
            except Exception as error:
                status.update(status="failed", error_type=type(error).__name__, message=str(error)[:450])
        else:
            status["status"] = "up_to_date"
    write_json(run / "refresh.json", status)
    if selected is None:
        return _failed(output, run, replay, "截至请求时点没有通过校验的独立行情批次", root=root, refresh=status, phase=phase)
    try:
        return _screen(root, output, run, config, config_hash, code_hashes, selected, expected, phase, today, replay, status,
                       model_path=model_path)
    except (OSError, ValueError, KeyError) as exc:
        return _failed(output, run, replay, str(exc), root=root, refresh=status)


def replay_daily(root, output, run, reference, code_hashes):
    source = Path(reference)
    if not source.is_absolute():
        source = output / "runs" / source
    try:
        evidence = json.loads((source / "audit.json").read_text(encoding="utf-8"))
        target = json.loads((source / "screen.json").read_text(encoding="utf-8"))
        policy = source / "config.json"
        if sha256_file(policy) != evidence["archived_config_sha256"] or code_hashes != evidence["code_sha256"]:
            raise ValueError("Frozen configuration or program differs from the original run")
        config = json.loads(policy.read_text(encoding="utf-8"))
        path = root / evidence["release_path"]
        if not path.is_file():
            # An imported batch can have a content-addressed folder name on a new machine.
            copies = (root / config["data"]["daily_updates"]).glob("*/manifest.json")
            path = next((copy for copy in copies if sha256_file(copy) == evidence["manifest_sha256"]), path)
        if not path.is_file() or sha256_file(path) != evidence["manifest_sha256"]:
            raise ValueError("Frozen release manifest checksum mismatch")
        expected = target["session"]["expected_session"]
        write_json(run / "config.json", config)
        return _screen(root, output, run, config, evidence["config_sha256"], code_hashes, path,
                       pd.Timestamp(expected) if expected else None, target["session"]["phase"],
                       pd.Timestamp(target["run_date"]), True, {"status": "pinned_replay", "source_run": source.name},
                       target["selection_sha256"])
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return _failed(output, run, True, str(exc))


def _screen(root, output, run, config, config_hash, code_hashes, path, expected, phase, today, replay, refresh,
            expected_selection=None, model_path=None):
    meta = json.loads(path.read_text(encoding="utf-8"))
    verify_release(path, meta)
    day = pd.Timestamp(meta["last_session"])
    schedule = load_sessions(path.parent / "sessions.csv")
    quotes = pd.read_parquet(path.parent / "quotes.parquet")
    market = read_table(path.parent / "benchmark.csv")
    market["date"] = dates(market["date"])
    market = market.set_index("date").sort_index()
    snapshot, disclosed = load_current([str(path)], day)
    if snapshot is None:
        return _failed(output, run, replay, "缺信号日官方中证800成分", root=root)
    # Only the dated snapshot can establish membership. Earlier bars are factor warm-up observations.
    active = pd.Series(False, index=quotes.index)
    industry = pd.Series(pd.NA, index=quotes.index, dtype="string")
    coverage = pd.DataFrame(index=market.index, columns=["expected", "observed", "coverage"], dtype=float)
    active, industry, coverage = apply_current_membership(quotes, day, active, industry, coverage, snapshot, disclosed)
    current = quotes.loc[active]
    confirmed = pd.Series(False, index=quotes.index)
    if quote_completion([str(path)], day, config["data"]["settled_after_cst"]):
        confirmed.loc[active] = trade_flags(current)["bar_ok"] | current["trade_status"].eq(0)
    verified = int(confirmed.sum())
    share = verified / len(snapshot)
    session = readiness(day, expected, phase, share >= config["data"]["minimum_member_coverage"], replay)
    session.update(verified_codes=verified, verified_share=share)
    audit = quote_audit(quotes, market.index)
    audit.update(created_utc=utc_now(), source="independent_daily_release", release_path=os.path.relpath(path, root),
                 manifest_sha256=sha256_file(path), config_sha256=config_hash, code_sha256=code_hashes,
                 archived_config_sha256=sha256_file(run / "config.json"), session_readiness=session,
                 dependency_versions={k: version(k) for k in ("numpy", "pandas", "pyarrow", "baostock", "requests", "xlrd")},
                 sources=meta["sources"], refresh=refresh, history_start=meta["history_start"],
                 current_financials=meta["current_financials"], missing_members=meta["missing_members"])
    from .quality_study import load_quality_history
    qh = load_quality_history(root / config["data"]["quality_history"])
    if qh is None and disclosed is not None and not disclosed.empty:
        qh = disclosed.copy()
        if "version_known_date" not in qh.columns and "ann_date" in qh.columns:
            qh["version_known_date"] = qh["ann_date"]
    features = build_features(quotes, market, active, industry, coverage, None, config, quality_history=qh)
    apply_current_quality(features.panel, day, snapshot, disclosed, config["eligibility"]["minimum_cash_conversion"],
                          config["eligibility"]["fundamental_max_age_days"],
                          profile=config["eligibility"].get("quality_profile", "value_cash_quality"))
    features.panel["closed_bar_verified"] = confirmed
    bundle = None
    configured_model = model_path or config.get("screening", {}).get("model_path") or None
    if configured_model:
        from .models import load_verified_model
        resolved_model = Path(configured_model)
        if not resolved_model.is_absolute():
            resolved_model = root / resolved_model
        bundle, model_digest = load_verified_model(resolved_model, features.columns)
        audit["screening_model"] = {"path": portable_path(resolved_model, root), "sha256": model_digest,
                                    "train_stats": bundle.train_stats}
    write_json(run / "audit.json", audit)
    picks, result, decisions = generate(features.panel, features.market, day, today, coverage, config, bundle, session)
    table = "research_watchlist.csv" if result["state"] == "RESEARCH_ONLY" else "quality_screen.csv"
    write_csv(run / table, picks)
    digests = {"raw": _selection_digest(picks)}
    spread = None
    if table == "quality_screen.csv":
        spread = diversified_screen(decisions, config)
        write_csv(run / "quality_screen_diversified.csv", spread)
        result["diversified_names"] = len(spread)
        digests["diversified"] = _selection_digest(spread)
    result["selection_sha256"] = digests
    if expected_selection is not None and digests != expected_selection:
        return _failed(output, run, replay, "回放的代码、顺序或分数与原名单不同", root=root, original=expected_selection, actual=digests)
    missing = snapshot.loc[~snapshot["code"].isin(decisions["code"]), ["code", "name"]]
    if not missing.empty:
        missing = missing.assign(date=day, member=True, eligible=False, quality_qualified=False, selected=False,
                                 run_state=result["state"], exclusion_reason="当日缺行情")
        decisions = pd.concat([decisions, missing], ignore_index=True)
    write_csv(run / "screening_universe.csv", decisions)
    write_csv(run / "coverage.csv", coverage.loc[[day]].rename_axis("date").reset_index())
    write_csv(run / "market.csv", features.market.reset_index())
    if not replay and result["state"] == "FUNDAMENTAL_SCREEN":
        cohorts = prospective_variants(decisions, picks, spread, config)
        write_csv(run / "prospective_cohorts.csv", cohorts)
        result["prospective"] = record_prospective(output, run, result, picks, audit, horizon=config["research"]["horizon"],
                                                   sessions=schedule, variants=cohorts)
    if not replay:
        result["prospective_observations"] = prospective_outcomes(output, run, quotes, market, schedule, config["research"]["horizon"])
    write_json(run / "screen.json", result)
    report_html(run / "report.html", result, audit, "独立行情批次 " + path.parent.name, picks, None, None, table,
                diversified_picks=spread)
    publish_report_pointer(output, make_pointer(root, run, result["state"], result["signal_date"]), replay)
    print(run / "report.html", flush=True)
    return 2 if result["state"] == "RESEARCH_ONLY" else 0
