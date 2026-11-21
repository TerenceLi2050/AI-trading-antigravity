import argparse
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

import pandas as pd

from .config import load_config
from .current import apply_current_membership, apply_current_quality, load_current
from .data import (
    apply_canonical_quotes,
    apply_raw_history,
    assemble_quotes,
    benchmark_for,
    load_research_base,
    load_status,
    market_calendar,
    quote_audit,
)
from .features import build_features
from .io import atomic_path, sha256_file, utc_now, write_csv, write_json
from .labels import executable_labels
from .market import trade_flags
from .pit import fundamentals, membership
from .protocol import lockbox_request, make_pointer, prospective_outcomes, publish_report_pointer, record_prospective
from .report import report_html
from .screen import diversified_screen, generate
from .sessions import CST, last_completed, load_sessions, quote_completion, readiness


def main(argv=None):
    parser = argparse.ArgumentParser(description="中证 800 自动化研究与量化初筛")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path)
    parser.add_argument("--mode", choices=["daily", "all", "screen", "audit"], default="daily")
    parser.add_argument("--replay-run", help="按原运行归档的独立行情、日历和参数回放，传入运行目录或编号")
    parser.add_argument("--import-release", type=Path, help="离线校验并导入独立行情批次，按归档行情和日历筛选")
    parser.add_argument("--watch", action="store_true", help="持续运行，收盘后自动出榜，数据未就绪时重试")
    parser.add_argument("--as-of", help="信号截止日 YYYY-MM-DD，默认使用最新完成的交易日")
    parser.add_argument("--refresh", action="store_true", help="更新公开行情，失败时使用已落盘数据")
    parser.add_argument("--refresh-disclosures", action="store_true", help="单独更新当期公开财报和成分快照")
    parser.add_argument("--probe", action="store_true", help="只执行公开数据源小样本检查")
    parser.add_argument("--refresh-calendar", action="store_true", help="刷新独立交易日历后结束")
    parser.add_argument("--collect-history", action="store_true", help="归档公开历史样本和全收益指数后结束")
    parser.add_argument("--collect-raw-history", action="store_true", help="补齐现有历史样本的原始行情和 ST 状态后结束")
    parser.add_argument("--audit-universe", action="store_true", help="核对开发期历史成分抽样和行情缺口")
    parser.add_argument("--collect-universe-prices", action="store_true", help="自动补采历史成分抽样缺口证券的原始及复权日线")
    parser.add_argument("--build-history", action="store_true", help="无旧 CSV 从零构建独立历史库（日历、指数、动态成分、多年行情、PIT 财报）")
    parser.add_argument("--assemble-local", action="store_true", help="配合 --build-history 使用：将本机已有分片/归档合成为独立 data/input/ 规范库")
    parser.add_argument("--history-start", help="历史建库起始日期 YYYY-MM-DD，默认取配置 data.history_start")
    parser.add_argument("--history-end", help="历史建库截止日期 YYYY-MM-DD，默认取最近完成交易日")
    parser.add_argument("--skip-bars", action="store_true", help="配合 --build-history 使用：跳过多年个股日线拉取")
    parser.add_argument("--skip-financials", action="store_true", help="配合 --build-history 使用：跳过多年 PIT 财报拉取")
    parser.add_argument("--bar-provider", choices=["auto", "baostock", "eastmoney", "akshare"], help="历史日线采集源（auto 自动在 BaoStock / Eastmoney / AkShare 间切换）")
    parser.add_argument("--fill-gap", nargs=2, metavar=("START", "END"), help="补采 data/input/quotes.parquet 指定区间 [START, END] 的缺失日线")
    parser.add_argument("--factor-research", action="store_true", help="在开发期动态无偏差池上对全量候选因子库执行单因子体检、相关性去重与消融赛马")
    parser.add_argument("--standalone", action="store_true", help="强制仅使用 data/input/quotes.parquet 独立规范库，不读取旧 CSV")
    parser.add_argument("--model", type=Path, help="日常选股加载已验证冻结模型 model.joblib 进行打分")
    parser.add_argument("--doctor", action="store_true", help="检查依赖、数据源退避、日常/历史就绪状态和报告指针")
    parser.add_argument("--lockbox", choices=["freeze", "evaluate"], help="独立冻结或执行已冻结的正式验证")
    args = parser.parse_args(argv)
    collection_commands = (args.probe, args.refresh_calendar, args.collect_history, args.collect_raw_history,
                           args.audit_universe, args.collect_universe_prices, args.build_history, args.fill_gap,
                           args.factor_research, args.doctor, args.refresh, args.refresh_disclosures)
    if args.import_release and (args.mode not in {"daily", "screen"} or args.replay_run or args.lockbox or args.watch
                                or any(collection_commands)):
        parser.error("--import-release is an offline daily screening operation")
    root = args.root.resolve()
    if args.config:
        policy_path = root / args.config
    else:
        import sysconfig
        options = [root / "configs/default.toml", Path(__file__).resolve().parents[2] / "configs/default.toml",
                   Path(sysconfig.get_path("data")) / "share/csi800-quant-screen/default.toml"]
        policy_path = next((p for p in options if p.is_file()), options[0])
    config = load_config(policy_path)
    data = config["data"]
    if args.doctor:
        import json
        from .doctor import run_doctor
        report = run_doctor(root, config)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return 0
    if args.watch:
        if args.mode != "daily" or args.as_of or args.replay_run or args.lockbox or any(collection_commands):
            parser.error("--watch requires live daily mode")
        from .watch import watch_daily
        watch_cmd = ["--root", str(root), "--config", str(policy_path), "--mode", "daily"]
        if args.model:
            watch_cmd += ["--model", str(args.model)]
        return watch_daily(lambda: main(watch_cmd), root / data["session_calendar"], data["settled_after_cst"],
                           output_dir=root / data["output"])
    now = datetime.now(CST)
    today = pd.Timestamp(now.date())
    requested = pd.Timestamp(args.as_of) if args.as_of else today
    replay = args.as_of is not None
    if args.lockbox and (args.mode != "all" or replay):
        parser.error("--lockbox requires --mode all with a fixed, verified live dataset")
    if requested > today or requested != requested.normalize():
        parser.error("--as-of must be a date no later than today in China Standard Time")
    output = root / data["output"]
    run = output / "runs" / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    run.mkdir(parents=True, exist_ok=False)
    code_fingerprints = {path.name: sha256_file(path) for path in sorted(Path(__file__).parent.glob("*.py"))}
    config_fingerprint = sha256_file(policy_path)
    write_json(run / "config.json", config)
    if args.import_release:
        import shutil
        from .daily import import_release
        from .daily_run import _failed, run_daily
        try:
            imported = import_release(args.import_release.resolve(), root / data["daily_updates"])
            schedule = load_sessions(imported.parent / "sessions.csv")
            if schedule is None:
                raise ValueError("Imported release lacks a trading calendar")
            calendar_path = root / data["session_calendar"]
            if not calendar_path.exists():
                with atomic_path(calendar_path) as temp:
                    shutil.copyfile(imported.parent / "sessions.csv", temp)
            write_json(run / "import.json", {"manifest_sha256": sha256_file(imported), "release": imported.parent.name})
        except (OSError, ValueError, TypeError, KeyError) as exc:
            return _failed(output, run, replay, str(exc))
        expected, phase = last_completed(schedule, requested, now, data["settled_after_cst"])
        return run_daily(root, output, run, config, config_fingerprint, code_fingerprints, schedule, requested,
                         expected, phase, today, False, replay)
    if args.lockbox == "evaluate" and (output / "lockbox_protocol.json").is_file():
        import json
        from .protocol import evaluate_frozen
        if "frozen_files" in json.loads((output / "lockbox_protocol.json").read_text(encoding="utf-8")):
            outcome = evaluate_frozen(output, run, code_fingerprints)
            write_json(run / "lockbox_request.json", outcome)
            print(run / "lockbox_request.json", flush=True)
            return 0 if outcome["state"] == "PASSED" else 2
    if args.replay_run and (args.refresh or args.refresh_disclosures or args.lockbox or args.as_of or args.mode not in {"daily", "screen"}):
        parser.error("--replay-run is an offline daily screening operation")
    if args.replay_run:
        from .daily_run import replay_daily
        return replay_daily(root, output, run, args.replay_run, code_fingerprints)
    session_path = root / data["session_calendar"]
    try:
        calendar = load_sessions(session_path)
    except (ValueError, OSError) as exc:
        calendar = None
        write_json(run / "calendar_error.json", {"error": str(exc)})
    if calendar is None and args.mode in {"daily", "screen"} and not replay:
        from .daily import available_releases
        archived, _ = available_releases(root / data["daily_updates"], requested)
        if archived is not None:
            calendar = load_sessions(archived.parent / "sessions.csv")
    universe_task = args.audit_universe or args.collect_universe_prices
    auto = args.mode == "daily" and not replay and not (args.probe or args.collect_history or args.collect_raw_history or universe_task)
    needs_cold_screen = args.mode == "screen" and not (root / data["legacy_csv"]).is_file() and not (root / data["cache"] / "legacy.json").is_file()
    if args.refresh_calendar or ((auto or args.refresh or needs_cold_screen) and not replay and (calendar is None or requested not in calendar.index)):
        from .history import download_sessions
        from .providers import _error

        try:
            try:
                calendar_status = download_sessions(session_path, pd.Timestamp("2019-03-01"), requested + pd.Timedelta(days=45))
            except ValueError:
                calendar_status = download_sessions(session_path, pd.Timestamp("2019-03-01"), requested)
            calendar = load_sessions(session_path)
        except Exception as exc:
            calendar_status = _error(exc)
        write_json(run / "calendar_refresh.json", calendar_status)
        if args.refresh_calendar:
            print(run / "calendar_refresh.json", flush=True)
            return int(calendar is None or requested not in calendar.index)
    expected, phase = last_completed(calendar, requested, now, data["settled_after_cst"])
    cutoff = expected if expected is not None else requested
    if args.probe:
        from .providers import probe
        result = probe(cutoff)
        write_json(run / "provider_probe.json", result)
        print(run / "provider_probe.json", flush=True)
        return 0
    if args.fill_gap:
        from .history_builder import fill_quote_gap
        g_start, g_end = pd.Timestamp(args.fill_gap[0]), pd.Timestamp(args.fill_gap[1])
        provider = args.bar_provider or data["bar_provider"]
        result = fill_quote_gap(root, config, g_start, g_end, provider=provider)
        write_json(run / "gap_fill.json", result)
        print(run / "gap_fill.json", flush=True)
        return 0 if result.get("status") == "published" else 2
    if args.build_history:
        from .history_builder import build_history
        h_start = pd.Timestamp(args.history_start or data["history_start"])
        h_end = pd.Timestamp(args.history_end) if args.history_end else cutoff
        provider = args.bar_provider or data["bar_provider"]
        result = build_history(
            root, config, h_start, h_end,
            bar_provider=provider, skip_bars=args.skip_bars,
            skip_financials=args.skip_financials, assemble_local=args.assemble_local,
        )
        write_json(run / "history_build.json", result)
        print(run / "history_build.json", flush=True)
        quotes_state = (result.get("quotes") or {}).get("status")
        return 2 if quotes_state == "failed" else 0
    if args.collect_history:
        from .history import collect_history, normalize_total_return
        sampled = root / "data" / "historical_observations" / run.name
        collected = collect_history(sampled, cutoff)
        total = sampled / "h00906_raw.csv"
        if total.is_file():
            metadata = normalize_total_return(pd.read_csv(total), root / data["total_return"])
            write_json(run / "total_return_collection.json", metadata)
        print(sampled / "manifest.json", flush=True)
        return 2 if any(item.get("status") == "failed" for item in collected["findings"].values()) else 0

    daily_folder = root / data["daily_updates"]
    has_research_source = (
        (root / data["legacy_csv"]).is_file()
        or (root / data["cache"] / "legacy.json").is_file()
        or (root / data["canonical_quotes"]).is_file()
    )
    if (not (args.collect_raw_history or universe_task or args.factor_research) and (args.mode == "daily" or
            (args.mode == "screen" and (args.refresh or replay or any(daily_folder.glob("*/manifest.json"))
              or not has_research_source)))):
        from .daily_run import run_daily
        return run_daily(root, output, run, config, config_fingerprint, code_fingerprints, calendar,
                         requested, expected, phase, today,
                         (auto or args.refresh or args.refresh_disclosures or not any(daily_folder.glob("*/manifest.json")))
                         and not replay, replay, force=args.refresh or args.refresh_disclosures,
                         model_path=args.model)

    print("Loading verified historical quote source", flush=True)
    quotes, legacy = load_research_base(root, data, prefer_canonical=args.standalone)
    if universe_task:
        from .universe_audit import run_universe_audit
        return run_universe_audit(root, run, quotes, config, args.collect_universe_prices)
    if args.collect_raw_history:
        from .history import collect_raw_history
        historical_end = min(cutoff, pd.Timestamp(config["research"]["lockbox_start"]) - pd.Timedelta(days=1))
        result = collect_raw_history(quotes["code"].unique(), quotes["date"].min(), historical_end, root / data["raw_history"])
        write_json(run / "raw_history_collection.json", result)
        print(run / "raw_history_collection.json", flush=True)
        return int(result["status"] != "published")
    known_by = (requested.tz_localize(CST) + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)) if replay else None
    quotes, releases = assemble_quotes(quotes, root / data["updates"], known_by)
    refresh_status = {"status": "not_requested"}
    if (args.refresh or args.refresh_disclosures) and not replay:
        from .providers import refresh, refresh_disclosures
        if expected is None:
            refresh_status = {"status": "failed", "error_type": "CalendarUnverified"}
        elif args.refresh:
            trusted = set(quote_completion(releases, cutoff, data["settled_after_cst"])) if not replay else None
            refresh_status = refresh(quotes, root / data["updates"], cutoff, trusted)
        else:
            refresh_status = refresh_disclosures(quotes, root / data["updates"], cutoff)
        write_json(run / "refresh.json", refresh_status)
        print(f"Data refresh: {refresh_status['status']}", flush=True)

    if refresh_status["status"] == "published":
        quotes, releases = assemble_quotes(load_research_base(root, data, prefer_canonical=args.standalone)[0], root / data["updates"])
    quotes = apply_canonical_quotes(quotes, root / data["canonical_quotes"])
    quotes = quotes.loc[quotes["date"].le(cutoff)].reset_index(drop=True)
    if quotes.empty:
        parser.error("No quotes available before the selected cutoff date")
    quotes, raw_history = apply_raw_history(quotes, root / data["raw_history"])
    quotes = load_status(quotes, root / data["status"])
    market = market_calendar(quotes, root / data["calendar"], root / data["total_return"]).loc[:cutoff]
    # Benchmark-only calendar rows can extend coverage checks. An actual stock quote day is needed for scoring.
    latest = quotes["date"].max()
    member, industry, coverage = membership(quotes, market.index, root / data["membership"])
    snapshot, disclosed = load_current(releases, latest)
    live_member, live_industry, live_coverage = apply_current_membership(quotes, latest, member, industry, coverage, snapshot, disclosed)
    confirmed = quotes["_release"].isin(quote_completion(releases, latest, data["settled_after_cst"]))
    active = quotes["date"].eq(latest) & live_member
    eligible_feed = quotes.loc[active]
    valid_feed = trade_flags(eligible_feed)["bar_ok"] | eligible_feed["trade_status"].eq(0)
    confirmed.loc[active] &= valid_feed
    expected_count = live_coverage.at[latest, "expected"]
    verified_count = int((confirmed & active).sum())
    verified_share = verified_count / expected_count if pd.notna(expected_count) and expected_count > 0 else 0
    session_state = readiness(latest, expected, phase, verified_share >= data["minimum_member_coverage"], replay)
    session_state.update(verified_codes=verified_count, verified_share=verified_share)
    audit = quote_audit(quotes, market.index)
    audit["legacy"] = legacy
    audit["raw_history"] = raw_history
    audit["releases"] = releases
    audit["refresh"] = refresh_status
    audit["config_sha256"] = config_fingerprint
    audit["inputs"] = {key: sha256_file(root / data[key]) if (root / data[key]).is_file() else None
                       for key in ("membership", "financials", "status", "calendar", "canonical_quotes",
                                   "session_calendar", "total_return", "quality_history", "raw_history")}
    audit["created_utc"] = utc_now()
    audit["session_readiness"] = session_state
    if calendar is not None:
        open_days = calendar.loc[market.index[0]:cutoff, "is_session"]
        audit["benchmark_missing_sessions"] = [str(date.date()) for date in open_days[open_days.eq(1)].index.difference(market.index)]
    audit["code_sha256"] = code_fingerprints
    dependencies = ("numpy", "pandas", "pyarrow", "scipy", "scikit-learn", "lightgbm", "catboost") if args.mode == "all" else ("numpy", "pandas", "pyarrow")
    audit["dependency_versions"] = {name: version(name) for name in dependencies}
    audit["same_session_official_snapshot"] = snapshot is not None
    audit["same_session_financial_snapshot_rows"] = len(disclosed) if disclosed is not None else 0
    write_csv(run / "coverage.csv", live_coverage.rename_axis("date").reset_index())
    write_json(run / "audit.json", audit)
    if args.factor_research:
        from .factor_tournament import run_factor_tournament
        from .quality_study import load_quality_history
        qh = load_quality_history(root / data["quality_history"])
        summary = run_factor_tournament(quotes, market, member, industry, coverage, qh, config, run)
        print(run / "factor_selected.json", flush=True)
        return 0
    if args.mode == "audit":
        print(run / "audit.json", flush=True)
        return 0

    print("Building daily factors and disclosure-aligned quality features", flush=True)
    finance = fundamentals(quotes, root / data["financials"], config["eligibility"]["fundamental_max_age_days"])
    from .quality_study import load_quality_history
    history = load_quality_history(root / data["quality_history"])
    features = build_features(quotes, market, live_member, live_industry, live_coverage, finance, config, quality_history=history)
    apply_current_quality(features.panel, latest, snapshot, disclosed, config["eligibility"]["minimum_cash_conversion"],
                          config["eligibility"]["fundamental_max_age_days"],
                          profile=config["eligibility"].get("quality_profile", "value_cash_quality"))
    write_csv(run / "market.csv", features.market.reset_index())
    settings = config["research"]
    lockbox = pd.Timestamp(settings["lockbox_start"])
    if args.mode == "all":
        import joblib
        from .models import samples, train, training_sample
        from .quality_study import load_quality_history
        from .study import run_development
        development = quotes.loc[quotes["date"].lt(lockbox)]
        dev_market = benchmark_for(market.loc[market.index < lockbox])
        labels = executable_labels(development, dev_market, settings["horizon"])
        audit["development_eligible_labels"] = {name: int(count) for name, count in
            labels.loc[features.panel.loc[labels.index, "eligible"], "label_status"].value_counts().items()}
    if args.mode == "all" and settings["exit_delay_sensitivity"]:
        delayed = executable_labels(development, dev_market, settings["horizon"], settings["exit_delay_sensitivity"])
        eligible = features.panel.loc[labels.index, "eligible"]
        recovered = eligible & labels["label_status"].eq("exit_blocked_or_missing") & delayed["label_status"].eq("realized")
        audit["delayed_exit_sensitivity"] = {"max_delay_sessions": settings["exit_delay_sensitivity"],
                                             "recovered_labels": int(recovered.sum()),
                                             "mean_recovered_return": delayed.loc[recovered, "forward_return"].mean()}
    write_json(run / "audit.json", audit)
    bundle = None
    if args.mode == "all":
        sample = samples(features.panel.loc[labels.index], labels, features.columns, settings)
        model_date = min(latest, lockbox)
        trainable = training_sample(sample, market.index, model_date, settings)
        if trainable["date"].nunique() >= settings["min_train_dates"]:
            print(f"Fitting frozen screening model through {model_date.date()}", flush=True)
            bundle = train(trainable, features.columns, model_date, settings)
            with atomic_path(run / "model.joblib") as temp:
                joblib.dump(bundle, temp, compress=3)
            write_json(run / "model.json", {**bundle.train_stats, "features": bundle.columns})

    features.panel["closed_bar_verified"] = confirmed
    picks, screen, decisions = generate(features.panel, features.market, latest, today, live_coverage, config, bundle, session_state)
    table_file = "quality_screen.csv" if screen["state"] in {"FUNDAMENTAL_SCREEN", "HISTORICAL_SCREEN"} else "research_watchlist.csv"
    write_csv(run / table_file, picks)
    diversified = None
    if table_file == "quality_screen.csv":
        diversified = diversified_screen(decisions, config)
        write_csv(run / "quality_screen_diversified.csv", diversified)
        screen["diversified_names"] = len(diversified)
    write_csv(run / "screening_universe.csv", decisions)
    if screen["state"] == "FUNDAMENTAL_SCREEN" and not replay and args.mode == "screen":
        screen["prospective"] = record_prospective(output, run, screen, picks, audit, horizon=settings["horizon"], sessions=calendar)
    if not replay and args.mode == "screen":
        screen["prospective_observations"] = prospective_outcomes(output, run, quotes, market, calendar, settings["horizon"])
    if args.mode == "all":
        screen["publication_scope"] = "development_research"
    write_json(run / "screen.json", screen)
    research, scores, quality_diag = None, None, None
    if args.mode == "all":
        import json
        research, scores = run_development(quotes, features, sample, coverage, config, run, history,
                                          audit.get("benchmark_missing_sessions", []))
        if (run / "quality_research.json").is_file():
            quality_diag = json.loads((run / "quality_research.json").read_text(encoding="utf-8"))
        if args.lockbox:
            research["lockbox_request"] = lockbox_request(args.lockbox, research, audit, bundle, quotes, features, config,
                                                          coverage, output, run)
        write_json(run / "research.json", research)
    message = "使用已落盘的行情"
    if refresh_status["status"] == "published":
        message = "行情更新已完成" if refresh_status["rows"] else "当期成分和财报已更新"
    if refresh_status["status"] == "failed":
        message += "，公开源更新失败"
    report_html(run / "report.html", screen, audit, message, picks, research, scores, table_file,
                diversified_picks=diversified, quality_research=quality_diag)
    pointer = make_pointer(root, run, screen["state"], screen["signal_date"])
    if args.mode == "all":
        write_json(output / "latest_research.json", {**pointer, "scope": "development_research"})
    else:
        publish_report_pointer(output, pointer, replay)
    print(run / "report.html", flush=True)
    if args.mode == "all":
        return 0
    return 0 if screen["state"] != "RESEARCH_ONLY" else 2
