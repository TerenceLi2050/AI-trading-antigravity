from math import ceil
from pathlib import Path

import pandas as pd

from .backtest import performance, simulate, yearly_performance
from .data import benchmark_for
from .features import BASELINE_GROUPS, EXTENDED_PRICE_FACTORS, PIT_VALUATION_FACTORS
from .io import write_csv, write_json
from .models import active_strategies, ModelSet, score, walk_forward
from .quality_study import run_quality_comparison
from .report import nav_plot
from .validation import deflated_sharpe_and_pbo, permutation_diagnostic, purged_diagnostic, rank_ic, ridge_family_ablation


def run_development(quotes, features, development_sample, coverage, config, output: Path,
                    quality_history=None, missing_sessions=()):
    settings = config["research"]
    market = benchmark_for(features.market.loc[features.market.index < settings["lockbox_start"]])
    predictions, fitted = walk_forward(development_sample, features.market.index, features.columns, settings)
    fits = [bundle.train_stats for _, bundle in fitted]
    if predictions.empty:
        quality = run_quality_comparison(quotes, features, market, predictions, development_sample, coverage,
                                         quality_history, config, output)
        return {"lockbox_state": "SEALED", "research_gate_reasons": ["可训练的开发区间不足"], "fits": fits,
                "quality_comparison": quality["state"]}, pd.DataFrame()
    strategies = active_strategies(settings)
    keys = ["date", "code"]
    realized = predictions.merge(development_sample[[*keys, *features.columns, "excess_return"]], on=keys, validate="one_to_one")
    lag = max(1, ceil(settings["horizon"] / settings["sample_every"]))
    model_stats, model_ic = rank_ic(realized, strategies, config["data"]["minimum_cross_section"], lag)
    factor_stats, _ = rank_ic(realized, features.columns, config["data"]["minimum_cross_section"], lag)
    cv, cv_paths = purged_diagnostic(development_sample, features.columns, features.market.index, settings, build_paths=True)
    write_csv(output / "signal_diagnostics.csv", model_stats)
    write_csv(output / "signal_ic_timeseries.csv", model_ic)
    write_csv(output / "factor_diagnostics.csv", factor_stats)
    factor_corr = realized[features.columns].corr().rename_axis("signal").reset_index()
    write_csv(output / "factor_correlations.csv", factor_corr)
    write_csv(output / "purged_cv.csv", cv)
    summaries, path_nav = [], []
    for name, path in cv_paths.items():
        validation_path = path.loc[path["date"].ge(settings["oos_start"])]
        ic, _ = rank_ic(validation_path, ["ridge"], config["data"]["minimum_cross_section"], lag)
        nav, _ = simulate(quotes, market, validation_path, features.panel, "ridge", settings, config["portfolio"])
        summaries.append({"path": name, "rank_ic": ic["rank_ic"].iat[0], "facets": len(validation_path["date"].unique()),
                          **performance(nav)})
        if not nav.empty:
            path_nav.append(nav.assign(path=name))
    write_csv(output / "cpcv_paths.csv", pd.DataFrame(summaries))
    if path_nav:
        write_csv(output / "cpcv_path_nav.csv", pd.concat(path_nav, ignore_index=True))
    importance = permutation_diagnostic(fitted, development_sample, config["data"]["minimum_cross_section"], settings["seed"])
    write_csv(output / "permutation_diagnostics.csv", importance)
    families = {name: {f"rank_{column}" for column in columns}
                for name, columns in zip(("momentum", "risk", "liquidity", "reversal_and_structure"), BASELINE_GROUPS)}
    ext_cols = {f"rank_{col}" for col in EXTENDED_PRICE_FACTORS if f"rank_{col}" in features.columns}
    if ext_cols:
        families["extended_microstructure"] = ext_cols
    pit_cols = {f"rank_{col}" for col in PIT_VALUATION_FACTORS if f"rank_{col}" in features.columns}
    if pit_cols:
        families["pit_valuation_quality"] = pit_cols
    if "ridge" in fitted[0][1].models:
        ablation = ridge_family_ablation(fitted, development_sample, features.market.index, settings, families,
                                         config["data"]["minimum_cross_section"])
        write_csv(output / "factor_family_ablation.csv", ablation)

    paths, performance_rows, fills = {}, {}, pd.DataFrame()
    default_alloc = config["portfolio"].get("default_allocation", "equal_weight")
    variants = {"ensemble_equal_weight": "equal_weight", "ensemble_inverse_volatility": "inverse_volatility", "ensemble_hrp": "hrp"}
    for strategy in strategies + ["ensemble_2x_cost", *variants]:
        name = strategy if strategy in strategies else "ensemble"
        allocation = variants.get(strategy, default_alloc)
        multiplier = 2.0 if strategy == "ensemble_2x_cost" else 1.0
        nav, ledger = simulate(quotes, market, predictions, features.panel, name, settings,
                               config["portfolio"], multiplier, allocation)
        paths[strategy] = nav
        performance_rows[strategy] = performance(nav)
        if strategy == "ensemble":
            fills = ledger
        print(f"portfolio {strategy}: {len(ledger):,} fills", flush=True)
    write_csv(output / "portfolio_nav.csv", pd.concat(paths, names=["strategy"]).reset_index(level=0).reset_index(drop=True))
    write_csv(output / "fills.csv", fills)
    yearly = yearly_performance(paths)
    if not yearly.empty:
        write_csv(output / "yearly_performance.csv", yearly)
    nav_plot(output / "development_nav.png", paths, bool(market["benchmark_total_return"].all()))
    multiple_testing = deflated_sharpe_and_pbo({name: paths[name] for name in strategies})
    quality = run_quality_comparison(quotes, features, market, predictions, realized, coverage, quality_history, config, output)
    write_json(output / "experiment_manifest.json", {
        "development_cutoff": settings["lockbox_start"], "selection_trials": strategies,
        "sensitivity_only": ["ensemble_2x_cost", *variants], "cpcv_ridge_paths": list(cv_paths),
        "ridge_factor_family_ablation": list(families),
        "population": "registered strategies from this run; earlier notebook trials are uncounted",
        "estimator_parameters": {name: model.get_params() for name, model in fitted[0][1].models.items()},
        "quality_comparison": quality["state"],
    })

    oos = quotes.loc[quotes["date"].between(predictions["date"].min(), market.index[-1])]
    covered = coverage.loc[coverage.index.intersection(market.loc[predictions["date"].min():].index), "coverage"]
    cfg = config["data"]
    gate_reasons = []
    prior_date = settings.get("prior_notebook_signal")
    if prior_date and pd.Timestamp(settings["lockbox_start"]) <= pd.Timestamp(prior_date) <= features.market.index.max():
        gate_reasons.append(f"旧版 Notebook 已在预留区间的 {prior_date} 展示选股结果，整段独立性无法确认")
    if not (covered.ge(cfg["minimum_member_coverage"])
            & coverage.loc[covered.index, "expected"].ge(cfg["minimum_member_count"])).all():
        gate_reasons.append("历史成分股覆盖未达到门槛，开发回测存在股票池偏差")
    if not market["benchmark_total_return"].all():
        gate_reasons.append("基准只有价格指数，缺中证 800 全收益指数序列")
    if any(str(market.index[0].date()) <= day <= str(market.index[-1].date()) for day in missing_sessions):
        gate_reasons.append("独立交易日历存在缺失指数行情的开发交易日")
    if (oos["is_st"].notna() & oos["trade_status"].notna()).mean() < 0.95:
        gate_reasons.append("历史 ST 和交易状态覆盖低于 95%")
    if oos["price_source"].ne("legacy_reconstructed").mean() < 0.95:
        gate_reasons.append("可追溯的不复权开盘行情覆盖低于 95%")
    if multiple_testing["dsr_ensemble"] is None or multiple_testing["dsr_ensemble"] < 0.95:
        gate_reasons.append("集成策略 DSR 低于预设的 0.95 或样本不足")
    if multiple_testing["pbo"] is None or multiple_testing["pbo"] >= 0.50:
        gate_reasons.append("PBO 未达到低于 0.50 的门槛或样本不足")
    model_rank_ic = model_stats.set_index("signal").loc["ensemble", "rank_ic"]
    ensemble = performance_rows["ensemble"]
    baseline = performance_rows["baseline"]
    if ensemble.get("max_unpriced_weight", 1) > 0.05:
        gate_reasons.append("组合曾有超过 5% 净值的持仓只能沿用旧估值")
    if not (model_rank_ic > 0 and ensemble.get("annual_excess", 0) > max(0, baseline.get("annual_excess", 0))
            and performance_rows["ensemble_2x_cost"].get("annual_excess", 0) > 0):
        gate_reasons.append("集成策略未同时通过正 RankIC、超过规则基线和双倍成本费后超额门槛")
    summary = {"lockbox_state": "SEALED" if gate_reasons else "READY", "research_gate_reasons": gate_reasons,
               "first_oos_signal": str(predictions["date"].min().date()), "last_oos_signal": str(predictions["date"].max().date()),
               "last_oos_realized_signal": str(realized.loc[realized["excess_return"].notna(), "date"].max().date()),
               "backtest_end": str(market.index[-1].date()), "fits": fits,
               "performance": performance_rows, "multiple_testing": multiple_testing,
               "cpcv_splits": int(cv["scheme"].eq("cpcv_split").sum()) if not cv.empty else 0,
               "cpcv_paths": len(summaries), "quality_comparison": quality["state"],
               "label_open_benchmark": "CSI 800 total return" if market["benchmark_open_total_return"].all() else "CSI 800 price index",
               "fixed_model_trials": strategies, "holding_period_sessions": settings["horizon"],
               "benchmark": "CSI 800 total return" if market["benchmark_total_return"].all() else "CSI 800 price index",
               "labels_executable_share": float(realized["excess_return"].notna().mean())}
    return summary, model_stats


def evaluate_locked(quotes, features, config, bundle: ModelSet, coverage, out_root: Path, run: Path) -> str:
    record = out_root / "lockbox_once.json"
    if record.exists():
        return "PREVIOUSLY_CLAIMED"
    settings = config["research"]
    start = pd.Timestamp(settings["lockbox_start"])
    market = benchmark_for(features.market.loc[start:])
    dates = market.index
    if len(dates) < 126:
        return "SEALED_INSUFFICIENT_PERIOD"
    lock_quotes = quotes.loc[quotes["date"].ge(start)]
    heads = coverage["coverage"].reindex(dates)
    verified = (lock_quotes["trade_status"].notna() & lock_quotes["is_st"].notna()
                & lock_quotes["price_source"].ne("legacy_reconstructed"))
    members = coverage["expected"].reindex(dates)
    if (not (heads.ge(config["data"]["minimum_member_coverage"]) & members.ge(config["data"]["minimum_member_count"])).all()
            or not market.loc[dates, "benchmark_total_return"].all() or verified.mean() < 0.95):
        return "SEALED_INSUFFICIENT_DATA"
    # Creation is exclusive. Repeated runs cannot silently replace a completed or interrupted evaluation.
    try:
        with record.open("x", encoding="utf-8") as claim:
            claim.write('{"state":"CLAIMED"}\n')
    except FileExistsError:
        return "PREVIOUSLY_CLAIMED"
    try:
        signal_dates = dates[::settings["sample_every"]]
        rows = features.panel.loc[features.panel["date"].isin(signal_dates) & features.panel["eligible"]]
        predictions = score(rows, bundle)
        nav, fills = simulate(quotes, market, predictions, features.panel, "ensemble", settings, config["portfolio"])
        stress, _ = simulate(quotes, market, predictions, features.panel, "ensemble", settings, config["portfolio"], 2)
        write_csv(run / "lockbox_nav.csv", nav)
        write_csv(run / "lockbox_fills.csv", fills)
        normal, stressed = performance(nav), performance(stress)
        passed = (normal.get("annual_excess", 0) > 0 and stressed.get("annual_excess", 0) > 0
                  and normal.get("max_unpriced_weight", 1) <= 0.05 and stressed.get("max_unpriced_weight", 1) <= 0.05)
        outcome = "PASSED" if passed else "FAILED"
        write_json(record, {"state": outcome, "run": str(run), "portfolio": normal, "2x_cost": stressed,
                            "fixed_model": bundle.train_stats})
        return outcome
    except Exception as error:
        write_json(record, {"state": "INTERRUPTED", "run": str(run), "reason": type(error).__name__})
        raise
