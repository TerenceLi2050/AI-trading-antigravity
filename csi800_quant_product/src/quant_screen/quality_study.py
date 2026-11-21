from pathlib import Path

import pandas as pd

from .backtest import performance, simulate
from .current import ytd_quality
from .io import dates, numeric, read_table, require, stock_codes, write_csv, write_json
from .validation import rank_ic


QUALITY_FIELDS = ["roe_ytd_pct", "book_per_share", "profit_ytd", "eps_ytd", "operating_cash_per_share_ytd",
                  "profit_growth_pct", "revenue_growth_pct"]
SCORES = ["quality_only", "rules_same_pool", "quality_mix", "ridge", "lgbm_huber", "ensemble"]


def load_quality_history(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    frame = read_table(path)
    stamp = ["report_date", "ann_date", "version_known_date"]
    keys = ["code", "report_date", "version_known_date"]
    fields = ["code", *stamp, "source", "version", *QUALITY_FIELDS]
    require(frame, fields, keys, "versioned YTD history")
    frame["code"] = stock_codes(frame["code"])
    for name in stamp:
        frame[name] = dates(frame[name])
    numeric(frame, QUALITY_FIELDS)
    require(frame, fields, keys, "normalized YTD history")
    if (frame[stamp].isna().any().any() or not frame["report_date"].dt.is_quarter_end.all()
            or not frame["ann_date"].ge(frame["report_date"]).all()
            or not frame["version_known_date"].ge(frame["ann_date"]).all()
            or frame[["source", "version"]].astype("string").fillna("").apply(lambda x: x.str.strip().eq("")).any().any()):
        raise ValueError("Historical quality rows need original publication and version availability dates")
    return frame.sort_values("version_known_date")


def historical_quality_signals(frame: pd.DataFrame, history: pd.DataFrame, config: dict):
    """Score each cross-section with the latest common period already visible at that time."""
    settings, data = config["eligibility"], config["data"]
    output, coverage = [], []
    for date, section in frame.groupby("date", sort=True):
        known = history.loc[history["version_known_date"].lt(date)
                            & history["report_date"].ge(date - pd.Timedelta(days=settings["fundamental_max_age_days"]))]
        section = section.loc[section["eligible"] & section["industry"].notna() & section["is_st"].eq(0)
                              & section["price_source"].ne("legacy_reconstructed")].copy()
        max_complete = 0
        qualified_count, selected_period = 0, pd.NaT
        for period in sorted(known["report_date"].unique(), reverse=True):
            values = known.loc[known["report_date"].eq(period)].drop_duplicates("code", keep="last")
            view = section.join(values.set_index("code")[QUALITY_FIELDS], on="code", validate="one_to_one")
            complete, ranks = ytd_quality(
                view, pd.Series(True, index=view.index), settings["minimum_cash_conversion"],
                profile=settings.get("quality_profile", "value_cash_quality"),
            )
            max_complete = max(max_complete, int(complete.sum()))
            if len(view) < data["minimum_cross_section"] or complete.mean() < data["minimum_fundamental_coverage"]:
                continue
            selected_period = pd.Timestamp(period)
            quality = ranks.ge(settings["minimum_quality_rank"])
            qualified_count = int(quality.sum())
            valid = view.loc[quality, ["date", "code", "baseline", "industry"]].copy()
            valid["report_date"] = selected_period
            valid["quality_only"] = ranks[quality]
            valid["rules_same_pool"] = valid["baseline"]
            weight = settings["quality_weight"]
            valid["quality_mix"] = weight * valid["quality_only"] + (1 - weight) * valid["rules_same_pool"]
            if qualified_count >= config["portfolio"]["top_k"]:
                output.append(valid)
            break
        coverage.append({"date": date, "universe_rows": len(section), "complete_rows": max_complete,
                         "quality_rows": qualified_count, "report_date": selected_period})
    scores = pd.concat(output, ignore_index=True) if output else pd.DataFrame()
    return scores, pd.DataFrame(coverage)


def run_quality_comparison(quotes, features, market, predictions, realized, coverage, history, config, output):
    settings = config["research"]
    state = {"state": "BLOCKED", "reasons": [], "rule": "versioned_ytd_quality_mix",
             "calendar_scope": "development_only"}
    if history is None:
        state["reasons"].append("缺逐公告版本的年内累计指标 quality_history.csv")
    if coverage["expected"].isna().all():
        state["reasons"].append("缺可核验的历史有效成分和行业")
    if state["reasons"]:
        write_json(output / "quality_research.json", state)
        return state
    columns = [c for c in ("date", "code", "eligible", "industry", "is_st", "price_source", "baseline", "raw_close")
               if c in features.panel.columns]
    dates_to_score = (pd.DatetimeIndex(predictions["date"].unique()) if not predictions.empty
                      else market.index[::settings["sample_every"]])
    dates_to_score = dates_to_score[(dates_to_score >= pd.Timestamp(settings["oos_start"]))
                                    & (dates_to_score < pd.Timestamp(settings["lockbox_start"]))]
    observed = coverage.get("coverage")
    if observed is None:
        panel = features.panel
        if "member" in panel:
            panel = panel.loc[panel["member"]]
        observed = panel.groupby("date")["code"].nunique().div(coverage["expected"])
    covered = coverage["expected"].ge(config["data"]["minimum_member_count"]) & observed.ge(config["data"]["minimum_member_coverage"])
    dates_to_score = dates_to_score.intersection(coverage.index[covered.fillna(False)])
    panel = features.panel.loc[features.panel["date"].isin(dates_to_score), columns]
    ranks, dates_checked = historical_quality_signals(panel, history, config)
    write_csv(output / "quality_research_coverage.csv", dates_checked)
    if ranks.empty:
        state["reasons"].append("历史截面财务或交易状态覆盖不足")
        write_json(output / "quality_research.json", state)
        return state
    common = ranks
    strategies = SCORES[:3]
    if not predictions.empty:
        common = ranks.merge(predictions.drop(columns="baseline"), on=["date", "code"], validate="one_to_one")
        model_names = [m for m in ("ridge", "lgbm_rank", "lgbm_huber", "catboost") if m in predictions.columns]
        if model_names:
            common["ensemble"] = common.groupby("date")[model_names].rank(pct=True).mean(axis=1)
        strategies = [s for s in SCORES if s in common.columns]
    date_count = common["date"].nunique()
    if date_count < settings["min_train_dates"]:
        state["reasons"].append("共同可比的历史评分日期不足")
        state["signal_dates"] = date_count
        write_json(output / "quality_research.json", state)
        return state
    metrics, diversified_metrics, navs, ledgers = {}, {}, [], []
    # The published raw list is the exact Top K. No retention buffer or implicit industry/financial cap.
    portfolio = {**config["portfolio"], "keep_rank": config["portfolio"]["top_k"],
                 "max_industry_names": config["portfolio"]["top_k"],
                 "max_financial_names": config["portfolio"]["top_k"]}
    diversified_portfolio = {**config["portfolio"], "keep_rank": config["portfolio"]["top_k"]}
    for name in strategies + ["quality_mix_2x_cost"]:
        strategy = "quality_mix" if name.endswith("_2x_cost") else name
        nav, ledger = simulate(quotes, market, common, features.panel, strategy, settings, portfolio,
                               2.0 if name.endswith("_2x_cost") else 1.0)
        metrics[name] = performance(nav)
        navs.append(nav.assign(strategy=name))
        ledgers.append(ledger.assign(strategy=name))
    for name, mult in (("quality_mix_diversified", 1.0), ("quality_mix_diversified_2x_cost", 2.0)):
        nav, ledger = simulate(quotes, market, common, features.panel, "quality_mix", settings, diversified_portfolio, mult)
        diversified_metrics[name] = performance(nav)
        navs.append(nav.assign(strategy=name))
        ledgers.append(ledger.assign(strategy=name))
    free = panel.loc[panel["date"].isin(common["date"]) & panel["eligible"] & panel["is_st"].eq(0)
                     & panel["price_source"].ne("legacy_reconstructed")].rename(columns={"baseline": "price_all"})
    nav, ledger = simulate(quotes, market, free, features.panel, "price_all", settings, portfolio)
    controls = {"price_all": performance(nav)}
    navs.append(nav.assign(strategy="price_all"))
    ledgers.append(ledger.assign(strategy="price_all"))
    write_csv(output / "quality_strategy_nav.csv", pd.concat(navs, ignore_index=True))
    write_csv(output / "quality_strategy_fills.csv", pd.concat(ledgers, ignore_index=True))
    if not realized.empty:
        diagnostic = common.merge(realized[["date", "code", "excess_return"]], on=["date", "code"], validate="one_to_one")
        ic, _ = rank_ic(diagnostic, strategies, max(20, config["portfolio"]["top_k"]),
                        max(1, (settings["horizon"] + settings["sample_every"] - 1) // settings["sample_every"]))
        write_csv(output / "quality_strategy_rank_ic.csv", ic)
    write_csv(output / "quality_strategy_scores.csv", common[["date", "code", "report_date", *strategies]])
    state.update(state="DEVELOPMENT_DIAGNOSTIC", performance=metrics, diversified_performance=diversified_metrics,
                 signal_dates=date_count, full_universe_control=controls, model_comparison=not predictions.empty,
                 first_signal=str(common["date"].min().date()), last_signal=str(common["date"].max().date()))
    write_json(output / "quality_research.json", state)
    return state
