from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .models import ModelSet


OUTPUT_FIELDS = ["rank", "date", "code", "name", "industry", "raw_close", "adv20", "vol60", "close_to_high",
                 "baseline", "ridge", "lgbm_rank", "lgbm_huber", "catboost", "ensemble", "quality_score",
                 "final_score", "quality_component", "price_component", "fundamentals_ok", "report_date", "report_ann_date", "visible_after",
                 "report_source", "report_version", "fundamental_basis", "profit_ytd", "roe_ytd_pct", "book_per_share",
                 "operating_cash_per_share_ytd", "is_st", "price_source", "review_status", "risk_flags",
                 "tree_contributions"]


def generate(panel: pd.DataFrame, market: pd.DataFrame, day: pd.Timestamp, today: pd.Timestamp,
             coverage: pd.DataFrame, config: dict, bundle: ModelSet | None, session_state: dict | None = None):
    current = panel.loc[panel["date"].eq(day)].copy()
    candidates = current.loc[current["eligible"] & ~current["negative_equity"] & ~current["ttm_loss"] & ~current["ytd_loss"]].copy()
    candidates["final_score"] = candidates["baseline"]
    if bundle is not None:
        from .models import explain, score
        raw = score(candidates, bundle) if not candidates.empty else pd.DataFrame(index=candidates.index)
        model_columns = raw.columns.difference(["date", "code", "baseline"])
        candidates[model_columns] = raw[model_columns]
        if "ensemble" in candidates:
            candidates["final_score"] = candidates["ensemble"]

    replay = session_state is not None and session_state["scope"] == "historical_replay"
    reasons = [] if session_state is None else list(session_state["reasons"])
    if session_state is None:
        stale = (today - day).days > config["data"]["maximum_staleness_days"]
    else:
        wanted = session_state["expected_session"]
        stale = not replay and wanted is not None and str(day.date()) < wanted
    if stale:
        reasons.append(f"行情滞后 {(today - day).days} 个自然日")
    if day < market.index[-1]:
        reasons.append("有更新的指数交易日缺个股报价")
    active_coverage = coverage.at[day, "coverage"]
    active_count = coverage.at[day, "expected"]
    if (pd.isna(active_coverage) or active_coverage < config["data"]["minimum_member_coverage"]
            or active_count < config["data"]["minimum_member_count"]):
        reasons.append("当期指数成分历史及覆盖率未通过")
    qualified = (candidates["fundamentals_ok"] & candidates["quality_score"].ge(config["eligibility"]["minimum_quality_rank"])
                 & candidates["is_st"].eq(0) & candidates["industry"].notna()
                 & candidates["price_source"].ne("legacy_reconstructed") & np.isfinite(candidates["baseline"]))
    if session_state is not None:
        qualified &= candidates["closed_bar_verified"]
    if len(candidates) < config["data"]["minimum_cross_section"]:
        reasons.append("可评分横截面不足")
    eligible_count = int(current["eligible"].sum())
    financial_count = int((current["eligible"] & current["fundamentals_ok"]).sum())
    if not eligible_count or financial_count / eligible_count < config["data"]["minimum_fundamental_coverage"]:
        reasons.append("当期财务质量字段的覆盖不足")
    if qualified.sum() < config["portfolio"]["top_k"]:
        reasons.append("具备完整财报、交易风险状态和原始行情的质量候选不足")
    official = not reasons
    state = ("HISTORICAL_SCREEN" if replay else "FUNDAMENTAL_SCREEN") if official else "RESEARCH_ONLY"
    candidates["review_status"] = state
    model_weight = float(config.get("screening", {}).get("model_weight", 0.0))
    use_model = bundle is not None and "ensemble" in candidates.columns and candidates["ensemble"].notna().any() and model_weight > 0
    if official:
        candidates = candidates.loc[qualified].copy()
        weight = config["eligibility"]["quality_weight"]
        price_signal = (
            (1 - model_weight) * candidates["baseline"] + model_weight * candidates["ensemble"]
            if use_model else candidates["baseline"]
        )
        candidates["price_signal"] = price_signal
        candidates["quality_component"] = weight * candidates["quality_score"]
        candidates["price_component"] = (1 - weight) * price_signal
        candidates["final_score"] = candidates["quality_component"] + candidates["price_component"]
    top = candidates.sort_values(["final_score", "code"], ascending=[False, True]).head(config["portfolio"]["top_k"]).copy()
    risks = []
    for row in top.itertuples():
        issues = []
        if stale:
            issues.append("行情滞后")
        if not row.fundamentals_ok:
            issues.append("财务未验证")
        if pd.isna(row.is_st):
            issues.append("ST 状态未知")
        if pd.isna(row.industry):
            issues.append("历史行业未知")
        if row.price_source == "legacy_reconstructed":
            issues.append("原始开盘价由复权比例重建")
        if row.close_to_high < -0.3:
            issues.append("较过去一年高点回撤逾30%")
        if row.vol60 * np.sqrt(252) > 0.6:
            issues.append("60日年化波动逾60%")
        risks.append("、".join(issues))
    top["risk_flags"] = risks
    top["rank"] = np.arange(1, len(top) + 1)
    if bundle is not None:
        preferred = config.get("research", {}).get("explain_model", "auto")
        top["tree_contributions"] = explain(panel.loc[top.index], bundle, preferred=preferred)
    market_row = market.loc[day]
    audit = {"signal_date": str(day.date()), "run_date": str(today.date()), "state": state, "session": session_state,
             "reasons": reasons, "quoted_names": len(current), "eligible_names": int(current["eligible"].sum()),
             "watchlist_names": len(top), "quality_qualified_names": int(qualified.sum()),
             "fundamentally_covered_names": int(current["fundamentals_ok"].sum()), "financially_covered_eligible_names": financial_count,
             "excluded_ttm_loss": int((current["eligible"] & current["ttm_loss"]).sum()),
             "excluded_ytd_loss": int((current["eligible"] & current["ytd_loss"]).sum()),
             "excluded_negative_equity": int((current["eligible"] & current["negative_equity"]).sum()),
             "csi800_close": float(market_row["market_close"]), "index_return_20": market_row["return_20"],
             "index_annualized_volatility_60": market_row["annualized_vol_60"],
             "index_above_ma120": bool(market_row["above_ma120"]), "positive_momentum_breadth": market_row["positive_momentum_breadth"],
             "current_membership_coverage": active_coverage, "current_expected_members": active_count,
             "quality_weight": config["eligibility"]["quality_weight"],
             "model_active": use_model, "model_weight": model_weight if use_model else 0.0}
    logs = current.reindex(columns=["date", "code", "name", "member", "industry", "bar_ok", "closed_bar_verified", "is_st", "price_source", "raw_close",
                                    "tradable_history", "valid_bars_60", "adv20", "recent_extreme_move", "eligible",
                                    "fundamentals_ok", "fundamental_basis", "negative_equity", "ttm_loss", "ytd_loss", "quality_score", "baseline"])
    if use_model:
        logs["ensemble"] = candidates["ensemble"].reindex(logs.index)
        logs["price_signal"] = candidates["price_signal"].reindex(logs.index)
    else:
        logs["price_signal"] = logs["baseline"]
    logs["quality_qualified"] = qualified.reindex(logs.index, fill_value=False)
    logs["selected"] = logs.index.isin(top.index)
    logs["run_state"] = audit["state"]
    rules = config["eligibility"]
    logs["exclusion_reason"] = np.select(
        [~current["member"], ~current["bar_ok"], current["is_st"].eq(1), ~current["tradable_history"],
         current["adv20"].lt(rules["min_median_amount"]), current["raw_close"].lt(rules["min_raw_price"]),
         current["recent_extreme_move"].gt(rules["max_jump"]), ~current["eligible"], ~current["fundamentals_ok"],
         current["negative_equity"], current["ttm_loss"] | current["ytd_loss"], current["industry"].isna(),
         current["is_st"].isna(), current["quality_score"].isna(),
         current["quality_score"].lt(rules["minimum_quality_rank"]), ~logs["quality_qualified"], ~logs["selected"]],
        ["非当日成分", "无有效成交行情", "风险警示", "交易历史不足", "成交额不足", "股价低于门槛", "近期价格跳变",
         "量价横截面不足", "财报字段或时点不足", "净资产非正", "累计或滚动亏损", "行业缺失", "风险状态未知",
         "盈利、现金转化或评分字段未通过", "财务排名低于门槛", "行情凭证或评分不足", "综合分位于名单之外"], default="")
    return top.reindex(columns=OUTPUT_FIELDS), audit, logs


def prospective_variants(decisions, picks, spread, config):
    """Freeze an unrestricted price comparator as well as the quality-conditioned comparators."""
    k = config["portfolio"]["top_k"]
    tradable = (decisions["member"] & decisions["eligible"] & decisions["closed_bar_verified"]
                & decisions["is_st"].eq(0) & decisions["price_source"].ne("legacy_reconstructed"))
    cohorts = [picks.assign(variant="quality_mix"), spread.assign(variant="quality_mix_diversified")]
    for name, mask, column in [("price_all", tradable, "baseline"),
                               ("price_quality_pool", decisions["quality_qualified"], "baseline"),
                               ("quality_only", decisions["quality_qualified"], "quality_score")]:
        ranked = decisions.loc[mask].dropna(subset=[column]).sort_values([column, "code"], ascending=[False, True]).head(k)
        cohorts.append(ranked.assign(rank=np.arange(1, len(ranked) + 1), final_score=ranked[column], variant=name))
    columns = ["variant", "date", "rank", "code", "name", "industry", "quality_score", "baseline", "final_score"]
    return pd.concat([frame[columns] for frame in cohorts], ignore_index=True)


def diversified_screen(decisions: pd.DataFrame, config: dict) -> pd.DataFrame:
    cols = ["code", "date", "name", "industry", "quality_score", "baseline"]
    if "price_signal" in decisions.columns:
        cols.append("price_signal")
    rank = decisions.loc[decisions["quality_qualified"], cols].copy()
    weight = config["eligibility"]["quality_weight"]
    price_col = rank["price_signal"] if "price_signal" in rank.columns else rank["baseline"]
    rank["final_score"] = weight * rank["quality_score"] + (1 - weight) * price_col
    rank = rank.sort_values(["final_score", "code"], ascending=[False, True])
    positions, industries, financial_count = [], {}, 0
    is_financial = rank["industry"].str.contains("银行|证券|金融|保险|多元金融", na=False)
    settings = config["portfolio"]
    for item in rank.itertuples():
        financial = bool(is_financial.at[item.Index])
        if industries.get(item.industry, 0) >= settings["max_industry_names"]:
            continue
        if financial and financial_count >= settings["max_financial_names"]:
            continue
        positions.append(item.Index)
        industries[item.industry] = industries.get(item.industry, 0) + 1
        financial_count += int(financial)
        if len(positions) >= settings["top_k"]:
            break
    result = rank.loc[positions].copy()
    result.insert(0, "rank", np.arange(1, len(result) + 1))
    return result
