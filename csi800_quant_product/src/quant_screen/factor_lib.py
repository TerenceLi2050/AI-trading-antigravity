"""Comprehensive, strictly causal (zero-lookahead) candidate factor library and PIT quality/valuation alignment."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .market import trade_flags


FACTOR_FAMILIES: dict[str, list[str]] = {
    "reversal_momentum": [
        "ret_5d", "ret_20d", "resid_rev_5d", "resid_rev_20d",
        "mom_60_5", "mom_120_20", "resid_mom_60", "resid_mom_120_20",
        "near_high_250d", "trend_efficiency_60d",
    ],
    "volatility_risk": [
        "low_vol_20", "low_vol_60", "low_idio_vol_60", "low_downside_risk_60",
        "low_max_ret_20", "low_max_min_spread_20", "neg_skew_60",
        "drawdown_60", "low_beta_60",
    ],
    "microstructure": [
        "overnight_ret_20", "intraday_rev_20", "overnight_minus_intraday_20",
        "gap_rejection_20", "neg_close_loc_20", "vwap_reversal_20", "low_hl_range_20",
    ],
    "liquidity_turnover": [
        "neg_log_adv_20", "amihud_20", "steady_turnover_20", "neg_turnover_shock_5_60",
        "low_turnover_20", "low_pv_corr_20", "low_vol_ret_corr_20",
    ],
    "size_valuation": [
        "neg_log_float_mv", "bp_pit", "ep_ttm_pit", "cfp_ttm_pit", "profit_mv_pit",
    ],
    "fundamental_quality": [
        "roe_ytd_pit", "roe_ttm_pit", "cash_conversion_pit",
        "profit_growth_pit", "revenue_growth_pit", "margin_expansion_pit",
    ],
    "interactions": [
        "value_x_quality", "quality_x_lowvol", "lowvol_x_intraday_rev", "value_x_lowturnover",
    ],
}

ALL_CANDIDATE_FACTORS: list[str] = [f for fam in FACTOR_FAMILIES.values() for f in fam]
FACTOR_TO_FAMILY: dict[str, str] = {f: fam for fam, cols in FACTOR_FAMILIES.items() for f in cols}


def _ttm_flow(history: dict, period: pd.Timestamp, field: str) -> float:
    current = history.get(period, {}).get(field, np.nan)
    if pd.isna(current):
        return np.nan
    if period.month == 12:
        return float(current)
    prev_y4 = period - pd.offsets.YearEnd(1)
    prev_q = period - pd.offsets.QuarterEnd(4)
    y4_val = history.get(prev_y4, {}).get(field, np.nan)
    pq_val = history.get(prev_q, {}).get(field, np.nan)
    if pd.isna(y4_val) or pd.isna(pq_val):
        return np.nan
    return float(current + y4_val - pq_val)


def align_pit_quality(quotes: pd.DataFrame, quality_history: pd.DataFrame | None, max_age: int = 190) -> pd.DataFrame:
    """Align versioned quarterly financial history onto daily quotes with strict `version_known_date < date` PIT rule."""
    out_cols = [
        "pit_report_date", "pit_book_per_share", "pit_roe_ytd_pct",
        "pit_profit_ttm", "pit_eps_ttm", "pit_ocfps_ttm",
        "pit_profit_growth_pct", "pit_revenue_growth_pct",
    ]
    if quality_history is None or quality_history.empty:
        return pd.DataFrame({c: np.nan for c in out_cols}, index=quotes.index)

    qh = quality_history.copy()
    qh["code"] = qh["code"].astype("string")
    qh["report_date"] = pd.to_datetime(qh["report_date"])
    qh["version_known_date"] = pd.to_datetime(qh["version_known_date"])
    qh = qh.sort_values(["code", "version_known_date", "report_date"])

    events = []
    for code, grp in qh.groupby("code", sort=False):
        hist: dict[pd.Timestamp, dict] = {}
        for known_date, batch in grp.groupby("version_known_date", sort=True):
            for row in batch.to_dict("records"):
                hist[row["report_date"]] = row
            latest = max(hist)
            rec = hist[latest]
            events.append({
                "code": code,
                "visible_after": known_date,
                "pit_report_date": latest,
                "pit_book_per_share": rec.get("book_per_share", np.nan),
                "pit_roe_ytd_pct": rec.get("roe_ytd_pct", np.nan),
                "pit_profit_ttm": _ttm_flow(hist, latest, "profit_ytd"),
                "pit_eps_ttm": _ttm_flow(hist, latest, "eps_ytd"),
                "pit_ocfps_ttm": _ttm_flow(hist, latest, "operating_cash_per_share_ytd"),
                "pit_profit_growth_pct": rec.get("profit_growth_pct", np.nan),
                "pit_revenue_growth_pct": rec.get("revenue_growth_pct", np.nan),
            })
    ev_df = pd.DataFrame(events)
    if ev_df.empty:
        return pd.DataFrame({c: np.nan for c in out_cols}, index=quotes.index)
    ev_df["code"] = ev_df["code"].astype(quotes["code"].dtype)
    left = quotes[["code", "date"]].assign(_row=quotes.index).sort_values("date")
    aligned = (
        pd.merge_asof(
            left,
            ev_df.sort_values("visible_after"),
            left_on="date",
            right_on="visible_after",
            by="code",
            direction="backward",
            allow_exact_matches=False,
        )
        .set_index("_row")
        .reindex(quotes.index)
    )
    fresh = aligned["date"].sub(aligned["pit_report_date"]).dt.days.le(max_age)
    for col in out_cols:
        if col != "pit_report_date":
            aligned[col] = pd.to_numeric(aligned[col], errors="coerce").where(fresh)
    aligned["pit_report_date"] = aligned["pit_report_date"].where(fresh)
    return aligned[out_cols]


def neutralize_cross_section(
    panel: pd.DataFrame,
    factor_cols: list[str],
    *,
    eligible_col: str = "eligible",
    industry_col: str = "industry",
    size_col: str = "log_float_mv",
) -> pd.DataFrame:
    """Cross-sectionally neutralize factors against industry dummies and log float market cap on each date."""
    result = pd.DataFrame(np.nan, index=panel.index, columns=factor_cols, dtype="float32")
    for _, grp in panel.groupby("date", sort=False):
        elig = grp.loc[grp[eligible_col]]
        if len(elig) < 30:
            continue
        size_raw = elig[size_col].to_numpy(dtype=float) if size_col in elig.columns else np.full(len(elig), np.nan)
        size_ok = np.isfinite(size_raw)
        has_size = size_ok.sum() >= 30
        if has_size:
            med = np.nanmedian(size_raw)
            std = np.nanstd(size_raw)
            z_size = np.where(size_ok, (size_raw - med) / (std if std > 1e-8 else 1.0), 0.0)
        else:
            z_size = np.zeros(len(elig))

        # Build industry design matrix (collapse rare industries < 5 into 'OTHER')
        ind = elig[industry_col].astype("string").fillna("UNKNOWN") if industry_col in elig.columns else pd.Series("UNKNOWN", index=elig.index)
        counts = ind.value_counts()
        major = counts[counts >= 5].index
        ind_clean = ind.where(ind.isin(major), "OTHER")
        dummies = pd.get_dummies(ind_clean, drop_first=True, dtype=float).to_numpy()
        intercept = np.ones((len(elig), 1), dtype=float)
        X_full = np.hstack([intercept, z_size[:, None], dummies]) if has_size else np.hstack([intercept, dummies])
        X_ind_only = np.hstack([intercept, dummies])

        for col in factor_cols:
            y = elig[col].to_numpy(dtype=float)
            valid = np.isfinite(y)
            if valid.sum() < 20:
                continue
            X = X_ind_only[valid] if col == "neg_log_float_mv" else X_full[valid]
            yv = y[valid]
            # Rank-normalizing y before OLS makes neutralization robust to extreme financial outliers.
            ranks = pd.Series(yv).rank(pct=True).to_numpy(dtype=float)
            coef, _, _, _ = np.linalg.lstsq(X, ranks, rcond=None)
            resid = ranks - X @ coef
            pct_resid = pd.Series(resid).rank(pct=True).to_numpy(dtype=np.float32)
            out_vals = np.full(len(elig), np.nan, dtype=np.float32)
            out_vals[valid] = pct_resid
            result.loc[elig.index, col] = out_vals
    return result


def build_candidate_factors(
    quotes: pd.DataFrame,
    calendar: pd.DataFrame,
    member: pd.Series,
    industry: pd.Series,
    coverage: pd.DataFrame,
    quality_history: pd.DataFrame | None,
    config: dict,
    *,
    neutralize: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """Compute the full 48-factor candidate library with strictly causal rolling windows and PIT financials."""
    flags = trade_flags(quotes)
    sessions = calendar.index
    codes = pd.Index(quotes["code"].unique()).sort_values()
    row = sessions.get_indexer(quotes["date"])
    col = codes.get_indexer(quotes["code"])

    def matrix(column: str, mask: pd.Series = flags["bar_ok"]) -> pd.DataFrame:
        values = np.full((len(sessions), len(codes)), np.nan)
        values[row, col] = quotes[column].where(mask).to_numpy(dtype=float, na_value=np.nan)
        return pd.DataFrame(values, index=sessions, columns=codes)

    close = matrix("adj_close")
    raw_close = matrix("raw_close")
    opened = matrix("adj_open")
    high = matrix("adj_high")
    low = matrix("adj_low")
    volume = matrix("volume")
    amount = matrix("amount")
    turnover = matrix("turnover").where(lambda df: df.gt(0))
    total_shares = matrix("total_shares") if "total_shares" in quotes.columns else pd.DataFrame(np.nan, index=sessions, columns=codes)

    returns = close.div(close.shift(1)).sub(1)
    log_ret = np.log(close.div(close.shift(1)).where(lambda df: df.gt(0)))
    market_ret = calendar["market_close"].pct_change(fill_method=None)
    mkt_var60 = market_ret.rolling(60, min_periods=45).var()
    beta = returns.rolling(60, min_periods=45).cov(market_ret).div(mkt_var60, axis=0)
    residual = returns - beta.shift(1).mul(market_ret, axis=0)

    vol20 = returns.rolling(20, min_periods=15).std()
    vol60 = returns.rolling(60, min_periods=48).std()
    adv20 = amount.rolling(20, min_periods=15).median()
    range_price = (high - low).where(lambda df: df.gt(0))
    adj_ratio = close.div(raw_close.where(raw_close.gt(0)))
    adj_vwap = amount.div(volume.where(volume.gt(0))).mul(adj_ratio)

    # Zero-lookahead daily float market cap: amount / turnover (5-day median smoothed), fallback raw_close * total_shares
    float_mv_raw = amount.div(turnover).where(lambda df: df.gt(0))
    float_mv_smooth = float_mv_raw.rolling(5, min_periods=1).median()
    total_mv_fallback = raw_close.mul(total_shares).where(lambda df: df.gt(0))
    float_mv = float_mv_smooth.combine_first(total_mv_fallback)

    o_log = np.log(opened.div(close.shift(1)).where(lambda df: df.gt(0)))
    d_log = np.log(close.div(opened).where(lambda df: df.gt(0)))
    overnight_20 = o_log.rolling(20, min_periods=15).sum()
    intraday_rev_20 = -d_log.rolling(20, min_periods=15).sum()

    resid_std_60 = residual.rolling(60, min_periods=45).std()
    resid_std_100 = residual.shift(20).rolling(100, min_periods=80).std()

    raw_map: dict[str, pd.DataFrame] = {
        # Family 1: Reversal & Residual Momentum
        "ret_5d": close.shift(5).div(close).sub(1),
        "ret_20d": close.shift(20).div(close).sub(1),
        "resid_rev_5d": -residual.rolling(5, min_periods=4).sum(),
        "resid_rev_20d": -residual.rolling(20, min_periods=15).sum(),
        "mom_60_5": close.shift(5).div(close.shift(60)).sub(1),
        "mom_120_20": close.shift(20).div(close.shift(120)).sub(1),
        "resid_mom_60": residual.rolling(60, min_periods=45).mean().div(resid_std_60.where(resid_std_60.gt(0))),
        "resid_mom_120_20": residual.shift(20).rolling(100, min_periods=80).mean().div(resid_std_100.where(resid_std_100.gt(0))),
        "near_high_250d": close.div(close.rolling(250, min_periods=120).max()),
        "trend_efficiency_60d": log_ret.rolling(60, min_periods=48).sum().div(
            log_ret.abs().rolling(60, min_periods=48).sum().where(lambda df: df.gt(0))
        ),
        # Family 2: Volatility, Tail Risk & Lottery
        "low_vol_20": -vol20,
        "low_vol_60": -vol60,
        "low_idio_vol_60": -resid_std_60,
        "low_downside_risk_60": -returns.clip(upper=0).pow(2).rolling(60, min_periods=48).mean().pow(0.5),
        "low_max_ret_20": -returns.rolling(20, min_periods=15).max(),
        "low_max_min_spread_20": -(returns.rolling(20, min_periods=15).max() - returns.rolling(20, min_periods=15).min()),
        "neg_skew_60": -returns.rolling(60, min_periods=48).skew(),
        "drawdown_60": close.div(close.rolling(60, min_periods=48).max()).sub(1),
        "low_beta_60": -beta,
        # Family 3: Microstructure (Overnight vs Intraday)
        "overnight_ret_20": overnight_20,
        "intraday_rev_20": intraday_rev_20,
        "overnight_minus_intraday_20": overnight_20 + intraday_rev_20,
        "gap_rejection_20": (-np.sign(o_log) * d_log).rolling(20, min_periods=15).mean(),
        "neg_close_loc_20": -((close - low).div(range_price)).rolling(20, min_periods=15).mean(),
        "vwap_reversal_20": -(close.div(adj_vwap.where(adj_vwap.gt(0))).sub(1)).rolling(20, min_periods=15).mean(),
        "low_hl_range_20": -((high - low).div(close.where(close.gt(0)))).rolling(20, min_periods=15).mean(),
        # Family 4: Liquidity, Turnover & Price-Volume
        "neg_log_adv_20": -np.log(adv20.where(adv20.gt(0))),
        "amihud_20": returns.abs().div(amount.where(amount.gt(0))).rolling(20, min_periods=15).mean(),
        "steady_turnover_20": -turnover.rolling(20, min_periods=15).std().div(turnover.rolling(20, min_periods=15).mean()),
        "neg_turnover_shock_5_60": -turnover.rolling(5, min_periods=4).mean().div(turnover.rolling(60, min_periods=48).mean()),
        "low_turnover_20": -turnover.rolling(20, min_periods=15).mean(),
        "low_pv_corr_20": -close.rolling(20, min_periods=15).corr(volume),
        "low_vol_ret_corr_20": -returns.rolling(20, min_periods=15).corr(np.log1p(volume.clip(lower=0))),
        # Size
        "log_float_mv": np.log(float_mv.where(float_mv.gt(0))),
        "neg_log_float_mv": -np.log(float_mv.where(float_mv.gt(0))),
    }

    def values(frame: pd.DataFrame) -> np.ndarray:
        return frame.to_numpy()[row, col]

    panel = pd.DataFrame({name: values(frame) for name, frame in raw_map.items()}, index=quotes.index)
    panel["float_mv"] = values(float_mv)
    panel = panel.replace([np.inf, -np.inf], np.nan)
    panel[["date", "code", "raw_close", "is_st", "price_source"]] = quotes[["date", "code", "raw_close", "is_st", "price_source"]]
    panel["industry"] = industry
    panel["member"] = member
    panel["adv20"] = values(adv20)
    panel["vol60"] = values(vol60)

    settings = config["eligibility"]
    valid_bars_60 = values(close.rolling(60, min_periods=1).count())
    cum_bars = values(close.notna().cumsum())
    extreme_move = values(returns.abs().rolling(20, min_periods=15).max())
    good = (
        member & flags["bar_ok"] & ~panel["is_st"].eq(1)
        & pd.Series(valid_bars_60, index=quotes.index).ge(settings["min_valid_bars_60"])
        & pd.Series(cum_bars, index=quotes.index).ge(settings["min_listing_sessions"])
        & panel["adv20"].ge(settings["min_median_amount"])
        & panel["raw_close"].ge(settings["min_raw_price"])
        & pd.Series(extreme_move, index=quotes.index).le(settings["max_jump"])
    )
    covered = (
        coverage["expected"].isna()
        | (coverage["coverage"].ge(config["data"]["minimum_member_coverage"])
           & coverage["expected"].ge(config["data"]["minimum_member_count"]))
    )
    good &= panel["date"].map(covered)
    panel["eligible"] = good & good.groupby(panel["date"]).transform("sum").ge(config["data"]["minimum_cross_section"])

    # Family 5 & 6: PIT Valuation & Fundamental Quality
    pit_q = align_pit_quality(quotes, quality_history, max_age=settings["fundamental_max_age_days"])
    raw_px = panel["raw_close"].where(panel["raw_close"].gt(0))
    bps = pit_q["pit_book_per_share"]
    eps_ttm = pit_q["pit_eps_ttm"]
    ocfps_ttm = pit_q["pit_ocfps_ttm"]
    fmv = panel["float_mv"].where(panel["float_mv"].gt(0))

    panel["bp_pit"] = bps.div(raw_px)
    panel["ep_ttm_pit"] = eps_ttm.div(raw_px)
    panel["cfp_ttm_pit"] = ocfps_ttm.div(raw_px)
    panel["profit_mv_pit"] = pit_q["pit_profit_ttm"].div(fmv)
    panel["roe_ytd_pit"] = pit_q["pit_roe_ytd_pct"]
    panel["roe_ttm_pit"] = eps_ttm.div(bps.where(bps.gt(0)))
    panel["cash_conversion_pit"] = ocfps_ttm.div(eps_ttm.where(eps_ttm.gt(0))).clip(-5, 5)
    panel["profit_growth_pit"] = pit_q["pit_profit_growth_pct"]
    panel["revenue_growth_pit"] = pit_q["pit_revenue_growth_pct"]
    panel["margin_expansion_pit"] = pit_q["pit_profit_growth_pct"] - pit_q["pit_revenue_growth_pct"]

    # Family 7: Cross-Family Interactions (constructed from same-date eligible ranks)
    base_cols = [c for c in ALL_CANDIDATE_FACTORS if c not in FACTOR_FAMILIES["interactions"]]
    masked = panel[base_cols].where(panel["eligible"], axis=0)
    raw_ranks = masked.groupby(panel["date"]).rank(pct=True)

    panel["value_x_quality"] = raw_ranks["bp_pit"] * raw_ranks["roe_ttm_pit"]
    panel["quality_x_lowvol"] = raw_ranks["roe_ttm_pit"] * raw_ranks["low_idio_vol_60"]
    panel["lowvol_x_intraday_rev"] = raw_ranks["low_idio_vol_60"] * raw_ranks["intraday_rev_20"]
    panel["value_x_lowturnover"] = raw_ranks["bp_pit"] * raw_ranks["low_turnover_20"]
    panel = panel.replace([np.inf, -np.inf], np.nan)

    # Compute final cross-sectional ranks (either raw or industry+size neutralized)
    if neutralize:
        final_ranks = neutralize_cross_section(panel, ALL_CANDIDATE_FACTORS)
    else:
        final_ranks = panel[ALL_CANDIDATE_FACTORS].where(panel["eligible"], axis=0).groupby(panel["date"]).rank(pct=True).astype("float32")

    rank_cols = [f"cand_{c}" for c in ALL_CANDIDATE_FACTORS]
    panel[rank_cols] = final_ranks.to_numpy(dtype=np.float32)
    return panel, rank_cols
