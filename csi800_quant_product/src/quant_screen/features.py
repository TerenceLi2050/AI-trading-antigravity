from dataclasses import dataclass

import numpy as np
import pandas as pd

from .factor_lib import align_pit_quality
from .market import trade_flags
from .pit import FUNDAMENTAL_FACTORS


# Signs express the economic hypothesis behind the transparent baseline.
PRICE_FACTORS = {
    "momentum_120_20": 1, "momentum_60_5": 1, "residual_momentum_60": 1,
    "trend": 1, "frog_in_pan": 1, "scaled_momentum": 1, "reversal_5": 1,
    "low_volatility": 1, "drawdown_60": 1, "low_downside_risk": 1,
    "low_amihud": 1, "liquidity": 1, "steady_turnover": 1, "turnover_shock": -1,
    "overnight_return_20": -1, "intraday_return_20": 1, "close_location_20": 1,
}
EXTENDED_PRICE_FACTORS = {
    "low_idio_vol_60": 1,
    "low_max_ret_20": 1,
    "vwap_reversal_20": 1,
    "low_pv_corr_20": 1,
    "low_hl_range_20": 1,
    "neg_skew_60": 1,
    "low_turnover_20": 1,
    "low_vol_ret_corr_20": 1,
    "overnight_minus_intraday_20": 1,
    "resid_rev_20d": 1,
    "low_beta_60": 1,
}
PIT_VALUATION_FACTORS = {
    "bp_pit": 1,
    "cfp_ttm_pit": 1,
    "ep_ttm_pit": 1,
    "cash_conversion_pit": 1,
}
# Tournament-validated China A-share low-turnover multi-family baseline orientation.
ASHARE_BASELINE_SIGNS = {
    "low_idio_vol_60": 1, "low_hl_range_20": 1, "low_max_ret_20": 1, "neg_skew_60": 1, "low_downside_risk": 1,
    "overnight_minus_intraday_20": 1, "vwap_reversal_20": 1, "low_turnover_20": 1, "steady_turnover": 1, "low_vol_ret_corr_20": 1,
    "bp_pit": 1, "cfp_ttm_pit": 1, "cash_conversion_pit": 1,
    "low_volatility": 1, "reversal_5": 1, "intraday_return_20": -1, "close_location_20": -1,
    "overnight_return_20": 1, "turnover_shock": -1, "low_pv_corr_20": 1,
}
ASHARE_BASELINE_GROUPS = [
    ["low_idio_vol_60", "low_hl_range_20", "low_max_ret_20", "neg_skew_60", "low_downside_risk"],
    ["overnight_minus_intraday_20", "vwap_reversal_20", "low_turnover_20", "steady_turnover", "low_vol_ret_corr_20"],
    ["bp_pit", "cfp_ttm_pit", "cash_conversion_pit"],
]
BASELINE_GROUPS = [
    ["momentum_120_20", "momentum_60_5", "residual_momentum_60", "trend", "frog_in_pan", "scaled_momentum"],
    ["low_volatility", "drawdown_60", "low_downside_risk"],
    ["low_amihud", "liquidity", "steady_turnover", "turnover_shock"],
    ["reversal_5", "overnight_return_20", "intraday_return_20", "close_location_20"],
]
RANK_PREFIX = "rank_"


@dataclass
class Features:
    panel: pd.DataFrame
    columns: list[str]
    market: pd.DataFrame


def build_features(quotes: pd.DataFrame, calendar: pd.DataFrame, member: pd.Series, industry: pd.Series,
                   coverage: pd.DataFrame, financial: pd.DataFrame | None, config: dict,
                   quality_history: pd.DataFrame | None = None) -> Features:
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
    turnover = matrix("turnover").where(lambda frame: frame.ge(0))
    returns = close.div(close.shift()).sub(1)
    market_ret = calendar["market_close"].pct_change(fill_method=None)
    beta = returns.rolling(60, min_periods=45).cov(market_ret).div(market_ret.rolling(60, min_periods=45).var(), axis=0)
    residual = returns - beta.shift(1).mul(market_ret, axis=0)
    vol = returns.rolling(60, min_periods=48).std()
    adv = amount.rolling(20, min_periods=15).median()
    range_price = (high - low).where(lambda frame: frame.gt(0))
    mom = close.shift(5).div(close.shift(60)).sub(1)
    signs = np.sign(returns)
    adj_ratio = close.div(raw_close.where(raw_close.gt(0)))
    adj_vwap = amount.div(volume.where(volume.gt(0))).mul(adj_ratio)
    raw = {
        "momentum_120_20": close.shift(20).div(close.shift(120)).sub(1),
        "momentum_60_5": mom,
        "residual_momentum_60": residual.rolling(60, min_periods=45).mean(),
        "trend": close.rolling(20, min_periods=16).mean().div(close.rolling(120, min_periods=100).mean()).sub(1),
        "frog_in_pan": mom.abs() * signs.shift(5).rolling(55, min_periods=44).mean(),
        "scaled_momentum": mom.div(vol.shift(5).where(lambda frame: frame.gt(0))),
        "reversal_5": close.shift(5).div(close).sub(1),
        "low_volatility": -vol,
        "drawdown_60": close.div(close.rolling(60, min_periods=48).max()).sub(1),
        "low_downside_risk": -returns.clip(upper=0).pow(2).rolling(60, min_periods=48).mean().pow(0.5),
        "low_amihud": -returns.abs().div(amount).rolling(20, min_periods=15).mean(),
        "liquidity": np.log(adv),
        "steady_turnover": -turnover.rolling(20, min_periods=15).std().div(turnover.rolling(20, min_periods=15).mean()),
        "turnover_shock": turnover.rolling(5, min_periods=4).mean().div(turnover.rolling(60, min_periods=48).mean()),
        "overnight_return_20": opened.div(close.shift()).sub(1).rolling(20, min_periods=15).mean(),
        "intraday_return_20": close.div(opened).sub(1).rolling(20, min_periods=15).mean(),
        "close_location_20": (close - low).div(range_price).rolling(20, min_periods=15).mean(),
    }
    use_extended = bool(config["eligibility"].get("use_extended_factors", True))
    active_factors = dict(PRICE_FACTORS)
    if use_extended:
        o_log = np.log(opened.div(close.shift(1)).where(lambda df: df.gt(0)))
        d_log = np.log(close.div(opened).where(lambda df: df.gt(0)))
        raw.update({
            "low_idio_vol_60": -residual.rolling(60, min_periods=45).std(),
            "low_max_ret_20": -returns.rolling(20, min_periods=15).max(),
            "vwap_reversal_20": -(close.div(adj_vwap.where(adj_vwap.gt(0))).sub(1)).rolling(20, min_periods=15).mean(),
            "low_pv_corr_20": -close.rolling(20, min_periods=15).corr(volume),
            "low_hl_range_20": -((high - low).div(close.where(close.gt(0)))).rolling(20, min_periods=15).mean(),
            "neg_skew_60": -returns.rolling(60, min_periods=48).skew(),
            "low_turnover_20": -turnover.rolling(20, min_periods=15).mean(),
            "low_vol_ret_corr_20": -returns.rolling(20, min_periods=15).corr(np.log1p(volume.clip(lower=0))),
            "overnight_minus_intraday_20": (o_log - d_log).rolling(20, min_periods=15).sum(),
            "resid_rev_20d": -residual.rolling(20, min_periods=15).sum(),
            "low_beta_60": -beta,
        })
        active_factors.update(EXTENDED_PRICE_FACTORS)

    def values(frame: pd.DataFrame) -> np.ndarray:
        return frame.to_numpy()[row, col]

    settings = config["eligibility"]
    panel = pd.DataFrame({name: values(frame) for name, frame in raw.items()}, index=quotes.index)
    panel = panel.replace([np.inf, -np.inf], np.nan)
    panel[["date", "code", "raw_close", "is_st", "price_source", "snapshot_industry"]] = quotes[
        ["date", "code", "raw_close", "is_st", "price_source", "snapshot_industry"]]
    panel["industry"] = industry
    panel["member"] = member
    panel["bar_ok"] = flags["bar_ok"]
    panel["adv20"] = values(adv)
    panel["vol60"] = values(vol)
    panel["valid_bars_60"] = values(close.rolling(60, min_periods=1).count())
    panel["cumulative_bars"] = values(close.notna().cumsum())
    panel["recent_extreme_move"] = values(returns.abs().rolling(20, min_periods=15).max())
    panel["close_to_high"] = values(close.div(close.rolling(252, min_periods=120).max()).sub(1))
    panel["tradable_history"] = (panel["valid_bars_60"].ge(settings["min_valid_bars_60"])
                                 & panel["cumulative_bars"].ge(settings["min_listing_sessions"]))
    good = (member & flags["bar_ok"] & ~panel["is_st"].eq(1) & panel["tradable_history"]
            & panel["adv20"].ge(settings["min_median_amount"]) & panel["raw_close"].ge(settings["min_raw_price"])
            & panel["recent_extreme_move"].le(settings["max_jump"]))
    covered = (coverage["expected"].isna() | (coverage["coverage"].ge(config["data"]["minimum_member_coverage"])
               & coverage["expected"].ge(config["data"]["minimum_member_count"])))
    good &= panel["date"].map(covered)
    panel["eligible"] = good & good.groupby(panel["date"]).transform("sum").ge(config["data"]["minimum_cross_section"])

    use_pit_val = bool(settings.get("use_pit_valuation_features", True)) and quality_history is not None and not quality_history.empty
    if use_pit_val:
        pit_q = align_pit_quality(quotes, quality_history, max_age=settings["fundamental_max_age_days"])
        raw_px = panel["raw_close"].where(panel["raw_close"].gt(0))
        bps = pit_q["pit_book_per_share"]
        eps_ttm = pit_q["pit_eps_ttm"]
        ocfps_ttm = pit_q["pit_ocfps_ttm"]
        panel["bp_pit"] = bps.div(raw_px)
        panel["cfp_ttm_pit"] = ocfps_ttm.div(raw_px)
        panel["ep_ttm_pit"] = eps_ttm.div(raw_px)
        panel["cash_conversion_pit"] = ocfps_ttm.div(eps_ttm.where(eps_ttm.gt(0))).clip(-5, 5)
        panel = panel.replace([np.inf, -np.inf], np.nan)
        active_factors.update(PIT_VALUATION_FACTORS)

    names = [f"{RANK_PREFIX}{name}" for name in active_factors]
    # Percentile ranks are computed independently after the same-day eligibility check.
    masked_raw = panel.loc[:, list(active_factors)].where(panel["eligible"], axis=0)
    universe_ranks = masked_raw.groupby(panel["date"]).rank(pct=True)
    if settings.get("industry_neutralize", False) and panel["industry"].notna().any():
        ind_mask = panel["eligible"] & panel["industry"].notna()
        ind_count = ind_mask.groupby([panel["date"], panel["industry"]]).transform("sum")
        ind_ranks = masked_raw.groupby([panel["date"], panel["industry"]]).rank(pct=True)
        ranks = ind_ranks.where(ind_count.ge(8), universe_ranks, axis=0)
    else:
        ranks = universe_ranks
    panel[names] = ranks.astype("float32")

    classic_groups = []
    for group in BASELINE_GROUPS:
        oriented = [panel[f"{RANK_PREFIX}{name}"] if PRICE_FACTORS[name] > 0 else 1 - panel[f"{RANK_PREFIX}{name}"]
                    for name in group]
        classic_groups.append(pd.concat(oriented, axis=1).mean(axis=1))
    panel["classic_baseline"] = pd.concat(classic_groups, axis=1).mean(axis=1)

    if settings.get("baseline_profile", "ashare") == "ashare":
        ashare_groups = []
        for group in ASHARE_BASELINE_GROUPS:
            present = [name for name in group if f"{RANK_PREFIX}{name}" in panel.columns]
            if not present:
                continue
            oriented = [
                panel[f"{RANK_PREFIX}{name}"] if ASHARE_BASELINE_SIGNS[name] > 0 else 1 - panel[f"{RANK_PREFIX}{name}"]
                for name in present
            ]
            group_mean = pd.concat(oriented, axis=1).mean(axis=1)
            if group_mean.notna().any():
                ashare_groups.append(group_mean)
        panel["baseline"] = pd.concat(ashare_groups, axis=1).mean(axis=1)
    else:
        panel["baseline"] = panel["classic_baseline"]

    if financial is None:
        panel["fundamentals_ok"] = panel["negative_equity"] = panel["ttm_loss"] = False
        panel["quality_score"] = np.nan
    else:
        panel = pd.concat([panel, financial], axis=1)
        peer_mask = panel["eligible"] & panel["fundamentals_ok"]
        profitability = ["roe_ttm", "roa_ttm", "gross_profit_assets", "cash_return_assets", "low_accruals"]
        values = panel[FUNDAMENTAL_FACTORS].where(peer_mask, axis=0)
        peer_count = peer_mask.groupby([panel["date"], panel["industry"]]).transform("sum")
        f_ranks = values.groupby([panel["date"], panel["industry"]]).rank(pct=True)
        f_ranks = f_ranks.where(peer_count.ge(8), values.groupby(panel["date"]).rank(pct=True), axis=0)
        financial_company = panel["industry"].str.contains("银行|证券|金融|保险|多元金融", na=False)
        cash_conversion = panel["cfo_ttm"].div(panel["net_income_ttm"].where(panel["net_income_ttm"].gt(0)))
        operating_score = f_ranks[profitability].mean(axis=1).where(
            f_ranks[profitability].count(axis=1).ge(3) & cash_conversion.ge(settings["minimum_cash_conversion"]))
        finance_score = f_ranks[["roe_ttm", "roa_ttm", "revenue_growth"]].mean(axis=1)
        panel["quality_score"] = operating_score.where(~financial_company, finance_score).where(panel["net_income_ttm"].gt(0))

    market = calendar.copy()
    market["return_20"] = calendar["market_close"].pct_change(20)
    market["annualized_vol_60"] = market_ret.rolling(60).std() * np.sqrt(252)
    market["above_ma120"] = calendar["market_close"].gt(calendar["market_close"].rolling(120).mean())
    market["eligible_count"] = panel["eligible"].groupby(panel["date"]).sum().reindex(sessions, fill_value=0)
    market["positive_momentum_breadth"] = panel["momentum_60_5"].gt(0).where(panel["eligible"]).groupby(panel["date"]).mean()
    return Features(panel, names, market)
