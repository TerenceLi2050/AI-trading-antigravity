from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from .allocation import hierarchical_weights
from .market import order_lot, trade_flags


@dataclass
class Position:
    units: float
    mark: float
    acquired: pd.Timestamp
    industry: str | None


def cost(date: pd.Timestamp, side: str, notional: float, config: dict, multiplier: float) -> float:
    transfer = config["transfer_fee_after_2022_04_29"] if date >= pd.Timestamp("2022-04-29") else config["transfer_fee"]
    stamp = config["stamp_duty_after_2023_08_28"] if date >= pd.Timestamp("2023-08-28") else config["stamp_duty"]
    return multiplier * (max(config["min_commission"], notional * config["commission"])
                         + notional * (transfer + (stamp if side == "sell" else 0)))


FINANCIAL_PATTERN = "银行|证券|金融|保险|多元金融"


def _is_financial(sector) -> bool:
    if pd.isna(sector):
        return False
    text = str(sector)
    return any(token in text for token in ("银行", "证券", "金融", "保险", "多元金融"))


def _target(picks: pd.DataFrame, holdings: dict, config: dict, name: str, allocation: str,
            past_returns: pd.DataFrame | None = None):
    ranked = picks.sort_values([name, "code"], ascending=[False, True]).dropna(subset=[name])
    keep = ranked.head(config["keep_rank"])
    order = pd.concat([keep.loc[keep["code"].isin(holdings)], ranked]).drop_duplicates("code")
    chosen, sectors = [], {}
    fin_cap = config.get("max_financial_names")
    fin_count = 0
    for row in order.itertuples(index=False):
        sector = row.industry
        is_fin = _is_financial(sector)
        if pd.notna(sector) and sectors.get(sector, 0) >= config["max_industry_names"]:
            continue
        if fin_cap is not None and is_fin and fin_count >= fin_cap:
            continue
        chosen.append(row)
        if pd.notna(sector):
            sectors[sector] = sectors.get(sector, 0) + 1
        if is_fin:
            fin_count += 1
        if len(chosen) == config["top_k"]:
            break
    if not chosen:
        return {}
    raw = np.ones(len(chosen))
    if allocation in ("inverse_volatility", "hrp"):
        raw = np.array([1 / max(row.vol60, 0.008) if pd.notna(row.vol60) else 0 for row in chosen])
        if raw.sum() == 0:
            raw[:] = 1
    if past_returns is not None:
        estimate = hierarchical_weights(past_returns[[row.code for row in chosen]])
        if estimate is not None:
            raw = estimate.to_numpy()
    capped = np.minimum(raw / raw.sum(), config["max_weight"])
    return {row.code: weight for row, weight in zip(chosen, capped) if weight > 0}


def simulate(quotes: pd.DataFrame, market: pd.DataFrame, scores: pd.DataFrame, eligible: pd.DataFrame,
             name: str, research: dict, config: dict, cost_multiplier: float = 1.0,
             allocation: str = "equal_weight", end: pd.Timestamp | None = None):
    if scores.empty:
        return pd.DataFrame(), pd.DataFrame()
    # A scheduled rebalance snaps to the first scored session at or after each elapsed interval.
    score_dates = pd.DatetimeIndex(scores["date"].unique()).sort_values()
    positions = market.index.get_indexer(score_dates)
    if (positions < 0).any():
        raise ValueError("Backtest signal date falls outside the trading calendar")
    chosen_signals = []
    for position in positions:
        if not chosen_signals or position - chosen_signals[-1] >= research["rebalance_every"]:
            chosen_signals.append(position)
    chosen_dates = market.index[chosen_signals]
    start = chosen_dates[0]
    if end is None:
        end = market.index[-1]
    calendar = market.loc[start:end]
    keys = ["date", "code"]
    candidates = scores.loc[scores["date"].isin(chosen_dates), [*keys, name]].merge(
        eligible[[*keys, "industry", "adv20", "vol60"]], on=keys, how="inner", validate="one_to_one")
    signals = {date: frame for date, frame in candidates.groupby("date", sort=False)}
    trades = quotes.loc[quotes["date"].between(start, end), [*keys, "raw_open", "raw_close", "adj_open", "adj_close"]].copy()
    flags = trade_flags(quotes.loc[trades.index], market.index)
    trades[["buy_open", "sell_open", "bar_ok", "open_ok"]] = flags[["buy_open", "sell_open", "bar_ok", "open_ok"]]
    dates = market.index.to_series().shift(1)
    previous_adv = eligible[["date", "code", "adv20"]].rename(columns={"date": "previous_date"})
    trades["previous_date"] = trades["date"].map(dates)
    trades = trades.merge(previous_adv, on=["previous_date", "code"], how="left", validate="many_to_one")
    trades = trades.drop(columns="previous_date").sort_values(keys)
    # Iterate only by session. All price and status checks at a fill use the open and previous ADV.
    by_day = {date: frame.set_index("code").drop(columns="date") for date, frame in trades.groupby("date", sort=False)}
    asset_returns = None
    if allocation == "hrp":
        warmup = market.index[max(0, market.index.get_loc(start) - 121)]
        history = quotes.loc[quotes["date"].between(warmup, end) & quotes["volume"].gt(0)]
        closes = history.pivot(index="date", columns="code", values="adj_close").reindex(market.loc[warmup:end].index)
        asset_returns = closes.pct_change(fill_method=None)
    cash = config["cash"]
    holdings: dict[str, Position] = {}
    target = {}
    target_industries = {}
    ordered = False
    deferred_buys = set()
    nav, ledger = [], []
    slippage = cost_multiplier * config["slippage_bps"] / 10_000
    for date, market_bar in calendar.iterrows():
        day = by_day.get(date)
        orders = []
        turnover, fee_total, blocked = 0.0, 0.0, 0
        equity_at_open = cash
        for code, holding in holdings.items():
            mark = holding.mark
            if day is not None and code in day.index and day.at[code, "open_ok"]:
                mark = day.at[code, "adj_open"]
            equity_at_open += holding.units * mark
        liquidations = set(holdings) - set(target)
        for code in sorted(set(holdings) if ordered else liquidations):
            holding = holdings[code]
            if holding.acquired >= date:
                blocked += 1
                continue
            if day is None or code not in day.index or not day.at[code, "sell_open"]:
                blocked += int(code in liquidations)
                continue
            bar = day.loc[code]
            ratio = bar.adj_open / bar.raw_open
            shares = int(round(holding.units * ratio))
            desired = max(0.0, equity_at_open * target.get(code, 0.0) / bar.raw_open)
            amount = bar.adv20 * config["max_adv_participation"]
            capacity = int(amount / bar.raw_open) if np.isfinite(amount) else 0
            quantity = order_lot(code, min(max(0, int(shares - desired)), capacity), "sell", shares)
            if quantity:
                orders.append((code, "sell", quantity, bar, holding))
        if (ordered or deferred_buys) and day is not None:
            for code, weight in target.items():
                if not ordered and code not in deferred_buys:
                    continue
                if code not in day.index or not day.at[code, "buy_open"]:
                    blocked += 1
                    continue
                bar = day.loc[code]
                holding = holdings.get(code)
                current = holding.units * bar.adj_open if holding else 0.0
                limit = bar.adv20 * config["max_adv_participation"]
                needed = min(max(0, weight * equity_at_open - current), limit) if np.isfinite(limit) else 0
                quantity = order_lot(code, int(needed / bar.raw_open), "buy")
                if quantity:
                    orders.append((code, "buy", quantity, bar, holding))
        ordered = False
        deferred_buys = set()
        for code, side, quantity, bar, holding in orders:
            fill_price = bar.raw_open
            ratio = bar.adj_open / bar.raw_open
            if side == "buy":
                sector = target_industries.get(code)
                sector_count = sum(item.industry == sector for item in holdings.values()) if sector is not None else 0
                fin_cap = config.get("max_financial_names")
                fin_count = sum(_is_financial(item.industry) for item in holdings.values())
                fin_blocked = fin_cap is not None and _is_financial(sector) and fin_count >= fin_cap
                if code not in holdings and (len(holdings) >= config["top_k"] or sector_count >= config["max_industry_names"] or fin_blocked):
                    blocked += 1
                    deferred_buys.add(code)
                    continue
                quantity = order_lot(code, min(quantity, int(cash / (fill_price * (1 + slippage)))), "buy")
                while quantity and quantity * fill_price * (1 + slippage) + cost(date, side, quantity * fill_price, config, cost_multiplier) > cash:
                    quantity = order_lot(code, quantity - (1 if code.startswith("688") else 100), "buy")
            if not quantity:
                continue
            notional = quantity * fill_price
            fees = cost(date, side, notional, config, cost_multiplier)
            impact = notional * slippage
            if side == "sell" and fees + impact >= cash + notional:
                blocked += 1
                continue
            turnover += notional
            fee_total += fees + impact
            if side == "buy":
                cash -= notional + fees + impact
                units = (holding.units if holding else 0) + quantity / ratio
                holdings[code] = Position(units, bar.adj_open, date, sector)
            else:
                cash += notional - fees - impact
                shares_before = round(holding.units * ratio)
                if quantity >= shares_before:
                    del holdings[code]
                else:
                    holding.units -= quantity / ratio
            ledger.append({"date": date, "code": code, "side": side, "shares": quantity,
                           "fill_price": fill_price, "value": notional, "cash_fee": fees, "slippage_cost": impact})

        exposure, unpriced, frozen = 0.0, 0.0, 0
        industries = {}
        for code, holding in holdings.items():
            if day is not None and code in day.index and day.at[code, "bar_ok"]:
                holding.mark = day.at[code, "adj_close"]
            else:
                unpriced += holding.mark * holding.units
                frozen += 1
            exposure += holding.mark * holding.units
            industries[holding.industry] = industries.get(holding.industry, 0.0) + holding.mark * holding.units
        equity = cash + exposure
        nav.append({"date": date, "nav": equity / config["cash"], "benchmark": market_bar.benchmark_close,
                    "cash_weight": cash / equity, "turnover": turnover / equity_at_open, "cost": fee_total,
                    "positions": len(holdings), "unpriced_weight": unpriced / equity, "frozen_holdings": frozen,
                    "blocked_orders": blocked, "rebalance_signal": date in signals, "target_positions": len(target),
                    "unknown_industry_weight": industries.pop(None, 0.0) / equity,
                    "largest_known_industry_weight": max(industries.values(), default=0.0) / equity})
        if date in signals:
            deferred_buys.clear()
            past_returns = asset_returns.loc[:date].tail(120) if asset_returns is not None else None
            target = _target(signals[date], holdings, config, name, allocation, past_returns)
            target_industries = {row.code: row.industry if pd.notna(row.industry) else None
                                 for row in signals[date].itertuples()}
            for code in holdings.keys() & target_industries.keys():
                holdings[code].industry = target_industries[code]
            ordered = True
    series = pd.DataFrame(nav)
    series["benchmark"] /= series["benchmark"].iat[0]
    return series, pd.DataFrame(ledger, columns=["date", "code", "side", "shares", "fill_price", "value", "cash_fee", "slippage_cost"])


def performance(nav: pd.DataFrame) -> dict:
    if len(nav) < 2:
        return {}
    daily = nav["nav"].pct_change().dropna()
    excess = daily - nav["benchmark"].pct_change().dropna()
    periods = (len(nav) - 1) / 252
    unpriced_days = int(nav["unpriced_weight"].gt(0).sum()) if "unpriced_weight" in nav else 0
    peak_unpriced_date = None
    if "unpriced_weight" in nav and nav["unpriced_weight"].gt(0).any():
        peak_unpriced_date = str(nav.loc[nav["unpriced_weight"].idxmax(), "date"].date())
    return {"sessions": len(nav), "total_return": float(nav["nav"].iat[-1] - 1),
            "benchmark_return": float(nav["benchmark"].iat[-1] - 1),
            "annual_return": float(nav["nav"].iat[-1] ** (1 / periods) - 1),
            "annual_excess": float((nav["nav"].iat[-1] / nav["benchmark"].iat[-1]) ** (1 / periods) - 1),
            "annual_volatility": float(daily.std() * math.sqrt(252)),
            "sharpe_zero_cash_rate": float(daily.mean() / daily.std() * math.sqrt(252)) if daily.std() > 0 else None,
            "information_ratio": float(excess.mean() / excess.std() * math.sqrt(252)) if excess.std() > 0 else None,
            "max_drawdown": float(nav["nav"].div(nav["nav"].cummax()).min() - 1),
            "annual_turnover": float(nav["turnover"].sum() / periods), "paid_cost": float(nav["cost"].sum()),
            "max_unpriced_weight": float(nav["unpriced_weight"].max()), "unpriced_days": unpriced_days,
            "peak_unpriced_date": peak_unpriced_date, "blocked_orders": int(nav["blocked_orders"].sum())}


def yearly_performance(paths: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Compute calendar-year return, excess return, max drawdown, and turnover for each strategy."""
    rows = []
    for strategy, nav in paths.items():
        if nav.empty or len(nav) < 2:
            continue
        frame = nav.copy()
        frame["year"] = pd.to_datetime(frame["date"]).dt.year
        for year, part in frame.groupby("year", sort=True):
            if len(part) < 2:
                continue
            ret = float(part["nav"].iat[-1] / part["nav"].iat[0] - 1)
            bench = float(part["benchmark"].iat[-1] / part["benchmark"].iat[0] - 1)
            mdd = float(part["nav"].div(part["nav"].cummax()).min() - 1)
            rows.append({
                "strategy": strategy,
                "year": int(year),
                "sessions": len(part),
                "return": ret,
                "benchmark_return": bench,
                "excess_return": (1 + ret) / (1 + bench) - 1 if bench > -1 else np.nan,
                "max_drawdown": mdd,
                "turnover": float(part["turnover"].sum()),
            })
    return pd.DataFrame(rows)
