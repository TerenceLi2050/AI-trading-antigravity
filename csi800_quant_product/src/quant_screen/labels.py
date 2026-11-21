import numpy as np
import pandas as pd

from .market import trade_flags


def executable_labels(quotes: pd.DataFrame, market: pd.DataFrame, horizon: int, exit_delay: int = 0) -> pd.DataFrame:
    sessions = market.index
    positions = sessions.get_indexer(quotes["date"])
    schedule = pd.Series(sessions, index=sessions)
    result = quotes[["date", "code"]].copy()
    result["entry_date"] = result["date"].map(schedule.shift(-1))
    result["planned_exit_date"] = result["date"].map(schedule.shift(-(1 + horizon)))
    result["exit_date"] = result["planned_exit_date"]
    flags = trade_flags(quotes, sessions)
    keys = pd.MultiIndex.from_frame(quotes[["date", "code"]])
    buy = pd.Series(quotes["adj_open"].where(flags["buy_open"]).to_numpy(), index=keys)
    sell = pd.Series(quotes["adj_open"].where(flags["sell_open"]).to_numpy(), index=keys)
    entry = buy.reindex(pd.MultiIndex.from_frame(result[["entry_date", "code"]])).to_numpy()
    exit_price = sell.reindex(pd.MultiIndex.from_frame(result[["exit_date", "code"]])).to_numpy()
    for delay in range(1, exit_delay + 1):
        next_date = result["date"].map(schedule.shift(-(1 + horizon + delay)))
        appropriate = result["planned_exit_date"].notna() & ~np.isfinite(exit_price) & np.isfinite(entry) & next_date.notna()
        future = sell.reindex(pd.MultiIndex.from_arrays([next_date[appropriate], result.loc[appropriate, "code"]])).to_numpy()
        filled = appropriate[appropriate].index[np.isfinite(future)]
        result.loc[filled, "exit_date"] = next_date.loc[filled]
        exit_price[result.index.get_indexer(filled)] = future[np.isfinite(future)]
    realized = result["exit_date"].notna() & np.isfinite(entry) & np.isfinite(exit_price)
    result["label_status"] = np.select([result["exit_date"].isna(), ~np.isfinite(entry), ~np.isfinite(exit_price)],
                                       ["pending", "entry_blocked_or_missing", "exit_blocked_or_missing"], default="realized")
    open_index = market["benchmark_open"]
    benchmark = result["exit_date"].map(open_index).div(result["entry_date"].map(open_index))
    result["forward_return"] = pd.Series(exit_price / entry - 1, index=result.index).where(realized)
    result["excess_return"] = result["forward_return"].sub(benchmark - 1)
    result["session"] = positions
    return result
