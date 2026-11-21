import numpy as np
import pandas as pd


def trade_flags(quotes: pd.DataFrame, sessions: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    adj = quotes[["adj_open", "adj_high", "adj_low", "adj_close"]]
    raw = quotes[["raw_open", "raw_high", "raw_low", "raw_close"]]
    price_ok = (adj.gt(0).all(axis=1) & raw.gt(0).all(axis=1) & np.isfinite(adj).all(axis=1)
                & np.isfinite(raw).all(axis=1) & raw["raw_high"].ge(raw[["raw_open", "raw_close", "raw_low"]].max(axis=1) - 0.005)
                & raw["raw_low"].le(raw[["raw_open", "raw_close"]].min(axis=1) + 0.005))
    tradable = (price_ok & quotes["volume"].gt(0) & quotes["amount"].gt(0)
                & quotes["trade_status"].ne(0))
    st = quotes["is_st"].eq(1)
    wide = quotes["code"].str.startswith("688") | (quotes["code"].str.startswith(("300", "301"))
                                                     & quotes["date"].ge("2020-08-24"))
    main_st_cap = np.where(quotes["date"].ge("2026-07-06"), 0.10, 0.05)
    cap = np.where(wide, 0.20, np.where(st | quotes["is_st"].isna(), main_st_cap, 0.10))
    preclose = quotes.get("raw_preclose", pd.Series(np.nan, index=quotes.index))
    reference = preclose.where(preclose.gt(0))
    if sessions is not None and reference.isna().any():
        previous = pd.Series(sessions, index=sessions).shift(1)
        keys = pd.MultiIndex.from_frame(quotes[["date", "code"]])
        prior_keys = pd.MultiIndex.from_arrays([quotes["date"].map(previous), quotes["code"]])
        prior_close = pd.Series(quotes["raw_close"].to_numpy(), index=keys).reindex(prior_keys)
        prior_close.index = quotes.index
        reference = reference.combine_first(prior_close.where(prior_close.gt(0)))
    synthetic_up = np.floor(reference.mul(1 + cap).mul(100) + 0.5000001).div(100)
    synthetic_down = np.floor(reference.mul(1 - cap).mul(100) + 0.5000001).div(100)
    up = quotes["limit_up"].combine_first(synthetic_up)
    down = quotes["limit_down"].combine_first(synthetic_down)
    limit_up = quotes["raw_open"].ge(up - 0.005)
    limit_down = quotes["raw_open"].le(down + 0.005)
    # A daily bar can support research at its reported open, while auction fillability needs separate inputs.
    open_ok = (quotes["trade_status"].ne(0) & raw["raw_open"].gt(0) & adj["adj_open"].gt(0)
               & np.isfinite(raw["raw_open"]) & np.isfinite(adj["adj_open"]))
    within = open_ok & up.notna() & down.notna() & raw["raw_open"].between(down - 0.005, up + 0.005)
    return pd.DataFrame({"bar_ok": tradable, "open_ok": open_ok,
                         "buy_open": within & ~limit_up & ~st, "sell_open": within & ~limit_down,
                         "open_upper": up, "open_lower": down,
                         "missing_open_limit": up.isna() | down.isna(), "is_st_known": quotes["is_st"].notna()},
                        index=quotes.index)


def order_lot(code: str, quantity: int, side: str, holding: int = 0) -> int:
    if side == "sell" and quantity >= holding:
        return holding
    if code.startswith("688"):
        return quantity if quantity >= 200 else 0
    return quantity // 100 * 100
