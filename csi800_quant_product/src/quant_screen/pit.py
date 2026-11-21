from pathlib import Path

import numpy as np
import pandas as pd

from .io import dates, numeric, read_table, require, stock_codes


FLOWS = ["revenue_ytd", "net_income_ytd", "cfo_ytd", "gross_profit_ytd"]
BALANCES = ["assets", "equity", "liabilities"]
FUNDAMENTAL_FACTORS = ["roe_ttm", "roa_ttm", "gross_profit_assets", "cash_return_assets",
                       "low_accruals", "earnings_yield", "book_yield", "revenue_growth"]
REPORT_COLUMNS = ["code", "report_date", "ann_date", "source", "version", *FLOWS, *BALANCES]


def membership(quotes: pd.DataFrame, calendar: pd.DatetimeIndex, path: Path):
    keys = ["code", "start_date"]
    columns = keys + ["end_date", "ann_date", "industry", "source"]
    if not path.is_file():
        return (pd.Series(True, index=quotes.index), pd.Series(pd.NA, index=quotes.index, dtype="string"),
                pd.DataFrame(index=calendar, columns=["expected", "observed", "coverage"], dtype=float))
    intervals = read_table(path)
    require(intervals, columns, keys, "CSI 800 membership")
    intervals["code"] = stock_codes(intervals["code"])
    for column in ("start_date", "end_date", "ann_date"):
        intervals[column] = dates(intervals[column])
    intervals = intervals.sort_values(["code", "start_date"])
    require(intervals, columns, keys, "normalized CSI 800 membership")
    ends = intervals["end_date"].fillna(pd.Timestamp.max.normalize())
    previous_end = ends.groupby(intervals["code"], sort=False).shift()
    if (intervals["start_date"].lt(previous_end).any() or ends.le(intervals["start_date"]).any()
            or not intervals["ann_date"].lt(intervals["start_date"]).all()
            or intervals["source"].fillna("").str.strip().eq("").any()):
        raise ValueError("Invalid membership intervals, publication dates, or provenance")
    lookup = pd.merge_asof(quotes[["date", "code"]].assign(row=quotes.index).sort_values("date"),
                           intervals[columns].sort_values("start_date"),
                           left_on="date", right_on="start_date", by="code", direction="backward")
    lookup = lookup.set_index("row").reindex(quotes.index)
    active = lookup["start_date"].notna() & (lookup["end_date"].isna() | lookup["date"].lt(lookup["end_date"]))
    start_pos = calendar.searchsorted(intervals["start_date"])
    end_pos = calendar.searchsorted(ends)
    changes = np.zeros(len(calendar) + 1, dtype=int)
    np.add.at(changes, start_pos, 1)
    np.add.at(changes, end_pos, -1)
    coverage = pd.DataFrame({"expected": changes.cumsum()[:-1]}, index=calendar)
    coverage["observed"] = lookup.loc[active].groupby("date").size().reindex(calendar, fill_value=0)
    coverage["coverage"] = coverage["observed"].div(coverage["expected"].where(coverage["expected"].gt(0)))
    return active, lookup["industry"].astype("string").where(active), coverage


def _ttm(history: dict, period: pd.Timestamp, field: str) -> float:
    current = history.get(period, {}).get(field, np.nan)
    if period.month == 12:
        return current
    previous_year = period - pd.offsets.YearEnd(1)
    previous_same_quarter = period - pd.offsets.QuarterEnd(4)
    return current + history.get(previous_year, {}).get(field, np.nan) - history.get(previous_same_quarter, {}).get(field, np.nan)


def _events(reports: pd.DataFrame) -> pd.DataFrame:
    events = []
    for code, filings in reports.groupby("code", sort=False):
        history = {}
        for ann_date, releases in filings.groupby("ann_date", sort=True):
            for row in releases.to_dict("records"):
                history[row["report_date"]] = row
            latest = max(history)
            record = history[latest]
            last_year = latest - pd.offsets.QuarterEnd(4)
            events.append({"code": code, "visible_after": ann_date, "report_date": latest,
                           "report_ann_date": record["ann_date"], "report_source": record["source"],
                           "report_version": record["version"], **{key: record[key] for key in BALANCES},
                           **{key.replace("_ytd", "_ttm"): _ttm(history, latest, key) for key in FLOWS},
                           "revenue_ttm_prior": _ttm(history, last_year, "revenue_ytd")})
    return pd.DataFrame(events)


def fundamentals(quotes: pd.DataFrame, path: Path, max_age: int) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    reports = read_table(path)
    keys = ["code", "report_date", "ann_date"]
    require(reports, REPORT_COLUMNS, keys, "financial releases")
    reports["code"] = stock_codes(reports["code"])
    for column in ("report_date", "ann_date"):
        reports[column] = dates(reports[column])
    numeric(reports, FLOWS + BALANCES)
    require(reports, REPORT_COLUMNS, keys, "normalized financial releases")
    if (reports["ann_date"].lt(reports["report_date"]).any() or not reports["report_date"].dt.is_quarter_end.all()
            or reports[["source", "version"]].astype("string").fillna("").apply(lambda x: x.str.strip().eq("")).any().any()):
        raise ValueError("Financial rows require calendar quarter ends, real announcement dates and version provenance")
    releases = _events(reports)
    releases["code"] = releases["code"].astype(quotes["code"].dtype)
    aligned = pd.merge_asof(quotes[["code", "date"]].assign(row=quotes.index).sort_values("date"),
                            releases.sort_values("visible_after"), left_on="date", right_on="visible_after",
                            by="code", direction="backward", allow_exact_matches=False).set_index("row").reindex(quotes.index)
    fresh = aligned["date"].sub(aligned["report_date"]).dt.days.le(max_age)
    equity = aligned["equity"].where(aligned["equity"].gt(0))
    assets = aligned["assets"].where(aligned["assets"].gt(0))
    capitalization = quotes["raw_close"].mul(quotes["total_shares"]).where(quotes["total_shares"].gt(0))
    income, cash_flow = aligned["net_income_ttm"], aligned["cfo_ttm"]
    previous_revenue = aligned["revenue_ttm_prior"].where(aligned["revenue_ttm_prior"].gt(0))
    values = {
        "roe_ttm": income.div(equity), "roa_ttm": income.div(assets),
        "gross_profit_assets": aligned["gross_profit_ttm"].div(assets),
        "cash_return_assets": cash_flow.div(assets), "low_accruals": cash_flow.sub(income).div(assets),
        "earnings_yield": income.div(capitalization.where(capitalization.gt(0))),
        "book_yield": aligned["equity"].div(capitalization.where(capitalization.gt(0))),
        "revenue_growth": aligned["revenue_ttm"].div(previous_revenue).sub(1),
    }
    for key, value in values.items():
        aligned[key] = value.where(fresh)
    aligned["fundamentals_ok"] = fresh & equity.notna() & assets.notna() & income.notna() & cash_flow.notna()
    aligned["negative_equity"] = fresh & aligned["equity"].le(0)
    aligned["ttm_loss"] = fresh & income.le(0)
    result = [*FUNDAMENTAL_FACTORS, "fundamentals_ok", "negative_equity", "ttm_loss", "report_date", "report_ann_date",
              "report_source", "report_version", "visible_after", "net_income_ttm", "cfo_ttm", "equity"]
    return aligned[result]
