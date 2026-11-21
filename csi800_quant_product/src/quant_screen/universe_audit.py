"""Measure and repair quote coverage at archived constituent observations."""

from contextlib import ExitStack
import json

import pandas as pd

from .io import dates, require, sha256_file, stock_codes, write_csv, write_json, write_parquet
from .providers import BaoStockAccessError, _bars, _baostock, _error, _normalize


def observed_gaps(samples, quotes, start, end):
    samples = samples.copy()
    samples["code"] = stock_codes(samples["code"].str.split(".").str[-1])
    samples["requested_date"] = dates(samples["requested_date"])
    samples["vendor_date"] = dates(samples["updateDate"])
    days = pd.DatetimeIndex(quotes["date"].unique()).sort_values()
    local = pd.MultiIndex.from_frame(quotes[["date", "code"]])
    records, members = [], []
    for day, part in samples.loc[samples["requested_date"].between(start, end)].groupby("requested_date"):
        counts = part.groupby("index")["code"].nunique().to_dict()
        complete = counts == {"000300": 300, "000905": 500} and len(part) == part["code"].nunique() == 800
        complete &= bool(part["vendor_date"].le(day).all())
        pos = days.searchsorted(day, side="right") - 1
        quote_day = days[pos] if pos >= 0 else pd.NaT
        if pd.isna(quote_day) or quote_day < start:
            continue
        present = pd.MultiIndex.from_arrays([[quote_day] * len(part), part["code"]]).isin(local)
        records.append({"requested_date": day, "vendor_date_min": part["vendor_date"].min(), "vendor_date_max": part["vendor_date"].max(),
                        "quote_date": quote_day, "complete_sample": complete, "sample_codes": part["code"].nunique(),
                        "legacy_covered": int(present.sum()), "legacy_share": float(present.mean())})
        if complete and pd.notna(quote_day):
            members.append(part[["code", "code_name"]].assign(requested_date=day, date=quote_day, legacy_covered=present))
    observations = pd.concat(members, ignore_index=True) if members else pd.DataFrame(
        columns=["code", "code_name", "requested_date", "date", "legacy_covered"])
    missing = observations.loc[observations["legacy_covered"].eq(False)]
    queue = (missing.groupby("code").agg(name=("code_name", "last"), missing_observations=("date", "size"),
             first_missing=("date", "min"), last_missing=("date", "max")).reset_index()
             .sort_values(["missing_observations", "code"], ascending=[False, True]))
    return pd.DataFrame(records), observations, queue


def _collect_prices(codes, start, end, folder):
    problems = {}
    with ExitStack() as stack:
        client = None
        for code in codes:
            path = folder / f"{code}.parquet"
            for attempt in range(3):
                try:
                    if client is None:
                        client = stack.enter_context(_baostock())
                    raw, adjusted = _bars(client, code, start, end)
                    rows = _normalize(raw, adjusted, pd.DataFrame(), code)
                    require(rows, ["raw_open", "adj_close", "trade_status", "is_st"], ["code", "date"], "historical bars")
                    if not rows["date"].between(start, end).all():
                        raise ValueError("Provider returned bars outside the development interval")
                    write_parquet(path, rows)
                    path.with_suffix(".sha256").write_text(sha256_file(path), encoding="ascii")
                    break
                except (OSError, ValueError, KeyError) as exc:
                    stack.close()
                    client = None
                    if attempt == 2 or (isinstance(exc, (ValueError, KeyError)) and not isinstance(exc, UnicodeError)):
                        problems[code] = _error(exc)
                        break
    return problems


def _checked(path):
    try:
        return path.is_file() and sha256_file(path) == path.with_suffix(".sha256").read_text(encoding="ascii")
    except OSError:
        return False


def momentum_sensitivity(observations, quotes, additions, count):
    """Compare candidate sets using identical 60-to-5-session momentum on observed dates."""
    dates_to_score = pd.DatetimeIndex(observations["date"].unique()).sort_values()
    days = pd.DatetimeIndex(quotes["date"].unique()).sort_values()
    window = days[max(0, days.get_loc(dates_to_score[0]) - 60):days.get_loc(dates_to_score[-1]) + 1]

    def score(bars):
        valid = bars.loc[bars["date"].isin(window) & bars["volume"].gt(0) & bars["adj_close"].gt(0)]
        wide = valid.pivot(index="date", columns="code", values="adj_close").reindex(window)
        return wide.shift(5).div(wide.shift(60)).sub(1).where(wide.notna()).reindex(dates_to_score)

    def leaders(values):
        # Codes are pre-sorted, providing deterministic tie breaking for identical momentum.
        return set(values.dropna().sort_index().sort_values(ascending=False, kind="stable").head(count).index)

    legacy = score(quotes)
    repaired = legacy.combine_first(score(additions)) if not additions.empty else legacy
    result = []
    for requested, sample in observations.groupby("requested_date"):
        date = sample["date"].iat[0]
        fixed = leaders(legacy.loc[date])
        intersection = leaders(legacy.loc[date].reindex(sample["code"]))
        observed_scores = repaired.loc[date].reindex(sample["code"])
        rebuilt = leaders(observed_scores)
        result.append({"requested_date": requested, "quote_date": date, "members": len(sample),
                       "members_with_signal": int(observed_scores.notna().sum()), "fixed_pool_top": len(fixed),
                       "overlap_fixed_vs_observed": len(fixed & rebuilt),
                       "overlap_intersection_vs_observed": len(intersection & rebuilt),
                       "observed_top": len(rebuilt), "observed_top_missing_from_fixed": len(rebuilt - fixed)})
    return pd.DataFrame(result)


def run_universe_audit(root, run, quotes, config, collect):
    base = root / "data/historical_observations"
    sources = sorted(base.glob("*/component_samples.csv"))
    if not sources:
        write_json(run / "universe_audit.json", {"state": "MISSING_OBSERVATIONS", "command": "--collect-history"})
        return 2
    source = sources[-1]
    signature = sha256_file(source)
    manifest = json.loads((source.parent / "manifest.json").read_text(encoding="utf-8"))
    if signature != manifest["sha256"][source.name]:
        raise ValueError("Constituent observation checksum mismatch")
    samples = pd.read_csv(source, dtype={"index": str, "code": str})
    settings = config["research"]
    start = pd.Timestamp(settings["oos_start"])
    end = pd.Timestamp(settings["lockbox_start"]) - pd.Timedelta(days=1)
    dates_table, observations, queue = observed_gaps(samples, quotes[["date", "code"]], start, end)
    write_csv(run / "universe_missing_codes.csv", queue)
    if observations.empty:
        write_json(run / "universe_audit.json", {"state": "NO_COMPLETE_OBSERVATIONS"})
        return 2
    begin = start - pd.Timedelta(days=425)
    folder = base / "development_bars" / f"{begin:%Y%m%d}-{end:%Y%m%d}-{signature[:12]}"
    problems = {}
    codes = queue["code"].tolist()
    pending = [c for c in codes if not _checked(folder / f"{c}.parquet")]
    if collect and pending:
        folder.mkdir(parents=True, exist_ok=True)
        print(f"Downloading development history for {len(pending)} missing securities", flush=True)
        try:
            problems = _collect_prices(pending, begin, end, folder)
        except BaoStockAccessError as exc:
            problems["provider"] = _error(exc)
    good = [c for c in codes if _checked(folder / f"{c}.parquet")]
    keys = ["date", "code"]
    observations["backfill_covered"] = False
    additions = pd.DataFrame()
    if good:
        frames = [pd.read_parquet(folder / f"{c}.parquet", columns=keys + ["adj_close", "volume"]) for c in good]
        additions = pd.concat(frames, ignore_index=True)
        repaired = pd.MultiIndex.from_frame(additions[keys])
        observations["backfill_covered"] = pd.MultiIndex.from_frame(observations[keys]).isin(repaired)
    observations["combined_covered"] = observations["legacy_covered"] | observations["backfill_covered"]
    summary = observations.groupby("requested_date").agg(combined_covered=("combined_covered", "sum"),
                                                        combined_share=("combined_covered", "mean"))
    dates_table = dates_table.join(summary, on="requested_date")
    write_csv(run / "universe_observation_coverage.csv", dates_table)
    write_csv(run / "universe_unresolved_rows.csv", observations.loc[~observations["combined_covered"]])
    complete = dates_table.loc[dates_table["complete_sample"]]
    result = {"state": "OBSERVED_COVERAGE_AUDIT", "scope": "sampled dates only; quote presence includes suspensions",
              "source_sha256": signature, "development_end": str(end.date()), "complete_samples": len(complete),
              "incomplete_samples": len(dates_table) - len(complete), "missing_distinct_codes": len(queue),
              "collected_codes": len(good), "backfill_sources": "BaoStock raw, HFQ and trading status",
              "median_legacy_share": complete["legacy_share"].median(), "median_combined_share": complete["combined_share"].median(),
              "min_combined_share": complete["combined_share"].min(), "unresolved_stock_dates": int((~observations["combined_covered"]).sum()),
              "collection_errors": problems, "research_inputs_ready": False}
    if good:
        sensitivity = momentum_sensitivity(observations, quotes, additions, config["portfolio"]["top_k"])
        write_csv(run / "momentum_universe_sensitivity.csv", sensitivity)
        result["momentum_sensitivity"] = {
            "factor": "close[t-5]/close[t-60]-1, positive volume at endpoints and on signal day",
            "top_k": config["portfolio"]["top_k"], "dates": len(sensitivity),
            "median_replaced_vs_fixed_pool": sensitivity["observed_top_missing_from_fixed"].median(),
            "median_overlap_intersection_vs_observed": sensitivity["overlap_intersection_vs_observed"].median(),
            "minimum_member_signal_coverage": (sensitivity["members_with_signal"] / sensitivity["members"]).min()}
    write_json(run / "universe_audit.json", result)
    print(run / "universe_audit.json", flush=True)
    return 2 if collect and len(good) != len(codes) else 0
