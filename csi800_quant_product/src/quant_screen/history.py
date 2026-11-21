from contextlib import ExitStack
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .io import dates, numeric, require, sha256_file, stock_codes, utc_now, write_csv, write_json, write_parquet
from .providers import BaoStockAccessError, _bao_code, _baostock, _error, _get, _rows, RAW_FIELDS


CSI_PERFORMANCE = "https://www.csindex.com.cn/csindex-home/perf/index-perf"
RAW_HISTORY_FIELDS = ["raw_open", "raw_high", "raw_low", "raw_close", "raw_preclose", "trade_status", "is_st"]


def _raw_history_batch(codes, start, end, work):
    fields = "date,code,open,high,low,close,preclose,tradestatus,isST"
    names = {"open": "raw_open", "high": "raw_high", "low": "raw_low", "close": "raw_close",
             "preclose": "raw_preclose", "tradestatus": "trade_status", "isST": "is_st"}
    failures = {}
    with ExitStack() as connection:
        client = None
        for number, code in enumerate(codes, 1):
            path = work / f"{code}.parquet"
            if path.is_file():
                continue
            for attempt in range(3):
                try:
                    if client is None:
                        client = connection.enter_context(_baostock())
                    raw = _rows(client.query_history_k_data_plus(_bao_code(code), fields, start, end, adjustflag="3"))
                    require(raw, ["date", "code", *names], ["date", "code"], "historical raw bars")
                    if not raw["code"].eq(_bao_code(code)).all():
                        raise ValueError("Historical response returned another security")
                    raw = raw.rename(columns=names)
                    raw["date"] = dates(raw["date"])
                    if not raw["date"].between(start, end).all():
                        raise ValueError("Historical response returned an out-of-range session")
                    raw["code"] = code
                    raw[RAW_HISTORY_FIELDS] = raw[RAW_HISTORY_FIELDS].replace("", np.nan)
                    numeric(raw, RAW_HISTORY_FIELDS)
                    if not (raw["trade_status"].isin([0, 1]) & raw["is_st"].isin([0, 1])).all():
                        raise ValueError("Historical response returned invalid status flags")
                    write_parquet(path, raw[["date", "code", *RAW_HISTORY_FIELDS]])
                    break
                except (OSError, UnicodeDecodeError) as error:
                    connection.close()
                    client = None
                    if attempt == 2:
                        failures[code] = _error(error)
                    else:
                        time.sleep(attempt + 1)
                except ValueError as error:
                    failures[code] = _error(error)
                    break
            if number % 50 == 0:
                print(f"Historical raw shard {number}/{len(codes)}, rejected {len(failures)}", flush=True)
            if len(failures) > 20:
                break
    return failures


def collect_raw_history(codes, start: pd.Timestamp, end: pd.Timestamp, target: Path) -> dict:
    codes = stock_codes(pd.Series(sorted(set(codes)))).tolist()
    work = target.parent / ".history_work" / f"{start:%Y%m%d}-{end:%Y%m%d}"
    work.mkdir(parents=True, exist_ok=True)
    remaining = [code for code in codes if not (work / f"{code}.parquet").is_file()]
    failed = {}
    if remaining:
        try:
            failed = _raw_history_batch(remaining, str(start.date()), str(end.date()), work)
        except BaoStockAccessError as error:
            return {"status": "source_access_refused", "provider": _error(error),
                    "requested": len(codes), "cached": sum((work / f"{c}.parquet").is_file() for c in codes)}
    files = [work / f"{code}.parquet" for code in codes if (work / f"{code}.parquet").is_file()]
    result = {"scope": "observed securities only", "requested": len(codes), "collected": len(files),
              "start": str(start.date()), "end": str(end.date()), "failures": failed, "created_utc": utc_now()}
    if not files or len(files) / len(codes) < 0.95:
        return {**result, "status": "insufficient_coverage"}
    frame = pd.concat([pd.read_parquet(file) for file in files], ignore_index=True)
    write_parquet(target, frame)
    result.update(status="published", rows=len(frame), sha256=sha256_file(target), source="BaoStock unadjusted historical bars")
    write_json(target.with_suffix(".json"), result)
    return result


def normalize_total_return(raw: pd.DataFrame, path: Path) -> dict:
    require(raw, ["tradeDate", "indexCode", "close", "open"], ["tradeDate"], "official H00906")
    if not raw["indexCode"].eq("H00906").all():
        raise ValueError("Unexpected benchmark code in CSI response")
    out = raw[["tradeDate", "close", "open"]].rename(columns={"tradeDate": "date", "close": "total_return_close",
                                                                "open": "total_return_open"})
    out["date"] = pd.to_datetime(out["date"].astype(str), format="%Y%m%d", errors="raise")
    numeric(out, ["total_return_open", "total_return_close"])
    require(out, out.columns, ["date"], "normalized H00906")
    if not out["total_return_close"].gt(0).all():
        raise ValueError("Official series contains invalid closing levels")
    out["source"] = CSI_PERFORMANCE
    write_csv(path, out.sort_values("date"))
    return {"rows": len(out), "opened": int(out["total_return_open"].gt(0).sum()), "sha256": sha256_file(path)}


def download_sessions(path: Path, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    with _baostock() as client:
        raw = _rows(client.query_trade_dates(str(start.date()), str(end.date())))
    require(raw, ["calendar_date", "is_trading_day"], ["calendar_date"], "BaoStock trading calendar")
    normalized = raw.rename(columns={"calendar_date": "date", "is_trading_day": "is_session"})
    normalized["date"] = dates(normalized["date"])
    normalized["is_session"] = pd.to_numeric(normalized["is_session"], errors="raise")
    normalized = normalized.sort_values("date").reset_index(drop=True)
    if (not pd.DatetimeIndex(normalized["date"]).equals(pd.date_range(start, end))
            or not normalized["is_session"].isin([0, 1]).all()):
        raise ValueError("BaoStock calendar did not cover the full requested range")
    normalized["source"] = "BaoStock query_trade_dates"
    write_csv(path, normalized)
    return {"rows": len(normalized), "sessions": int(normalized["is_session"].sum()), "sha256": sha256_file(path),
            "start": str(start.date()), "end": str(end.date())}


def component_coverage(samples: pd.DataFrame, quotes: pd.DataFrame, official: pd.DataFrame | None = None) -> pd.DataFrame:
    groups = {day: set(rows["code"].astype(str)) for day, rows in quotes.groupby("date", sort=True)}
    days = pd.DatetimeIndex(sorted(groups))
    samples = samples.assign(requested_date=dates(samples["requested_date"]), vendor_date=dates(samples["updateDate"]))
    samples["code"] = stock_codes(samples["code"].str.split(".").str[-1])
    official_codes = set(official["code"]) if official is not None else set()
    official_day = dates(official["snapshot_date"]).max() if official is not None else pd.NaT
    result = []
    for date, part in samples.groupby("requested_date", sort=True):
        sample_codes = set(part["code"])
        index_counts = part.groupby("index")["code"].nunique().to_dict()
        count_ok = index_counts.get("000300") == 300 and index_counts.get("000905") == 500 and len(sample_codes) == 800
        position = days.searchsorted(date, side="right") - 1
        market_date = days[position] if position >= 0 else pd.NaT
        observed = len(groups[market_date] & sample_codes) if pd.notna(market_date) else 0
        result.append({"requested_date": date, "first_vendor_date": part["vendor_date"].min(),
                       "last_vendor_date": part["vendor_date"].max(), "complete_components": count_ok,
                       "distinct_codes": len(sample_codes), "comparison_date": market_date, "matched_quotes": observed,
                       "coverage_of_sample": observed / len(sample_codes),
                       "matches_same_day_official": len(official_codes & sample_codes) if date == official_day else None})
    return pd.DataFrame(result)


def collect_history(output: Path, day: pd.Timestamp) -> dict:
    """Archive public observations with their provider dates. Monthly samples carry no effective-interval guarantee."""
    output.mkdir(parents=True, exist_ok=False)
    findings = {}
    try:
        result = _get(CSI_PERFORMANCE, indexCode="H00906", startDate="20190301", endDate=day.strftime("%Y%m%d")).json()
        rows = pd.DataFrame(result["data"])
        if rows.empty:
            raise ValueError("The CSI official endpoint returned no total-return history")
        write_csv(output / "h00906_raw.csv", rows)
        findings["h00906"] = {"status": "collected", "rows": len(rows), "columns": list(rows.columns),
                              "source": CSI_PERFORMANCE}
    except Exception as error:
        findings["h00906"] = _error(error)

    try:
        with _baostock() as client:
            dates_to_check = list(pd.date_range("2019-03-01", min(day, pd.Timestamp("2025-05-31")), freq="MS"))
            dates_to_check += [pd.Timestamp("2022-01-06"), pd.Timestamp("2024-06-17"), day]
            frames, errors = [], {}
            for date in sorted(set(dates_to_check)):
                for index, loader in [("000300", client.query_hs300_stocks), ("000905", client.query_zz500_stocks)]:
                    try:
                        rows = _rows(loader(date=str(date.date())))
                        require(rows, ["updateDate", "code", "code_name"], ["code"], "component snapshot")
                        frames.append(rows.assign(index=index, requested_date=date))
                    except (OSError, ValueError) as error:
                        errors[f"{index}:{date.date()}"] = _error(error)
            if frames:
                snapshots = pd.concat(frames, ignore_index=True)
                write_csv(output / "component_samples.csv", snapshots)
                findings["components"] = {"rows": len(snapshots), "security_codes": snapshots["code"].nunique(),
                                           "requested_dates": snapshots["requested_date"].nunique(),
                                           "source": "BaoStock monthly observations for 000300 and 000905",
                                           "scope": "snapshot_observation", "errors": errors}
            for code in ["000001", "000792", "600519", "000022"]:
                source = {
                    "bars": lambda: _rows(client.query_history_k_data_plus(_bao_code(code), RAW_FIELDS,
                                                                          "2024-04-18", "2024-04-23", adjustflag="3")),
                    "adjustments": lambda: _rows(client.query_adjust_factor(code=_bao_code(code), start_date="2019-03-01",
                                                                           end_date=str(day.date()))),
                    "basic": lambda: _rows(client.query_stock_basic(code=_bao_code(code))),
                }
                for key, fetch in source.items():
                    try:
                        rows = fetch()
                        if not rows.empty:
                            write_csv(output / f"{code}_{key}.csv", rows)
                        findings[f"{code}_{key}"] = {"rows": len(rows), "source": "BaoStock"}
                    except (OSError, ValueError) as error:
                        findings[f"{code}_{key}"] = _error(error)
    except (OSError, ValueError, BaoStockAccessError) as error:
        findings["baostock"] = _error(error)
    result = {"created_utc": utc_now(), "findings": findings,
              "sha256": {file.name: sha256_file(file) for file in output.glob("*.csv")}}
    write_json(output / "manifest.json", result)
    return result
