"""Standalone, resumable builder for multi-year historical research inputs without legacy CSV."""

from contextlib import ExitStack
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .data import BASE_QUOTE_COLUMNS
from .history import CSI_PERFORMANCE, download_sessions, normalize_total_return
from .io import dates, numeric, require, sha256_file, stock_codes, utc_now, write_csv, write_json, write_parquet
from .providers import (
    BaoStockAccessError,
    _bars,
    _bars_akshare,
    _bars_eastmoney,
    _baostock,
    _benchmark,
    _error,
    _get,
    _normalize,
    _normalize_akshare,
    _normalize_eastmoney,
    _quarter_snapshot_history,
    _rows,
    _sessions_akshare,
    _snapshot,
    baostock_cooldown_status,
)


QUALITY_HISTORY_COLUMNS = [
    "code", "report_date", "ann_date", "version_known_date", "source", "version",
    "roe_ytd_pct", "book_per_share", "profit_ytd", "eps_ytd",
    "operating_cash_per_share_ytd", "profit_growth_pct", "revenue_growth_pct",
]


def ensure_sessions(path: Path, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    """Download or verify trading sessions calendar, with AkShare fallback when BaoStock is unavailable."""
    if path.is_file():
        try:
            existing = pd.read_csv(path, parse_dates=["date"])
            if not existing.empty and existing["date"].min() <= start and existing["date"].max() >= end:
                return {"status": "cached", "rows": len(existing), "sha256": sha256_file(path)}
        except Exception:
            pass
    try:
        return download_sessions(path, start, end)
    except Exception as first_exc:
        frame = _sessions_akshare(start, end)
        if frame.empty or not frame["is_session"].isin([0, 1]).all():
            raise first_exc
        write_csv(path, frame)
        return {
            "rows": len(frame),
            "sessions": int(frame["is_session"].sum()),
            "sha256": sha256_file(path),
            "start": str(start.date()),
            "end": str(end.date()),
            "source": "AkShare tool_trade_date_hist_sina",
            "fallback_from": type(first_exc).__name__,
        }


def build_benchmark_calendar(calendar_path: Path, total_return_path: Path, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    """Collect CSI 800 price index open/close and official H00906 total-return series."""
    result = {}
    if calendar_path.is_file():
        try:
            existing = pd.read_csv(calendar_path, parse_dates=["date"])
            if not existing.empty and existing["date"].min() <= start + pd.Timedelta(days=10) and existing["date"].max() >= end - pd.Timedelta(days=5):
                result.update(calendar_status="cached", calendar_rows=len(existing), calendar_sha256=sha256_file(calendar_path))
        except Exception:
            pass
    if "calendar_status" not in result:
        index = _benchmark(start, end).copy()
        index["source"] = index.attrs.get("source", "CSI 800 benchmark")
        write_csv(calendar_path, index[["date", "market_open", "market_close", "source"]])
        result.update(
            calendar_rows=len(index),
            calendar_start=str(index["date"].min().date()),
            calendar_end=str(index["date"].max().date()),
            calendar_source=index["source"].iat[0],
            calendar_sha256=sha256_file(calendar_path),
        )
    if total_return_path.is_file():
        try:
            tr = pd.read_csv(total_return_path, parse_dates=["date"])
            if not tr.empty and tr["date"].max() >= end - pd.Timedelta(days=5):
                result["total_return"] = {"status": "cached", "rows": len(tr), "sha256": sha256_file(total_return_path)}
                return result
        except Exception:
            pass
    try:
        payload = _get(CSI_PERFORMANCE, indexCode="H00906",
                       startDate=start.strftime("%Y%m%d"), endDate=end.strftime("%Y%m%d")).json()
        rows = pd.DataFrame(payload.get("data") or [])
        if not rows.empty:
            result["total_return"] = normalize_total_return(rows, total_return_path)
    except Exception as exc:
        result["total_return_error"] = _error(exc)
    return result


def collect_quality_history(
    target: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    codes: set[str] | None = None,
    work_dir: Path | None = None,
) -> tuple[dict, pd.Series]:
    """Fetch and cache quarterly PIT financial disclosures from Eastmoney RPT_LICO_FN_CPD."""
    cache = work_dir or (target.parent / ".history_work" / "quarters")
    cache.mkdir(parents=True, exist_ok=True)
    # Include one year prior to `start` so cross-sections at `start` have fresh reports.
    periods = pd.date_range(start - pd.DateOffset(years=1), end, freq="QE")
    frames, errors = [], {}
    for period in periods:
        shard = cache / f"{period:%Y%m%d}.parquet"
        frame = None
        if shard.is_file():
            try:
                frame = pd.read_parquet(shard)
            except Exception:
                frame = None
        if frame is None:
            try:
                frame = _quarter_snapshot_history(period, codes=None)
                if not frame.empty:
                    write_parquet(shard, frame)
                time.sleep(0.25)
            except Exception as exc:
                errors[str(period.date())] = _error(exc)
                continue
        if frame is not None and not frame.empty:
            frames.append(frame)
    if not frames:
        return {"status": "failed", "periods_requested": len(periods), "errors": errors}, pd.Series(dtype="string")
    combined = pd.concat(frames, ignore_index=True)
    combined["code"] = stock_codes(combined["code"])
    if codes is not None:
        combined = combined.loc[combined["code"].isin(codes)].copy()
    industry_map = (
        combined.dropna(subset=["industry"])
        .assign(industry=lambda df: df["industry"].astype("string").str.strip())
        .loc[lambda df: df["industry"].ne("")]
        .sort_values(["code", "report_date", "ann_date"])
        .drop_duplicates("code", keep="last")
        .set_index("code")["industry"]
    )
    out = (
        combined[QUALITY_HISTORY_COLUMNS]
        .dropna(subset=["report_date", "ann_date", "version_known_date"])
        .sort_values(["code", "report_date", "version_known_date"])
        .drop_duplicates(["code", "report_date", "version_known_date"], keep="last")
        .reset_index(drop=True)
    )
    write_csv(target, out)
    summary = {
        "status": "published",
        "rows": len(out),
        "securities": int(out["code"].nunique()),
        "periods_collected": int(out["report_date"].nunique()),
        "periods_requested": len(periods),
        "first_report_date": str(out["report_date"].min().date()),
        "last_report_date": str(out["report_date"].max().date()),
        "sha256": sha256_file(target),
        "errors": errors,
        "caveat": (
            "Eastmoney RPT_LICO_FN_CPD supplies original NOTICE_DATE per quarter; "
            "subsequent restatements overwrite historical values in-place on the vendor side."
        ),
    }
    write_json(target.with_suffix(".json"), summary)
    return summary, industry_map


def build_membership_intervals(
    samples: pd.DataFrame,
    target: Path,
    industry_map: pd.Series | None = None,
    source_label: str = "baostock_monthly_observation_reconstructed",
) -> dict:
    """Convert dated constituent snapshots into non-overlapping half-open membership intervals."""
    frame = samples.copy()
    code_col = "code"
    frame[code_col] = stock_codes(frame[code_col].astype(str).str.split(".").str[-1])
    date_col = "requested_date" if "requested_date" in frame.columns else "snapshot_date"
    frame["obs_date"] = dates(frame[date_col])
    if "index" in frame.columns:
        valid_dates = []
        for day, part in frame.groupby("obs_date", sort=True):
            counts = part.groupby("index")[code_col].nunique().to_dict()
            if counts.get("000300", 0) >= 295 and counts.get("000905", 0) >= 490:
                valid_dates.append(day)
        frame = frame.loc[frame["obs_date"].isin(valid_dates)].copy()
    obs_dates = sorted(frame["obs_date"].unique())
    if not obs_dates:
        raise ValueError("No complete constituent observation dates available to build membership intervals")
    members_by_date = {
        day: set(part[code_col].tolist())
        for day, part in frame.groupby("obs_date", sort=True)
    }
    if industry_map is None:
        industry_map = pd.Series(dtype="string")
    if "industry" in frame.columns:
        sample_ind = (
            frame.dropna(subset=["industry"])
            .assign(industry=lambda d: d["industry"].astype("string").str.strip())
            .loc[lambda d: d["industry"].ne("")]
            .sort_values("obs_date")
            .drop_duplicates(code_col, keep="last")
            .set_index(code_col)["industry"]
        )
        industry_map = sample_ind.combine_first(industry_map)

    all_codes = sorted(set().union(*members_by_date.values()))
    records = []
    for code in all_codes:
        present_flags = [code in members_by_date[d] for d in obs_dates]
        in_interval = False
        start_day = None
        for idx, flag in enumerate(present_flags):
            if flag and not in_interval:
                start_day = obs_dates[idx]
                in_interval = True
            elif not flag and in_interval:
                end_day = obs_dates[idx]
                records.append({
                    "code": code,
                    "start_date": start_day,
                    "end_date": end_day,
                    "ann_date": start_day - pd.Timedelta(days=1),
                    "industry": industry_map.get(code, pd.NA),
                    "source": source_label,
                })
                in_interval = False
        if in_interval and start_day is not None:
            records.append({
                "code": code,
                "start_date": start_day,
                "end_date": pd.NaT,
                "ann_date": start_day - pd.Timedelta(days=1),
                "industry": industry_map.get(code, pd.NA),
                "source": source_label,
            })

    out = pd.DataFrame(records).sort_values(["code", "start_date"]).reset_index(drop=True)
    write_csv(target, out)
    summary = {
        "status": "published",
        "intervals": len(out),
        "distinct_codes": int(out["code"].nunique()),
        "observation_dates": len(obs_dates),
        "first_start_date": str(out["start_date"].min().date()),
        "last_observation_date": str(obs_dates[-1].date()),
        "industry_covered_share": float(out["industry"].notna().mean()),
        "sha256": sha256_file(target),
        "source": source_label,
    }
    write_json(target.with_suffix(".json"), summary)
    return summary


def _valid_history_shard(path: Path, code: str) -> bool:
    try:
        sig = path.with_suffix(".sha256")
        if not path.is_file() or not sig.is_file():
            return False
        if sha256_file(path) != sig.read_text(encoding="ascii").strip():
            return False
        keys = pd.read_parquet(path, columns=["code", "date"])
        return not keys.empty and keys["code"].eq(code).all() and not keys["date"].duplicated().any()
    except Exception:
        return False


def _download_stock_history(
    codes: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    work_dir: Path,
    provider: str = "auto",
    references: dict[str, pd.DataFrame] | None = None,
) -> dict:
    """Download multi-year raw + HFQ bars for each code with resumable Parquet shards and provider fallback."""
    work_dir.mkdir(parents=True, exist_ok=True)
    problems = {}
    refs = references or {}
    use_baostock = provider in {"auto", "baostock"} and not baostock_cooldown_status().get("in_cooldown", False)
    allow_eastmoney = provider in {"auto", "eastmoney"}
    allow_akshare = provider in {"auto", "akshare"}

    with ExitStack() as stack:
        bs_client = None
        consecutive_failures = 0
        for idx, code in enumerate(codes, 1):
            shard = work_dir / f"{code}.parquet"
            if _valid_history_shard(shard, code):
                consecutive_failures = 0
                continue
            ref = refs.get(code, pd.DataFrame())
            saved = False
            if use_baostock:
                for attempt in range(3):
                    try:
                        if bs_client is None:
                            bs_client = stack.enter_context(_baostock())
                        raw, adj = _bars(bs_client, code, start, end)
                        rows = _normalize(raw, adj, ref, code)
                        require(rows, ["raw_open", "adj_close", "trade_status", "is_st"], ["code", "date"], "history bars")
                        write_parquet(shard, rows)
                        shard.with_suffix(".sha256").write_text(sha256_file(shard), encoding="ascii")
                        saved = True
                        break
                    except BaoStockAccessError as exc:
                        stack.close()
                        bs_client = None
                        use_baostock = False
                        if not (allow_eastmoney or allow_akshare):
                            problems[code] = _error(exc)
                            return problems
                        break
                    except (OSError, UnicodeError) as exc:
                        stack.close()
                        bs_client = None
                        if attempt == 2 and not (allow_eastmoney or allow_akshare):
                            problems[code] = _error(exc)
                        else:
                            time.sleep(attempt + 1)
                    except (ValueError, KeyError) as exc:
                        if not (allow_eastmoney or allow_akshare):
                            problems[code] = _error(exc)
                        break
            if not saved and allow_eastmoney:
                for attempt in range(2):
                    try:
                        raw, adj = _bars_eastmoney(code, start, end)
                        try:
                            rows = _normalize_eastmoney(raw, adj, ref, code)
                        except ValueError:
                            rows = _normalize_eastmoney(raw, adj, pd.DataFrame(), code)
                        require(rows, ["raw_open", "adj_close", "trade_status", "is_st"], ["code", "date"], "eastmoney bars")
                        write_parquet(shard, rows)
                        shard.with_suffix(".sha256").write_text(sha256_file(shard), encoding="ascii")
                        saved = True
                        time.sleep(0.06)
                        break
                    except Exception as exc:
                        if attempt == 1 and not allow_akshare:
                            problems[code] = _error(exc)
                        else:
                            time.sleep(0.3)
            if not saved and allow_akshare:
                for attempt in range(2):
                    try:
                        raw, adj = _bars_akshare(code, start, end)
                        try:
                            rows = _normalize_akshare(raw, adj, ref, code)
                        except ValueError:
                            rows = _normalize_akshare(raw, adj, pd.DataFrame(), code)
                        require(rows, ["raw_open", "adj_close", "trade_status", "is_st"], ["code", "date"], "akshare bars")
                        write_parquet(shard, rows)
                        shard.with_suffix(".sha256").write_text(sha256_file(shard), encoding="ascii")
                        saved = True
                        time.sleep(0.35)
                        break
                    except Exception as exc:
                        if attempt == 1:
                            problems[code] = _error(exc)
                        else:
                            time.sleep(1.5)
            consecutive_failures = 0 if saved else consecutive_failures + 1
            if consecutive_failures >= 5 and idx == consecutive_failures:
                break
            if idx % 100 == 0:
                print(f"History bars: {idx}/{len(codes)} processed, {len(problems)} failures", flush=True)
    return problems


def fill_quote_gap(
    root: Path,
    config: dict,
    gap_start: pd.Timestamp,
    gap_end: pd.Timestamp,
    provider: str = "auto",
) -> dict:
    """Fill a targeted date gap in data/input/quotes.parquet while reconciling HFQ scale against existing bars."""
    data_cfg = config["data"]
    quotes_path = root / data_cfg["canonical_quotes"]
    if not quotes_path.is_file():
        raise FileNotFoundError(f"Canonical quotes not found at {quotes_path}")
    existing = pd.read_parquet(quotes_path)
    existing["date"] = pd.to_datetime(existing["date"], format="mixed").dt.normalize()
    existing["code"] = stock_codes(existing["code"])

    # Determine target codes: active members in [gap_start, gap_end] union any daily snapshot members
    codes = set()
    mem_path = root / data_cfg["membership"]
    if mem_path.is_file():
        mem = pd.read_csv(mem_path, dtype={"code": str})
        mem["code"] = stock_codes(mem["code"])
        mem["start_date"] = pd.to_datetime(mem["start_date"])
        mem["end_date"] = pd.to_datetime(mem["end_date"])
        active = mem["start_date"].le(gap_end) & (mem["end_date"].isna() | mem["end_date"].gt(gap_start))
        codes.update(mem.loc[active, "code"].tolist())
    if not codes:
        codes.update(existing["code"].unique().tolist())
    code_list = sorted(codes)

    fetch_start = gap_start - pd.Timedelta(days=30)
    fetch_end = gap_end + pd.Timedelta(days=15)
    # Build overlap references per security from existing quotes around the gap boundary
    boundary = existing.loc[existing["date"].between(fetch_start, fetch_end), ["code", "date", "raw_close", "adj_close"]]
    references = {c: grp for c, grp in boundary.groupby("code", sort=False)}

    work_dir = root / "data/input/.history_work" / f"gap_{gap_start:%Y%m%d}_{gap_end:%Y%m%d}"
    problems = _download_stock_history(code_list, fetch_start, fetch_end, work_dir, provider=provider, references=references)
    shards = [work_dir / f"{c}.parquet" for c in code_list if _valid_history_shard(work_dir / f"{c}.parquet", c)]
    if not shards:
        return {"status": "failed", "requested_codes": len(code_list), "failures": len(problems),
                "sample_errors": dict(list(problems.items())[:10])}

    additions = pd.concat([pd.read_parquet(p) for p in shards], ignore_index=True)
    additions["date"] = pd.to_datetime(additions["date"], format="mixed").dt.normalize()
    additions["code"] = stock_codes(additions["code"])
    additions = additions.loc[additions["date"].between(gap_start, gap_end)].copy()
    for col in BASE_QUOTE_COLUMNS:
        if col not in additions.columns:
            additions[col] = pd.Series(pd.NA, index=additions.index, dtype="string") if col in {"snapshot_industry", "price_source"} else np.nan
    # Keep existing verified rows on conflict (only fill genuinely missing (date, code) pairs)
    existing_keys = pd.MultiIndex.from_frame(existing[["date", "code"]])
    add_keys = pd.MultiIndex.from_frame(additions[["date", "code"]])
    new_only = additions.loc[~add_keys.isin(existing_keys), BASE_QUOTE_COLUMNS].copy()

    if not new_only.empty:
        non_empty = [df.dropna(axis=1, how="all") for df in (existing[BASE_QUOTE_COLUMNS], new_only) if not df.empty]
        merged = pd.concat(non_empty, ignore_index=True)
        for col in BASE_QUOTE_COLUMNS:
            if col not in merged.columns:
                merged[col] = pd.Series(pd.NA, index=merged.index, dtype="string") if col in {"snapshot_industry", "price_source"} else np.nan
        merged = merged[BASE_QUOTE_COLUMNS].drop_duplicates(["date", "code"], keep="first").sort_values(["date", "code"], ignore_index=True)
        write_parquet(quotes_path, merged)
    else:
        merged = existing

    summary = {
        "status": "published",
        "gap_start": str(gap_start.date()),
        "gap_end": str(gap_end.date()),
        "provider": provider,
        "requested_codes": len(code_list),
        "collected_codes": len(shards),
        "failures": len(problems),
        "added_rows": len(new_only),
        "total_rows": len(merged),
        "total_sessions": int(merged["date"].nunique()),
        "sha256": sha256_file(quotes_path),
    }
    write_json(root / "data/input/gap_fill.json", summary)
    return summary


def _load_latest_component_samples(root: Path) -> pd.DataFrame | None:
    sources = sorted((root / "data/historical_observations").glob("*/component_samples.csv"))
    if not sources:
        return None
    return pd.read_csv(sources[-1], dtype={"index": str, "code": str})


def _collect_constituent_union(root: Path, start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame, list[str]]:
    """Use archived monthly observations if available, or fetch fresh observations + current snapshot."""
    samples = _load_latest_component_samples(root)
    if samples is None:
        frames = []
        if not baostock_cooldown_status().get("in_cooldown", False):
            with _baostock() as client:
                check_dates = list(pd.date_range(start, end, freq="MS")) + [end]
                for day in sorted(set(check_dates)):
                    for idx_code, loader in [("000300", client.query_hs300_stocks), ("000905", client.query_zz500_stocks)]:
                        try:
                            rows = _rows(loader(date=str(day.date())))
                            if not rows.empty:
                                frames.append(rows.assign(index=idx_code, requested_date=day))
                        except Exception:
                            continue
        if frames:
            samples = pd.concat(frames, ignore_index=True)
        else:
            snap, _ = _snapshot()
            samples = snap.rename(columns={"snapshot_date": "requested_date"})
    codes = sorted(stock_codes(samples["code"].astype(str).str.split(".").str[-1]).unique().tolist())
    return samples, codes


def assemble_local_canonical_history(root: Path, config: dict) -> dict:
    """Assemble standalone canonical quotes/calendar/membership from locally archived shards without requiring legacy CSV at runtime."""
    data_cfg = config["data"]
    input_dir = root / "data/input"
    input_dir.mkdir(parents=True, exist_ok=True)
    summary = {"mode": "assemble_local", "created_utc": utc_now()}

    frames = []
    # 1. Include development_bars shards (full raw + HFQ + status for 388 gap securities)
    dev_dirs = sorted((root / "data/historical_observations/development_bars").glob("*"))
    dev_count = 0
    for folder in dev_dirs:
        if not folder.is_dir():
            continue
        for parquet_file in sorted(folder.glob("*.parquet")):
            code = parquet_file.stem
            if _valid_history_shard(parquet_file, code):
                frames.append(pd.read_parquet(parquet_file))
                dev_count += 1

    # 2. Include verified daily releases (quotes + benchmark + disclosures industry)
    daily_releases = sorted((root / data_cfg["daily_updates"]).glob("*/manifest.json"))
    daily_benchmarks = []
    daily_industries = []
    for manifest_path in daily_releases:
        try:
            q_path = manifest_path.parent / "quotes.parquet"
            b_path = manifest_path.parent / "benchmark.csv"
            f_path = manifest_path.parent / "current_financials.csv"
            if q_path.is_file():
                frames.append(pd.read_parquet(q_path))
            if b_path.is_file():
                b_df = pd.read_csv(b_path)
                b_df["source"] = f"daily_release:{manifest_path.parent.name}"
                daily_benchmarks.append(b_df)
            if f_path.is_file():
                f_df = pd.read_csv(f_path, dtype={"code": str})
                if "industry" in f_df.columns:
                    daily_industries.append(f_df[["code", "industry"]].dropna())
        except Exception:
            continue

    # 3. Include verified historical work shards if present
    work_bars = root / "data/input/.history_work/canonical_bars"
    if work_bars.is_dir():
        for parquet_file in sorted(work_bars.glob("*.parquet")):
            if _valid_history_shard(parquet_file, parquet_file.stem):
                frames.append(pd.read_parquet(parquet_file))

    # 4. If verified legacy parquet cache + raw_history exist, project verified non-legacy rows
    raw_hist_path = root / data_cfg["raw_history"]
    legacy_manifest = root / data_cfg["cache"] / "legacy.json"
    if legacy_manifest.is_file() and raw_hist_path.is_file():
        from .data import apply_raw_history, import_legacy
        legacy_quotes, _ = import_legacy(root / data_cfg["legacy_csv"], root / data_cfg["cache"])
        repaired, raw_info = apply_raw_history(legacy_quotes, raw_hist_path)
        clean_repaired = repaired.loc[repaired["price_source"].ne("legacy_reconstructed")].copy()
        if not clean_repaired.empty:
            frames.insert(0, clean_repaired)
            summary["repaired_legacy_rows"] = len(clean_repaired)
            summary["raw_history_info"] = raw_info
        # Extract verified benchmark series from legacy quotes if calendar.csv does not exist yet
        cal_path = root / data_cfg["calendar"]
        if not cal_path.is_file():
            b_legacy = (
                legacy_quotes.dropna(subset=["market_open", "market_close"])
                .groupby("date")[["market_open", "market_close"]]
                .first()
                .reset_index()
            )
            b_legacy["source"] = "assembled_local_benchmark"
            daily_benchmarks.insert(0, b_legacy)

    if daily_benchmarks:
        norm_benchmarks = []
        for bdf in daily_benchmarks:
            part = bdf.copy()
            part["date"] = pd.to_datetime(part["date"], format="mixed").dt.normalize()
            norm_benchmarks.append(part[["date", "market_open", "market_close", "source"]])
        cal_df = pd.concat(norm_benchmarks, ignore_index=True)
        cal_df = cal_df.sort_values("date").drop_duplicates("date", keep="last")
        cal_path = root / data_cfg["calendar"]
        write_csv(cal_path, cal_df[["date", "market_open", "market_close", "source"]])
        summary["calendar_rows"] = len(cal_df)

    if not frames:
        raise ValueError("No local bar archives found to assemble canonical quotes")

    standardized = []
    for df in frames:
        part = df.copy()
        part["date"] = pd.to_datetime(part["date"], format="mixed").dt.normalize()
        part["code"] = stock_codes(part["code"])
        for col in BASE_QUOTE_COLUMNS:
            if col not in part.columns:
                part[col] = pd.Series(pd.NA, index=part.index, dtype="string") if col in {"snapshot_industry", "price_source"} else np.nan
        standardized.append(part[BASE_QUOTE_COLUMNS])

    non_empty = [df.dropna(axis=1, how="all") for df in standardized if not df.empty]
    merged = pd.concat(non_empty, ignore_index=True)
    for col in BASE_QUOTE_COLUMNS:
        if col not in merged.columns:
            merged[col] = pd.Series(pd.NA, index=merged.index, dtype="string") if col in {"snapshot_industry", "price_source"} else np.nan
    merged = (
        merged[BASE_QUOTE_COLUMNS]
        .drop_duplicates(["date", "code"], keep="last")
        .sort_values(["date", "code"], ignore_index=True)
    )
    valid_rows = (
        merged["price_source"].fillna("").astype("string").str.strip().ne("")
        & merged["price_source"].ne("legacy_reconstructed")
        & merged["trade_status"].isin([0, 1])
        & merged["is_st"].isin([0, 1])
    )
    merged = merged.loc[valid_rows].reset_index(drop=True)
    quotes_path = root / data_cfg["canonical_quotes"]
    write_parquet(quotes_path, merged)
    summary["quotes"] = {
        "rows": len(merged),
        "securities": int(merged["code"].nunique()),
        "sessions": int(merged["date"].nunique()),
        "first_date": str(merged["date"].min().date()),
        "last_date": str(merged["date"].max().date()),
        "dev_bar_shards": dev_count,
        "sha256": sha256_file(quotes_path),
    }

    # Build membership.csv if component_samples.csv is available
    samples = _load_latest_component_samples(root)
    if samples is not None:
        ind_series = pd.Series(dtype="string")
        # Priority: quarterly financial cache / quality_history -> daily current_financials -> snapshot_industry
        qh_path = root / data_cfg["quality_history"]
        quarter_dir = input_dir / ".history_work" / "quarters"
        if quarter_dir.is_dir():
            q_inds = []
            for p in sorted(quarter_dir.glob("*.parquet")):
                qf = pd.read_parquet(p, columns=["code", "industry"]).dropna()
                q_inds.append(qf)
            if q_inds:
                ind_series = (
                    pd.concat(q_inds, ignore_index=True)
                    .assign(code=lambda d: stock_codes(d["code"]),
                            industry=lambda d: d["industry"].astype("string").str.strip())
                    .loc[lambda d: d["industry"].ne("")]
                    .drop_duplicates("code", keep="last")
                    .set_index("code")["industry"]
                )
        if daily_industries:
            d_ind = (
                pd.concat(daily_industries, ignore_index=True)
                .assign(code=lambda d: stock_codes(d["code"]),
                        industry=lambda d: d["industry"].astype("string").str.strip())
                .loc[lambda d: d["industry"].ne("")]
                .drop_duplicates("code", keep="last")
                .set_index("code")["industry"]
            )
            ind_series = d_ind.combine_first(ind_series)
        snap_ind = (
            merged.dropna(subset=["snapshot_industry"])
            .assign(snapshot_industry=lambda d: d["snapshot_industry"].astype("string").str.strip())
            .loc[lambda d: d["snapshot_industry"].ne("")]
            .drop_duplicates("code", keep="last")
            .set_index("code")["snapshot_industry"]
        )
        ind_series = ind_series.combine_first(snap_ind)
        summary["membership"] = build_membership_intervals(samples, root / data_cfg["membership"], ind_series)
    elif qh_path := (root / data_cfg["quality_history"]):
        if qh_path.is_file():
            summary["quality_history_existing"] = True

    write_json(input_dir / "history_build.json", summary)
    return summary


def build_history(
    root: Path,
    config: dict,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    bar_provider: str = "auto",
    skip_bars: bool = False,
    skip_financials: bool = False,
    assemble_local: bool = False,
) -> dict:
    """End-to-end standalone historical data builder requiring zero legacy CSV files."""
    if assemble_local:
        return assemble_local_canonical_history(root, config)

    data_cfg = config["data"]
    input_dir = root / "data/input"
    input_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "mode": "online_build_history",
        "start": str(start.date()),
        "end": str(end.date()),
        "bar_provider": bar_provider,
        "created_utc": utc_now(),
    }

    report["sessions"] = ensure_sessions(root / data_cfg["session_calendar"], start, end)
    report["benchmark"] = build_benchmark_calendar(
        root / data_cfg["calendar"], root / data_cfg["total_return"], start, end
    )
    samples, codes = _collect_constituent_union(root, start, end)
    report["constituent_union_codes"] = len(codes)

    industry_map = pd.Series(dtype="string")
    if not skip_financials:
        fin_summary, industry_map = collect_quality_history(
            root / data_cfg["quality_history"], start, end, codes=set(codes)
        )
        report["quality_history"] = fin_summary

    # Backfill industry map from daily disclosures if any codes are still unmapped
    daily_inds = []
    for manifest_path in sorted((root / data_cfg["daily_updates"]).glob("*/manifest.json")):
        f_path = manifest_path.parent / "current_financials.csv"
        if f_path.is_file():
            try:
                fdf = pd.read_csv(f_path, dtype={"code": str})
                if "industry" in fdf.columns:
                    daily_inds.append(fdf[["code", "industry"]].dropna())
            except Exception:
                continue
    if daily_inds:
        d_map = (
            pd.concat(daily_inds, ignore_index=True)
            .assign(code=lambda d: stock_codes(d["code"]),
                    industry=lambda d: d["industry"].astype("string").str.strip())
            .loc[lambda d: d["industry"].ne("")]
            .drop_duplicates("code", keep="last")
            .set_index("code")["industry"]
        )
        industry_map = industry_map.combine_first(d_map)

    report["membership"] = build_membership_intervals(samples, root / data_cfg["membership"], industry_map)

    if not skip_bars:
        work_bars = input_dir / ".history_work" / "canonical_bars"
        failures = _download_stock_history(codes, start, end, work_bars, provider=bar_provider)
        good_files = [work_bars / f"{c}.parquet" for c in codes if _valid_history_shard(work_bars / f"{c}.parquet", c)]
        if not good_files:
            report["quotes"] = {"status": "failed", "failures": len(failures), "sample_errors": dict(list(failures.items())[:10])}
        else:
            frames = [pd.read_parquet(p) for p in good_files]
            merged = pd.concat(frames, ignore_index=True)
            merged["date"] = dates(merged["date"])
            merged["code"] = stock_codes(merged["code"])
            merged = merged.drop_duplicates(["date", "code"], keep="last").sort_values(["date", "code"], ignore_index=True)
            quotes_path = root / data_cfg["canonical_quotes"]
            write_parquet(quotes_path, merged)
            report["quotes"] = {
                "status": "published" if len(good_files) / max(len(codes), 1) >= 0.95 else "partial",
                "collected_codes": len(good_files),
                "requested_codes": len(codes),
                "rows": len(merged),
                "failures": len(failures),
                "sha256": sha256_file(quotes_path),
            }

    write_json(input_dir / "history_build.json", report)
    return report
