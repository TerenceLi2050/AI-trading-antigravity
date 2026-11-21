import csv
import json
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from .io import dates, numeric, read_table, require, sha256_file, stock_codes, utc_now, write_json, write_parquet


LEGACY_COLUMNS = {
    1: ("日期", "date"), 2: ("股票代码", "code"),
    3: ("market_open", "market_open"), 4: ("market_close", "market_close"),
    5: ("开盘", "adj_open"), 6: ("最高", "adj_high"),
    7: ("最低", "adj_low"), 8: ("收盘", "adj_close"),
    9: ("成交量", "volume"), 10: ("成交额", "amount"),
    12: ("换手率", "turnover"), 13: ("vwap", "adj_vwap"),
    179: ("当日收盘价", "raw_close"), 180: ("当日涨跌幅", "vendor_pct"),
    183: ("总股本", "total_shares"), 191: ("所属行业", "snapshot_industry"),
}
PRICE_FIELDS = ["raw_open", "raw_high", "raw_low", "raw_close", "adj_open", "adj_high",
                "adj_low", "adj_close", "volume", "amount", "turnover"]
STATUS_COLUMNS = ["trade_status", "is_st", "limit_up", "limit_down"]
BASE_QUOTE_COLUMNS = [
    "date", "code", "market_open", "market_close",
    "adj_open", "adj_high", "adj_low", "adj_close",
    "volume", "amount", "turnover", "adj_vwap",
    "raw_close", "vendor_pct", "total_shares", "snapshot_industry",
    "raw_open", "raw_high", "raw_low", "price_source",
    "trade_status", "is_st", "limit_up", "limit_down", "raw_preclose",
]


def empty_quotes_frame() -> pd.DataFrame:
    data = {}
    for col in BASE_QUOTE_COLUMNS:
        if col == "date":
            data[col] = pd.Series(dtype="datetime64[ns]")
        elif col in {"code", "snapshot_industry", "price_source"}:
            data[col] = pd.Series(dtype="string")
        else:
            data[col] = pd.Series(dtype="float64")
    return pd.DataFrame(data)


def load_research_base(root: Path, data_cfg: dict, prefer_canonical: bool = False) -> tuple[pd.DataFrame, dict]:
    """Load historical base quotes from canonical quotes alone or legacy archive."""
    canonical = root / data_cfg["canonical_quotes"]
    legacy_csv = root / data_cfg["legacy_csv"]
    legacy_manifest = root / data_cfg["cache"] / "legacy.json"
    if prefer_canonical:
        if not canonical.is_file():
            raise FileNotFoundError(
                f"Standalone canonical quotes not found at {canonical}. Run --build-history first."
            )
        return empty_quotes_frame(), {
            "source": "canonical_only",
            "sha256": sha256_file(canonical),
            "path": str(canonical),
        }
    if legacy_csv.is_file() or legacy_manifest.is_file():
        return import_legacy(legacy_csv, root / data_cfg["cache"])
    if canonical.is_file():
        return empty_quotes_frame(), {
            "source": "canonical_only",
            "sha256": sha256_file(canonical),
            "path": str(canonical),
        }
    raise FileNotFoundError(
        "Historical research needs data/input/quotes.parquet (run --build-history) "
        "or the legacy CSV/cache. Use --mode daily for a clean daily start."
    )


def import_legacy(source: Path, folder: Path) -> tuple[pd.DataFrame, dict]:
    key = source.stat() if source.is_file() else None
    signature = ({"path": str(source.resolve()), "size": key.st_size, "mtime_ns": key.st_mtime_ns, "schema": 2}
                 if key is not None else None)
    manifest = folder / "legacy.json"
    if manifest.exists():
        metadata = json.loads(manifest.read_text(encoding="utf-8"))
        cache = folder / metadata["parquet"]
        recorded = metadata["signature"]
        relocated = (signature and signature["size"] == recorded["size"] and signature["path"] != recorded["path"]
                     and sha256_file(source) == metadata["sha256"])
        if (signature == recorded or relocated or key is None) and cache.is_file() and recorded.get("schema") == 2:
            cache_hash = sha256_file(cache)
            if (metadata.get("parquet_sha256", cache_hash) != cache_hash
                    or key is None and "parquet_sha256" not in metadata):
                raise ValueError("Legacy Parquet cache checksum mismatch")
            if "parquet_sha256" not in metadata:
                if sha256_file(source) != metadata["sha256"]:
                    raise ValueError("Original CSV checksum changed")
                metadata["parquet_sha256"] = cache_hash
                write_json(manifest, metadata)
            panel = pd.read_parquet(cache)
            if "raw_preclose" not in panel:
                panel["raw_preclose"] = np.nan
            return panel, metadata

    if key is None:
        raise FileNotFoundError("Historical research needs the legacy source or its verified cache. Use daily for a clean start.")
    with source.open(encoding="utf-8-sig", newline="") as stream:
        header = next(csv.reader(stream))
    if any(position >= len(header) or header[position] != pair[0] for position, pair in LEGACY_COLUMNS.items()):
        raise ValueError("Legacy CSV schema changed. Inspect its columns before importing.")
    selected = list(LEGACY_COLUMNS)
    columns = [pair[1] for pair in LEGACY_COLUMNS.values()]
    frames = []
    for chunk in pd.read_csv(source, encoding="utf-8-sig", usecols=selected, chunksize=100_000,
                             dtype={"股票代码": "string", "所属行业": "string"}):
        chunk.columns = columns
        chunk["date"] = dates(chunk["date"])
        chunk["code"] = stock_codes(chunk["code"])
        numeric(chunk, set(columns) - {"date", "code", "snapshot_industry"})
        frames.append(chunk)
    panel = pd.concat(frames, ignore_index=True)
    require(panel, columns, ("date", "code"), "legacy quotes")
    scale = panel["adj_close"].div(panel["raw_close"]).where(panel["raw_close"].gt(0))
    vwap_error = panel["adj_vwap"].div(scale).div(panel["amount"].div(panel["volume"])).sub(1).abs()
    check = panel["volume"].gt(0) & panel["amount"].gt(0)
    verified = check & vwap_error.le(0.01) & scale.gt(0)
    for leg in ("open", "high", "low"):
        panel[f"raw_{leg}"] = panel[f"adj_{leg}"].div(scale).round(2).where(verified)
    panel["price_source"] = "legacy_reconstructed"
    panel["trade_status"] = np.where(panel["volume"].eq(0) | panel["amount"].eq(0), 0.0, np.nan)
    panel["is_st"] = panel["limit_up"] = panel["limit_down"] = panel["raw_preclose"] = np.nan
    panel = panel.sort_values(["date", "code"], ignore_index=True)
    file_hash = sha256_file(source)
    current = source.stat()
    if (current.st_size, current.st_mtime_ns) != (key.st_size, key.st_mtime_ns):
        raise RuntimeError("Legacy source changed during import")
    cache = folder / f"legacy-{uuid4().hex}.parquet"
    metadata = {
        "signature": signature, "parquet": cache.name, "sha256": file_hash,
        "imported_utc": utc_now(), "rows": len(panel), "securities": panel["code"].nunique(),
        "first_date": str(panel["date"].min().date()), "last_date": str(panel["date"].max().date()),
        "sessions": panel["date"].nunique(), "empty_turnover_rows": int(panel["turnover"].isna().sum()),
        "zero_volume_or_amount": int(panel["trade_status"].eq(0).sum()),
        "tradable_rows_failed_vwap_check": int((check & ~verified).sum()),
        "excluded_legacy_columns": ["all statement/valuation fields", "snapshot industry in historical models"],
    }
    write_parquet(cache, panel)
    metadata["parquet_sha256"] = sha256_file(cache)
    write_json(manifest, metadata)
    return panel, metadata


def assemble_quotes(base: pd.DataFrame, updates: Path, known_by=None) -> tuple[pd.DataFrame, list[str]]:
    releases = [(p, json.loads(p.read_text(encoding="utf-8"))) for p in updates.glob("*/manifest.json")]
    if known_by is not None:
        releases = [(p, m) for p, m in releases if pd.Timestamp(m["created_utc"]) <= known_by]
    releases.sort(key=lambda pair: (pd.Timestamp(pair[1]["created_utc"]), str(pair[0])))
    base = base.assign(_release=pd.Series(pd.NA, index=base.index, dtype="string"))
    if not releases:
        return base, []
    frames = [base]
    sources = []
    for release, manifest in releases:
        path = release.parent / "quotes.parquet"
        if sha256_file(path) != manifest["quotes_sha256"]:
            raise ValueError(f"Quote release checksum mismatch: {path}")
        frame = pd.read_parquet(path)
        if not frame.empty:
            frame["_release"] = str(release)
            if "raw_preclose" not in frame:
                frame["raw_preclose"] = np.nan
            require(frame, base.columns, ("date", "code"), str(path))
            frames.append(frame[base.columns])
        sources.append(str(release))
    joined = (pd.concat(frames, ignore_index=True).drop_duplicates(["date", "code"], keep="last")
              .sort_values(["date", "code"], ignore_index=True))
    joined["code"] = stock_codes(joined["code"])
    return joined, sources


def apply_canonical_quotes(base: pd.DataFrame, path: Path) -> pd.DataFrame:
    if not path.is_file():
        return base
    rows = read_table(path)
    keys = ["date", "code"]
    required = [*keys, *PRICE_FIELDS, "trade_status", "is_st", "price_source", "vendor_pct"]
    require(rows, required, keys, "canonical quotes")
    rows["date"] = dates(rows["date"])
    rows["code"] = stock_codes(rows["code"])
    rows["_release"] = pd.Series(pd.NA, index=rows.index, dtype="string")
    if "adj_vwap" not in rows.columns:
        ratio = rows["adj_close"].div(rows["raw_close"].where(rows["raw_close"].gt(0)))
        rows["adj_vwap"] = rows["amount"].div(rows["volume"].where(rows["volume"].gt(0))) * ratio
    for name in set(base.columns) - set(rows.columns):
        rows[name] = pd.Series(pd.NA, index=rows.index, dtype="string") if name in {"snapshot_industry", "_release"} else np.nan
    numeric(rows, set(base.columns) - {"date", "code", "price_source", "snapshot_industry", "_release"})
    require(rows, required, keys, "normalized canonical quotes")
    if (rows["price_source"].fillna("").astype("string").str.strip().isin(["", "legacy_reconstructed"]).any()
            or not rows["trade_status"].isin([0, 1]).all() or not rows["is_st"].isin([0, 1]).all()):
        raise ValueError("Canonical quotes require provenance and explicit binary trade and ST status")
    ordered = rows[base.columns].copy()
    if base.empty:
        return ordered.sort_values(keys, ignore_index=True)
    non_empty = [frame.dropna(axis=1, how="all") for frame in (base, ordered) if not frame.empty]
    combined = pd.concat(non_empty, ignore_index=True)
    for col in base.columns:
        if col not in combined.columns:
            combined[col] = pd.Series(pd.NA, index=combined.index, dtype="string") if col in {"snapshot_industry", "_release", "price_source"} else np.nan
    return combined[base.columns].drop_duplicates(keys, keep="last").sort_values(keys, ignore_index=True)


def load_status(quotes: pd.DataFrame, path: Path) -> pd.DataFrame:
    if not path.is_file():
        return quotes
    status = read_table(path)
    keys = ["date", "code"]
    columns = keys + STATUS_COLUMNS + ["source"]
    require(status, columns, keys, "daily trade status")
    status["code"] = stock_codes(status["code"])
    status["date"] = dates(status["date"])
    numeric(status, STATUS_COLUMNS)
    require(status, columns, keys, "normalized trade status")
    invalid_limit = (status["limit_up"].le(0) | status["limit_down"].le(0)
                     | status["limit_up"].lt(status["limit_down"]))
    if (not status["trade_status"].isin([0, 1]).all() or not status["is_st"].isin([0, 1]).all()
            or invalid_limit.any() or status["source"].fillna("").str.strip().eq("").any()):
        raise ValueError("Invalid binary trade/ST flags, limit prices or source")
    marked = status[keys + STATUS_COLUMNS].rename(columns={name: f"{name}_official" for name in STATUS_COLUMNS})
    merged = quotes.merge(marked, on=keys, how="left", validate="one_to_one")
    for name in STATUS_COLUMNS:
        merged[name] = merged.pop(f"{name}_official").combine_first(merged[name])
    return merged


def apply_raw_history(quotes: pd.DataFrame, path: Path) -> tuple[pd.DataFrame, dict]:
    if not path.is_file():
        return quotes, {"status": "unavailable"}
    from .history import RAW_HISTORY_FIELDS

    keys = ["date", "code"]
    metadata_path = path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
    if metadata and sha256_file(path) != metadata.get("sha256"):
        raise ValueError("Historical raw bar archive checksum mismatch")
    raw = read_table(path)
    require(raw, [*keys, *RAW_HISTORY_FIELDS], keys, "raw history")
    raw["date"] = dates(raw["date"])
    raw["code"] = stock_codes(raw["code"])
    numeric(raw, RAW_HISTORY_FIELDS)
    require(raw, [*keys, *RAW_HISTORY_FIELDS], keys, "normalized raw history")
    focused = quotes.loc[quotes["price_source"].eq("legacy_reconstructed"), [*keys, "raw_close", "adj_open", "adj_close"]]
    joined = focused.assign(row=focused.index).merge(raw, on=keys, suffixes=("_legacy", ""), validate="one_to_one")
    price_match = joined["raw_close"].sub(joined["raw_close_legacy"]).abs().le(0.011)
    ratio = joined["raw_open"].div(joined["raw_close"].where(joined["raw_close"].gt(0)))
    old_ratio = joined["adj_open"].div(joined["adj_close"].where(joined["adj_close"].gt(0)))
    adjusted_match = ratio.sub(old_ratio).abs().le(0.005)
    eligible = price_match & (adjusted_match | joined["trade_status"].eq(0))
    eligible &= joined["trade_status"].isin([0, 1]) & joined["is_st"].isin([0, 1])
    replacement = joined.loc[eligible].set_index("row")
    quotes = quotes.copy()
    quotes.loc[replacement.index, RAW_HISTORY_FIELDS] = replacement[RAW_HISTORY_FIELDS]
    quotes.loc[replacement.index, "price_source"] = "baostock_raw_legacy_adjusted" if metadata else "supplied_raw_legacy_adjusted"
    return quotes, {"status": "applied", "matched_keys": len(joined), "accepted": len(replacement),
                    "rejected": int((~eligible).sum()), "not_in_legacy": len(raw) - len(joined)}


def market_calendar(quotes: pd.DataFrame, path: Path, total_path: Path | None = None) -> pd.DataFrame:
    columns = ["market_open", "market_close"]
    has_quote_index = (set(columns).issubset(quotes.columns)
                       and quotes[columns].notna().all(axis=1).any())
    if has_quote_index:
        valid_quotes = quotes.dropna(subset=columns)
        cardinality = valid_quotes.groupby("date")[columns].nunique()
        if cardinality.gt(1).any().any():
            raise ValueError("Conflicting CSI 800 benchmark values on the same session")
        calendar = valid_quotes.groupby("date")[columns].first()
    else:
        session_dates = pd.DatetimeIndex(quotes["date"].dropna().unique()).sort_values()
        calendar = pd.DataFrame(np.nan, index=session_dates, columns=columns)
    total_return_fields = ["total_return_open", "total_return_close"]
    if path.is_file():
        official = read_table(path)
        require(official, ["date", *columns, "source"], ["date"], "calendar")
        official["date"] = dates(official["date"])
        official_columns = columns + [name for name in total_return_fields if name in official]
        require(official, official_columns, ["date"], "benchmark fields")
        numeric(official, official_columns)
        require(official, ["date", *columns, "source"], ["date"], "normalized calendar")
        if official["source"].fillna("").str.strip().eq("").any():
            raise ValueError("Calendar observations require source provenance")
        official = official.set_index("date")[official_columns]
        calendar = official.combine_first(calendar)
    if total_path is not None and total_path.is_file():
        total = read_table(total_path)
        required = ["date", "total_return_close", "source"]
        require(total, required, ["date"], "total-return benchmark")
        total["date"] = dates(total["date"])
        require(total, required, ["date"], "normalized total-return benchmark")
        fields = [name for name in total_return_fields if name in total]
        numeric(total, fields)
        if not total["total_return_close"].gt(0).all() or total["source"].fillna("").str.strip().eq("").any():
            raise ValueError("Total-return series requires valid closing levels and provenance")
        calendar = total.set_index("date")[fields].reindex(calendar.index).combine_first(calendar)
    if calendar.empty or not calendar[columns].gt(0).all().all():
        raise ValueError("Every session needs a positive CSI 800 open and close")
    return calendar.sort_index().rename_axis("date")


def benchmark_for(calendar: pd.DataFrame) -> pd.DataFrame:
    calendar = calendar.copy()
    use_close = "total_return_close" in calendar and calendar["total_return_close"].gt(0).all()
    use_open = "total_return_open" in calendar and calendar["total_return_open"].gt(0).all()
    calendar["benchmark_open"] = calendar["total_return_open"] if use_open else calendar["market_open"]
    calendar["benchmark_close"] = calendar["total_return_close"] if use_close else calendar["market_close"]
    calendar["benchmark_total_return"] = use_close
    calendar["benchmark_open_total_return"] = use_open
    return calendar


def quote_audit(quotes: pd.DataFrame, calendar: pd.DatetimeIndex) -> dict:
    ordered = quotes.sort_values(["code", "date"])
    locations = pd.Series(calendar.get_indexer(ordered["date"]), index=ordered.index)
    gaps = locations.groupby(ordered["code"], sort=False).diff().sub(1).clip(lower=0)
    return {
        "rows": len(quotes), "securities": quotes["code"].nunique(), "sessions": len(calendar),
        "internal_missing_stock_sessions": int(gaps.sum()), "gap_events": int(gaps.gt(0).sum()),
        "longest_internal_gap": int(gaps.max()) if gaps.notna().any() else 0,
        "missing_explicit_trade_status": int(quotes["trade_status"].isna().sum()),
        "missing_historical_st": int(quotes["is_st"].isna().sum()),
        "source_rows": {str(key): int(value) for key, value in quotes["price_source"].value_counts().items()},
    }
