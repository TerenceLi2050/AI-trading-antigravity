"""Self-contained, dated market-data releases for the daily screening rule."""

from contextlib import ExitStack
from datetime import datetime, time
from hashlib import sha256
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from uuid import uuid4

import pandas as pd
import pyarrow as pa

from .io import file_lock, require, sha256_file, utc_now, write_csv, write_json, write_parquet
from .providers import _bars, _baostock, _benchmark, _disclosures, _error, _normalize, _snapshot
from .sessions import CST


def available_releases(folder: Path, day: pd.Timestamp, known_by=None):
    """Return complete releases newest first. A corrupt version never hides a usable predecessor."""
    candidates, rejected = [], []
    for path in folder.glob("*/manifest.json"):
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
            observed = pd.Timestamp(meta["created_utc"])
            if observed.tz is None:
                raise ValueError("Daily release timestamp lacks a timezone")
            end = pd.Timestamp(meta["last_session"])
            if end > day or (known_by is not None and observed > known_by):
                continue
            finance_ok = meta["current_financials"]["status"] == "ok"
            candidates.append((end, finance_ok, observed, path, meta))
        except (ValueError, OSError, KeyError, TypeError) as exc:
            rejected.append({"release": path.parent.name, "error": str(exc)})
    for _, _, _, path, meta in sorted(candidates, reverse=True):
        try:
            verify_release(path, meta)
        except (ValueError, OSError, KeyError, TypeError) as exc:
            rejected.append({"release": path.parent.name, "error": str(exc)})
            continue
        return path, rejected
    return None, rejected


def verify_release(path: Path, meta: dict) -> None:
    files = meta["files"]
    required = {"quotes.parquet", "benchmark.csv", "sessions.csv", "current_membership_snapshot.csv",
                "current_membership_snapshot.xls"}
    if meta["current_financials"]["status"] == "ok":
        required.add("current_financials.csv")
    if meta["status"] != "published" or not required.issubset(files):
        raise ValueError("Incomplete daily release")
    for name, digest in files.items():
        if Path(name).name != name or sha256_file(path.parent / name) != digest:
            raise ValueError(f"Daily release checksum mismatch: {name}")
    if (meta["quotes_sha256"] != files["quotes.parquet"]
            or meta["members_sha256"] != files["current_membership_snapshot.csv"]
            or ("current_financials.csv" in files and meta["current_financials"]["sha256"] != files["current_financials.csv"])):
        raise ValueError("Inconsistent release hashes")


def import_release(source: Path, folder: Path) -> Path:
    """Install an intact native release, preserving its original observation time and bytes."""
    manifest = source / "manifest.json" if source.is_dir() else source
    source_bytes = manifest.read_bytes()
    meta = json.loads(source_bytes)
    if meta["schema"] != 1:
        raise ValueError("Unsupported daily archive format")
    verify_release(manifest, meta)
    work = folder / ".work"
    work.mkdir(parents=True, exist_ok=True)
    with file_lock(folder / ".collection.lock", wait_for_collection=True):
        if manifest.resolve().parent.parent == folder.resolve() and manifest.name == "manifest.json":
            return manifest
        day = pd.Timestamp(meta["last_session"])
        target = folder / f"{day:%Y%m%d}-{sha256(source_bytes).hexdigest()[:12]}"
        if target.exists():
            if (target / "manifest.json").read_bytes() != source_bytes:
                raise ValueError("Conflicting archive at the import destination")
            verify_release(target / "manifest.json", meta)
        else:
            with TemporaryDirectory(dir=work) as temp:
                staged = Path(temp) / "release"
                staged.mkdir()
                for name in meta["files"]:
                    shutil.copyfile(manifest.parent / name, staged / name)
                verify_release(staged / "manifest.json", meta)
                (staged / "manifest.json").write_bytes(source_bytes)
                staged.rename(target)
        return target / "manifest.json"


def _valid_shard(path: Path, code: str, day: pd.Timestamp) -> bool:
    try:
        if not path.is_file() or sha256_file(path) != path.with_suffix(".sha256").read_text(encoding="ascii"):
            return False
        keys = pd.read_parquet(path, columns=["code", "date"])
        return (not keys.empty and keys["code"].eq(code).all() and not keys["date"].duplicated().any()
                and keys["date"].max() == day)
    except (OSError, ValueError, pa.ArrowException):
        return False


def _download_daily(codes, start, day, cache, prior):
    failures = {}
    with ExitStack() as stack:
        client = None
        for number, code in enumerate(codes, 1):
            path = cache / f"{code}.parquet"
            old = prior.get(code)
            for attempt in range(3):
                try:
                    if client is None:
                        client = stack.enter_context(_baostock())
                    since = start
                    reference = pd.DataFrame()
                    if old is not None and len(old) >= 3 and old["date"].max() >= start:
                        since = max(start, old["date"].max() - pd.Timedelta(days=45))
                        reference = old.loc[old["date"].ge(since)]
                    raw, adjusted = _bars(client, code, since, day)
                    try:
                        rows = _normalize(raw, adjusted, reference, code)
                    except ValueError:
                        if reference.empty:
                            raise
                        # A split, a long suspension or insufficient overlap calls for a fresh full window.
                        raw, adjusted = _bars(client, code, start, day)
                        rows = _normalize(raw, adjusted, pd.DataFrame(), code)
                        since = start
                    if since > start:
                        rows = pd.concat([old.loc[old["date"].between(start, since, inclusive="left")], rows],
                                         ignore_index=True)
                    require(rows, ["date", "code", "trade_status", "is_st"], ["date", "code"], "daily stock history")
                    if rows["date"].max() != day or not rows["date"].between(start, day).all():
                        raise ValueError("Missing target bar or unexpected session")
                    write_parquet(path, rows)
                    path.with_suffix(".sha256").write_text(sha256_file(path), encoding="ascii")
                    break
                except (OSError, UnicodeError, ValueError, KeyError) as exc:
                    stack.close()
                    client = None
                    if attempt == 2 or isinstance(exc, (ValueError, KeyError)) and not isinstance(exc, UnicodeError):
                        failures[code] = _error(exc)
                        break
            if number % 100 == 0:
                print(f"Daily history: {number}/{len(codes)} checked, {len(failures)} rejected", flush=True)
            if len(failures) >= 15 and len(failures) == number:
                break
    return failures


def collect_daily(folder, schedule, day, config, previous=None, tracking_codes=()):
    data = config["data"]
    sessions = schedule.index[(schedule["is_session"].eq(1)) & (schedule.index <= day)]
    if len(sessions) < data["warmup_sessions"] or sessions[-1] != day:
        raise ValueError("Independent calendar lacks the configured warm-up sessions")
    sessions = sessions[-data["warmup_sessions"]:]
    start = sessions[0]
    members, payload = _snapshot()
    if not members["snapshot_date"].eq(day).all():
        raise ValueError("Official constituent snapshot is not for the expected session")
    index = _benchmark(start, day).set_index("date").sort_index()
    if not index.index.equals(sessions):
        raise ValueError(f"Benchmark differs from the independent calendar ({len(sessions.difference(index.index))} missing sessions)")
    member_codes = set(members["code"])
    pool = sorted(member_codes) + sorted(set(tracking_codes) - member_codes)
    prior = {}
    parent = "cold"
    if previous:
        meta = json.loads(previous.read_text(encoding="utf-8"))
        parent = meta["quotes_sha256"]
        old_index = pd.read_csv(previous.parent / "benchmark.csv", parse_dates=["date"]).set_index("date")
        overlap = index.index.intersection(old_index.index)[-10:]
        if len(overlap) and index.loc[overlap].sub(old_index.loc[overlap]).abs().gt(0.02).any().any():
            raise ValueError("CSI 800 benchmark disagrees with the verified previous release on overlapping sessions")
        prior_rows = pd.read_parquet(previous.parent / "quotes.parquet").drop(columns=["market_open", "market_close"])
        prior = dict(tuple(prior_rows.groupby("code", sort=False)))
    cache = folder / ".work" / f"v1-{start:%Y%m%d}-{day:%Y%m%d}-{parent[:16]}"
    cache.mkdir(parents=True, exist_ok=True)
    # Persist the earliest request time. A resumed batch must retain its original close evidence.
    stamp = cache / "first_request.txt"
    try:
        with stamp.open("x", encoding="ascii") as f:
            f.write(utc_now())
    except FileExistsError:
        pass
    started = stamp.read_text(encoding="ascii")
    if datetime.fromisoformat(started) < datetime.combine(day.date(), time.fromisoformat(data["settled_after_cst"]), CST):
        raise ValueError("Daily bars were requested before the configured close")
    pending = [code for code in pool if not _valid_shard(cache / f"{code}.parquet", code, day)]
    print(f"Daily history {start.date()} to {day.date()}: {len(pool)-len(pending)} cached, {len(pending)} to fetch", flush=True)
    failures = {}
    if pending:
        failures = _download_daily(pending, start, day, cache, prior)
    frames = [pd.read_parquet(cache / f"{code}.parquet") for code in pool
              if _valid_shard(cache / f"{code}.parquet", code, day)]
    if not frames:
        raise ValueError(f"No usable daily bars ({len(failures)} source failures)")
    rows = pd.concat(frames, ignore_index=True)
    if not rows["date"].isin(sessions).all():
        raise ValueError("Stock bars outside the independent trading calendar")
    rows = rows.join(index, on="date", validate="many_to_one").sort_values(["date", "code"], ignore_index=True)
    covered = len(member_codes & set(rows.loc[rows["date"].eq(day), "code"])) / len(members)
    if covered < data["minimum_member_coverage"]:
        raise ValueError(f"Target session quote coverage too low: {covered:.2%}; {len(failures)} source failures")
    try:
        financial = _disclosures(day, member_codes)
        financial_status = {"status": "ok", "rows": len(financial)}
    except Exception as exc:
        financial, financial_status = None, _error(exc)
    release = folder / f"{day:%Y%m%d}-{uuid4().hex[:12]}"
    release.mkdir(parents=True, exist_ok=False)
    write_parquet(release / "quotes.parquet", rows)
    write_csv(release / "benchmark.csv", index.reset_index())
    write_csv(release / "sessions.csv", schedule.rename_axis("date").reset_index())
    write_csv(release / "current_membership_snapshot.csv", members)
    (release / "current_membership_snapshot.xls").write_bytes(payload)
    if financial is not None:
        write_csv(release / "current_financials.csv", financial)
        financial_status["sha256"] = sha256_file(release / "current_financials.csv")
    files = {p.name: sha256_file(p) for p in release.iterdir()}
    meta = {"status": "published", "schema": 1, "created_utc": utc_now(), "bars_requested_utc": started,
            "last_session": str(day.date()), "history_start": str(start.date()), "history_sessions": len(sessions),
            "rows": len(rows), "minimum_coverage": covered, "failed_codes": failures,
            "missing_members": sorted(member_codes - set(rows.loc[rows["date"].eq(day), "code"])),
            "members_sha256": files["current_membership_snapshot.csv"], "quotes_sha256": files["quotes.parquet"],
            "current_financials": financial_status, "files": files, "parent_quotes_sha256": parent,
            "sources": {"bars": "BaoStock raw and HFQ", "calendar": "BaoStock query_trade_dates",
                        "benchmark": index.attrs.get("source", "CSI 800 daily open/close"),
                        "membership": "CSI 000906 official snapshot", "financials": "Eastmoney RPT_LICO_FN_CPD"}}
    path = release / "manifest.json"
    write_json(path, meta)
    return path
