from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@contextmanager
def file_lock(path: Path, wait_for_collection=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as lock:
        if os.name == "nt":
            import msvcrt
            lock.seek(0)
            if wait_for_collection:
                import errno
                import time
                while True:
                    try:
                        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
                        break
                    except OSError as exc:
                        if exc.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                            raise
                        time.sleep(1)
            else:
                msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == "nt":
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock, fcntl.LOCK_UN)


@contextmanager
def atomic_path(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temp = Path(name)
    try:
        yield temp
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def write_text(path: Path, content: str) -> None:
    with atomic_path(path) as temp:
        temp.write_text(content, encoding="utf-8")


def write_json(path: Path, content: dict) -> None:
    write_text(path, json.dumps(_json_value(content), ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def _json_value(value):
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    with atomic_path(path) as temp:
        frame.to_csv(temp, index=False, encoding="utf-8-sig")


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    with atomic_path(path) as temp:
        frame.to_parquet(temp, index=False, engine="pyarrow", compression="zstd")


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, dtype={"code": "string"}, encoding="utf-8-sig")


def stock_codes(values: pd.Series) -> pd.Series:
    codes = values.astype("string").str.strip().str.zfill(6)
    if not codes.str.fullmatch(r"(?:00|30|60|68)\d{4}").fillna(False).all():
        raise ValueError("Expected six-digit Shanghai/Shenzhen A-share security codes")
    return codes


def dates(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values.astype("string"), format="%Y-%m-%d", errors="raise").astype("datetime64[ns]")


def numeric(frame: pd.DataFrame, columns) -> None:
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise").replace([np.inf, -np.inf], np.nan)


def require(frame: pd.DataFrame, names, key, label: str) -> None:
    missing = set(names) - set(frame.columns)
    if missing:
        raise ValueError(f"{label}: missing columns {sorted(missing)}")
    if frame.empty or frame[list(key)].isna().any().any() or frame.duplicated(list(key)).any():
        raise ValueError(f"{label}: table must be nonempty and key {list(key)} unique and complete")
