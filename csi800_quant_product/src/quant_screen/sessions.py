from datetime import datetime, time, timedelta, timezone
import json
from pathlib import Path

import pandas as pd

from .io import dates, read_table, require


CST = timezone(timedelta(hours=8))


def load_sessions(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    frame = read_table(path)
    fields = ["date", "is_session", "source"]
    require(frame, fields, ["date"], "independent calendar")
    frame["date"] = dates(frame["date"])
    frame = frame.set_index("date").sort_index()
    if (frame.index.has_duplicates or not frame.index.equals(pd.date_range(frame.index[0], frame.index[-1], name="date"))
            or not frame["is_session"].isin([0, 1]).all() or frame["source"].fillna("").astype(str).str.strip().eq("").any()):
        raise ValueError("Independent calendar must cover every civil date, with binary flags and sources")
    return frame


def last_completed(calendar: pd.DataFrame | None, requested: pd.Timestamp, now: datetime,
                   settled_after: str) -> tuple[pd.Timestamp | None, str]:
    local = now.astimezone(CST)
    today = pd.Timestamp(local.date())
    if calendar is None or requested not in calendar.index:
        return None, "CALENDAR_UNVERIFIED"
    open_today = calendar.at[requested, "is_session"] == 1
    before_close = requested == today and open_today and local.time() < time.fromisoformat(settled_after)
    end = requested - pd.Timedelta(days=1) if before_close else requested
    open_dates = calendar.loc[:end, "is_session"]
    recent = open_dates.loc[open_dates.eq(1)]
    if recent.empty:
        return None, "CALENDAR_UNVERIFIED"
    phase = "AWAITING_CLOSE" if before_close else "AFTER_CLOSE" if open_today else "NON_TRADING_DAY"
    return recent.index[-1], phase


def quote_completion(releases: list[str], day: pd.Timestamp, settled_after: str) -> list[str]:
    finalization = datetime.combine(day.date(), time.fromisoformat(settled_after), tzinfo=CST)
    verified = []
    for entry in releases:
        manifest = json.loads(Path(entry).read_text(encoding="utf-8"))
        after = manifest.get("bars_requested_utc")
        if (manifest.get("last_session") == str(day.date()) and manifest.get("rows", 0) > 0 and after
                and datetime.fromisoformat(after) >= finalization):
            verified.append(entry)
    return verified


def readiness(day: pd.Timestamp, expected: pd.Timestamp | None, phase: str, verified: bool, replay: bool) -> dict:
    reasons = []
    if expected is None:
        reasons.append("独立交易日历未覆盖运行日")
    elif day != expected:
        reasons.append(f"应使用 {expected.date()} 的收盘数据，实际行情为 {day.date()}")
    if not verified:
        reasons.append("当前行情批次缺少收盘后采集证据")
    return {"phase": phase, "scope": "historical_replay" if replay else "live",
            "expected_session": str(expected.date()) if expected is not None else None,
            "close_verified": verified, "reasons": reasons}
