from datetime import datetime, time, timedelta
from pathlib import Path
import time as clock

import pandas as pd

from .io import utc_now, write_json
from .sessions import CST, load_sessions


def next_check(now, status, schedule, settled_after):
    if status:
        return now + timedelta(minutes=30)
    today = pd.Timestamp(now.date())
    dates = pd.date_range(today, periods=15)
    for day in dates:
        when = datetime.combine(day.date(), time.fromisoformat(settled_after), CST)
        if when > now and (schedule is None or day not in schedule.index or schedule.at[day, "is_session"] == 1):
            return when
    return datetime.combine((today + pd.Timedelta(days=1)).date(), time.fromisoformat(settled_after), CST)


def _log_event(output_dir: Path | None, message: str, payload: dict) -> None:
    print(message, flush=True)
    if output_dir is None:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "watch.log").open("a", encoding="utf-8") as stream:
        stream.write(f"[{datetime.now(CST).isoformat(timespec='seconds')}] {message}\n")
    write_json(output_dir / "watch_status.json", payload)


def watch_daily(run, calendar_path, settled_after, output_dir: Path | None = None):
    cycles = 0
    try:
        while True:
            cycles += 1
            error_text = None
            try:
                status = run()
                calendar = load_sessions(calendar_path)
            except (OSError, ValueError, RuntimeError) as error:
                error_text = str(error)
                status, calendar = 2, None
            now = datetime.now(CST)
            when = next_check(now, status, calendar, settled_after)
            summary = f"Cycle {cycles}: exit={status}" + (f" error={error_text}" if error_text else "") + f" | next={when.isoformat(timespec='minutes')}"
            _log_event(output_dir, summary, {
                "updated_utc": utc_now(),
                "cycle": cycles,
                "last_exit_code": status,
                "last_error": error_text,
                "next_check_cst": when.isoformat(timespec="seconds"),
            })
            clock.sleep(max(1, (when - now).total_seconds()))
    except KeyboardInterrupt:
        _log_event(output_dir, "Watcher stopped by user", {"updated_utc": utc_now(), "state": "stopped", "cycles": cycles})
        return 0
