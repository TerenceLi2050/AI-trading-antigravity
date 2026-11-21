from contextlib import ExitStack, contextmanager
from io import BytesIO
import json
from pathlib import Path
import socket
import tempfile
import time
from unittest.mock import patch
from uuid import uuid4

import numpy as np
import pandas as pd

from .io import dates, file_lock, numeric, require, sha256_file, stock_codes, utc_now, write_csv, write_json, write_parquet


CSI_SNAPSHOT = "https://oss-ch.csindex.com.cn/static/html/csindex/public/uploads/file/autofile/cons/000906cons.xls"
INDEX_API = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
FINANCIAL_API = "https://datacenter-web.eastmoney.com/api/data/v1/get"
SAMPLES = {"000001": "银行", "000333": "制造业", "300750": "创业板", "688027": "科创板", "000792": "历史停复牌"}
RAW_FIELDS = "date,code,open,high,low,close,preclose,volume,amount,turn,tradestatus,pctChg,isST"
ADJ_FIELDS = "date,code,open,high,low,close"


def _get(url: str, **params):
    import requests

    for attempt in range(3):
        try:
            response = requests.get(url, params=params, timeout=(7, 20))
            response.raise_for_status()
            return response
        except (requests.ConnectionError, requests.Timeout):
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def _quarter_snapshot(day: pd.Timestamp, codes: set[str], period: pd.Timestamp) -> pd.DataFrame:
    params = dict(sortColumns="SECURITY_CODE", sortTypes="1", pageSize=500, pageNumber=1,
                  reportName="RPT_LICO_FN_CPD", columns="ALL", filter=f"(REPORTDATE='{period:%Y-%m-%d}')")
    first = _get(FINANCIAL_API, **params).json()["result"]
    if not first or not first["pages"]:
        return pd.DataFrame()
    if not 1 <= first["pages"] <= 30:
        raise ValueError("Unexpected number of quarterly disclosure pages")
    records = list(first["data"])
    for page in range(2, first["pages"] + 1):
        params["pageNumber"] = page
        records.extend(_get(FINANCIAL_API, **params).json()["result"]["data"])
    frame = pd.DataFrame(records)
    selection = {"SECURITY_CODE": "code", "REPORTDATE": "report_date", "NOTICE_DATE": "ann_date",
                 "UPDATE_DATE": "vendor_update", "PARENT_NETPROFIT": "profit_ytd", "BASIC_EPS": "eps_ytd",
                 "WEIGHTAVG_ROE": "roe_ytd_pct", "BPS": "book_per_share", "MGJYXJJE": "operating_cash_per_share_ytd",
                 "SJLTZ": "profit_growth_pct", "YSTZ": "revenue_growth_pct", "PUBLISHNAME": "industry"}
    chosen = frame.loc[frame["SECURITY_CODE"].isin(codes)].copy()
    if "PUBLISHNAME" not in chosen and "TRADE_MARKET" in chosen:
        chosen["PUBLISHNAME"] = pd.NA
    for optional in ("PUBLISHNAME", "SJLTZ", "YSTZ"):
        if optional not in chosen:
            chosen[optional] = pd.NA
    require(chosen, selection, ["SECURITY_CODE"], "Eastmoney current financial snapshot")
    result = chosen[list(selection)].rename(columns=selection)
    for name in ("report_date", "ann_date", "vendor_update"):
        result[name] = pd.to_datetime(result[name], errors="raise", format="mixed").dt.normalize()
    numeric(result, list(selection.values())[4:-1])
    valid = (result["report_date"].eq(period) & result["ann_date"].ge(result["report_date"])
             & result["ann_date"].lt(day) & result["vendor_update"].le(day))
    return result.loc[valid].assign(snapshot_date=day, source="Eastmoney RPT_LICO_FN_CPD")


def _disclosures(day: pd.Timestamp, codes: set[str]) -> pd.DataFrame:
    best = pd.DataFrame()
    core = ["profit_ytd", "eps_ytd", "roe_ytd_pct", "book_per_share", "industry"]
    for quarters in (1, 2, 3):
        period = day - pd.offsets.QuarterEnd(quarters)
        if (day - period).days > 190:
            break
        frame = _quarter_snapshot(day, codes, period)
        if frame.empty:
            continue
        if frame[core].notna().all(axis=1).sum() >= 0.80 * len(codes):
            return frame
        if len(frame) > len(best):
            best = frame
    if best.empty:
        raise ValueError("No recent common-period financial snapshot is available")
    return best


def _index(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    response = _get(INDEX_API, secid="1.000906", fields1="f1,f2,f3,f4,f5",
                    fields2="f51,f52,f53", klt="101", fqt="0", beg=start.strftime("%Y%m%d"), end=end.strftime("%Y%m%d"))
    payload = response.json()["data"]
    if not payload or not payload.get("klines") or str(payload.get("code", "")) != "000906":
        raise ValueError("CSI 800 index response was empty or unexpected")
    frame = pd.DataFrame([row.split(",") for row in payload["klines"]], columns=["date", "market_open", "market_close"])
    frame["date"] = dates(frame["date"])
    numeric(frame, ["market_open", "market_close"])
    require(frame, frame.columns, ["date"], "Eastmoney CSI 800 closes")
    if not (frame[["market_open", "market_close"]].gt(0).all().all() and frame["date"].between(start, end).all()):
        raise ValueError("Index feed dates or prices failed validation")
    frame.attrs["source"] = "Eastmoney 000906 open/close"
    return frame.sort_values("date")


def _benchmark(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    try:
        return _index(start, end)
    except Exception as first_error:
        with _baostock() as bs:
            frame = _rows(bs.query_history_k_data_plus("sh.000906", "date,code,open,close",
                                                       str(start.date()), str(end.date()), adjustflag="3"))
        if frame.empty or not frame["code"].eq("sh.000906").all():
            raise first_error
        frame = frame.rename(columns={"open": "market_open", "close": "market_close"}).drop(columns="code")
        frame["date"] = dates(frame["date"])
        numeric(frame, ["market_open", "market_close"])
        require(frame, frame.columns, ["date"], "BaoStock CSI 800 index")
        if not frame[["market_open", "market_close"]].gt(0).all().all():
            raise first_error
        frame.attrs["source"] = "BaoStock sh.000906 open/close"
        return frame.sort_values("date")


def _snapshot() -> tuple[pd.DataFrame, bytes]:
    value = _get(CSI_SNAPSHOT).content
    table = pd.read_excel(BytesIO(value), dtype=str)
    if len(table.columns) != 9 or len(table) != 800:
        raise ValueError("Unexpected official CSI 800 constituent snapshot schema or size")
    result = table.iloc[:, [0, 1, 4, 5]].copy()
    result.columns = ["snapshot_date", "index_code", "code", "name"]
    result["snapshot_date"] = pd.to_datetime(result["snapshot_date"], format="%Y%m%d", errors="raise")
    result["code"] = stock_codes(result["code"])
    require(result, result.columns, ["code"], "CSI current snapshot")
    if not result["index_code"].str.zfill(6).eq("000906").all() or result["snapshot_date"].nunique() != 1:
        raise ValueError("Provided constituents are not one official CSI 800 snapshot")
    return result, value


class _BaoConnection:
    """The vendor's receive loop needs an exception on EOF as well as a per-message deadline."""
    def __init__(self, connection):
        self.connection = connection
        self.deadline = 0.0
        self.next_send = 0.0

    def send(self, message):
        pause = self.next_send - time.monotonic()
        if pause > 0:
            time.sleep(pause)
        self.next_send = time.monotonic() + 0.25
        self.connection.settimeout(12)
        self.deadline = time.monotonic() + 45
        self.connection.sendall(message)
        return len(message)

    def recv(self, size):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("BaoStock message deadline exceeded")
        self.connection.settimeout(min(12, remaining))
        block = self.connection.recv(size)
        if not block:
            raise ConnectionResetError("BaoStock closed the connection before completing the message")
        return block

    def close(self):
        self.connection.close()


class BaoStockAccessError(RuntimeError):
    def __init__(self, code, retry_after=None):
        self.code = code
        detail = f"; local retry after {retry_after}" if retry_after else ""
        super().__init__(f"BaoStock access refused ({code}){detail}")


def _check_bao_result(result):
    code = str(result.error_code)
    if code in {"10001002", "10001004", "10001005", "10001006", "10001007", "10001011"}:
        raise BaoStockAccessError(code)
    if code != "0":
        raise ConnectionError(f"BaoStock returned {code}")


def _bao_state_dir():
    return Path(tempfile.gettempdir()) / "csi800-baostock-provider"


def baostock_cooldown_status() -> dict:
    pause = _bao_state_dir() / "access.json"
    if not pause.is_file():
        return {"in_cooldown": False}
    try:
        info = json.loads(pause.read_text(encoding="utf-8"))
        retry_after = pd.Timestamp(info["retry_after"])
        active = retry_after > pd.Timestamp.now(tz="UTC")
        return {"in_cooldown": bool(active), **info}
    except (OSError, ValueError, KeyError):
        return {"in_cooldown": False}


@contextmanager
def _baostock():
    import baostock as bs
    from baostock.common import contants, context
    from baostock.util.socketutil import SocketUtil

    state = _bao_state_dir()
    pause = state / "access.json"
    with file_lock(state / "connection.lock", wait_for_collection=True):
        if pause.is_file():
            previous = json.loads(pause.read_text(encoding="utf-8"))
            if pd.Timestamp(previous["retry_after"]) > pd.Timestamp.now(tz="UTC"):
                raise BaoStockAccessError(previous["code"], previous["retry_after"])
        try:
            with socket.create_connection((contants.BAOSTOCK_SERVER_IP, contants.BAOSTOCK_SERVER_PORT), timeout=8) as wire:
                connection = _BaoConnection(wire)
                with patch.object(SocketUtil, "connect", lambda _: setattr(context, "default_socket", connection)):
                    _check_bao_result(bs.login())
                pause.unlink(missing_ok=True)
                yield bs
                bs.logout()
        except BaoStockAccessError as exc:
            # A local minimum retry interval, independent of the provider's own enforcement.
            retry = (pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=1)).isoformat()
            write_json(pause, {"code": exc.code, "observed_utc": utc_now(), "retry_after": retry})
            raise BaoStockAccessError(exc.code, retry) from exc
        finally:
            context.default_socket = None


def _rows(result) -> pd.DataFrame:
    rows = []
    while result.error_code == "0" and result.next():
        rows.append(result.get_row_data())
    _check_bao_result(result)
    return pd.DataFrame(rows, columns=result.fields)


def _bao_code(code: str) -> str:
    return f"{'sh' if code.startswith('6') else 'sz'}.{code}"


def _ak_code(code: str) -> str:
    return f"{'sh' if code.startswith('6') else 'sz'}{code}"


def _em_secid(code: str) -> str:
    return f"{'1' if code.startswith('6') else '0'}.{code}"


def _bars_eastmoney(code: str, start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch raw (fqt=0) and HFQ (fqt=2) daily stock bars directly from Eastmoney push2his API."""
    secid = _em_secid(code)
    beg, fin = start.strftime("%Y%m%d"), end.strftime("%Y%m%d")
    common = dict(
        secid=secid,
        fields1="f1,f2,f3,f4,f5,f6",
        fields2="f51,f52,f53,f54,f55,f56,f57,f59,f61",
        klt="101",
        beg=beg,
        end=fin,
    )
    raw_json = _get(INDEX_API, fqt="0", **common).json().get("data")
    adj_json = _get(INDEX_API, fqt="2", **common).json().get("data")
    if not raw_json or not raw_json.get("klines") or not adj_json or not adj_json.get("klines"):
        raise ValueError(f"Eastmoney returned empty klines for {code}")
    cols = ["date", "open", "close", "high", "low", "volume", "amount", "vendor_pct", "turnover"]
    raw = pd.DataFrame([row.split(",") for row in raw_json["klines"]], columns=cols)
    adj = pd.DataFrame([row.split(",") for row in adj_json["klines"]], columns=cols)
    return raw, adj


def _normalize_eastmoney(raw: pd.DataFrame, adjusted: pd.DataFrame, reference: pd.DataFrame, code: str) -> pd.DataFrame:
    keys = ["date", "code"]
    legs = ["open", "high", "low", "close"]
    raw = raw.copy()
    adjusted = adjusted.copy()
    raw["date"] = dates(raw["date"])
    adjusted["date"] = dates(adjusted["date"])
    raw["code"] = code
    adjusted["code"] = code
    raw = raw.rename(columns={leg: f"raw_{leg}" for leg in legs})
    adjusted = adjusted[["date", "code", *legs]].rename(columns={leg: f"adj_{leg}" for leg in legs})
    num_raw = [f"raw_{l}" for l in legs] + ["volume", "amount", "vendor_pct", "turnover"]
    numeric(raw, num_raw)
    numeric(adjusted, [f"adj_{l}" for l in legs])
    # Eastmoney reports volume in lots (100 shares) and turnover in percent.
    raw["volume"] = raw["volume"] * 100.0
    raw["turnover"] = raw["turnover"] / 100.0
    merged = raw[[*keys, *num_raw]].merge(adjusted, on=keys, how="inner", validate="one_to_one").sort_values("date")
    if merged.empty:
        raise ValueError("Eastmoney raw and HFQ bars have no overlapping dates")
    merged["raw_preclose"] = merged["raw_close"].shift(1)
    merged["trade_status"] = np.where(merged["volume"].gt(0) & merged["amount"].gt(0), 1.0, 0.0)
    merged["is_st"] = 0.0
    if not reference.empty:
        overlap = (merged.loc[merged["trade_status"].eq(1), ["date", "raw_close", "adj_close"]]
                   .merge(reference[["date", "raw_close", "adj_close"]], on="date", suffixes=("", "_existing"))
                   .sort_values("date").tail(8))
        ratios = overlap["adj_close_existing"].div(overlap["adj_close"])
        common_price = overlap["raw_close_existing"].sub(overlap["raw_close"]).abs().le(0.02)
        if len(ratios) < 2 or not common_price.all() or not ratios.gt(0).all() or ratios.div(ratios.median()).sub(1).abs().max() > 0.015:
            raise ValueError("Cannot reconcile Eastmoney corporate-action scale on overlapping sessions")
        merged[[f"adj_{leg}" for leg in legs]] *= ratios.median()
    merged["adj_vwap"] = (merged["amount"].div(merged["volume"].where(merged["volume"].gt(0)))
                          * merged["adj_close"].div(merged["raw_close"].where(merged["raw_close"].gt(0))))
    merged["price_source"] = "eastmoney_raw_and_hfq"
    merged["total_shares"] = merged["limit_up"] = merged["limit_down"] = np.nan
    merged["snapshot_industry"] = pd.NA
    return merged


def _bars_akshare(code: str, start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fallback bar loader inspired by the original AkShare workflow, with strict raw/HFQ separation."""
    import akshare as ak

    symbol = _ak_code(code)
    s_date, e_date = start.strftime("%Y%m%d"), end.strftime("%Y%m%d")
    raw = ak.stock_zh_a_daily(symbol=symbol, start_date=s_date, end_date=e_date, adjust="")
    adjusted = ak.stock_zh_a_daily(symbol=symbol, start_date=s_date, end_date=e_date, adjust="hfq")
    if raw is None or raw.empty or adjusted is None or adjusted.empty:
        raise ValueError(f"AkShare returned empty bars for {code}")
    return raw, adjusted


def _normalize_akshare(raw: pd.DataFrame, adjusted: pd.DataFrame, reference: pd.DataFrame, code: str) -> pd.DataFrame:
    keys = ["date", "code"]
    legs = ["open", "high", "low", "close"]
    raw = raw.copy()
    adjusted = adjusted.copy()
    raw["date"] = dates(raw["date"])
    adjusted["date"] = dates(adjusted["date"])
    raw["code"] = code
    adjusted["code"] = code
    raw = raw.rename(columns={leg: f"raw_{leg}" for leg in legs})
    adjusted = adjusted[["date", "code", *legs]].rename(columns={leg: f"adj_{leg}" for leg in legs})
    require(raw, [*keys, *[f"raw_{l}" for l in legs], "volume", "amount"], keys, "AkShare raw bars")
    require(adjusted, [*keys, *[f"adj_{l}" for l in legs]], keys, "AkShare HFQ bars")
    if "turnover" not in raw.columns:
        raw["turnover"] = np.nan
    num_cols = [f"raw_{l}" for l in legs] + ["volume", "amount", "turnover"]
    numeric(raw, num_cols)
    numeric(adjusted, [f"adj_{l}" for l in legs])
    merged = raw[[*keys, *num_cols]].merge(adjusted, on=keys, how="inner", validate="one_to_one").sort_values("date")
    if merged.empty:
        raise ValueError("AkShare raw and HFQ bars have no overlapping dates")
    merged["raw_preclose"] = merged["raw_close"].shift(1)
    merged["vendor_pct"] = merged["raw_close"].pct_change().mul(100).fillna(0.0)
    merged["trade_status"] = np.where(merged["volume"].gt(0) & merged["amount"].gt(0), 1.0, 0.0)
    # AkShare daily bars do not carry historical ST flags; keep 0.0 only if caller overlays official status,
    # otherwise mark explicitly in price_source so downstream audits stay transparent.
    merged["is_st"] = 0.0
    if not reference.empty:
        overlap = (merged.loc[merged["trade_status"].eq(1), ["date", "raw_close", "adj_close"]]
                   .merge(reference[["date", "raw_close", "adj_close"]], on="date", suffixes=("", "_existing"))
                   .sort_values("date").tail(8))
        ratios = overlap["adj_close_existing"].div(overlap["adj_close"])
        common_price = overlap["raw_close_existing"].sub(overlap["raw_close"]).abs().le(0.02)
        if len(ratios) < 3 or not common_price.all() or not ratios.gt(0).all() or ratios.div(ratios.median()).sub(1).abs().max() > 0.01:
            raise ValueError("Cannot establish a consistent AkShare corporate-action scale on overlapping sessions")
        merged[[f"adj_{leg}" for leg in legs]] *= ratios.median()
    merged["adj_vwap"] = (merged["amount"].div(merged["volume"].where(merged["volume"].gt(0)))
                          * merged["adj_close"].div(merged["raw_close"].where(merged["raw_close"].gt(0))))
    merged["price_source"] = "akshare_raw_and_hfq"
    merged["total_shares"] = merged["limit_up"] = merged["limit_down"] = np.nan
    merged["snapshot_industry"] = pd.NA
    return merged


def _quarter_snapshot_history(period: pd.Timestamp, codes: set[str] | None = None) -> pd.DataFrame:
    """Fetch a historical quarter-end disclosure batch from Eastmoney with original announcement dates."""
    params = dict(sortColumns="SECURITY_CODE", sortTypes="1", pageSize=500, pageNumber=1,
                  reportName="RPT_LICO_FN_CPD", columns="ALL", filter=f"(REPORTDATE='{period:%Y-%m-%d}')")
    payload = _get(FINANCIAL_API, **params).json().get("result")
    if not payload or not payload.get("pages"):
        return pd.DataFrame()
    if not 1 <= payload["pages"] <= 35:
        raise ValueError("Unexpected number of historical quarterly disclosure pages")
    records = list(payload["data"])
    for page in range(2, payload["pages"] + 1):
        params["pageNumber"] = page
        records.extend(_get(FINANCIAL_API, **params).json()["result"]["data"])
    frame = pd.DataFrame(records)
    if frame.empty:
        return pd.DataFrame()
    frame["SECURITY_CODE"] = frame["SECURITY_CODE"].astype("string").str.strip()
    frame = frame.loc[frame["SECURITY_CODE"].str.fullmatch(r"(?:00|30|60|68)\d{4}", na=False)].copy()
    if codes is not None:
        frame = frame.loc[frame["SECURITY_CODE"].isin(codes)].copy()
    if frame.empty:
        return pd.DataFrame()
    selection = {"SECURITY_CODE": "code", "REPORTDATE": "report_date", "NOTICE_DATE": "ann_date",
                 "UPDATE_DATE": "vendor_update", "PARENT_NETPROFIT": "profit_ytd", "BASIC_EPS": "eps_ytd",
                 "WEIGHTAVG_ROE": "roe_ytd_pct", "BPS": "book_per_share", "MGJYXJJE": "operating_cash_per_share_ytd",
                 "SJLTZ": "profit_growth_pct", "YSTZ": "revenue_growth_pct", "PUBLISHNAME": "industry"}
    for optional in ("PUBLISHNAME", "SJLTZ", "YSTZ", "UPDATE_DATE"):
        if optional not in frame:
            frame[optional] = pd.NA
    require(frame, list(selection), ["SECURITY_CODE"], "Eastmoney historical financial batch")
    result = frame[list(selection)].rename(columns=selection).copy()
    result["code"] = stock_codes(result["code"])
    for name in ("report_date", "ann_date", "vendor_update"):
        result[name] = pd.to_datetime(result[name], errors="coerce", format="mixed").dt.normalize()
    numeric(result, list(selection.values())[4:-1])
    valid = result["report_date"].eq(period) & result["ann_date"].notna() & result["ann_date"].ge(result["report_date"])
    result = result.loc[valid].copy()
    result["version_known_date"] = result["ann_date"]
    result["source"] = "Eastmoney RPT_LICO_FN_CPD"
    result["version"] = result["ann_date"].dt.strftime("%Y-%m-%d")
    return result


def _sessions_akshare(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    import akshare as ak

    raw = ak.tool_trade_date_hist_sina()
    open_days = pd.DatetimeIndex(pd.to_datetime(raw["trade_date"])).normalize()
    full = pd.date_range(start, end, name="date")
    frame = pd.DataFrame({"date": full, "is_session": full.isin(open_days).astype(int),
                          "source": "AkShare tool_trade_date_hist_sina"})
    return frame


def _bars(bs, code: str, start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    iso = (str(start.date()), str(end.date()))
    raw = _rows(bs.query_history_k_data_plus(_bao_code(code), RAW_FIELDS, *iso, adjustflag="3"))
    adjusted = _rows(bs.query_history_k_data_plus(_bao_code(code), ADJ_FIELDS, *iso, adjustflag="1"))
    return raw, adjusted


def _sample(frame: pd.DataFrame) -> dict:
    return {"rows": len(frame), "fields": list(frame.columns), "nonempty": (frame.notna() & frame.ne("")).sum().to_dict(),
            "first_two_rows": frame.head(2).astype("string").fillna("").to_dict("records")}


def _error(exc: Exception) -> dict:
    return {"status": "failed", "error_type": type(exc).__name__,
            "errno": getattr(exc, "errno", None),
            "message": str(exc)[:350] if isinstance(exc, (ValueError, TypeError, KeyError, BaoStockAccessError)) else None}


def probe(day: pd.Timestamp) -> dict:
    checks = {"created_utc": utc_now(), "samples": SAMPLES}
    for name, load in (("csi_official_snapshot", lambda: _snapshot()[0]),
                       ("eastmoney_benchmark", lambda: _index(day - pd.Timedelta(days=60), day)),
                       ("eastmoney_financial_schema", lambda: pd.DataFrame(_get(FINANCIAL_API, pageSize=1, pageNumber=1,
                        reportName="RPT_LICO_FN_CPD", columns="ALL", filter=f"(REPORTDATE='{day - pd.offsets.QuarterEnd(1):%Y-%m-%d}')").json()["result"]["data"]))):
        try:
            checks[name] = {"status": "ok", **_sample(load())}
        except Exception as exc:
            checks[name] = _error(exc)
    try:
        quote_checks = {}
        with _baostock() as bs:
            for code in SAMPLES:
                raw, adjusted = _bars(bs, code, day - pd.Timedelta(days=20), day)
                fields = {"raw": _sample(raw), "adjusted": _sample(adjusted)}
                for name in ("query_profit_data", "query_cash_flow_data", "query_balance_data", "query_growth_data"):
                    fields[name] = _sample(_rows(getattr(bs, name)(_bao_code(code), day.year - 1, 4)))
                quote_checks[code] = fields
        checks["baostock"] = {"status": "ok", "securities": quote_checks,
                             "acceptance": "Bar joins require code/date match; filings require source revision and announcement evidence."}
    except Exception as exc:
        checks["baostock"] = _error(exc)
    return checks


def _normalize(raw: pd.DataFrame, adjusted: pd.DataFrame, reference: pd.DataFrame, code: str):
    keys = ["date", "code"]
    legs = ["open", "high", "low", "close"]
    raw = raw.rename(columns={**{leg: f"raw_{leg}" for leg in legs}, "turn": "turnover", "isST": "is_st",
                              "tradestatus": "trade_status", "pctChg": "vendor_pct"}).copy()
    adjusted = adjusted.rename(columns={leg: f"adj_{leg}" for leg in legs}).copy()
    for frame in (raw, adjusted):
        require(frame, keys, keys, "BaoStock bars")
        if not frame["code"].eq(_bao_code(code)).all():
            raise ValueError("BaoStock response contains a different security")
        frame["date"] = dates(frame["date"])
        frame["code"] = code
        fields = set(frame) - set(keys)
        frame[list(fields)] = frame[list(fields)].replace("", np.nan)
        numeric(frame, fields)
    available = raw.merge(adjusted, on=keys, how="left", validate="one_to_one")
    open_rows = available["trade_status"].eq(1)
    if (not available.loc[open_rows, [f"adj_{leg}" for leg in legs]].gt(0).all().all()
            or not available["trade_status"].isin([0, 1]).all()
            or not available["is_st"].isin([0, 1]).all()):
        raise ValueError("Raw and adjusted bars or trading states do not match")
    if not reference.empty:
        overlap = (available.loc[open_rows, ["date", "raw_close", "adj_close"]].merge(
            reference[["date", "raw_close", "adj_close"]], on="date", suffixes=("", "_existing"))
                   .sort_values("date").tail(8))
        ratios = overlap["adj_close_existing"].div(overlap["adj_close"])
        common_price = overlap["raw_close_existing"].sub(overlap["raw_close"]).abs().le(0.011)
        if len(ratios) < 3 or not common_price.all() or not ratios.gt(0).all() or ratios.div(ratios.median()).sub(1).abs().max() > 0.005:
            raise ValueError("Cannot establish a consistent corporate-action scale on overlapping sessions")
        available[[f"adj_{leg}" for leg in legs]] *= ratios.median()
    available["turnover"] /= 100
    available["adj_vwap"] = (available["amount"].div(available["volume"].where(available["volume"].gt(0)))
                             * available["adj_close"].div(available["raw_close"].where(available["raw_close"].gt(0))))
    available["price_source"] = "baostock_raw_and_hfq"
    available["total_shares"] = available["limit_up"] = available["limit_down"] = np.nan
    available["snapshot_industry"] = pd.NA
    return available.rename(columns={"preclose": "raw_preclose"})


def _download_batch(codes, reference, anchor, day, cache, revalidate=False):
    failures = {}
    with ExitStack() as connection:
        bs = None
        for number, code in enumerate(codes, 1):
            path = cache / f"{code}.parquet"
            if path.is_file():
                continue
            prior = reference.get(code, pd.DataFrame())
            security_anchor = prior["date"].max() if not prior.empty else anchor
            start = security_anchor - pd.Timedelta(days=45 if len(prior) else 240)
            for attempt in range(3):
                try:
                    if bs is None:
                        bs = connection.enter_context(_baostock())
                    raw, adjusted = _bars(bs, code, start, day)
                    normalized = _normalize(raw, adjusted, prior, code)
                    if len(prior):
                        after = normalized["date"].gt(security_anchor)
                        if revalidate:
                            after |= normalized["date"].eq(day)
                        normalized = normalized.loc[after]
                    if normalized.empty:
                        raise ValueError("No new rows for this security")
                    write_parquet(path, normalized)
                    break
                except (OSError, UnicodeDecodeError) as exc:
                    connection.close()
                    bs = None
                    if attempt == 2:
                        failures[code] = _error(exc)
                    else:
                        time.sleep(attempt + 1)
                except (ValueError, KeyError) as exc:
                    failures[code] = _error(exc)
                    break
            if number % 50 == 0:
                print(f"BaoStock shard: {number}/{len(codes)} examined, {len(failures)} rejected", flush=True)
            if len(failures) > 20:
                break
    return failures


def refresh(base: pd.DataFrame, output: Path, day: pd.Timestamp, closed_releases: set[str] | None = None) -> dict:
    release_ids = base["_release"] if "_release" in base else pd.Series(pd.NA, index=base.index)
    base = base.drop(columns="_release", errors="ignore")
    known = base.loc[base["date"].le(day)]
    anchor = known["date"].max()
    if pd.isna(anchor):
        return {"status": "failed", "error_type": "NoAnchorBeforeCutoff"}
    stage = "benchmark"
    try:
        index = _benchmark(anchor - pd.Timedelta(days=14), day)
        upcoming = index.loc[index["date"].gt(anchor)]
        stage = "constituents"
        snapshot, snapshot_bytes = _snapshot()
        if snapshot["snapshot_date"].iat[0] > day:
            raise ValueError("Current snapshot was released after the requested cutoff")
        latest = upcoming["date"].max() if not upcoming.empty else anchor
        if snapshot["snapshot_date"].iat[0] != latest:
            raise ValueError("Official snapshot date differs from the last index session in the feed")
        stage = "current_financials"
        try:
            financial = _disclosures(latest, set(snapshot["code"]))
            financial_status = {"status": "ok", "rows": len(financial)}
        except Exception as exc:
            financial = None
            financial_status = _error(exc)
        pool = snapshot["code"].tolist()
        if upcoming.empty:
            confirmed = known["date"].eq(anchor)
            if closed_releases is not None:
                confirmed &= release_ids.reindex(known.index).isin(closed_releases)
            present = set(known.loc[confirmed, "code"])
            pool = [code for code in pool if code not in present]
        if upcoming.empty and not pool:
            existing = sorted(output.glob(f"{latest:%Y%m%d}-*/manifest.json"))
            if existing and financial is None:
                return {"status": "up_to_date", "anchor": str(anchor.date()), "current_financials": financial_status}
            release = output / f"{latest:%Y%m%d}-{pd.Timestamp.now(tz='UTC'):%H%M%S}-{uuid4().hex[:8]}"
            release.mkdir(parents=True, exist_ok=False)
            return _publish(release, base.iloc[:0], snapshot, snapshot_bytes, financial, financial_status, anchor, latest, 1.0, {})
        local = known.loc[known["date"].eq(anchor), ["market_open", "market_close"]].iloc[0]
        common = index.loc[index["date"].eq(anchor), ["market_open", "market_close"]]
        if common.empty or not common.sub(local).abs().lt(0.02).all().all():
            raise ValueError("External index benchmark does not reconcile with the local anchor")
        stage = "bars"
        anchor_rows = (known.loc[known["code"].isin(pool), ["code", "date", "raw_close", "adj_close"]]
                       .sort_values("date").groupby("code").tail(20))
        anchor_id = int(pd.util.hash_pandas_object(anchor_rows, index=False).sum())
        reference = {code: rows for code, rows in anchor_rows.groupby("code")}
        reconfirm = upcoming.empty and closed_releases is not None
        cache = output / ".work" / f"v3-{anchor:%Y%m%d}-{latest:%Y%m%d}-{anchor_id:016x}-{int(reconfirm)}"
        cache.mkdir(parents=True, exist_ok=True)
        # Preserve the earliest request timestamp when resuming a partial batch.
        started = cache / "first_request.txt"
        try:
            with started.open("x", encoding="utf-8") as stream:
                stream.write(utc_now())
        except FileExistsError:
            pass
        failures = {}
        pending = [code for code in pool if not (cache / f"{code}.parquet").is_file()]
        if pending:
            extra = (True,) if reconfirm else ()
            failures = _download_batch(pending, reference, anchor, latest, cache, *extra)
        frames = [pd.read_parquet(path) for code in pool if (path := cache / f"{code}.parquet").is_file()]
        if not frames or len(failures) > 40:
            return {"status": "failed", "error_type": "TooManyQuoteFailures", "failed_codes": failures}
        appended = pd.concat(frames, ignore_index=True)
        # Newly entering names may bring earlier history. Reuse the existing benchmark for past sessions.
        old_index = known.groupby("date")[["market_open", "market_close"]].first().reset_index()
        benchmarks = pd.concat([old_index, upcoming]).drop_duplicates("date", keep="last")
        appended = appended.merge(benchmarks, on="date", how="inner", validate="many_to_one")
        check_sessions = upcoming["date"] if not upcoming.empty else pd.DatetimeIndex([latest])
        previous = known.loc[known["date"].isin(check_sessions) & known["code"].isin(snapshot["code"]), ["date", "code"]]
        covered = pd.concat([previous, appended[["date", "code"]]], ignore_index=True).groupby("date")["code"].nunique()
        coverage = covered.reindex(check_sessions, fill_value=0) / len(snapshot)
        if coverage.lt(0.95).any():
            return {"status": "failed", "error_type": "NewSessionCoverageTooLow", "minimum_coverage": float(coverage.min()),
                    "failed_codes": failures}
        release = output / f"{latest:%Y%m%d}-{pd.Timestamp.now(tz='UTC'):%H%M%S}-{uuid4().hex[:8]}"
        release.mkdir(parents=True, exist_ok=False)
        return _publish(release, appended[base.columns], snapshot, snapshot_bytes, financial, financial_status,
                        anchor, latest, float(coverage.min()), failures, started.read_text(encoding="utf-8"))
    except Exception as exc:
        return {**_error(exc), "stage": stage}


def refresh_disclosures(base: pd.DataFrame, output: Path, cutoff: pd.Timestamp) -> dict:
    try:
        snapshot, payload = _snapshot()
        day = snapshot["snapshot_date"].iat[0]
        if day > cutoff:
            raise ValueError("Official constituent snapshot is after cutoff")
        disclosures = _disclosures(day, set(snapshot["code"]))
        release = output / f"{day:%Y%m%d}-{pd.Timestamp.now(tz='UTC'):%H%M%S}-{uuid4().hex[:8]}"
        release.mkdir(parents=True, exist_ok=False)
        return _publish(release, base.iloc[:0], snapshot, payload, disclosures,
                        {"status": "ok", "rows": len(disclosures)}, day, day, None, {})
    except Exception as exc:
        return {**_error(exc), "stage": "current_financials"}


def _publish(release, quotes, snapshot, snapshot_bytes, financial, financial_status, anchor, latest, coverage, failures,
             bars_requested_utc=None):
    path = release / "quotes.parquet"
    write_parquet(path, quotes)
    members = release / "current_membership_snapshot.csv"
    write_csv(members, snapshot)
    (release / "current_membership_snapshot.xls").write_bytes(snapshot_bytes)
    if financial is not None:
        financial_path = release / "current_financials.csv"
        write_csv(financial_path, financial)
        financial_status["sha256"] = sha256_file(financial_path)
    manifest = {"status": "published", "created_utc": utc_now(), "bars_requested_utc": bars_requested_utc,
                "anchor": str(anchor.date()),
                "last_session": str(latest.date()), "rows": len(quotes), "minimum_coverage": coverage,
                "failed_codes": failures, "current_financials": financial_status, "members_sha256": sha256_file(members),
                "sources": ["BaoStock", INDEX_API, CSI_SNAPSHOT, FINANCIAL_API], "quotes_sha256": sha256_file(path)}
    write_json(release / "manifest.json", manifest)
    return manifest
