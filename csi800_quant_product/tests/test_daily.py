from datetime import datetime, time
import json
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest

from quant_screen import daily, protocol, study
from quant_screen.config import load_config
from quant_screen.data import import_legacy
from quant_screen.io import sha256_file, write_json
from quant_screen.protocol import prospective_outcomes, record_prospective
from quant_screen.providers import _BaoConnection
from quant_screen.screen import prospective_variants
from quant_screen.sessions import CST
from quant_screen.universe_audit import momentum_sensitivity
from quant_screen.quality_study import run_quality_comparison
from quant_screen.watch import next_check
from test_pipeline import quote_fixture


def settings():
    return load_config(Path(__file__).resolve().parents[1] / "configs/default.toml")


def test_vendor_receive_loop_terminates_on_eof_and_on_message_deadline(monkeypatch):
    from quant_screen import providers
    now = [10.]
    monkeypatch.setattr(providers.time, "monotonic", lambda: now[0])
    wire = SimpleNamespace(settimeout=lambda x: None, sendall=lambda x: None, recv=lambda size: b"")
    bounded = _BaoConnection(wire)
    bounded.send(b"query")
    with pytest.raises(ConnectionResetError):
        bounded.recv(8192)
    now[0] = 56.
    with pytest.raises(TimeoutError):
        bounded.recv(8192)


def test_explicit_vendor_refusal_stops_requests_across_runs(monkeypatch, tmp_path):
    from contextlib import nullcontext
    import baostock
    from quant_screen import providers
    connections = []
    def connect(*args, **kwargs):
        connections.append(1)
        return nullcontext(object())
    monkeypatch.setattr(providers, "_bao_state_dir", lambda: tmp_path)
    monkeypatch.setattr(providers.socket, "create_connection", connect)
    monkeypatch.setattr(baostock, "login", lambda: SimpleNamespace(error_code="10001011"))
    for _ in range(2):
        with pytest.raises(providers.BaoStockAccessError, match="10001011"):
            with providers._baostock():
                pytest.fail("The provider explicitly refused access")
    assert len(connections) == 1
    assert json.loads((tmp_path / "access.json").read_text())["code"] == "10001011"


def test_clean_collection_increment_revision_and_recovery(monkeypatch, tmp_path):
    cfg = settings()
    days = pd.bdate_range("2023-01-02", periods=261, name="date")
    sessions = pd.date_range(days[0], days[-1], name="date")
    schedule = pd.DataFrame({"is_session": sessions.isin(days).astype(int), "source": "fixture"}, index=sessions)
    codes = ["000001", "000002", "600001"]
    day = days[-2]
    calls = []

    def members():
        return pd.DataFrame({"snapshot_date": day, "index_code": "000906", "code": codes, "name": codes}), b"fixture"

    def index(start, end):
        selected = days[(days >= start) & (days <= end)]
        return pd.DataFrame({"date": selected, "market_open": 100., "market_close": 101.})

    def bars(_, code, start, end):
        calls.append((code, start, end))
        dates = days[(days >= start) & (days <= end)]
        bs_code = ("sh." if code.startswith("6") else "sz.") + code
        frame = pd.DataFrame({"code": bs_code, "date": dates.strftime("%Y-%m-%d"), "open": 11., "high": 12.,
            "low": 10., "close": 11., "volume": 100000, "amount": 1100000, "turn": 1., "tradestatus": 1,
            "isST": 0, "preclose": 11., "pctChg": 0.})
        return frame, frame[["date", "code", "open", "high", "low", "close"]].copy()

    from contextlib import nullcontext
    monkeypatch.setattr(daily, "_bars", bars)
    monkeypatch.setattr(daily, "_baostock", lambda: nullcontext(object()))
    monkeypatch.setattr(daily, "_snapshot", members)
    monkeypatch.setattr(daily, "_benchmark", index)
    monkeypatch.setattr(daily, "_disclosures", lambda d, c: pd.DataFrame({"code": sorted(c)}))
    monkeypatch.setattr(daily, "utc_now", lambda: datetime.combine(day.date(), time(16), CST).isoformat())
    first = daily.collect_daily(tmp_path, schedule, day, cfg)
    assert len(calls) == 3 and len(pd.read_parquet(first.parent / "quotes.parquet")) == 780
    destination = tmp_path / "moved" / "daily"
    restored = daily.import_release(first.parent, destination)
    assert first.read_bytes() == restored.read_bytes() and restored == daily.import_release(first, destination)
    assert restored == daily.import_release(restored, destination)
    (restored.parent / "quotes.parquet").write_bytes(b"damaged")
    with pytest.raises(ValueError, match="checksum mismatch"):
        daily.import_release(first, destination)
    # An interrupted cache is revalidated independently for each security.
    next(tmp_path.glob(".work/*/000002.parquet")).write_bytes(b"partial")
    daily.collect_daily(tmp_path, schedule, day, cfg)
    assert len(calls) == 4 and calls[-1][0] == "000002"
    day = days[-1]
    second = daily.collect_daily(tmp_path, schedule, day, cfg, first)
    assert len(calls) == 7 and all(start > days[0] for _, start, _ in calls[-3:])
    panel = pd.read_parquet(second.parent / "quotes.parquet")
    assert len(panel) == 780 and panel["date"].min() == days[1] and not panel.duplicated(["date", "code"]).any()
    at = pd.Timestamp(datetime.combine(day.date(), time(15), CST))
    earlier, _ = daily.available_releases(tmp_path, day, at)
    assert earlier.parent != second.parent
    (second.parent / "current_membership_snapshot.csv").write_bytes(b"corrupt")
    previous, rejected = daily.available_releases(tmp_path, day)
    assert previous.parent != second.parent and rejected[0]["release"] == second.parent.name


def test_local_cache_is_portable_without_the_original_csv(tmp_path):
    quotes, _ = quote_fixture(4)
    path = tmp_path / "cache.parquet"
    quotes.to_parquet(path)
    write_json(tmp_path / "legacy.json", {"parquet": path.name, "parquet_sha256": sha256_file(path),
               "signature": {"schema": 2, "size": 123, "path": "old-machine"}, "sha256": "source"})
    got, _ = import_legacy(tmp_path / "absent.csv", tmp_path)
    assert len(got) == len(quotes) and "raw_preclose" in got


def test_prospective_controls_include_price_winner_excluded_by_financials():
    decisions = pd.DataFrame({"code": ["000001", "000002"], "date": pd.Timestamp("2024-06-03"),
               "member": True, "eligible": True, "is_st": 0, "closed_bar_verified": True, "price_source": "BaoStock",
               "quality_qualified": [True, False], "name": ["A", "B"], "industry": "I",
               "quality_score": [0.9, np.nan], "baseline": [0.4, 0.95]})
    picks = decisions.iloc[:1].assign(rank=1, final_score=.725)
    result = prospective_variants(decisions, picks, picks, {"portfolio": {"top_k": 1}})
    assert result.set_index("variant")["code"].to_dict() == {
        "quality_mix": "000001", "quality_mix_diversified": "000001", "price_all": "000002",
        "price_quality_pool": "000001", "quality_only": "000001"}


def test_changing_transport_keeps_the_registered_economic_rule(tmp_path):
    picks = pd.DataFrame({"code": ["000001"], "rank": [1]})
    screen = {"signal_date": "2026-11-20", "session": {"verified_codes": 800}}
    audit = {"source": "independent_daily_release", "config_sha256": "policy",
             "code_sha256": {"screen.py": "rule", "providers.py": "old-transport"}}
    def register():
        return record_prospective(tmp_path, tmp_path, screen, picks, audit, horizon=20)
    first = register()
    audit["code_sha256"]["providers.py"] = "new-transport"
    assert register() == {"status": "already_recorded", "path": first["path"]}
    audit["code_sha256"]["screen.py"] = "new-rule"
    assert register()["path"] != first["path"]


def test_frozen_horizon_duplicate_cohorts_and_interrupted_record(tmp_path, monkeypatch):
    quotes, market = quote_fixture(8)
    days = market.index
    schedule = pd.DataFrame({"is_session": 1}, index=days)
    monkeypatch.setattr(protocol, "utc_now", lambda: datetime.combine(days[0].date(), time(16), CST).isoformat())
    meta = {"signal_date": str(days[0].date()), "session": {"verified_codes": 3}}
    picks = pd.DataFrame({"code": ["000001", "000001"], "rank": [1, 1], "variant": ["quality_mix", "price_all"]})
    args = dict(horizon=2, sessions=schedule, variants=picks)
    result = record_prospective(tmp_path, tmp_path, meta, picks, {"config_sha256": "a"}, **args)
    path = Path(result["path"])
    path.unlink()  # Simulate a crash before the final signal manifest was committed.
    assert record_prospective(tmp_path, tmp_path, meta, picks, {"config_sha256": "a"}, **args)["status"] == "recorded"
    assert record_prospective(tmp_path, tmp_path, meta, picks, {"config_sha256": "a"}, **args)["status"] == "already_recorded"
    state = prospective_outcomes(tmp_path, tmp_path, quotes, market, schedule, 6)
    rows = pd.read_csv(tmp_path / "prospective_outcomes.csv")
    assert state["matured"] == 2 and set(rows["planned_exit"]) == {str(days[3].date())}
    assert rows["forward_return"].nunique() == 1 and set(rows["horizon_sessions"]) == {2}
    assert (path.parent / "settlement.json").exists()
    # A later warm-up window can no longer supply the original quotes.
    assert prospective_outcomes(tmp_path, tmp_path, quotes.iloc[:0], market.iloc[-2:], schedule, 6)["matured"] == 2


def test_lockbox_uses_frozen_window_after_dataset_advances(tmp_path, monkeypatch):
    quotes, market = quote_fixture(8)
    coverage = pd.DataFrame({"expected": 3, "coverage": 1.}, index=market.index)
    cfg = settings()
    cfg["research"]["lockbox_start"] = str(market.index[1].date())
    run = tmp_path / "runs" / "freeze"
    run.mkdir(parents=True)
    model = SimpleNamespace(train_stats={"source": "fixture"})
    joblib.dump(model, run / "model.joblib")
    audit = {"config_sha256": "cfg", "code_sha256": {"source": "hash"}, "inputs": {}, "releases": [], "legacy": {"sha256": "data"}}
    features = SimpleNamespace(panel=quotes, market=market.iloc[:5])
    assert protocol.lockbox_request("freeze", {"lockbox_state": "READY"}, audit, model, quotes.iloc[:15], features,
                                    cfg, coverage.iloc[:5], tmp_path, run)["state"] == "FROZEN"
    seen = []
    def evaluate(q, f, c, b, cover, root, out):
        seen.append(f.market.index[-1])
        return "PASSED"
    monkeypatch.setattr(study, "evaluate_locked", evaluate)
    features.market = market
    check = protocol.lockbox_request("evaluate", {"lockbox_state": "SEALED"}, audit, None, quotes, features, cfg,
                                     coverage, tmp_path, run)
    assert check["state"] == "PASSED" and seen == [market.index[4]]
    (run / "frozen_evaluation/market.parquet").write_bytes(b"changed")
    assert protocol.evaluate_frozen(tmp_path, run, audit["code_sha256"])["state"] == "PROTOCOL_MISMATCH"


def test_watch_uses_calendar_and_retries_while_data_lags():
    friday = datetime(2026, 11, 20, 16, tzinfo=CST)
    dates = pd.date_range("2026-11-20", periods=4)
    calendar = pd.DataFrame({"is_session": [1, 0, 0, 1]}, index=dates)
    assert next_check(friday, 0, calendar, "15:35") == datetime(2026, 11, 23, 15, 35, tzinfo=CST)
    assert next_check(friday, 2, calendar, "15:35") == datetime(2026, 11, 20, 16, 30, tzinfo=CST)


def test_resolved_names_survive_a_missing_peer_and_a_later_data_gap(tmp_path, monkeypatch):
    quotes, market = quote_fixture(6)
    days = market.index
    calendar = pd.DataFrame({"is_session": 1}, index=days)
    monkeypatch.setattr(protocol, "utc_now", lambda: datetime.combine(days[0].date(), time(16), CST).isoformat())
    picks = pd.DataFrame({"rank": [1, 2], "code": ["000001", "000002"]})
    meta = {"signal_date": str(days[0].date()), "session": {"verified_codes": 3}}
    record_prospective(tmp_path, tmp_path, meta, picks, {"config_sha256": "setup"}, horizon=2, sessions=calendar)
    hole = quotes["date"].eq(days[3]) & quotes["code"].eq("000002")
    assert prospective_outcomes(tmp_path, tmp_path, quotes.loc[~hole], market, calendar, 5)["matured"] == 1
    assert protocol.tracking_codes(tmp_path) == ["000002"]
    later = quotes.loc[quotes["code"].eq("000002")]
    assert prospective_outcomes(tmp_path, tmp_path, later, market, calendar, 9)["matured"] == 2
    assert protocol.tracking_codes(tmp_path) == []


def test_universe_sensitivity_reranks_new_members_on_the_same_sessions():
    quotes, market = quote_fixture(80)
    day = market.index[-1]
    member = pd.DataFrame({"date": day, "requested_date": day, "code": ["000001", "600999"]})
    new = quotes.loc[quotes["code"].eq("000001")].copy()
    new["code"] = "600999"
    new.loc[new["date"].ge(market.index[-10]), "adj_close"] *= 1.5
    rows = momentum_sensitivity(member, quotes, new, 1)
    assert rows["members_with_signal"].iat[0] == 2 and rows["observed_top_missing_from_fixed"].iat[0] == 1
    new.loc[new["date"].ge(market.index[-4]), "adj_close"] *= 2
    pd.testing.assert_frame_equal(rows, momentum_sensitivity(member, quotes, new, 1))


def test_real_rule_comparison_can_run_without_model_training(tmp_path):
    quotes, market = quote_fixture(12)
    cfg = settings()
    cfg["data"].update(minimum_cross_section=2, minimum_member_count=3)
    cfg["research"].update(min_train_dates=2, sample_every=1, rebalance_every=2)
    cfg["portfolio"].update(top_k=1, keep_rank=1)
    panel = quotes[["date", "code", "is_st", "price_source"]].assign(member=True, eligible=True, industry="I",
            adv20=1e8, vol60=.02, baseline=quotes["code"].map({"000001": .1, "000002": .2, "600001": .9}))
    hist = pd.DataFrame({"code": ["000001", "000002", "600001"], "report_date": pd.Timestamp("2023-12-31"),
            "version_known_date": pd.Timestamp("2024-01-01"), "roe_ytd_pct": [16, 8, -2], "book_per_share": 5.,
            "profit_ytd": [80., 24., -1.], "eps_ytd": 1., "operating_cash_per_share_ytd": 1.,
            "profit_growth_pct": [10, 3, -5], "revenue_growth_pct": [6, 1, -5]})
    coverage = pd.DataFrame({"expected": 3, "coverage": 1.}, index=market.index)
    result = run_quality_comparison(quotes, SimpleNamespace(panel=panel), market, pd.DataFrame(), pd.DataFrame(),
                                    coverage, hist, cfg, tmp_path)
    assert result["state"] == "DEVELOPMENT_DIAGNOSTIC" and not result["model_comparison"]
    fills = pd.read_csv(tmp_path / "quality_strategy_fills.csv", dtype={"code": "string"})
    assert set(fills.loc[fills["strategy"].eq("price_all"), "code"]) == {"600001"}
    assert set(fills.loc[fills["strategy"].eq("quality_mix"), "code"]) == {"000001"}
