from datetime import datetime, time
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from quant_screen.allocation import hierarchical_weights
from quant_screen import providers
from quant_screen.backtest import simulate
from quant_screen.config import load_config
from quant_screen.current import apply_current_membership, apply_current_quality
from quant_screen.data import apply_raw_history, benchmark_for
from quant_screen.features import build_features
from quant_screen.labels import executable_labels
from quant_screen.models import training_sample
from quant_screen.market import trade_flags
from quant_screen.pit import fundamentals, membership
from quant_screen.providers import _normalize
from quant_screen.screen import generate
from quant_screen.sessions import CST, last_completed, readiness
from quant_screen.quality_study import historical_quality_signals, run_quality_comparison
from quant_screen.protocol import lockbox_request, prospective_outcomes, publish_report_pointer, record_prospective


@pytest.fixture
def config():
    settings = load_config(Path(__file__).resolve().parents[1] / "configs/default.toml")
    settings["data"].update(minimum_member_count=2, minimum_cross_section=2)
    settings["portfolio"].update(top_k=1, keep_rank=1, max_weight=0.45)
    return settings


def quote_fixture(count=160):
    sessions = pd.bdate_range("2024-01-02", periods=count, name="date")
    codes = ["000001", "000002", "600001"]
    grid = pd.MultiIndex.from_product([sessions, codes], names=["date", "code"]).to_frame(index=False)
    grid["code"] = grid["code"].astype("string")
    time = np.repeat(np.arange(count), len(codes))
    shift = np.tile([0.0, 1.0, 2.0], count)
    price = 10 + shift + time * 0.01 + 0.05 * np.sin(time / 3 + shift)
    for prefix in ("raw", "adj"):
        grid[f"{prefix}_open"] = price
        grid[f"{prefix}_close"] = price * 1.001
        grid[f"{prefix}_high"] = price * 1.01
        grid[f"{prefix}_low"] = price * 0.99
    grid["volume"] = 20_000_000
    grid["amount"] = grid["raw_close"] * grid["volume"]
    grid["turnover"] = 0.01
    grid["vendor_pct"] = grid.groupby("code")["raw_close"].pct_change().fillna(0) * 100
    grid["total_shares"] = 1_000_000_000
    grid["trade_status"] = 1
    grid["is_st"] = 0
    grid["limit_up"] = grid["raw_open"] * 1.10
    grid["limit_down"] = grid["raw_open"] * 0.90
    grid["price_source"] = "fixture"
    grid["snapshot_industry"] = pd.NA
    benchmark = 3000 + np.arange(count) * 0.5 + np.sin(np.arange(count))
    market = pd.DataFrame({"market_open": benchmark, "market_close": benchmark * 1.002}, index=sessions)
    return grid, benchmark_for(market)


def test_truncating_future_bars_does_not_change_history(config, tmp_path):
    raw, market = quote_fixture()
    cutoff = market.index[140]
    prefix = raw.loc[raw["date"].le(cutoff)]
    member, industry, coverage = membership(raw, market.index, tmp_path / "missing.csv")
    full = build_features(raw, market, member, industry, coverage, None, config)
    truncated = build_features(prefix, market.loc[:cutoff], member.loc[prefix.index], industry.loc[prefix.index],
                               coverage.loc[:cutoff], None, config)
    fields = ["eligible", "baseline", *full.columns]
    pd.testing.assert_frame_equal(full.panel.loc[prefix.index, fields], truncated.panel[fields])
    assert truncated.panel.loc[truncated.panel["date"].eq(cutoff), "eligible"].all()


def test_announcement_and_restatement_take_effect_after_release(tmp_path):
    rows = []
    for period, released, profit in [("2023-03-31", "2023-04-20", 10), ("2023-12-31", "2024-03-15", 40),
                                      ("2024-03-31", "2024-04-20", 14), ("2024-03-31", "2024-05-06", 16)]:
        rows.append({"code": "000001", "report_date": period, "ann_date": released, "source": "test filing",
                     "version": released, "revenue_ytd": profit * 3, "net_income_ytd": profit,
                     "cfo_ytd": profit * 2, "gross_profit_ytd": profit * 2, "assets": 1000,
                     "equity": 500, "liabilities": 500})
    path = tmp_path / "filings.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    quotes = pd.DataFrame({"code": pd.Series(["000001"] * 6, dtype="string"), "raw_close": 10, "total_shares": 100,
                           "date": pd.to_datetime(["2024-04-01", "2024-04-19", "2024-04-20", "2024-04-22", "2024-05-06", "2024-05-07"])})
    result = fundamentals(quotes, path, 190)
    assert result["net_income_ttm"].tolist() == [40, 40, 40, 44, 44, 46]
    assert result.iloc[-1]["report_version"] == "2024-05-06"
    pd.DataFrame(rows[:-1]).to_csv(path, index=False)
    earlier = fundamentals(quotes.iloc[:-1], path, 190)
    pd.testing.assert_frame_equal(result.iloc[:-1], earlier)


def test_membership_effective_intervals_and_bad_schedule(tmp_path):
    quotes, market = quote_fixture(5)
    days = market.index
    path = tmp_path / "cons.csv"
    intervals = pd.DataFrame({"code": ["000001", "000002"], "start_date": [days[0], days[2]],
                              "end_date": [days[2], pd.NaT], "ann_date": [days[0] - pd.Timedelta(days=1), days[0]],
                              "industry": ["银行", "制造"], "source": ["official notice", "official notice"]})
    intervals.to_csv(path, index=False)
    active, sector, coverage = membership(quotes, days, path)
    assert quotes.loc[active, "code"].tolist() == ["000001", "000001", "000002", "000002", "000002"]
    assert sector.loc[active].tolist() == ["银行", "银行", "制造", "制造", "制造"]
    assert coverage["coverage"].eq(1).all()
    intervals.at[1, "ann_date"] = days[2]
    intervals.to_csv(path, index=False)
    with pytest.raises(ValueError, match="publication"):
        membership(quotes, days, path)


def test_labels_use_market_sessions_and_reject_blocked_open():
    quotes, market = quote_fixture(8)
    dates = market.index
    condition = quotes["date"].eq(dates[3]) & quotes["code"].eq("000001")
    quotes.loc[condition, "limit_down"] = quotes.loc[condition, "raw_open"]
    labels = executable_labels(quotes, market, 2).set_index(["date", "code"])
    row = labels.loc[(dates[0], "000001")]
    assert row["entry_date"] == dates[1] and row["exit_date"] == dates[3]
    assert row["label_status"] == "exit_blocked_or_missing" and pd.isna(row["excess_return"])
    assert labels.loc[(dates[0], "000002"), "label_status"] == "realized"
    assert labels.loc[(dates[-1], "000001"), "label_status"] == "pending"


def test_training_requires_matured_exit_and_embargo(config):
    _, market = quote_fixture(12)
    dates = market.index
    data = pd.DataFrame({"date": dates[:5], "session": range(5), "exit_date": [dates[4], dates[5], dates[6], dates[9], pd.NaT],
                         "excess_return": [0.1, 0.2, 0.3, 0.4, np.nan]})
    settings = config["research"] | {"embargo": 2}
    allowed = training_sample(data, dates, dates[8], settings)
    assert allowed["exit_date"].tolist() == [dates[4], dates[5]]


def test_replay_executes_next_open_and_retries_frozen_sell(config):
    data, market = quote_fixture(7)
    dates = market.index
    frozen = data["date"].eq(dates[2]) & data["code"].eq("000001")
    data.loc[frozen, "limit_down"] = data.loc[frozen, "raw_open"]
    scores = pd.DataFrame({"date": dates[:2], "code": ["000001", "000002"], "test_score": [1.0, 1.0]})
    eligible = data[["date", "code"]].assign(industry="A", adv20=1e9, vol60=0.03)
    replay = config["research"] | {"rebalance_every": 1}
    nav, fills = simulate(data, market, scores, eligible, "test_score", replay, config["portfolio"])
    first = fills.iloc[0]
    assert (first["date"], first["side"], first["code"]) == (dates[1], "buy", "000001")
    sales = fills.loc[fills["side"].eq("sell")]
    assert sales.iloc[0]["date"] == dates[3]
    assert fills["shares"].mod(100).eq(0).all() and fills["cash_fee"].ge(5).all()
    assert nav["cash_weight"].ge(0).all()


def test_refresh_matches_raw_and_adjusted_by_keys_not_row_order():
    day = pd.bdate_range("2024-04-01", periods=5)
    raw = pd.DataFrame({"date": day.strftime("%Y-%m-%d"), "code": "sz.000001", "open": np.arange(10, 15),
                        "high": np.arange(11, 16), "low": np.arange(9, 14), "close": np.arange(10, 15),
                        "preclose": 10, "volume": 1e6, "amount": 1e7, "turn": 1, "tradestatus": 1, "pctChg": 0, "isST": 0})
    adjusted = raw[["date", "code", "open", "high", "low", "close"]].copy().iloc[::-1]
    adjusted[["open", "high", "low", "close"]] *= 2
    overlap = pd.DataFrame({"date": day[:4], "raw_close": np.arange(10, 14), "adj_close": np.arange(10, 14) * 3})
    result = _normalize(raw, adjusted, overlap, "000001")
    np.testing.assert_allclose(result["adj_open"], raw["open"] * 3)
    assert result["turnover"].eq(0.01).all()
    with pytest.raises(ValueError, match="corporate-action"):
        _normalize(raw, adjusted, overlap.assign(raw_close=99), "000001")


def test_fresh_quality_screen_uses_only_current_disclosures_and_members(config, tmp_path):
    quotes, market = quote_fixture()
    day = market.index[-1]
    member, sector, coverage = membership(quotes, market.index, tmp_path / "missing.csv")
    snapshot = pd.DataFrame({"code": ["000001", "600001"], "name": ["甲", "乙"]})
    disclosures = pd.DataFrame({"code": ["000001", "600001"], "industry": ["制造", "制造"],
                                "report_date": "2024-06-30", "ann_date": "2024-07-05", "snapshot_date": day,
                                "vendor_update": "2024-07-05", "source": "fixture", "roe_ytd_pct": [12, 8],
                                "book_per_share": [5, 4], "profit_ytd": [1200, 800], "eps_ytd": [1.2, 0.8],
                                "operating_cash_per_share_ytd": [2.0, 0.7], "revenue_growth_pct": [6, 4],
                                "profit_growth_pct": [10, 5]})
    member, sector, coverage = apply_current_membership(quotes, day, member, sector, coverage, snapshot, disclosures)
    assert member.loc[quotes["date"].lt(day)].all()
    assert pd.isna(coverage.loc[market.index[-2], "expected"])
    features = build_features(quotes, market, member, sector, coverage, None, config)
    apply_current_quality(features.panel, day, snapshot, disclosures)
    picks, verdict, decisions = generate(features.panel, features.market, day, day, coverage, config, None)
    assert verdict["state"] == "FUNDAMENTAL_SCREEN"
    assert picks["code"].tolist() == ["000001"]
    assert picks["name"].tolist() == ["甲"]
    assert decisions["selected"].sum() == 1
    assert not features.panel.loc[features.panel["date"].lt(day), "fundamentals_ok"].any()
    _, stale, _ = generate(features.panel, features.market, day, day + pd.Timedelta(days=20), coverage, config, None)
    assert stale["state"] == "RESEARCH_ONLY"


def test_hierarchical_allocation_uses_normalized_positive_weights():
    rng = np.random.default_rng(42)
    returns = pd.DataFrame(rng.normal(size=(100, 3)) * [0.008, 0.015, 0.05], columns=["a", "b", "c"])
    weights = hierarchical_weights(returns)
    assert weights.sum() == pytest.approx(1.0)
    assert weights.gt(0).all() and weights.loc["a"] > weights.loc["c"]
    assert hierarchical_weights(returns.head(30)) is None


def test_refresh_recovers_missing_quotes_on_same_session(monkeypatch, tmp_path):
    quotes, market = quote_fixture(8)
    day = market.index[-1]
    quotes["market_open"] = quotes["date"].map(market["market_open"])
    quotes["market_close"] = quotes["date"].map(market["market_close"])
    missing = quotes["date"].eq(day) & quotes["code"].eq("000002")
    snapshot = pd.DataFrame({"snapshot_date": day, "code": quotes["code"].unique(), "name": ["a", "b", "c"]})
    monkeypatch.setattr(providers, "_snapshot", lambda: (snapshot, b"fixture"))
    monkeypatch.setattr(providers, "_benchmark", lambda *_: market[["market_open", "market_close"]].reset_index())
    monkeypatch.setattr(providers, "_disclosures", lambda *_: pd.DataFrame({"code": snapshot["code"]}))

    def download_retry(codes, reference, anchor, date, cache):
        assert codes == ["000002"] and list(reference) == codes and anchor == date == day
        quotes.loc[missing].drop(columns=["market_open", "market_close"]).to_parquet(cache / "000002.parquet", index=False)
        return {}

    monkeypatch.setattr(providers, "_download_batch", download_retry)
    info = providers.refresh(quotes.loc[~missing], tmp_path, day)
    assert info["status"] == "published" and info["minimum_coverage"] == 1 and info["rows"] == 1


def test_calendar_uses_completed_sessions_and_detects_stale_after_close():
    dates = pd.date_range("2026-11-20", "2026-11-23", name="date")
    schedule = pd.DataFrame({"is_session": [1, 0, 0, 1]}, index=dates)
    date = dates[-1]
    before = datetime.combine(date.date(), time(14, 50), tzinfo=CST)
    after = datetime.combine(date.date(), time(16), tzinfo=CST)
    assert last_completed(schedule, date, before, "15:35") == (dates[0], "AWAITING_CLOSE")
    expected, phase = last_completed(schedule, date, after, "15:35")
    assert expected == date and phase == "AFTER_CLOSE"
    assert readiness(dates[0], expected, phase, True, False)["reasons"]
    assert readiness(date, expected, phase, False, False)["reasons"]
    assert readiness(date, expected, phase, True, False)["reasons"] == []
    assert last_completed(None, date, after, "15:35")[0] is None


def test_open_decision_is_independent_of_same_day_close_and_high():
    quotes, calendar = quote_fixture(4)
    selected = quotes["date"].eq(calendar.index[1]) & quotes["code"].eq("000001")
    quotes.loc[selected, ["is_st", "limit_up", "limit_down"]] = np.nan
    quotes["raw_preclose"] = quotes["raw_open"]
    before = trade_flags(quotes, calendar.index).loc[selected, ["buy_open", "sell_open", "open_upper", "open_lower"]]
    quotes.loc[selected, ["raw_high", "raw_low", "raw_close", "vendor_pct", "volume", "amount"]] = [20., 1., 19., 90., 0., 0.]
    after = trade_flags(quotes, calendar.index).loc[selected, before.columns]
    pd.testing.assert_frame_equal(before, after)


def test_slippage_is_cost_and_cannot_move_price_past_limits(config):
    data, market = quote_fixture(7)
    d0, d1 = market.index[:2]
    selected = data["date"].eq(d1) & data["code"].eq("000001")
    for stem in ("raw", "adj"):
        data.loc[selected, [f"{stem}_open", f"{stem}_high", f"{stem}_low", f"{stem}_close"]] = [99.99, 100., 99.9, 99.98]
    data.loc[selected, ["limit_down", "limit_up"]] = [90., 100.]
    scores = pd.DataFrame({"date": [d0], "code": ["000001"], "signal": [1.]})
    features = data[["date", "code"]].assign(industry="制造", adv20=1e9, vol60=.03)
    _, fills = simulate(data, market, scores, features, "signal", config["research"], config["portfolio"])
    fill = fills.iloc[0]
    assert fill["fill_price"] == pytest.approx(99.99)
    assert fill["slippage_cost"] == pytest.approx(fill["value"] * .0008)


def test_frozen_position_uses_slot_until_delayed_sell(config):
    data, market = quote_fixture(7)
    days = market.index
    mask = data["date"].eq(days[2]) & data["code"].eq("000001")
    data.loc[mask, "limit_down"] = data.loc[mask, "raw_open"]
    scores = pd.DataFrame({"date": days[:2], "code": ["000001", "000002"], "signal": [1., 1.]})
    features = data[["date", "code"]].assign(industry="制造", adv20=1e9, vol60=.03)
    nav, fills = simulate(data, market, scores, features, "signal", config["research"] | {"rebalance_every": 1}, config["portfolio"])
    assert nav["positions"].max() == 1
    assert fills.loc[fills["code"].eq("000002"), "date"].tolist() == [days[3]]


def test_delayed_exit_label_uses_actual_exit_session():
    quotes, market = quote_fixture(8)
    wanted = quotes["date"].eq(market.index[3]) & quotes["code"].eq("000001")
    quotes.loc[wanted, "limit_down"] = quotes.loc[wanted, "raw_open"]
    labels = executable_labels(quotes, market, 2, exit_delay=1)
    row = labels.loc[labels["date"].eq(market.index[0]) & labels["code"].eq("000001")].iloc[0]
    assert row["planned_exit_date"] == market.index[3] and row["exit_date"] == market.index[4]
    assert row["label_status"] == "realized"


def test_historical_quality_rule_waits_for_disclosed_versions(config):
    days = pd.to_datetime(["2024-04-19", "2024-04-20", "2024-04-22", "2024-05-07"])
    codes = ["000001", "600001"]
    panel = pd.MultiIndex.from_product([days, codes], names=["date", "code"]).to_frame(index=False)
    panel = panel.assign(eligible=True, industry="制造", is_st=0, price_source="fixture", baseline=0.5)
    raw = []
    for date, release, roes in [("2023-12-31", "2024-03-30", [10., 6.]),
                                ("2024-03-31", "2024-04-20", [4., 12.]),
                                ("2024-03-31", "2024-05-07", [17., 3.])]:
        for code, roe in zip(codes, roes):
            raw.append({"code": code, "report_date": pd.Timestamp(date), "version_known_date": pd.Timestamp(release),
                        "roe_ytd_pct": roe, "book_per_share": 3., "profit_ytd": 10., "eps_ytd": 1.,
                        "operating_cash_per_share_ytd": 1., "profit_growth_pct": roe, "revenue_growth_pct": roe})
    signals, cover = historical_quality_signals(panel, pd.DataFrame(raw), config)
    assert cover["report_date"].tolist() == list(pd.to_datetime(["2023-12-31", "2023-12-31", "2024-03-31", "2024-03-31"]))
    assert signals.groupby("date")["code"].first().tolist() == ["000001", "000001", "600001", "600001"]


def test_sealed_lockbox_does_not_create_freeze_or_results_files(tmp_path):
    result = lockbox_request("freeze", {"lockbox_state": "SEALED", "research_gate_reasons": ["missing"]},
                             {}, None, None, None, {}, None, tmp_path, tmp_path)
    assert result == {"state": "SEALED", "reasons": ["missing"]}
    assert list(tmp_path.iterdir()) == []


def test_total_return_close_does_not_invent_open_levels():
    _, calendar = quote_fixture(5)
    fields = calendar[["market_open", "market_close"]].assign(total_return_close=lambda frame: frame["market_close"] * 2)
    computed = benchmark_for(fields)
    assert computed["benchmark_total_return"].all() and not computed["benchmark_open_total_return"].any()
    assert computed["benchmark_open"].eq(fields["market_open"]).all()
    assert computed["benchmark_close"].eq(fields["total_return_close"]).all()


def test_raw_history_repair_rejects_close_mismatch(tmp_path):
    quotes, _ = quote_fixture(4)
    quotes["price_source"] = "legacy_reconstructed"
    quotes["raw_preclose"] = np.nan
    columns = ["date", "code", "raw_open", "raw_high", "raw_low", "raw_close", "raw_preclose", "trade_status", "is_st"]
    raw = quotes[columns].iloc[:2].copy()
    raw["raw_preclose"] = 10.5
    raw.loc[raw.index[1], "raw_close"] = 99.
    file = tmp_path / "history.parquet"
    raw.to_parquet(file, index=False)
    repaired, audit = apply_raw_history(quotes, file)
    assert audit["accepted"] == 1 and audit["rejected"] == 1
    assert repaired["raw_preclose"].iloc[0] == 10.5 and pd.isna(repaired["raw_preclose"].iloc[1])


def test_quality_comparison_runs_end_to_end_with_matching_history(config, tmp_path):
    quotes, market = quote_fixture(8)
    config["research"].update(min_train_dates=2, rebalance_every=2, horizon=2)
    panel = quotes[["date", "code", "is_st", "price_source"]].assign(eligible=True, industry="制造", baseline=.6,
                                                                      adv20=1e9, vol60=.03)
    predictions = panel[["date", "code", "baseline"]].assign(ridge=.7, lgbm_rank=.8, lgbm_huber=.9, catboost=.5)
    known = pd.DataFrame({"code": quotes["code"].unique(), "report_date": pd.Timestamp("2023-12-31"),
                          "version_known_date": pd.Timestamp("2024-01-01"), "roe_ytd_pct": [15., 8., 10.],
                          "book_per_share": 4., "profit_ytd": 900., "eps_ytd": 1.3,
                          "operating_cash_per_share_ytd": 1.5, "profit_growth_pct": [10., 8., 9.],
                          "revenue_growth_pct": [9., 5., 7.]})
    coverage = pd.DataFrame({"expected": 3}, index=market.index)
    labels = executable_labels(quotes, market, 2)
    result = run_quality_comparison(quotes, SimpleNamespace(panel=panel), market, predictions, labels,
                                    coverage, known, config, tmp_path)
    assert result["state"] == "DEVELOPMENT_DIAGNOSTIC"
    assert len(result["performance"]) == 7 and result["performance"]["quality_mix"]["sessions"] == 8
    assert (tmp_path / "quality_strategy_scores.csv").exists()


def test_late_prospective_observation_cannot_be_scored(config, tmp_path, monkeypatch):
    from quant_screen import protocol

    quotes, market = quote_fixture(8)
    signal = market.index[0]
    entry = market.index[1]
    monkeypatch.setattr(protocol, "utc_now", lambda: datetime.combine(entry.date(), time(9, 16), CST).isoformat())
    picks = pd.DataFrame({"code": ["000001"], "rank": [1]})
    meta = {"signal_date": str(signal.date()), "session": {"verified_codes": 3}}
    record_prospective(tmp_path, tmp_path, meta, picks, {"config_sha256": "fixture"})
    calendar = pd.DataFrame({"is_session": 1}, index=market.index)
    state = prospective_outcomes(tmp_path, tmp_path, quotes, market, calendar, 2)
    assert state["matured"] == 0 and state["pending"] == 0
    assert pd.read_csv(tmp_path / "prospective_outcomes.csv")["label_status"].tolist() == ["OBSERVED_TOO_LATE"]


def test_failed_or_older_attempt_does_not_replace_last_valid_report(tmp_path):
    good = {"state": "FUNDAMENTAL_SCREEN", "signal_date": "2026-11-23", "run": "good"}
    publish_report_pointer(tmp_path, good, False)
    publish_report_pointer(tmp_path, good | {"signal_date": "2026-11-20", "run": "older"}, False)
    publish_report_pointer(tmp_path, good | {"state": "RESEARCH_ONLY", "run": "failed"}, False)
    assert json.loads((tmp_path / "latest.json").read_text(encoding="utf-8")) == good
    assert json.loads((tmp_path / "latest_run.json").read_text(encoding="utf-8"))["run"] == "failed"
