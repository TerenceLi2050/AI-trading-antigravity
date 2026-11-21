"""Tests for v0.3.0 standalone history, multi-provider fallback, upgraded quant signals, diversified backtest, and ops."""

import json
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest

from quant_screen import cli, doctor, history_builder, models, providers, screen
from quant_screen.backtest import performance, simulate, yearly_performance
from quant_screen.config import load_config
from quant_screen.data import apply_canonical_quotes, empty_quotes_frame, load_research_base, market_calendar
from quant_screen.features import build_features
from quant_screen.history_builder import build_membership_intervals
from quant_screen.labels import executable_labels
from quant_screen.pit import membership
from quant_screen.protocol import make_pointer, portable_path
from quant_screen.quality_study import run_quality_comparison
from test_pipeline import quote_fixture


def _cfg():
    settings = load_config(Path(__file__).resolve().parents[1] / "configs/default.toml")
    settings["data"].update(minimum_member_count=2, minimum_cross_section=2)
    settings["portfolio"].update(top_k=2, keep_rank=2, max_weight=0.5, max_industry_names=2, max_financial_names=1)
    return settings


def test_standalone_research_boots_without_legacy_csv(tmp_path):
    quotes, market = quote_fixture(12)
    input_dir = tmp_path / "data/input"
    input_dir.mkdir(parents=True)
    # Write canonical quotes WITHOUT market_open/market_close, paired with calendar.csv
    quotes_path = input_dir / "quotes.parquet"
    quotes.to_parquet(quotes_path, index=False)
    cal = market[["market_open", "market_close"]].reset_index()
    cal["source"] = "fixture_calendar"
    cal.to_csv(input_dir / "calendar.csv", index=False)

    cfg = _cfg()
    base, meta = load_research_base(tmp_path, cfg["data"])
    assert base.empty and meta["source"] == "canonical_only"
    loaded = apply_canonical_quotes(base, quotes_path)
    assert len(loaded) == len(quotes) and "adj_vwap" in loaded.columns
    rebuilt_cal = market_calendar(loaded, input_dir / "calendar.csv")
    assert len(rebuilt_cal) == len(market)
    np.testing.assert_allclose(rebuilt_cal["market_close"], market["market_close"])


def test_akshare_bar_normalization_matches_canonical_schema():
    days = pd.bdate_range("2024-03-01", periods=6)
    raw = pd.DataFrame({
        "date": days.strftime("%Y-%m-%d"),
        "open": [10.0, 10.2, 10.1, 10.4, 10.3, 10.5],
        "high": [10.3, 10.4, 10.3, 10.6, 10.5, 10.8],
        "low": [9.9, 10.0, 9.9, 10.2, 10.1, 10.3],
        "close": [10.1, 10.1, 10.3, 10.3, 10.4, 10.7],
        "volume": [1e6, 1e6, 0.0, 1.2e6, 1.1e6, 1.3e6],
        "amount": [1e7, 1e7, 0.0, 1.2e7, 1.1e7, 1.3e7],
        "turnover": [0.01, 0.01, 0.0, 0.012, 0.011, 0.013],
    })
    adj = raw[["date", "open", "high", "low", "close"]].copy()
    adj[["open", "high", "low", "close"]] *= 2.0
    out = providers._normalize_akshare(raw, adj, pd.DataFrame(), "000001")
    assert out["price_source"].eq("akshare_raw_and_hfq").all()
    assert out["trade_status"].tolist() == [1.0, 1.0, 0.0, 1.0, 1.0, 1.0]
    assert out["adj_vwap"].iloc[0] == pytest.approx(20.0)


def test_build_membership_intervals_produces_valid_pit_table(tmp_path):
    samples = pd.DataFrame({
        "requested_date": ["2024-01-01", "2024-01-01", "2024-02-01", "2024-02-01", "2024-03-01", "2024-03-01"],
        "code": ["sz.000001", "sh.600001", "sz.000001", "sz.000002", "sz.000001", "sh.600001"],
    })
    industries = pd.Series({"000001": "银行Ⅱ", "600001": "制造", "000002": "地产"})
    target = tmp_path / "membership.csv"
    summary = build_membership_intervals(samples, target, industries)
    assert summary["status"] == "published" and summary["distinct_codes"] == 3
    quotes, market = quote_fixture(5)
    # Validate directly through pit.membership strict interval rules
    quotes["date"] = pd.to_datetime(["2024-01-10", "2024-02-10", "2024-03-10", "2024-03-15", "2024-03-20"]).repeat(3)
    cal = pd.DatetimeIndex(quotes["date"].unique()).sort_values()
    active, sector, cov = membership(quotes, cal, target)
    feb = quotes["date"].eq("2024-02-10")
    assert set(quotes.loc[feb & active, "code"]) == {"000001", "000002"}
    assert sector.loc[feb & quotes["code"].eq("000001")].iat[0] == "银行Ⅱ"


def test_extended_factors_and_ashare_baseline_preserve_no_lookahead(tmp_path):
    cfg = _cfg()
    cfg["eligibility"]["use_extended_factors"] = True
    cfg["eligibility"]["baseline_profile"] = "ashare"
    cfg["eligibility"]["industry_neutralize"] = True
    raw, market = quote_fixture(160)
    cutoff = market.index[140]
    prefix = raw.loc[raw["date"].le(cutoff)]
    member = pd.Series(True, index=raw.index)
    industry = pd.Series("制造", index=raw.index, dtype="string")
    cov = pd.DataFrame({"expected": 3.0, "observed": 3.0, "coverage": 1.0}, index=market.index)
    full = build_features(raw, market, member, industry, cov, None, cfg)
    trunc = build_features(prefix, market.loc[:cutoff], member.loc[prefix.index], industry.loc[prefix.index],
                           cov.loc[:cutoff], None, cfg)
    assert "rank_low_idio_vol_60" in full.columns and "rank_low_max_ret_20" in full.columns
    assert "rank_vwap_reversal_20" in full.columns and "rank_low_pv_corr_20" in full.columns
    assert "classic_baseline" in full.panel.columns
    fields = ["eligible", "baseline", "classic_baseline", *full.columns]
    pd.testing.assert_frame_equal(full.panel.loc[prefix.index, fields], trunc.panel[fields])


def test_active_models_explain_and_daily_model_blending(tmp_path):
    rng = np.random.default_rng(7)
    days = pd.bdate_range("2024-01-02", periods=10)
    codes = [f"{i:06d}" for i in range(1, 31)]
    grid = pd.MultiIndex.from_product([days, codes], names=["date", "code"]).to_frame(index=False)
    cols = ["rank_low_volatility", "rank_reversal_5", "rank_low_pv_corr_20"]
    for c in cols:
        grid[c] = rng.random(len(grid)).astype(np.float32)
    grid["baseline"] = grid[cols].mean(axis=1)
    grid["excess_return"] = 0.2 * grid["rank_low_volatility"] - 0.1 * grid["rank_reversal_5"] + rng.normal(0, 0.02, len(grid))
    grid["exit_date"] = grid["date"] + pd.Timedelta(days=2)

    cfg = _cfg()
    cfg["research"]["active_models"] = ["ridge", "lgbm_huber"]
    bundle = models.train(grid, cols, days[-1], cfg["research"])
    assert set(bundle.models) == {"ridge", "lgbm_huber"}
    assert "lgbm_rank" not in bundle.models
    exp = models.explain(grid.head(5), bundle, preferred="auto")
    assert len(exp) == 5 and exp.notna().all()

    model_file = tmp_path / "model.joblib"
    joblib.dump(bundle, model_file)
    loaded, digest = models.load_verified_model(model_file, cols)
    assert digest and loaded.columns == cols
    with pytest.raises(ValueError, match="unavailable feature columns"):
        models.load_verified_model(model_file, ["rank_low_volatility"])


def test_backtest_enforces_financial_cap_and_yearly_performance():
    quotes, market = quote_fixture(10)
    days = market.index
    scores = pd.DataFrame({
        "date": [days[0], days[0], days[0]],
        "code": ["000001", "000002", "600001"],
        "sig": [0.99, 0.95, 0.50],
    })
    # Two top-ranked stocks are banks; max_financial_names=1 must pick only 1 bank + 1 non-financial
    eligible = quotes[["date", "code"]].assign(
        industry=quotes["code"].map({"000001": "银行Ⅱ", "000002": "证券Ⅱ", "600001": "制造"}),
        adv20=1e9, vol60=0.02,
    )
    cfg = _cfg()
    nav, fills = simulate(quotes, market, scores, eligible, "sig", cfg["research"], cfg["portfolio"])
    bought = set(fills.loc[fills["side"].eq("buy"), "code"])
    assert bought == {"000001", "600001"}
    yp = yearly_performance({"test": nav})
    assert len(yp) == 1 and yp["year"].iat[0] == 2024


def test_quality_study_includes_diversified_branch(tmp_path):
    quotes, market = quote_fixture(8)
    cfg = _cfg()
    cfg["research"].update(min_train_dates=2, rebalance_every=2, horizon=2)
    panel = quotes[["date", "code", "is_st", "price_source"]].assign(
        eligible=True,
        industry=quotes["code"].map({"000001": "银行Ⅱ", "000002": "证券Ⅱ", "600001": "制造"}),
        baseline=0.6, adv20=1e9, vol60=0.03,
    )
    predictions = panel[["date", "code", "baseline"]].assign(ridge=0.7, lgbm_huber=0.9, catboost=0.5)
    known = pd.DataFrame({
        "code": quotes["code"].unique(),
        "report_date": pd.Timestamp("2023-12-31"),
        "version_known_date": pd.Timestamp("2024-01-01"),
        "roe_ytd_pct": [15., 14., 10.], "book_per_share": 4., "profit_ytd": 900.,
        "eps_ytd": 1.3, "operating_cash_per_share_ytd": 1.5,
        "profit_growth_pct": [10., 9., 8.], "revenue_growth_pct": [9., 8., 7.],
    })
    cov = pd.DataFrame({"expected": 3}, index=market.index)
    labels = executable_labels(quotes, market, 2)
    res = run_quality_comparison(quotes, SimpleNamespace(panel=panel), market, predictions, labels, cov, known, cfg, tmp_path)
    assert res["state"] == "DEVELOPMENT_DIAGNOSTIC"
    assert "quality_mix_diversified" in res["diversified_performance"]
    assert "quality_mix_diversified_2x_cost" in res["diversified_performance"]


def test_portable_pointer_and_doctor(tmp_path):
    run = tmp_path / "reports/runs/20261121T000000Z-abc12345"
    run.mkdir(parents=True)
    (run / "report.html").write_text("ok", encoding="utf-8")
    ptr = make_pointer(tmp_path, run, "FUNDAMENTAL_SCREEN", "2026-11-20")
    assert ptr["run"] == "reports/runs/20261121T000000Z-abc12345"
    assert ptr["report"] == "reports/runs/20261121T000000Z-abc12345/report.html"
    (tmp_path / "reports/latest.json").write_text(json.dumps(ptr), encoding="utf-8")
    cfg = _cfg()
    diag = doctor.run_doctor(tmp_path, cfg)
    assert diag["pointers"]["latest.json"]["target_exists"] is True
    assert diag["pointers"]["latest.json"]["is_relative"] is True
    assert diag["historical"]["standalone_research_ready"] is False


def test_eastmoney_normalization_and_pit_candidate_factors():
    from quant_screen import factor_lib

    days = pd.bdate_range("2024-03-01", periods=5)
    raw = pd.DataFrame({
        "date": days.strftime("%Y-%m-%d"),
        "open": [10.0, 10.1, 10.2, 10.3, 10.4],
        "close": [10.1, 10.2, 10.3, 10.4, 10.5],
        "high": [10.2, 10.3, 10.4, 10.5, 10.6],
        "low": [9.9, 10.0, 10.1, 10.2, 10.3],
        "volume": [10000.0] * 5,  # lots -> 1,000,000 shares
        "amount": [1e7] * 5,
        "vendor_pct": [1.0] * 5,
        "turnover": [1.5] * 5,  # percent -> 0.015
    })
    adj = raw.copy()
    adj[["open", "close", "high", "low"]] *= 2.0
    norm = providers._normalize_eastmoney(raw, adj, pd.DataFrame(), "000001")
    assert norm["price_source"].eq("eastmoney_raw_and_hfq").all()
    assert norm["volume"].iloc[0] == pytest.approx(1_000_000.0)
    assert norm["turnover"].iloc[0] == pytest.approx(0.015)

    # Verify PIT alignment + 48-factor library respects strict version_known_date < date
    quotes, market = quote_fixture(160)
    qh = pd.DataFrame([
        {"code": "000001", "report_date": "2023-12-31", "ann_date": "2024-03-15", "version_known_date": "2024-03-15",
         "source": "em", "version": "2024-03-15", "roe_ytd_pct": 14.0, "book_per_share": 8.0, "profit_ytd": 1e9,
         "eps_ytd": 1.2, "operating_cash_per_share_ytd": 1.5, "profit_growth_pct": 10.0, "revenue_growth_pct": 8.0},
        {"code": "000001", "report_date": "2024-03-31", "ann_date": "2024-04-20", "version_known_date": "2024-04-20",
         "source": "em", "version": "2024-04-20", "roe_ytd_pct": 4.0, "book_per_share": 8.5, "profit_ytd": 3e8,
         "eps_ytd": 0.35, "operating_cash_per_share_ytd": 0.4, "profit_growth_pct": 12.0, "revenue_growth_pct": 9.0},
    ])
    aligned = factor_lib.align_pit_quality(quotes, qh, max_age=190)
    pre = quotes["code"].eq("000001") & quotes["date"].eq("2024-04-19")
    post = quotes["code"].eq("000001") & quotes["date"].eq("2024-04-22")
    assert aligned.loc[pre, "pit_book_per_share"].iat[0] == pytest.approx(8.0)
    assert aligned.loc[post, "pit_book_per_share"].iat[0] == pytest.approx(8.5)

    cfg = _cfg()
    member = pd.Series(True, index=quotes.index)
    industry = pd.Series("制造", index=quotes.index, dtype="string")
    cov = pd.DataFrame({"expected": 3.0, "observed": 3.0, "coverage": 1.0}, index=market.index)
    panel, rank_cols = factor_lib.build_candidate_factors(quotes, market, member, industry, cov, qh, cfg, neutralize=False)
    assert len(rank_cols) == len(factor_lib.ALL_CANDIDATE_FACTORS) == 48
    assert "bp_pit" in panel.columns and "log_float_mv" in panel.columns


def test_time_decay_weights_favor_recent_sessions():
    days = pd.bdate_range("2022-01-03", periods=500)
    df = pd.DataFrame({"date": days, "session": np.arange(500)})
    w = models._time_decay_weights(df, half_life=250)
    assert w is not None and len(w) == 500
    assert w[-1] > w[0] * 3.5 and np.mean(w) == pytest.approx(1.0, rel=1e-4)

