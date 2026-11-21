from datetime import time
from pathlib import Path
import tomllib


def load_config(path: Path) -> dict:
    with path.open("rb") as stream:
        config = tomllib.load(stream)
    data, eligibility, research, portfolio = (
        config[name] for name in ("data", "eligibility", "research", "portfolio")
    )
    for key, value in {"session_calendar": "data/input/sessions.csv", "total_return": "data/input/csi800_total_return.csv",
                       "quality_history": "data/input/quality_history.csv", "raw_history": "data/input/raw_history.parquet",
                       "settled_after_cst": "15:35", "daily_updates": "data/daily", "warmup_sessions": 260,
                       "history_start": "2019-03-15", "bar_provider": "auto"}.items():
        data.setdefault(key, value)
    eligibility.setdefault("minimum_cash_conversion", .5)
    eligibility.setdefault("quality_weight", .35)
    eligibility.setdefault("baseline_profile", "ashare")
    eligibility.setdefault("quality_profile", "value_cash_quality")
    eligibility.setdefault("industry_neutralize", False)
    eligibility.setdefault("use_extended_factors", True)
    eligibility.setdefault("use_pit_valuation_features", True)
    research.setdefault("exit_delay_sensitivity", 5)
    research.setdefault("active_models", ["ridge", "lgbm_huber", "catboost"])
    research.setdefault("explain_model", "auto")
    research.setdefault("sample_decay_half_life", 252)
    portfolio.setdefault("max_financial_names", 7)
    portfolio.setdefault("default_allocation", "inverse_volatility")
    screening = config.setdefault("screening", {})
    screening.setdefault("model_path", "")
    screening.setdefault("model_weight", 0.0)
    if data["bar_provider"] not in {"auto", "baostock", "eastmoney", "akshare"}:
        raise ValueError("bar_provider must be one of auto, baostock, eastmoney, akshare")
    if eligibility["baseline_profile"] not in {"ashare", "classic"}:
        raise ValueError("baseline_profile must be ashare or classic")
    if eligibility["quality_profile"] not in {"value_cash_quality", "classic_roe"}:
        raise ValueError("quality_profile must be value_cash_quality or classic_roe")
    if portfolio["default_allocation"] not in {"equal_weight", "inverse_volatility", "hrp"}:
        raise ValueError("default_allocation must be equal_weight, inverse_volatility, or hrp")
    if not research["active_models"] or not set(research["active_models"]).issubset({"ridge", "lgbm_rank", "lgbm_huber", "catboost"}):
        raise ValueError("active_models must be a non-empty subset of supported estimators")
    if not 0 <= screening["model_weight"] <= 1:
        raise ValueError("screening.model_weight must be in [0, 1]")
    if not research["oos_start"] < research["lockbox_start"]:
        raise ValueError("oos_start must precede lockbox_start")
    for key in ("horizon", "embargo", "sample_every", "rebalance_every", "train_sessions",
                "min_train_dates", "refit_months", "threads"):
        if not isinstance(research[key], int) or research[key] < (0 if key == "embargo" else 1):
            raise ValueError(f"Invalid research setting: {key}")
    if portfolio["top_k"] < 1 or portfolio["keep_rank"] < portfolio["top_k"]:
        raise ValueError("keep_rank must be at least top_k, and top_k must be positive")
    if not (0 < portfolio["max_weight"] <= 1 and 0 < portfolio["max_adv_participation"] <= 1):
        raise ValueError("Portfolio weight and participation limits must be in (0, 1]")
    if portfolio["cash"] <= 0 or portfolio["max_industry_names"] <= 0:
        raise ValueError("Initial cash and industry limit must be positive")
    if any(portfolio[key] < 0 for key in ("commission", "min_commission", "transfer_fee",
                                          "stamp_duty", "slippage_bps")):
        raise ValueError("Trading costs must be nonnegative")
    if not (0 < data["minimum_member_coverage"] <= 1 and data["minimum_cross_section"] >= 2
            and data["minimum_member_count"] >= data["minimum_cross_section"]):
        raise ValueError("Invalid coverage settings")
    if not 1 <= eligibility["min_valid_bars_60"] <= 60:
        raise ValueError("min_valid_bars_60 must be between 1 and 60")
    if data["warmup_sessions"] < max(252, eligibility["min_listing_sessions"]):
        raise ValueError("Daily history must cover the longest price and eligibility windows")
    if not 0 < data["minimum_fundamental_coverage"] <= 1:
        raise ValueError("minimum_fundamental_coverage must be in (0, 1]")
    if not all(0 <= eligibility[key] <= 1 for key in ("quality_weight", "minimum_quality_rank")):
        raise ValueError("Quality weight and rank threshold must be in [0, 1]")
    if not 0 <= eligibility["minimum_cash_conversion"] or portfolio["max_financial_names"] < 1:
        raise ValueError("Invalid financial screening limits")
    if not isinstance(research["exit_delay_sensitivity"], int) or not 0 <= research["exit_delay_sensitivity"] <= 20:
        raise ValueError("Exit delay sensitivity must be between zero and twenty sessions")
    if not time(15, 0) <= time.fromisoformat(data["settled_after_cst"]) < time(23, 59):
        raise ValueError("Daily close evidence must be collected at or after 15:00 CST")
    return config
