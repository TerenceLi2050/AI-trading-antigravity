from dataclasses import dataclass
from pathlib import Path
from typing import Any

from catboost import CatBoostRegressor
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from .io import sha256_file


ALL_MODELS = ["ridge", "lgbm_rank", "lgbm_huber", "catboost"]
STRATEGIES = ["baseline", "ridge", "lgbm_rank", "lgbm_huber", "catboost", "ensemble"]


def active_model_names(config: dict | None = None) -> list[str]:
    if not config:
        return ["ridge", "lgbm_huber", "catboost"]
    names = config.get("active_models") or ["ridge", "lgbm_huber", "catboost"]
    return [name for name in ALL_MODELS if name in names]


def active_strategies(config: dict | None = None) -> list[str]:
    return ["baseline", *active_model_names(config), "ensemble"]


@dataclass
class ModelSet:
    columns: list[str]
    models: dict[str, Any]
    train_stats: dict


def design(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    # All columns are already same-date percentiles. Neutral imputation is invariant to future rows.
    return frame[columns].fillna(0.5).to_numpy(dtype=np.float32)


def training_sample(frame: pd.DataFrame, sessions: pd.DatetimeIndex, fit_date: pd.Timestamp, config: dict):
    position = sessions.searchsorted(fit_date)
    end = position - config["embargo"]
    if end < 1:
        return frame.iloc[:0]
    start = max(0, position - config["train_sessions"])
    return frame.loc[frame["session"].ge(start) & frame["date"].lt(fit_date)
                     & frame["exit_date"].lt(sessions[end]) & frame["excess_return"].notna()]


def _time_decay_weights(frame: pd.DataFrame, half_life: int) -> np.ndarray | None:
    if not half_life or half_life <= 0:
        return None
    if "session" in frame.columns and frame["session"].notna().all():
        age = (frame["session"].max() - frame["session"]).to_numpy(dtype=float)
    else:
        days = (frame["date"].max() - frame["date"]).dt.days.to_numpy(dtype=float)
        age = days * (252.0 / 365.25)
    weights = np.power(0.5, np.maximum(age, 0.0) / float(half_life))
    mean_w = float(np.mean(weights))
    return (weights / mean_w).astype(np.float32) if mean_w > 0 else None


def train(frame: pd.DataFrame, columns: list[str], fit_date: pd.Timestamp, config: dict) -> ModelSet:
    frame = frame.sort_values(["date", "code"])
    targets = frame.groupby("date")["excess_return"].rank(pct=True)
    x = design(frame, columns)
    y = targets.to_numpy(dtype=np.float32)
    sample_weight = _time_decay_weights(frame, int(config.get("sample_decay_half_life", 252)))
    base = dict(n_estimators=160, learning_rate=0.03, num_leaves=7, max_depth=3,
                min_child_samples=350, reg_alpha=1.0, reg_lambda=15.0, colsample_bytree=0.7,
                subsample=0.8, subsample_freq=1, verbosity=-1, force_col_wise=True,
                random_state=config["seed"], n_jobs=config["threads"])
    wanted = active_model_names(config)
    fitted = {}
    if "ridge" in wanted:
        fitted["ridge"] = Ridge(alpha=200.0, solver="cholesky").fit(x, y, sample_weight=sample_weight)
    if "lgbm_rank" in wanted:
        rank_cfg = {**base, "lambdarank_truncation_level": 30}
        fitted["lgbm_rank"] = lgb.LGBMRanker(objective="lambdarank", **rank_cfg).fit(
            x, np.minimum((y * 10).astype(int), 9), group=frame.groupby("date", sort=False).size().to_numpy())
    if "lgbm_huber" in wanted:
        fitted["lgbm_huber"] = lgb.LGBMRegressor(objective="huber", **base).fit(x, y, sample_weight=sample_weight)
    if "catboost" in wanted:
        fitted["catboost"] = CatBoostRegressor(
            iterations=200, depth=3, learning_rate=0.035, l2_leaf_reg=25.0,
            thread_count=config["threads"], random_seed=config["seed"],
            allow_writing_files=False, verbose=False).fit(x, y, sample_weight=sample_weight)
    stats = {"fit_date": str(fit_date.date()), "rows": len(frame), "days": frame["date"].nunique(),
             "first_signal_date": str(frame["date"].min().date()), "last_signal_date": str(frame["date"].max().date()),
             "last_label_exit": str(frame["exit_date"].max().date()), "active_models": list(fitted),
             "sample_decay_half_life": int(config.get("sample_decay_half_life", 252))}
    return ModelSet(columns, fitted, stats)


def predict_model(name: str, model: Any, x: np.ndarray) -> np.ndarray:
    if name in ("ridge", "catboost"):
        return np.asarray(model.predict(x), dtype=float)
    return np.asarray(model.booster_.predict(x, num_threads=model.n_jobs), dtype=float)


def score(frame: pd.DataFrame, bundle: ModelSet) -> pd.DataFrame:
    x = design(frame, bundle.columns)
    predictions = frame[["date", "code", "baseline"]].copy()
    for name, model in bundle.models.items():
        predictions[name] = predict_model(name, model, x)
    rank = predictions.groupby("date")[list(bundle.models)].rank(pct=True)
    predictions["ensemble"] = rank.mean(axis=1)
    return predictions


def select_explain_model(bundle: ModelSet, preferred: str = "auto") -> str:
    if preferred != "auto" and preferred in bundle.models:
        return preferred
    for candidate in ("lgbm_huber", "catboost", "ridge", "lgbm_rank"):
        if candidate in bundle.models:
            return candidate
    return next(iter(bundle.models))


def explain(frame: pd.DataFrame, bundle: ModelSet, limit: int = 3, preferred: str = "auto") -> pd.Series:
    if frame.empty or not bundle.models:
        return pd.Series(dtype="string", index=frame.index)
    name = select_explain_model(bundle, preferred)
    model = bundle.models[name]
    x = design(frame, bundle.columns)
    if name in ("lgbm_huber", "lgbm_rank"):
        contributions = model.booster_.predict(x, pred_contrib=True, num_threads=model.n_jobs)[:, :-1]
    elif name == "catboost":
        from catboost import Pool
        contributions = np.asarray(model.get_feature_importance(Pool(x), type="ShapValues"))[:, :-1]
    else:
        contributions = (x - 0.5) * np.asarray(model.coef_, dtype=float)
    explanations = []
    for row in contributions:
        largest = np.argsort(-np.abs(row), kind="stable")[:limit]
        explanations.append(" / ".join(f"{bundle.columns[i].removeprefix('rank_')} {row[i]:+.4f}" for i in largest))
    return pd.Series(explanations, index=frame.index, dtype="string")


def load_verified_model(path: Path, available_columns: list[str]) -> tuple[ModelSet, str]:
    """Load a frozen ModelSet artifact and verify its feature schema against the live panel."""
    import joblib

    if not path.is_file():
        raise FileNotFoundError(f"Frozen model file not found: {path}")
    digest = sha256_file(path)
    bundle = joblib.load(path)
    if not isinstance(bundle, ModelSet) or not bundle.models:
        raise ValueError("Model artifact is not a valid ModelSet")
    missing = [col for col in bundle.columns if col not in available_columns]
    if missing:
        raise ValueError(f"Model expects unavailable feature columns: {missing[:5]}")
    return bundle, digest


def samples(panel: pd.DataFrame, labels: pd.DataFrame, columns: list[str], research: dict) -> pd.DataFrame:
    keep = panel["eligible"] & labels["session"].mod(research["sample_every"]).eq(0)
    fields = ["date", "code", "baseline", *columns]
    return panel.loc[keep, fields].join(labels.loc[keep, ["excess_return", "entry_date", "exit_date", "session"]])


def walk_forward(sample: pd.DataFrame, sessions: pd.DatetimeIndex, columns: list[str], config: dict):
    oos = sample.loc[sample["date"].ge(config["oos_start"]) & sample["date"].lt(config["lockbox_start"])]
    dates = pd.DatetimeIndex(oos["date"].unique()).sort_values()
    anchor = pd.Timestamp(config["oos_start"])
    blocks = ((dates.year - anchor.year) * 12 + dates.month - anchor.month) // config["refit_months"]
    frames, fitted = [], []
    for block in np.unique(blocks):
        block_dates = dates[blocks == block]
        fit_date = block_dates[0]
        rows = training_sample(sample, sessions, fit_date, config)
        if rows["date"].nunique() < config["min_train_dates"]:
            continue
        bundle = train(rows, columns, fit_date, config)
        predictions = score(oos.loc[oos["date"].isin(block_dates)], bundle)
        frames.append(predictions)
        fitted.append((block_dates, bundle))
        print(f"walk-forward {fit_date.date()} → {block_dates[-1].date()} ({len(rows):,} train rows)", flush=True)
    if not frames:
        return pd.DataFrame(columns=["date", "code", *active_strategies(config)]), fitted
    return pd.concat(frames).sort_values(["date", "code"]), fitted
