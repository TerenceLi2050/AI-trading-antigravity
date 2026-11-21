from itertools import combinations
import math

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew
from sklearn.linear_model import Ridge

from .models import design, predict_model, select_explain_model, training_sample


def ridge_family_ablation(fitted, sample, sessions, settings, groups, min_cross_section):
    tables = []
    for dates, bundle in fitted:
        fit = training_sample(sample, sessions, dates[0], settings)
        test = sample.loc[sample["date"].isin(dates)].copy()
        target = fit.groupby("date")["excess_return"].rank(pct=True)
        reference = bundle.models["ridge"]
        test["ridge_full"] = reference.predict(design(test, bundle.columns))
        for name, excluded in groups.items():
            columns = [column for column in bundle.columns if column not in excluded]
            estimator = Ridge(alpha=reference.alpha, solver="cholesky").fit(design(fit, columns), target)
            test[f"without_{name}"] = estimator.predict(design(test, columns))
        tables.append(test[["date", "excess_return", "ridge_full", *[f"without_{group}" for group in groups]]])
    combined = pd.concat(tables)
    lag = max(1, (settings["horizon"] + settings["sample_every"] - 1) // settings["sample_every"])
    result, _ = rank_ic(combined, ["ridge_full", *[f"without_{group}" for group in groups]], min_cross_section, lag)
    result["rank_ic_difference_from_full"] = result["rank_ic"] - result.loc[result["signal"].eq("ridge_full"), "rank_ic"].iat[0]
    return result


def _hac_t(series: pd.Series, lag: int) -> float:
    x = series.dropna().to_numpy()
    if len(x) <= lag + 2:
        return np.nan
    centered = x - x.mean()
    variance = centered @ centered / len(x)
    for j in range(1, lag + 1):
        variance += 2 * (1 - j / (lag + 1)) * (centered[j:] @ centered[:-j]) / len(x)
    return float(x.mean() * np.sqrt(len(x) / variance)) if variance > 0 else np.nan


def _correlations(pairs: pd.DataFrame, name: str) -> pd.Series:
    if pairs.empty:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([], name="date"))
    ranks = pairs.groupby("date")[[name, "excess_return"]].rank()
    ranks["date"] = pairs["date"]
    return ranks.groupby("date")[[name, "excess_return"]].corr().xs(name, level=1)["excess_return"]


def rank_ic(frame: pd.DataFrame, columns: list[str], min_cross_section: int, lag: int):
    rows, traces = [], []
    for name in columns:
        pairs = frame[["date", name, "excess_return"]].dropna()
        pairs = pairs.loc[pairs.groupby("date")[name].transform("size").ge(min_cross_section)]
        corr = _correlations(pairs, name)
        r = pairs.groupby("date")[name].rank(pct=True)
        buckets = np.ceil(r * 5).astype(int) - 1
        returns = pairs.groupby(["date", buckets])["excess_return"].mean().unstack().mean()
        traces.append(corr.rename(name))
        rows.append({"signal": name, "dates": corr.count(), "rank_ic": corr.mean(),
                     "icir_unannualized": corr.mean() / corr.std() if corr.std() > 0 else np.nan,
                     "positive_ic_share": corr.dropna().gt(0).mean(), "hac_t": _hac_t(corr, lag),
                     **{f"q{j + 1}_excess": returns.get(j, np.nan) for j in range(5)}})
    return pd.DataFrame(rows), pd.concat(traces, axis=1).rename_axis("date").reset_index()


def permutation_diagnostic(fitted, sample: pd.DataFrame, min_cross_section: int, seed: int) -> pd.DataFrame:
    results = []
    rng = np.random.default_rng(seed)
    for dates, bundle in fitted:
        # Use a fixed subsample of OOS sessions. No training rows or lockbox outcomes enter this diagnostic.
        rows = (sample.loc[sample["date"].isin(dates[::2])].dropna(subset=["excess_return"])
                .sort_values(["date", "code"]).reset_index(drop=True))
        rows = rows.loc[rows.groupby("date")["code"].transform("size").ge(min_cross_section)]
        if rows.empty:
            continue
        x = design(rows, bundle.columns)
        model_name = select_explain_model(bundle)
        model = bundle.models[model_name]
        pairs = rows[["date", "excess_return"]].copy()
        pairs["prediction"] = predict_model(model_name, model, x)
        before = _correlations(pairs, "prediction")
        grouped_positions = list(rows.groupby("date", sort=False).indices.values())
        for position, name in enumerate(bundle.columns):
            original = x[:, position].copy()
            for indexes in grouped_positions:
                x[indexes, position] = rng.permutation(original[indexes])
            pairs["prediction"] = predict_model(model_name, model, x)
            difference = before - _correlations(pairs, "prediction")
            results.append({"fit_date": bundle.train_stats["fit_date"], "model": model_name, "factor": name,
                            "oos_sessions": difference.count(), "mean_rank_ic_drop": difference.mean()})
            x[:, position] = original
    return pd.DataFrame(results, columns=["fit_date", "model", "factor", "oos_sessions", "mean_rank_ic_drop"])


def purged_diagnostic(sample: pd.DataFrame, columns: list[str], sessions: pd.DatetimeIndex,
                      settings: dict, partitions: int = 6, build_paths: bool = False):
    """Purged one-block and two-block CV, with optional chronological CPCV predictions."""
    pool = sample.dropna(subset=["excess_return", "exit_date"]).copy()
    dates = pd.DatetimeIndex(pool["date"].unique()).sort_values()
    if len(dates) < 2 * partitions:
        return (pd.DataFrame(), {}) if build_paths else pd.DataFrame()
    blocks = np.array_split(dates.to_numpy(), partitions)
    pool["target_rank"] = pool.groupby("date")["excess_return"].rank(pct=True)
    cutoff_locations = sessions.get_indexer(pool["exit_date"])
    signaled_locations = sessions.get_indexer(pool["date"])
    records, predictions = [], {}
    for group_count in (1, 2):
        for test_blocks in combinations(range(partitions), group_count):
            train_mask = np.ones(len(pool), dtype=bool)
            selected = np.zeros(len(pool), dtype=bool)
            for block in test_blocks:
                mask = pool["date"].isin(blocks[block]).to_numpy()
                selected |= mask
                start = int(signaled_locations[mask].min())
                exit_end = int(cutoff_locations[mask].max())
                train_mask &= ((cutoff_locations < start) | (signaled_locations > exit_end + settings["embargo"]))
            train, test = pool.loc[train_mask], pool.loc[selected].copy()
            if train["date"].nunique() < settings["min_train_dates"] or test.empty:
                continue
            estimator = Ridge(alpha=50, solver="cholesky").fit(design(train, columns), train["target_rank"])
            test["ridge"] = estimator.predict(design(test, columns))
            stats, _ = rank_ic(test, ["ridge"], 20, 1)
            records.append({"scheme": "purged_kfold" if group_count == 1 else "cpcv_split",
                            "test_blocks": ",".join(str(x + 1) for x in test_blocks),
                            "train_rows": len(train), "test_rows": len(test), "rank_ic": stats["rank_ic"].iat[0]})
            if build_paths and group_count == 2:
                full = sample.loc[sample["date"].isin(np.concatenate([blocks[i] for i in test_blocks]))]
                predictions[test_blocks] = full[["date", "code", "excess_return"]].assign(ridge=estimator.predict(design(full, columns)))
    splits = pd.DataFrame(records)
    if not build_paths:
        return splits
    paths = {}
    if partitions % 2 != 0:
        raise ValueError("Two-test-block path construction requires an even number of partitions")
    order = list(range(partitions))
    for i in range(partitions - 1):
        match = [tuple(sorted(pair)) for pair in zip(order[:partitions // 2], order[partitions // 2:][::-1])]
        if all(pair in predictions for pair in match):
            paths[f"cpcv_{i+1}"] = pd.concat([predictions[pair] for pair in match]).sort_values(["date", "code"])
        order = [order[0], order[-1], *order[1:-1]]
    return splits, paths


def deflated_sharpe_and_pbo(paths: dict[str, pd.DataFrame], blocks: int = 8) -> dict:
    """Use common, non-overlapping rebalance intervals and benchmark-relative log returns."""
    returns = {}
    for name, path in paths.items():
        nav = path.loc[path["rebalance_signal"]].set_index("date")
        returns[name] = np.log(nav["nav"] / nav["benchmark"]).diff()
    matrix = pd.DataFrame(returns).dropna()
    t, n = matrix.shape
    if t < 24 or n < 2 or matrix.std().le(0).any():
        return {"trials": n, "nonoverlapping_periods": t, "dsr_ensemble": None, "pbo": None}
    sr = matrix.mean().div(matrix.std())
    gamma = 0.5772156649015329
    expected_max = sr.std() * ((1 - gamma) * norm.ppf(1 - 1 / n) + gamma * norm.ppf(1 - 1 / (n * math.e)))
    x = matrix["ensemble"].to_numpy()
    estimate = sr["ensemble"]
    denominator = 1 - skew(x, bias=False) * estimate + (kurtosis(x, fisher=False, bias=False) - 1) * estimate ** 2 / 4
    dsr = float(norm.cdf((estimate - expected_max) * math.sqrt((t - 1) / denominator))) if denominator > 0 else None
    chunks = np.array_split(np.arange(t), blocks)
    bad = 0
    count = 0
    for chosen in combinations(range(blocks), blocks // 2):
        inside = np.concatenate([chunks[index] for index in chosen])
        outside = np.concatenate([chunks[index] for index in range(blocks) if index not in chosen])
        a, b = matrix.iloc[inside], matrix.iloc[outside]
        in_sr, out_sr = a.mean().div(a.std().where(a.std().gt(0))), b.mean().div(b.std().where(b.std().gt(0)))
        if in_sr.isna().any() or out_sr.isna().any():
            continue
        winning_rank = out_sr.rank(method="average").loc[in_sr.idxmax()] / (n + 1)
        bad += int(winning_rank <= 0.5)
        count += 1
    rng = np.random.default_rng(42)
    block_size = 3
    starts = rng.integers(0, t, size=(1000, math.ceil(t / block_size)))
    indexes = (starts[..., None] + np.arange(block_size)).reshape(1000, -1)[:, :t] % t
    bootstrap_mean = x[indexes].mean(axis=1)
    correlation = matrix.corr().to_numpy()[np.triu_indices(n, 1)]
    return {"trials": n, "nonoverlapping_periods": t, "dsr_ensemble": dsr,
            "pbo": bad / count if count else None, "cscv_splits": count,
            "mean_trial_correlation": float(np.nanmean(correlation)),
            "ensemble_relative_log_return_bootstrap": {"block_intervals": block_size, "resamples": 1000,
                                                        "period_mean_p025": float(np.quantile(bootstrap_mean, .025)),
                                                        "period_mean_p975": float(np.quantile(bootstrap_mean, .975))},
            "trial_set": list(matrix.columns), "assumptions": "CSCV and DSR are development diagnostics; correlated trials and serial dependence limit interpretation."}
