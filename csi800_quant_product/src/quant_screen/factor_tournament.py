"""Systematic single-factor audit, industry+size neutralization check, and correlation-cluster selection."""

from __future__ import annotations

import html
from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd

from .data import benchmark_for
from .factor_lib import ALL_CANDIDATE_FACTORS, FACTOR_TO_FAMILY, build_candidate_factors, neutralize_cross_section
from .io import write_csv, write_json, write_text
from .labels import executable_labels
from .validation import rank_ic


def _autocorr_lag(frame: pd.DataFrame, columns: list[str], step: int) -> dict[str, float]:
    """Measure cross-sectional rank stability across `step` sampled periods (higher = lower turnover)."""
    dates = sorted(frame["date"].unique())
    if len(dates) <= step:
        return {c: np.nan for c in columns}
    indexed = frame.set_index(["date", "code"])[columns]
    result = {}
    for col in columns:
        wide = indexed[col].unstack("code").reindex(dates)
        corrs = []
        for i in range(step, len(dates)):
            a, b = wide.iloc[i - step], wide.iloc[i]
            both = a.notna() & b.notna()
            if both.sum() >= 30:
                corrs.append(float(a[both].corr(b[both])))
        result[col] = float(np.mean(corrs)) if corrs else np.nan
    return result


def _select_orthogonal_factors(
    raw_table: pd.DataFrame,
    neutral_table: pd.DataFrame,
    corr_matrix: pd.DataFrame,
    *,
    max_corr: float = 0.70,
    min_coverage: float = 0.70,
    min_abs_neutral_ic: float = 0.015,
    min_abs_hac_t: float = 1.5,
) -> list[dict]:
    """Select strong, non-redundant factors based on Industry+Size neutralized ICIR and pairwise correlation."""
    merged = raw_table[["factor", "family", "coverage", "rank_ic", "icir_unannualized", "hac_t", "rank_autocorr_20d"]].merge(
        neutral_table[["factor", "rank_ic", "icir_unannualized", "hac_t", "q5_minus_q1"]],
        on="factor",
        suffixes=("_raw", "_neutral"),
    )
    merged["abs_neutral_icir"] = merged["icir_unannualized_neutral"].abs()
    merged["abs_raw_icir"] = merged["icir_unannualized_raw"].abs()
    # Rank candidates by neutralized ICIR (pure alpha), breaking ties with raw ICIR
    ordered = merged.sort_values(["abs_neutral_icir", "abs_raw_icir"], ascending=[False, False])

    selected: list[dict] = []
    chosen_cols: list[str] = []
    for row in ordered.itertuples(index=False):
        if pd.isna(row.rank_ic_neutral) or row.coverage < min_coverage:
            continue
        if abs(row.rank_ic_neutral) < min_abs_neutral_ic or abs(row.hac_t_neutral) < min_abs_hac_t:
            continue
        col = f"cand_{row.factor}"
        redundant_with = None
        for prev in chosen_cols:
            cval = abs(float(corr_matrix.at[col, prev]))
            if cval > max_corr:
                redundant_with = prev.removeprefix("cand_")
                break
        if redundant_with is not None:
            continue
        chosen_cols.append(col)
        selected.append({
            "factor": row.factor,
            "family": row.family,
            "recommended_sign": 1 if row.rank_ic_neutral > 0 else -1,
            "neutral_rank_ic": float(row.rank_ic_neutral),
            "neutral_icir": float(row.icir_unannualized_neutral),
            "neutral_hac_t": float(row.hac_t_neutral),
            "raw_rank_ic": float(row.rank_ic_raw),
            "raw_hac_t": float(row.hac_t_raw),
            "coverage": float(row.coverage),
            "rank_autocorr_20d": None if pd.isna(row.rank_autocorr_20d) else float(row.rank_autocorr_20d),
        })
    return selected


def run_factor_tournament(
    quotes: pd.DataFrame,
    market: pd.DataFrame,
    member: pd.Series,
    industry: pd.Series,
    coverage: pd.DataFrame,
    quality_history: pd.DataFrame | None,
    config: dict,
    output: Path,
) -> dict:
    """Evaluate all 48 candidate factors strictly on the development interval (< lockbox_start)."""
    settings = config["research"]
    lockbox = pd.Timestamp(settings["lockbox_start"])
    dev_quotes = quotes.loc[quotes["date"].lt(lockbox)].copy()
    dev_market = benchmark_for(market.loc[market.index < lockbox])
    dev_member = member.loc[dev_quotes.index]
    dev_industry = industry.loc[dev_quotes.index]
    dev_cov = coverage.loc[coverage.index < lockbox]

    print("Computing 48-factor candidate library on development universe...", flush=True)
    panel, rank_cols = build_candidate_factors(
        dev_quotes, dev_market, dev_member, dev_industry, dev_cov, quality_history, config, neutralize=False
    )
    labels = executable_labels(dev_quotes, dev_market, settings["horizon"])
    sample_mask = panel["eligible"] & labels["session"].mod(settings["sample_every"]).eq(0) & panel["date"].ge(settings["oos_start"])
    sample = panel.loc[sample_mask].join(labels.loc[sample_mask, ["excess_return", "session"]]).dropna(subset=["excess_return"])
    lag = max(1, ceil(settings["horizon"] / settings["sample_every"]))
    min_cs = config["data"]["minimum_cross_section"]

    print(f"Scoring {len(rank_cols)} raw candidate factors across {sample['date'].nunique()} development dates...", flush=True)
    raw_stats, _ = rank_ic(sample, rank_cols, min_cs, lag)
    raw_stats["factor"] = raw_stats["signal"].str.removeprefix("cand_")
    raw_stats["family"] = raw_stats["factor"].map(FACTOR_TO_FAMILY)
    raw_stats["coverage"] = [float(sample[c].notna().mean()) for c in raw_stats["signal"]]
    raw_stats["q5_minus_q1"] = raw_stats["q5_excess"] - raw_stats["q1_excess"]
    step = max(1, settings["horizon"] // settings["sample_every"])
    autocorr = _autocorr_lag(sample, rank_cols, step)
    raw_stats["rank_autocorr_20d"] = raw_stats["signal"].map(autocorr)
    raw_stats = raw_stats.sort_values("icir_unannualized", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)
    write_csv(output / "factor_tournament_raw.csv", raw_stats)

    print("Running Industry + Size OLS neutralization across sampled dates...", flush=True)
    neutral_ranks = neutralize_cross_section(sample, ALL_CANDIDATE_FACTORS)
    neutral_sample = sample[["date", "code", "excess_return"]].copy()
    for factor in ALL_CANDIDATE_FACTORS:
        neutral_sample[f"cand_{factor}"] = neutral_ranks[factor].to_numpy(dtype=np.float32)

    neutral_stats, _ = rank_ic(neutral_sample, rank_cols, min_cs, lag)
    neutral_stats["factor"] = neutral_stats["signal"].str.removeprefix("cand_")
    neutral_stats["family"] = neutral_stats["factor"].map(FACTOR_TO_FAMILY)
    neutral_stats["coverage"] = [float(neutral_sample[c].notna().mean()) for c in neutral_stats["signal"]]
    neutral_stats["q5_minus_q1"] = neutral_stats["q5_excess"] - neutral_stats["q1_excess"]
    neutral_stats = neutral_stats.sort_values("icir_unannualized", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)
    write_csv(output / "factor_tournament_neutral.csv", neutral_stats)

    corr_matrix = neutral_sample[rank_cols].corr()
    write_csv(output / "factor_tournament_correlations.csv", corr_matrix.rename_axis("signal").reset_index())

    selected = _select_orthogonal_factors(raw_stats, neutral_stats, corr_matrix)
    family_summary = (
        neutral_stats.groupby("family")
        .agg(
            candidates=("factor", "size"),
            mean_abs_neutral_ic=("rank_ic", lambda s: float(s.abs().mean())),
            best_factor=("factor", "first"),
            best_neutral_ic=("rank_ic", "first"),
            best_neutral_t=("hac_t", "first"),
        )
        .reset_index()
    )
    write_csv(output / "factor_family_summary.csv", family_summary)

    summary = {
        "development_window": f"{sample['date'].min().date()} to {sample['date'].max().date()}",
        "sampled_dates": int(sample["date"].nunique()),
        "total_candidates": len(ALL_CANDIDATE_FACTORS),
        "selected_count": len(selected),
        "selected_factors": selected,
        "family_summary": family_summary.to_dict(orient="records"),
    }
    write_json(output / "factor_selected.json", summary)

    # Generate HTML dashboard
    top_n = neutral_stats[["factor", "family", "coverage", "rank_ic", "icir_unannualized", "positive_ic_share", "hac_t", "q5_minus_q1"]].copy()
    top_n.columns = ["因子", "因子族", "覆盖率", "中性化 RankIC", "ICIR", "正IC占比", "HAC t值", "Q5-Q1超额"]
    sel_df = pd.DataFrame(selected)
    sel_html = sel_df.to_html(index=False, escape=True, float_format=lambda v: f"{v:.4f}") if not sel_df.empty else "<p>无满足门槛的因子</p>"
    page = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>中证 800 候选因子赛马报告</title>
<style>body{{margin:0;padding:28px;font:15px/1.6 'Microsoft YaHei',system-ui,sans-serif;background:#f7f8f6;color:#152b29}}
section{{background:#fff;border:1px solid #dfebe7;border-radius:8px;padding:18px;margin-top:20px;overflow:auto}}
table{{border-collapse:collapse;width:100%;font-size:0.86rem;font-variant-numeric:tabular-nums}}
th,td{{padding:7px 9px;border-bottom:1px solid #e3eae8;text-align:left}} th{{background:#e9f3f0}}</style></head>
<body><h1>中证 800 48 因子库赛马与行业+市值中性化体检</h1>
<p>评估区间：{html.escape(summary['development_window'])}（{summary['sampled_dates']} 个开发截面，严格位于锁箱之前）· 候选总数 {summary['total_candidates']} · 正交去重晋级 {summary['selected_count']} 个</p>
<section><h2>正交去重后晋级因子（|corr| ≤ 0.70，行业+市值中性化 |IC| ≥ 0.015 且 |t| ≥ 1.5）</h2>{sel_html}</section>
<section><h2>因子族汇总（行业+市值中性化口径）</h2>{family_summary.to_html(index=False, escape=True, float_format=lambda v: f'{v:.4f}')}</section>
<section><h2>全部 48 候选因子中性化榜单</h2>{top_n.to_html(index=False, escape=True, float_format=lambda v: f'{v:.4f}')}</section>
</body></html>"""
    write_text(output / "factor_tournament_report.html", page)
    return summary
