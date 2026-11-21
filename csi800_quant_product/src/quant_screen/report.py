import html
from pathlib import Path

import numpy as np
import pandas as pd

from .io import atomic_path, write_text


def _format(value, format_spec: str = ".4f") -> str:
    if value is None or pd.isna(value):
        return "—"
    if isinstance(value, (float, np.floating)):
        return format(value, format_spec)
    return str(value)


def nav_plot(path: Path, portfolios: dict[str, pd.DataFrame], total_return: bool) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.ticker import MultipleLocator

    figure = Figure(figsize=(10, 4.3), layout="constrained", facecolor="white")
    FigureCanvasAgg(figure)
    axes = figure.subplots()
    selected = portfolios["ensemble"]
    benchmark = "CSI 800 total return" if total_return else "CSI 800 price index"
    axes.plot(selected["date"], selected["benchmark"], color="#495862", label=benchmark, linewidth=1.7)
    for key, name, color, style in [("baseline", "Rules baseline", "#ae7640", "--"),
                                    ("ensemble", "Ensemble", "#116d65", "-"),
                                    ("ensemble_2x_cost", "Ensemble / double costs", "#4e77a6", ":")]:
        trace = portfolios[key]
        axes.plot(trace["date"], trace["nav"], label=name, color=color, linestyle=style, linewidth=1.8)
    axes.set_ylabel("NAV (start = 1.00)")
    axes.yaxis.set_major_locator(MultipleLocator(0.2))
    axes.grid(axis="y", color="#e7eeec")
    axes.spines[["right", "top"]].set_visible(False)
    axes.legend(loc="upper left", frameon=False, ncol=2, fontsize=9)
    with atomic_path(path) as temp:
        figure.savefig(temp, format="png", dpi=145)


def report_html(path: Path, screen: dict, data: dict, message: str, picks: pd.DataFrame,
                research: dict | None, scores: pd.DataFrame | None, table_file: str,
                diversified_picks: pd.DataFrame | None = None, quality_research: dict | None = None) -> None:
    status = {"FUNDAMENTAL_SCREEN": "基本面初筛", "HISTORICAL_SCREEN": "基本面初筛历史回放"}.get(screen["state"], "研究观察名单")
    if screen.get("publication_scope") == "development_research":
        status = "开发研究附表 · " + status
    notes = "".join(f"<li>{html.escape(reason)}</li>" for reason in screen["reasons"])
    view = picks[["rank", "code", "name", "industry", "raw_close", "adv20", "roe_ytd_pct", "final_score",
                  "quality_score", "risk_flags", "tree_contributions"]].copy()
    view["raw_close"] = view["raw_close"].map(lambda number: _format(number, ".2f"))
    view["adv20"] = view["adv20"].div(1e8).map(lambda number: _format(number, ".2f"))
    view["final_score"] = view["final_score"].map(_format)
    view["quality_score"] = view["quality_score"].map(_format)
    view["roe_ytd_pct"] = view["roe_ytd_pct"].map(lambda number: f"{number:.2f}%" if pd.notna(number) else "—")
    view.columns = ["排名", "股票代码", "名称", "行业", "收盘（元）", "20 日成交额中位（亿元）", "累计 ROE",
                    "综合分", "财务质量分", "风险标记", "模型特征贡献"]
    if view["模型特征贡献"].isna().all():
        view = view.drop(columns="模型特征贡献")
    view.insert(view.columns.get_loc("财务质量分") + 1, "量价规则分", picks["baseline"].map(_format))
    weight = screen["quality_weight"]
    model_note = (f"（量价腿已融合冻结模型，模型权重 {screen.get('model_weight', 0):.0%}）"
                  if screen.get("model_active") else "")
    score_note = (f"综合分由 {weight:.0%} 财务质量分和 {1-weight:.0%} 量价分{model_note}构成。逐股公告日期、评分分项和风险标记见下载表。"
                  if table_file == "quality_screen.csv" else "本次数据未达到正式名单门槛。表中展示同日可计算的研究排名。")
    diversified = ' · <a href="quality_screen_diversified.csv">行业分散名单</a>' if "diversified_names" in screen else ""
    diversified_html = ""
    if diversified_picks is not None and not diversified_picks.empty:
        dview = diversified_picks[["rank", "code", "name", "industry", "quality_score", "baseline", "final_score"]].copy()
        for col in ("quality_score", "baseline", "final_score"):
            dview[col] = dview[col].map(_format)
        dview.columns = ["排名", "股票代码", "名称", "行业", "财务质量分", "量价规则分", "综合分"]
        ind_counts = diversified_picks["industry"].fillna("未知").value_counts()
        ind_summary = " · ".join(f"{html.escape(str(k))} {int(v)} 只" for k, v in ind_counts.items())
        diversified_html = (
            f"<section><h2>行业分散候选名单（Top {len(dview)}）</h2>"
            f"<p>行业分布：{ind_summary}</p>"
            f"{dview.fillna('—').to_html(index=False, escape=True)}</section>"
        )
    session = screen.get("session") or {}
    phase = {"AFTER_CLOSE": "收盘后", "AWAITING_CLOSE": "当日尚未完成收盘确认", "NON_TRADING_DAY": "休市日",
             "CALENDAR_UNVERIFIED": "日历覆盖不足"}.get(session.get("phase"), "—")
    session_note = (f"交易状态 {phase} · 应使用 {session.get('expected_session') or '—'} 日行情"
                    f" · 收盘后验收覆盖 {session.get('verified_share', 0):.1%}")
    forward = ""
    if (path.parent / "prospective_summary.csv").is_file():
        note = screen.get("prospective_observations", {})
        forward = f"""<section><h2>前瞻跟踪</h2><p>归档 {note.get('records', 0)} 批信号，已取得开盘到开盘收益
{note.get('matured', 0)} 条，等待到期 {note.get('pending', 0)} 条。同一股票进入不同对照组时，各组分别计一条。
表中收益按可核验的到期开盘计算，未扣组合交易成本。</p><p><a href="prospective_summary.csv">分策略统计</a> ·
<a href="prospective_outcomes.csv">逐股到期及成交状态</a></p></section>"""
    quality_html = ""
    if quality_research is not None and quality_research.get("state") == "DEVELOPMENT_DIAGNOSTIC":
        q_rows = []
        for group_key in ("performance", "diversified_performance", "full_universe_control"):
            for name, row in (quality_research.get(group_key) or {}).items():
                q_rows.append((name, _format(row.get("total_return"), ".2%"), _format(row.get("annual_excess"), ".2%"),
                               _format(row.get("max_drawdown"), ".2%"), _format(row.get("annual_turnover"), ".2f")))
        if q_rows:
            q_df = pd.DataFrame(q_rows, columns=["规则/策略", "累计收益", "年化相对指数收益", "最大回撤", "年化双边换手"])
            quality_html = (
                f"<section><h2>实际发布规则历史对照（{html.escape(str(quality_research.get('first_signal', '')))} 至 "
                f"{html.escape(str(quality_research.get('last_signal', '')))}，共 {quality_research.get('signal_dates', 0)} 期）</h2>"
                f"{q_df.to_html(index=False, escape=True)}"
                f'<p><a href="quality_research.json">规则研究详情</a> · <a href="quality_strategy_nav.csv">逐日净值</a> · '
                f'<a href="quality_strategy_fills.csv">模拟成交</a></p></section>'
            )
    research_html = ""
    if research is not None and scores is not None and not scores.empty:
        comparisons = []
        for name, row in research["performance"].items():
            comparisons.append((name, _format(row.get("total_return"), ".2%"), _format(row.get("annual_excess"), ".2%"),
                                _format(row.get("max_drawdown"), ".2%"), _format(row.get("annual_turnover"), ".2f")))
        comp = pd.DataFrame(comparisons, columns=["策略", "累计收益", "年化相对指数收益", "最大回撤", "年化双边换手"])
        gate_notes = "".join(f"<li>{html.escape(reason)}</li>" for reason in research["research_gate_reasons"])
        metrics = research["multiple_testing"]
        ic = scores[["signal", "dates", "rank_ic", "hac_t"]].copy()
        ic.columns = ["模型", "评分期数", "平均 RankIC", "HAC t 值"]
        benchmark = "中证 800 全收益指数" if research["benchmark"] == "CSI 800 total return" else "中证 800 价格指数"
        yearly_link = ' · <a href="yearly_performance.csv">分年度绩效</a>' if (path.parent / "yearly_performance.csv").is_file() else ""
        research_html = f"""<section><h2>开发区间滚动验证</h2>
<p>{html.escape(research['first_oos_signal'])} 至 {html.escape(research['last_oos_signal'])} 生成信号。持有期末端和回测净值均截在锁箱之前。</p>
{ic.to_html(index=False, escape=True, float_format=lambda v: f'{v:.4f}')}
<p>{benchmark}同期收益 {_format(research['performance'].get('ensemble', {}).get('benchmark_return'), '.2%')}。DSR {_format(metrics.get('dsr_ensemble'))}，PBO {_format(metrics.get('pbo'))}。</p>
<img src="development_nav.png" alt="开发区间费后净值：中证 800、规则基线、集成模型与集成双倍成本">
{comp.to_html(index=False, escape=True)}<p>锁箱状态 <strong>{html.escape(research['lockbox_state'])}</strong></p><ul>{gate_notes}</ul>
<p><a href="research.json">研究指标与训练窗口</a> · <a href="quality_research.json">财务综合规则研究状态</a>{yearly_link} · <a href="cpcv_paths.csv">CPCV 路径</a> · <a href="factor_diagnostics.csv">单因子诊断</a> · <a href="factor_family_ablation.csv">因子族消融</a> · <a href="permutation_diagnostics.csv">样本外置乱诊断</a> · <a href="portfolio_nav.csv">净值和换手</a> · <a href="fills.csv">集成策略模拟成交</a></p></section>"""
    generated = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>中证 800 量化筛选 · {screen['signal_date']}</title>
<style>
body{{margin:0;background:#f7f8f6;color:#152b29;font:16px/1.6 'Microsoft YaHei','PingFang SC',system-ui,sans-serif}}
main{{max-width:1300px;margin:0 auto;padding:36px 24px}} header{{border-bottom:2px solid #22655b;padding-bottom:18px}}
h1{{font-size:clamp(1.8rem,3vw,2.8rem);margin:0}}h2{{font-size:1.4rem;margin:0 0 12px}}p{{margin:10px 0}} small{{color:#526262}}
section{{margin-top:26px;padding:20px;background:#fff;border:1px solid #dfebe7;border-radius:8px;overflow:auto}}
table{{border-collapse:collapse;width:100%;font-size:0.87rem;font-variant-numeric:tabular-nums;margin:12px 0}}
th,td{{text-align:left;padding:8px;border-bottom:1px solid #e3eae8}} th{{background:#e9f3f0;white-space:nowrap}}
img{{width:min(100%,900px);height:auto}}
tr:nth-child(even){{background:#f8faf9}} a{{color:#14594e}} .state{{font-weight:700;color:#81541b}} .summary{{display:flex;gap:0.8rem 1.4rem;flex-wrap:wrap}}
</style></head><body><main>
<header><h1>中证 800 量化筛选</h1><p class="state">{status} · 信号日 {screen['signal_date']}</p>
<small>运行日 {screen['run_date']} · {html.escape(message)}</small></header>
<section><h2>行情和覆盖</h2><p class="summary"><span>报价 {screen['quoted_names']} 只</span><span>通过量价门槛 {screen['eligible_names']} 只</span>
<span>可核对财务 {screen['fundamentally_covered_names']} 只</span><span>通过质量门槛 {screen['quality_qualified_names']} 只</span>
<span>中证 800 收盘 {_format(screen['csi800_close'], '.2f')}</span>
<span>指数近 20 日 {_format(screen['index_return_20'], '.2%')}</span><span>指数 60 日年化波动 {_format(screen['index_annualized_volatility_60'], '.2%')}</span>
<span>正动量广度 {_format(screen['positive_momentum_breadth'], '.2%')}</span></p><p>{html.escape(session_note)}</p><ul>{notes}</ul></section>
<section><h2>{status}</h2><p>{score_note}</p>
{view.fillna('—').to_html(index=False, escape=True)}<p><a href="{table_file}">下载筛选 CSV</a>{diversified} · <a href="screening_universe.csv">全股票池门槛记录</a> · <a href="screen.json">筛选状态</a> · <a href="audit.json">数据审计</a></p></section>
{diversified_html}{forward}{quality_html}{research_html}
<section><h2>数据记录</h2><p>合计 {data['rows']:,} 行 · {data['securities']:,} 只证券 · {data['sessions']:,} 个交易日 · 内部缺失 {data['internal_missing_stock_sessions']:,} 个股票日</p></section>
</main></body></html>"""
    write_text(path, generated)
