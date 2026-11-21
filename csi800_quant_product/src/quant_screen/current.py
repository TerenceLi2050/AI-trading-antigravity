import json
from pathlib import Path

import numpy as np
import pandas as pd

from .io import dates, read_table, require, sha256_file, stock_codes


def load_current(releases: list[str], day: pd.Timestamp) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    latest_members = None
    for entry in reversed(releases):
        path = Path(entry)
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest["last_session"] != str(day.date()) or "members_sha256" not in manifest:
            continue
        membership = path.parent / "current_membership_snapshot.csv"
        if sha256_file(membership) != manifest["members_sha256"]:
            raise ValueError("CSI membership snapshot checksum mismatch")
        members = read_table(membership)
        require(members, ["snapshot_date", "code", "name"], ["code"], "official snapshot")
        if len(members) != 800 or not dates(members["snapshot_date"]).eq(day).all():
            raise ValueError("Official membership snapshot date or count changed")
        members["code"] = stock_codes(members["code"])
        require(members, ["snapshot_date", "code", "name"], ["code"], "normalized official snapshot")
        if latest_members is None:
            latest_members = members
        elif set(latest_members["code"]) != set(members["code"]):
            continue
        financial = None
        if manifest["current_financials"]["status"] == "ok":
            path = path.parent / "current_financials.csv"
            if sha256_file(path) != manifest["current_financials"]["sha256"]:
                raise ValueError("Current financial snapshot checksum mismatch")
            financial = read_table(path)
            if financial.empty:
                financial = None
        if financial is not None:
            return latest_members, financial
    return latest_members, None


def apply_current_membership(quotes, day, member, industry, coverage, snapshot, disclosed):
    if snapshot is None:
        return member, industry, coverage
    present = quotes["date"].eq(day)
    verified = quotes["code"].isin(snapshot["code"])
    active = present & verified
    member = member.copy()
    industry = industry.copy()
    coverage = coverage.copy()
    member.loc[present] = verified.loc[present]
    coverage.loc[day, ["expected", "observed", "coverage"]] = (len(snapshot), active.sum(), active.sum() / len(snapshot))
    if disclosed is not None:
        lookup = disclosed.set_index("code")["industry"].astype("string").str.strip().replace("", pd.NA)
        industry.loc[active] = industry.loc[active].combine_first(quotes.loc[active, "code"].map(lookup).astype("string"))
    return member, industry, coverage


def ytd_quality(view: pd.DataFrame, visible: pd.Series, min_conversion: float,
                profile: str = "value_cash_quality") -> tuple[pd.Series, pd.Series]:
    financial = view["industry"].astype("string").str.contains("银行|证券|金融|保险|多元金融", na=False)
    fields = ["roe_ytd_pct", "book_per_share", "profit_ytd", "eps_ytd", "operating_cash_per_share_ytd"]
    values = view[fields].apply(pd.to_numeric, errors="raise")
    complete = values[fields[:-1]].notna().all(axis=1) & (financial | values[fields[-1]].notna()) & visible
    positive = values[fields[:-1]].gt(0).all(axis=1)
    ratio = values[fields[-1]].div(values["eps_ytd"].where(values["eps_ytd"].gt(0)))
    candidates = complete & view["eligible"] & positive & (financial | ratio.ge(min_conversion))
    has_price = "raw_close" in view.columns and pd.to_numeric(view["raw_close"], errors="coerce").gt(0).any()
    if profile == "value_cash_quality" and has_price:
        px = pd.to_numeric(view["raw_close"], errors="coerce").where(lambda s: s.gt(0))
        base = pd.DataFrame({
            "bp": values["book_per_share"].div(px),
            "cfp": values["operating_cash_per_share_ytd"].div(px).where(~financial),
            "cash": ratio.where(~financial),
            "roe": values["roe_ytd_pct"],
            "profit_growth": pd.to_numeric(view["profit_growth_pct"], errors="coerce"),
        })
        base = base.where(candidates, axis=0)
        group_size = candidates.groupby(view["industry"]).transform("sum")
        scores = base.groupby(view["industry"]).rank(pct=True).where(
            group_size.ge(8), base.groupby(financial).rank(pct=True), axis=0)
        non_fin = (0.35 * scores["bp"] + 0.30 * scores["cash"] + 0.20 * scores["cfp"] + 0.15 * scores["roe"])
        fin = (0.45 * scores["bp"] + 0.35 * scores["roe"] + 0.20 * scores["profit_growth"].fillna(0.5))
        rank = non_fin.where(~financial, fin)
        return complete, rank
    base = pd.DataFrame({"roe": values["roe_ytd_pct"], "cash": ratio.where(~financial),
                         "revenue_growth": view["revenue_growth_pct"], "profit_growth": view["profit_growth_pct"]})
    base = base.where(candidates, axis=0)
    group_size = candidates.groupby(view["industry"]).transform("sum")
    scores = base.groupby(view["industry"]).rank(pct=True).where(
        group_size.ge(8), base.groupby(financial).rank(pct=True), axis=0)
    rank = 0.55 * scores["roe"] + 0.45 * scores[["cash", "revenue_growth", "profit_growth"]].mean(axis=1)
    return complete, rank


def apply_current_quality(panel: pd.DataFrame, day: pd.Timestamp, snapshot, disclosed, min_conversion: float = 0.5,
                          max_age_days: int = 190, profile: str = "value_cash_quality") -> None:
    panel["fundamental_basis"] = np.where(panel["fundamentals_ok"], "filed_pit_ttm", "unavailable")
    panel["ytd_loss"] = False
    if snapshot is None:
        return
    current = panel["date"].eq(day)
    names = snapshot.set_index("code")["name"]
    panel.loc[current, "name"] = panel.loc[current, "code"].map(names)
    if disclosed is None:
        return
    needed = ["code", "industry", "report_date", "ann_date", "snapshot_date", "vendor_update", "source",
              "roe_ytd_pct", "book_per_share", "profit_ytd", "eps_ytd", "operating_cash_per_share_ytd",
              "profit_growth_pct", "revenue_growth_pct"]
    require(disclosed, needed, ["code"], "current financial snapshot")
    view_cols = [c for c in ("code", "industry", "eligible", "fundamentals_ok", "raw_close") if c in panel.columns]
    view = panel.loc[current, view_cols].join(
        disclosed.drop(columns="industry").set_index("code"), on="code", validate="many_to_one")
    for name in ("report_date", "ann_date", "snapshot_date", "vendor_update"):
        view[name] = dates(view[name])
    if view["report_date"].dropna().nunique() > 1:
        raise ValueError("Current YTD peers must share a reporting period")
    visible = (view["snapshot_date"].eq(day) & view["ann_date"].ge(view["report_date"])
               & view["report_date"].dt.is_quarter_end & view["report_date"].ge(day - pd.Timedelta(days=max_age_days))
               & view["ann_date"].lt(day) & view["vendor_update"].between(view["ann_date"], day)
               & view["source"].astype("string").str.strip().ne("").fillna(False) & ~view["fundamentals_ok"])
    valid, rank = ytd_quality(view, visible, min_conversion, profile=profile)
    rows = valid[valid].index
    panel.loc[rows, "fundamentals_ok"] = True
    panel.loc[rows, "quality_score"] = rank.loc[rows]
    panel.loc[rows, "negative_equity"] = view.loc[rows, "book_per_share"].le(0)
    panel.loc[rows, "ytd_loss"] = view.loc[rows, "profit_ytd"].le(0)
    panel.loc[rows, "fundamental_basis"] = "current_ytd_snapshot"
    for target, source in {"report_date": "report_date", "report_ann_date": "ann_date", "visible_after": "vendor_update",
                           "report_source": "source", "report_version": "snapshot_date", "roe_ytd_pct": "roe_ytd_pct",
                           "operating_cash_per_share_ytd": "operating_cash_per_share_ytd", "profit_ytd": "profit_ytd",
                           "book_per_share": "book_per_share"}.items():
        if target == "report_version":
            value = view.loc[rows, source].dt.strftime("%Y-%m-%d")
        else:
            value = view.loc[rows, source]
        panel.loc[rows, target] = value
