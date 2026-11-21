"""System health diagnostics for daily screening, historical research, and deployment."""

from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path

from .providers import baostock_cooldown_status


REQUIRED_CORE = ("numpy", "pandas", "pyarrow", "baostock", "requests", "xlrd")
OPTIONAL_PACKAGES = ("akshare", "scipy", "scikit-learn", "lightgbm", "catboost", "joblib", "matplotlib")


def _pkg_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _check_pointer(root: Path, output: Path, filename: str) -> dict:
    pointer_path = output / filename
    if not pointer_path.is_file():
        return {"exists": False}
    try:
        payload = json.loads(pointer_path.read_text(encoding="utf-8"))
        run_ref = payload.get("run")
        resolved = None
        if run_ref:
            candidate = Path(run_ref)
            if not candidate.is_absolute():
                candidate = (root / candidate).resolve()
            resolved = candidate
        return {
            "exists": True,
            "state": payload.get("state"),
            "signal_date": payload.get("signal_date"),
            "run": run_ref,
            "target_exists": bool(resolved and resolved.exists()),
            "is_relative": bool(run_ref and not Path(run_ref).is_absolute()),
        }
    except Exception as exc:
        return {"exists": True, "error": str(exc)}


def run_doctor(root: Path, config: dict) -> dict:
    data = config["data"]
    output = root / data["output"]
    core_deps = {pkg: _pkg_version(pkg) for pkg in REQUIRED_CORE}
    opt_deps = {pkg: _pkg_version(pkg) for pkg in OPTIONAL_PACKAGES}

    inputs = {}
    for key in (
        "canonical_quotes", "calendar", "total_return", "session_calendar",
        "membership", "quality_history", "financials", "status", "raw_history",
    ):
        path = root / data[key]
        inputs[key] = {
            "path": data[key],
            "exists": path.is_file(),
            "size_mb": round(path.stat().st_size / (1024 * 1024), 2) if path.is_file() else 0.0,
        }

    legacy_csv = root / data["legacy_csv"]
    legacy_cache = root / data["cache"] / "legacy.json"
    daily_releases = sorted((root / data["daily_updates"]).glob("*/manifest.json"))

    standalone_ready = inputs["canonical_quotes"]["exists"] and inputs["calendar"]["exists"]
    quality_research_ready = standalone_ready and inputs["membership"]["exists"] and inputs["quality_history"]["exists"]
    legacy_available = legacy_csv.is_file() or legacy_cache.is_file()

    screening_cfg = config.get("screening", {})
    model_path_str = screening_cfg.get("model_path", "")
    model_path = (root / model_path_str) if model_path_str else None
    model_ready = bool(model_path and model_path.is_file())

    pointers = {
        name: _check_pointer(root, output, name)
        for name in ("latest.json", "latest_run.json", "latest_research.json", "latest_replay.json")
    }
    cooldown = baostock_cooldown_status()

    recommendations = []
    if not daily_releases:
        recommendations.append("运行 `python run_daily_screen.py` 完成首次日常数据采集与出榜。")
    if not standalone_ready:
        recommendations.append(
            "运行 `python run_daily_screen.py --build-history`（或本地拼装 `--build-history --assemble-local`）以摆脱旧 CSV 依赖。"
        )
    if not inputs["membership"]["exists"]:
        recommendations.append("缺少 `data/input/membership.csv`（动态历史成分与行业），可由 `--build-history` 生成。")
    if not inputs["quality_history"]["exists"]:
        recommendations.append("缺少 `data/input/quality_history.csv`（PIT 历史财报），可由 `--build-history --skip-bars` 从东财按季度采集。")
    if cooldown.get("in_cooldown"):
        recommendations.append(
            f"BaoStock 当前处于本地退避期（至 {cooldown.get('retry_after')}），历史建库可加 `--bar-provider akshare` 绕过。"
        )
    for name, info in pointers.items():
        if info.get("exists") and not info.get("target_exists"):
            recommendations.append(f"`reports/{name}` 指向的目录不存在（目录搬迁遗留），建议刷新指针。")
    if not recommendations:
        recommendations.append("所有日常选股与独立历史研究模块均已就绪。")

    return {
        "root": str(root),
        "core_dependencies": core_deps,
        "optional_dependencies": opt_deps,
        "providers": {"baostock_cooldown": cooldown},
        "daily": {
            "releases_archived": len(daily_releases),
            "latest_release": daily_releases[-1].parent.name if daily_releases else None,
            "model_inference_enabled": model_ready and screening_cfg.get("model_weight", 0.0) > 0,
            "model_path": model_path_str or None,
        },
        "historical": {
            "standalone_research_ready": standalone_ready,
            "quality_research_ready": quality_research_ready,
            "legacy_available": legacy_available,
            "inputs": inputs,
        },
        "pointers": pointers,
        "recommendations": recommendations,
    }
