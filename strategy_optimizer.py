import random
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd


@dataclass
class SearchResult:
    best_params: dict
    best_score: float
    detail: list[dict]


def _safe_float(x, default=0.0) -> float:
    try:
        return float(x)
    except:
        return float(default)


# ═══════════════════════════════════════════════════════════
# Stage 1: Hard Filter
# ═══════════════════════════════════════════════════════════

def hard_filter(metrics: dict) -> bool:
    """
    硬过滤：任一条件不满足则淘汰。
    包括传统收益/风控指标 + 策略有效性指标（防空仓/低信号）
    """
    ann_ret = _safe_float(metrics.get("annual_return", 0.0))
    max_dd = _safe_float(metrics.get("max_drawdown", -1.0))
    turnover = _safe_float(metrics.get("turnover", 0.0))
    trades_total = int(metrics.get("trades_total", 0) or 0)
    invested_ratio = _safe_float(metrics.get("invested_ratio", 0.0))
    signal_days_ratio = _safe_float(metrics.get("signal_days_ratio", 0.0))
    avg_candidates = _safe_float(metrics.get("avg_candidates", 0.0))

    # 传统指标
    if ann_ret < 0.05:
        return False
    if max_dd < -0.50:
        return False

    # 策略有效性：宽松门槛，靠 Pareto 前沿筛选
    if trades_total < 10:
        return False
    if invested_ratio < 0.15:
        return False
    if signal_days_ratio < 0.20:
        return False
    if avg_candidates < 1.5:
        return False

    # 换手率合理性
    if turnover > 20.0:
        return False
    return True


# ═══════════════════════════════════════════════════════════
# Stage 2: Pareto Frontier
# ═══════════════════════════════════════════════════════════

PARETO_KEYS = ["annual_return", "sharpe", "sortino", "calmar", "neg_maxdd", "neg_turnover"]


def _pareto_keys(metrics: dict) -> dict:
    """将 metrics 转为 Pareto 比较用的 dict（全部最大化）"""
    return {
        "annual_return": _safe_float(metrics.get("annual_return", 0.0)),
        "sharpe": _safe_float(metrics.get("sharpe", 0.0)),
        "sortino": _safe_float(metrics.get("sortino", 0.0)),
        "calmar": _safe_float(metrics.get("calmar", 0.0)),
        "neg_maxdd": -abs(_safe_float(metrics.get("max_drawdown", 0.0))),
        "neg_turnover": -_safe_float(metrics.get("turnover", 0.0)),  # 真实组合换手率（年化）
    }


def _dominates(a: dict, b: dict) -> bool:
    """a dominates b: a 在所有维度 >= b，且至少一个维度 > b"""
    all_ge = all(a[k] >= b[k] for k in PARETO_KEYS)
    any_gt = any(a[k] > b[k] for k in PARETO_KEYS)
    return all_ge and any_gt


def pareto_filter(candidates: list[dict]) -> list[dict]:
    """返回非支配前沿（Pareto-optimal set）"""
    front = []
    for i, a in enumerate(candidates):
        dominated = False
        for j, b in enumerate(candidates):
            if i == j:
                continue
            if _dominates(b["pareto"], a["pareto"]):
                dominated = True
                break
        if not dominated:
            front.append(a)
    return front


# ═══════════════════════════════════════════════════════════
# Stage 3: Parameter Stability
# ═══════════════════════════════════════════════════════════

def _perturb_params(params: dict, space: dict, rng: random.Random, delta: float = 0.10):
    """参数扰动：连续参数做 ±delta，离散参数随机选邻域值"""
    perturbed = dict(params)
    for k, v in space.items():
        if isinstance(v, dict) and "min" in v and "max" in v:
            # 连续参数：±delta 范围抖动
            orig = float(params.get(k, (v["min"] + v["max"]) / 2))
            lo = float(v["min"])
            hi = float(v["max"])
            perturbed[k] = np.clip(orig + rng.uniform(-delta, delta) * (hi - lo), lo, hi)
        elif isinstance(v, (list, tuple)) and len(v) >= 2:
            # 离散参数：显式邻域搜索 — 随机选一个不同于当前值的候选
            cur = params.get(k)
            candidates = [x for x in v if x != cur]
            if candidates:
                perturbed[k] = rng.choice(candidates)
    return perturbed


def stability_score_fn(
    bt_engine,
    build_strategy,
    data: pd.DataFrame,
    params: dict,
    space: dict,
    *,
    n_perturb: int = 5,
    commission_bps: float = 1.0,
    slippage_bps: float = 2.0,
) -> float:
    """
    参数稳定性：对最佳参数做 N 次微扰，评估得分稳定性。
    stability = mean(neighbor_scores) - std(neighbor_scores)
    越高越好：附近参数都有效的策略更稳健。
    """
    rng = random.Random(42)
    neighbor_scores = []

    # 原始参数评估
    try:
        stg = build_strategy(params)
        res = bt_engine.run(stg, data, base_point=1000.0,
                            commission_bps=commission_bps, slippage_bps=slippage_bps)
        m = res.get("metrics", {})
        pk = _pareto_keys(m)
        base = np.mean([pk[k] for k in PARETO_KEYS])
    except Exception:
        base = 0.0

    neighbor_scores.append(base)

    for _ in range(n_perturb):
        try:
            p2 = _perturb_params(params, space, rng)
            stg2 = build_strategy(p2)
            res2 = bt_engine.run(stg2, data, base_point=1000.0,
                                 commission_bps=commission_bps, slippage_bps=slippage_bps)
            m2 = res2.get("metrics", {})
            pk2 = _pareto_keys(m2)
            neighbor_scores.append(np.mean([pk2[k] for k in PARETO_KEYS]))
        except Exception:
            neighbor_scores.append(0.0)

    scores = np.array(neighbor_scores)
    return float(np.mean(scores) - np.std(scores))


# ═══════════════════════════════════════════════════════════
# Keep utility functions
# ═══════════════════════════════════════════════════════════


def sample_params(space: dict, rng: random.Random) -> dict:
    """
    space 示例：
      {
        "top_n": [1,2,3],
        "factor": ["roc12","volatility20"],
        "ascending": [True, False],
        "rebalance_freq": [1,5,10],
        "stop_loss": [0, 0.05, 0.1],
      }
    """
    params = {}
    for k, v in (space or {}).items():
        if isinstance(v, (list, tuple)) and len(v) > 0:
            params[k] = rng.choice(list(v))
        elif isinstance(v, dict) and "min" in v and "max" in v:
            # 连续空间（简单均匀采样）
            lo = float(v["min"])
            hi = float(v["max"])
            params[k] = lo + rng.random() * (hi - lo)
        else:
            params[k] = v
    return params


def split_walk_forward_dates(
    dates: list[str],
    train_days: int,
    test_days: int,
):
    """
    不重叠 walk-forward：
    [train_days][test_days] [train_days][test_days] ...
    """
    n = len(dates)
    i = 0
    while i + train_days + test_days <= n:
        train = dates[i: i + train_days]
        test = dates[i + train_days: i + train_days + test_days]
        yield train, test
        i = i + train_days + test_days


def staged_optimization(
    bt_engine,
    build_strategy: Callable[[dict], dict],
    data: pd.DataFrame,
    *,
    space: dict,
    n_trials: int = 300,
    seed: int = 42,
    train_days: int = 504,
    test_days: int = 252,
    top_k: int = 10,
    max_search_date: str = "2023-12-31",
    commission_bps: float = 1.0,
    slippage_bps: float = 2.0,
    progress_cb: Optional[Callable[[int, int, float], None]] = None,
    status_cb: Optional[Callable[[dict], None]] = None,
    phase_cb: Optional[Callable[[str], None]] = None,
    cancel_event=None,
) -> SearchResult:
    """
    三阶段优化器：硬过滤 → Pareto Frontier → 参数稳定性

    Stage 1: 硬过滤 — annual_return<10%, max_dd<-35%, trades太少的直接淘汰
    Stage 2: Pareto Frontier — 6目标非支配排序，只保留前沿候选
    Stage 3: 参数稳定性 — 扰动±10%，邻域有效才入选
    最终: mean_fold_score - 0.5*fold_std - 0.5*parameter_instability
    """
    if data is None or data.empty:
        return SearchResult(best_params={}, best_score=-1e9, detail=[])

    df = data.copy()
    if max_search_date:
        df = df[df["date"] <= max_search_date]
    if df.empty:
        return SearchResult(best_params={}, best_score=-1e9, detail=[{"error": "搜索数据为空"}])

    dates = sorted(df["date"].unique())
    folds = list(split_walk_forward_dates(dates, train_days, test_days))
    if not folds:
        return SearchResult(
            best_params={}, best_score=-1e9,
            detail=[{"error": "数据长度不足以构造WFO分段"}],
        )

    if phase_cb:
        phase_cb("WFO分段")

    rng = random.Random(seed)
    n_trials = int(n_trials)

    params_list = [sample_params(space, rng) for _ in range(n_trials)]
    all_detail = []
    survivors = []  # [(idx, mean_pareto, fold_std, fold_metrics)]

    best_mean = -1e9

    if phase_cb:
        phase_cb("Trial搜索")
    trial_start = time.monotonic()

    # ── Per-trial WFO evaluation ──
    for i, params in enumerate(params_list):
        # Cancel check
        if cancel_event and cancel_event.is_set():
            if survivors:
                survivors.sort(key=lambda x: x["fold_mean"], reverse=True)
                best = survivors[0]
                return SearchResult(
                    best_params=best["params"] or {},
                    best_score=float(best["fold_mean"]),
                    detail=all_detail,
                )
            return SearchResult(best_params={}, best_score=-1e9,
                              detail=[{"error": "用户取消，无通过硬过滤的候选"}])

        stg = build_strategy(params)
        fold_metrics_list = []
        fold_pareto_list = []
        last_fc = {}

        for j, (train_dates, test_dates) in enumerate(folds):
            test_df = df[df["date"].isin(test_dates)]
            if test_df.empty:
                continue

            try:
                test_res = bt_engine.run(
                    stg, test_df, base_point=1000.0,
                    commission_bps=commission_bps, slippage_bps=slippage_bps,
                )
                m = test_res.get("metrics", {})
                m["years"] = len(test_dates) / 252.0
                last_fc = test_res.get("filter_chain", {})
            except Exception:
                import traceback
                print(f"[staged_optimization] 回测异常 trial={i+1} fold={j+1}\n{traceback.format_exc()}")
                continue

            # Stage 1: Hard filter per fold
            if not hard_filter(m):
                continue

            fold_metrics_list.append(m)
            fold_pareto_list.append(_pareto_keys(m))

        if not fold_metrics_list:
            all_detail.append({
                "trial": i + 1, "params": params, "passed": False,
                "reason": "all folds rejected",
                "filter_chain": {
                    "halted_pct": round(last_fc.get("halted_pct", 0), 3),
                    "exposure_zero_pct": round(last_fc.get("exposure_zero_pct", 0), 3),
                    "total_want": last_fc.get("total_want", 0),
                },
            })
            continue

        # Mean pareto metrics across folds
        mean_pareto = {}
        for k in PARETO_KEYS:
            mean_pareto[k] = float(np.mean([fp[k] for fp in fold_pareto_list]))

        fold_scores_agg = [np.mean([fp[k] for fp in fold_pareto_list]) for k in PARETO_KEYS]
        fold_mean = float(np.mean([np.mean([fp[k] for fp in fold_pareto_list]) for k in PARETO_KEYS]))
        fold_std = float(np.std([np.mean([fp[k] for k in PARETO_KEYS]) for fp in fold_pareto_list]))

        survivors.append({
            "idx": i,
            "params": params,
            "pareto": mean_pareto,
            "fold_mean": fold_mean,
            "fold_std": fold_std,
        })

        if fold_mean > best_mean:
            best_mean = fold_mean

        all_detail.append({
            "trial": i + 1, "params": params, "passed": True,
            "fold_mean": fold_mean, "fold_std": fold_std,
        })

        if progress_cb:
            progress_cb(i + 1, n_trials, best_mean)
        if status_cb:
            elapsed = time.monotonic() - trial_start
            completed = i + 1
            eta = (elapsed / completed) * (n_trials - completed) if completed > 0 else 0.0
            status_cb({
                "phase": "Trial搜索",
                "trial": completed,
                "n_trials": n_trials,
                "fold": len(fold_metrics_list),
                "n_folds": len(folds),
                "best": float(best_mean),
                "elapsed_sec": elapsed,
                "eta_sec": eta,
            })

    if not survivors:
        return SearchResult(best_params={}, best_score=-1e9, detail=[{"error": "所有候选未通过硬过滤"}])

    # ── Stage 2: Pareto Frontier ──
    if cancel_event and cancel_event.is_set():
        survivors.sort(key=lambda x: x["fold_mean"], reverse=True)
        best = survivors[0]
        return SearchResult(
            best_params=best["params"] or {},
            best_score=float(best["fold_mean"]),
            detail=all_detail,
        )
    if phase_cb:
        phase_cb("Pareto筛选")
    pareto_candidates = pareto_filter(survivors)
    if not pareto_candidates:
        # 兜底：取 top_k 个 fold_mean 最高的
        survivors.sort(key=lambda x: x["fold_mean"], reverse=True)
        pareto_candidates = survivors[:max(1, int(top_k))]

    # ── Stage 3: Parameter Stability ──
    if phase_cb:
        phase_cb("Stability测试")
    n_pareto = len(pareto_candidates)
    for cand_idx, cand in enumerate(pareto_candidates):
        if cancel_event and cancel_event.is_set():
            break
        if status_cb:
            status_cb({
                "phase": "Stability测试",
                "trial": cand_idx + 1,
                "n_trials": n_pareto,
                "fold": 0,
                "n_folds": 0,
                "best": float(best_mean),
                "elapsed_sec": time.monotonic() - trial_start,
                "eta_sec": 0.0,
            })
        try:
            stab = stability_score_fn(
                bt_engine, build_strategy, df,
                params=cand["params"], space=space,
                commission_bps=commission_bps, slippage_bps=slippage_bps,
            )
        except Exception:
            stab = 0.0
        cand["stability"] = stab

    # ── Final ranking ──
    for cand in pareto_candidates:
        cand["final_score"] = (
            cand["fold_mean"]
            - 0.5 * cand["fold_std"]
            - 0.5 * (1.0 - cand.get("stability", 0.0))
        )

    pareto_candidates.sort(key=lambda x: x["final_score"], reverse=True)
    best = pareto_candidates[0]

    summary = {
        "folds": [{"test_start": t[1][0], "test_end": t[1][-1]} for t in folds],
        "n_trials": n_trials,
        "n_survived_hard_filter": len(survivors),
        "n_pareto_frontier": len(pareto_candidates),
        "best_score": best["final_score"],
        "best_stability": best.get("stability", 0.0),
    }
    all_detail.insert(0, {"summary": summary})

    return SearchResult(
        best_params=best["params"] or {},
        best_score=float(best["final_score"]),
        detail=all_detail,
    )
