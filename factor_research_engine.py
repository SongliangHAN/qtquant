"""
Factor Research Engine — 全量因子研究引擎。

对 FACTOR_REGISTRY 中全部因子进行系统化研究，生成 factor_report_df。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.cluster import hierarchy as sch
from scipy.spatial.distance import squareform
from concurrent.futures import ThreadPoolExecutor, as_completed

from factor_lab import FactorLab
from factor_registry import (
    FACTOR_REGISTRY, FACTOR_GROUPS,
    get_all_factor_names, get_factor_info,
)


class FactorResearchEngine:
    """全量因子研究引擎。"""

    def __init__(self, data: pd.DataFrame):
        """
        data: long-form DataFrame with [date, code, close] + factor columns + regime.
        """
        self.data = data.sort_values(["code", "date"]).drop_duplicates(subset=["date", "code"]).copy()
        self.lab = FactorLab(data)
        self._available = self._find_available_factors()

    def _find_available_factors(self) -> list[str]:
        """找出 data 中实际存在且注册的因子列。"""
        registered = get_all_factor_names()
        return [f for f in registered if f in self.data.columns]

    # ═════════════════════════════════════════════════════════════════
    # Forward return 预计算（避免各 stage 重复 copy 全量数据）
    # ═════════════════════════════════════════════════════════════════

    def _prepare_forward_returns(self, horizons: list[int]) -> pd.DataFrame:
        """在 self.data 上一次性添加所有 fwd_ret_{h}d 列，返回含全部列的新 DataFrame。"""
        df = self.data.copy()
        for h in horizons:
            col = f"fwd_ret_{h}d"
            if col not in df.columns:
                df[col] = df.groupby("code")["close"].transform(
                    lambda x: x.shift(-h) / x - 1.0
                )
        return df

    # ═════════════════════════════════════════════════════════════════
    # 主入口
    # ═════════════════════════════════════════════════════════════════

    def run_full_research(
        self, horizon: int = 5, window: int = 60, n_quantiles: int = 5,
        progress_cb=None,
    ) -> dict:
        """
        对全部已注册因子进行完整研究。

        Returns dict:
          - "report": factor_report_df
          - "rolling_ic": wide DataFrame
          - "ic_decay": long DataFrame
          - "corr_matrix": DataFrame
          - "regime_ic": long DataFrame
          - "turnover": Series
          - "coverage": Series
        """
        factors = self._available
        if not factors:
            return {}

        total_stages = 7
        self._emit_progress(progress_cb, 0, total_stages, "开始全量因子研究...")

        # Stage 1: Basic stats + Coverage
        self._emit_progress(progress_cb, 1, total_stages, "计算基础统计...")
        basic = self._compute_all_basic_stats(factors)

        # 预计算所有 forward return 列（Stage 2/3/4/6 共用，仅 1 次全量 copy）
        decay_horizons = [1, 5, 10, 20]
        all_horizons = sorted(set(decay_horizons + [horizon]))
        data_fwd = self._prepare_forward_returns(all_horizons)

        # Stage 2: Rolling IC（优化：预建 date 查找表，避免每因子逐日扫描）
        self._emit_progress(progress_cb, 2, total_stages, "计算 Rolling RankIC...")
        rolling_ic = self._compute_rolling_ic(factors, data_fwd, horizon, window, progress_cb)

        # Stage 3: IC Decay (parallel per factor, with sub-progress)
        self._emit_progress(progress_cb, 3, total_stages, "计算 IC Decay...")
        ic_decay = self._compute_ic_decay_parallel(factors, data_fwd, decay_horizons, progress_cb)

        # Stage 4: Quantile returns (reuse pre-computed forward returns)
        self._emit_progress(progress_cb, 4, total_stages, "计算分位数收益...")
        quantile_results = self._compute_quantile_parallel(factors, data_fwd, horizon, n_quantiles)

        # Stage 5: Factor correlation
        self._emit_progress(progress_cb, 5, total_stages, "计算因子相关性...")
        corr_matrix = self.lab.compute_factor_correlation(factors)

        # Stage 6: Regime IC（同样使用预建查找表）
        self._emit_progress(progress_cb, 6, total_stages, "计算 Regime IC...")
        regime_ic = self._compute_regime_ic(factors, data_fwd, horizon, progress_cb)

        # Stage 7: Turnover + Stability + Monotonicity + Clusters
        self._emit_progress(progress_cb, 7, total_stages, "计算换手率/稳定性/聚类...")
        turnover = self._compute_all_turnover(factors)
        stability = self._compute_all_stability(factors)
        monotonicity = self._compute_all_monotonicity(quantile_results)
        clusters = self._compute_correlation_clusters(corr_matrix)

        # Assemble report
        report = self._assemble_report(
            factors, basic, rolling_ic, ic_decay,
            quantile_results, regime_ic,
            turnover, stability, monotonicity, clusters,
            corr_matrix=corr_matrix,
        )

        self._emit_progress(progress_cb, total_stages, total_stages, "完成")

        return {
            "report": report,
            "rolling_ic": rolling_ic,
            "ic_decay": ic_decay,
            "corr_matrix": corr_matrix,
            "regime_ic": regime_ic,
            "turnover": turnover,
            "coverage": pd.Series({f: basic[f]["coverage_pct"] for f in factors}),
            "stability": stability,
            "monotonicity": monotonicity,
            "clusters": clusters,
        }

    @staticmethod
    def _emit_progress(cb, cur: int, total: int, msg: str):
        if cb:
            cb(cur, total, msg)

    # ═════════════════════════════════════════════════════════════════
    # Basic Stats
    # ═════════════════════════════════════════════════════════════════

    def _compute_all_basic_stats(self, factors: list[str]) -> dict:
        result = {}
        for f in factors:
            col = self.data[f]
            valid = col.dropna()
            total = len(col)
            coverage = len(valid) / total if total > 0 else 0.0
            result[f] = {
                "mean_value": float(valid.mean()) if len(valid) > 0 else np.nan,
                "std_value": float(valid.std()) if len(valid) > 0 else np.nan,
                "coverage_pct": round(coverage * 100, 2),
            }
        return result

    # ═════════════════════════════════════════════════════════════════
    # Rolling IC（优化版：预建 date→Series 查找表，避免逐因子逐日全表扫描）
    # ═════════════════════════════════════════════════════════════════

    def _compute_rolling_ic(
        self, factors: list[str], data_fwd: pd.DataFrame,
        horizon: int, window: int, progress_cb=None,
    ) -> pd.DataFrame:
        fwd_col = f"fwd_ret_{horizon}d"
        dates = sorted(data_fwd["date"].unique())
        if not dates:
            return pd.DataFrame()

        # —— 一次性预建 date→Series 查找表（仅扫描 N 次，不是 N×M 次）——
        fwd_by_date: dict = {}
        factor_by_date: dict[str, dict] = {f: {} for f in factors}
        for d in dates:
            sub = data_fwd[data_fwd["date"] == d]
            fwd_vals = sub.set_index("code")[fwd_col].dropna()
            if len(fwd_vals) > 0:
                fwd_by_date[d] = fwd_vals
            for f in factors:
                f_vals = sub.set_index("code")[f].dropna()
                if len(f_vals) > 0:
                    factor_by_date[f][d] = f_vals

        all_ics: dict[str, pd.Series] = {}
        n_total = len(factors)
        for idx, factor in enumerate(factors):
            ic_series: dict = {}
            f_lookup = factor_by_date[factor]
            for i in range(horizon, len(dates)):
                t = dates[i]
                t_lag = dates[i - horizon]
                sub_f = f_lookup.get(t_lag)
                sub_r = fwd_by_date.get(t)
                if sub_f is None or sub_r is None:
                    ic_series[t] = np.nan
                    continue
                common = sub_f.index.intersection(sub_r.index)
                if len(common) < 5:
                    ic_series[t] = np.nan
                    continue
                try:
                    ic, _ = spearmanr(sub_f.loc[common], sub_r.loc[common])
                    ic_series[t] = abs(ic) if not np.isnan(ic) else np.nan
                except Exception:
                    ic_series[t] = np.nan

            raw = pd.Series(ic_series)
            all_ics[f"ic_{factor}"] = raw.rolling(window, min_periods=20).mean()
            if progress_cb:
                progress_cb(2, 7, f"Rolling IC {idx + 1}/{n_total} 因子")

        result = pd.DataFrame({"date": dates})
        for k, v in all_ics.items():
            result = result.merge(
                pd.DataFrame({"date": v.index, k: v.values}), on="date", how="left",
            )
        return result

    # ═════════════════════════════════════════════════════════════════
    # Regime IC（优化版：复用预建查找表思路）
    # ═════════════════════════════════════════════════════════════════

    def _compute_regime_ic(
        self, factors: list[str], data_fwd: pd.DataFrame,
        horizon: int, progress_cb=None,
    ) -> pd.DataFrame:
        fwd_col = f"fwd_ret_{horizon}d"
        dates = sorted(data_fwd["date"].unique())
        regimes_list = ["bull", "bear", "sideways", "panic", "bull_volatile"]

        # 预建查找表
        fwd_by_date: dict = {}
        regime_by_date: dict = {}
        factor_by_date: dict[str, dict] = {f: {} for f in factors}
        for d in dates:
            sub = data_fwd[data_fwd["date"] == d]
            fwd_vals = sub.set_index("code")[fwd_col].dropna()
            if len(fwd_vals) > 0:
                fwd_by_date[d] = fwd_vals
            if "regime" in sub.columns and len(sub) > 0:
                regime_by_date[d] = sub["regime"].iloc[0]
            for f in factors:
                f_vals = sub.set_index("code")[f].dropna()
                if len(f_vals) > 0:
                    factor_by_date[f][d] = f_vals

        # 按 regime 收集 IC
        result: dict[str, dict[str, list]] = {f: {r: [] for r in regimes_list} for f in factors}
        for i in range(horizon, len(dates)):
            t = dates[i]
            t_lag = dates[i - horizon]
            regime_val = regime_by_date.get(t_lag)
            if regime_val is None or regime_val not in regimes_list:
                continue
            for f in factors:
                sub_f = factor_by_date[f].get(t_lag)
                sub_r = fwd_by_date.get(t)
                if sub_f is None or sub_r is None:
                    continue
                common = sub_f.index.intersection(sub_r.index)
                if len(common) < 5:
                    continue
                try:
                    ic, _ = spearmanr(sub_f.loc[common], sub_r.loc[common])
                    if not np.isnan(ic):
                        result[f][regime_val].append(abs(ic))
                except Exception:
                    pass

        rows = []
        for f in factors:
            row = {"factor": f}
            for r in regimes_list:
                vals = result[f][r]
                row[r] = np.mean(vals) if vals else np.nan
            rows.append(row)
        return pd.DataFrame(rows)

    # ═════════════════════════════════════════════════════════════════
    # IC Decay (parallel)
    # ═════════════════════════════════════════════════════════════════

    def _compute_ic_decay_parallel(
        self, factors: list[str], data_fwd: pd.DataFrame,
        horizons: list[int], progress_cb=None,
    ) -> pd.DataFrame:
        n_total = len(factors)
        rows = []
        completed = 0
        # 每个线程直接使用 data_fwd（只读，无 copy）
        with ThreadPoolExecutor(max_workers=min(8, n_total)) as ex:
            futures = {
                ex.submit(self._ic_decay_one, f, data_fwd, horizons): f
                for f in factors
            }
            for fut in as_completed(futures):
                row = fut.result()
                if row is not None:
                    rows.append(row)
                completed += 1
                if progress_cb:
                    progress_cb(3, 7, f"IC Decay {completed}/{n_total} 因子")

        return pd.DataFrame(rows)

    @staticmethod
    def _ic_decay_one(factor: str, data: pd.DataFrame, horizons: list[int]) -> dict | None:
        """单因子 IC Decay 计算。data 已包含所有 fwd_ret_{h}d 列。"""
        if factor not in data.columns:
            return None
        row = {"factor": factor}
        dates = sorted(data["date"].unique())

        # 按 horizon 从大到小排序，大 horizon 的日期索引范围更小，可复用切片思路
        for h in horizons:
            fwd_col = f"fwd_ret_{h}d"
            if fwd_col not in data.columns:
                row[f"h{h}"] = np.nan
                continue

            # 预建 date→Series 查找表（只做一次 per factor）
            factor_by_date = {}
            fwd_by_date = {}
            for d in dates:
                sub = data[data["date"] == d]
                f_vals = sub.set_index("code")[factor].dropna()
                if len(f_vals) > 0:
                    factor_by_date[d] = f_vals
                if fwd_col in sub.columns:
                    r_vals = sub.set_index("code")[fwd_col].dropna()
                    if len(r_vals) > 0:
                        fwd_by_date[d] = r_vals

            ics = []
            for i in range(h, len(dates)):
                t = dates[i]
                t_lag = dates[i - h]
                sub_f = factor_by_date.get(t_lag)
                sub_r = fwd_by_date.get(t)
                if sub_f is None or sub_r is None:
                    continue
                common = sub_f.index.intersection(sub_r.index)
                if len(common) < 5:
                    continue
                try:
                    ic, _ = spearmanr(sub_f.loc[common], sub_r.loc[common])
                    if not np.isnan(ic):
                        ics.append(abs(ic))
                except Exception:
                    pass
            row[f"h{h}"] = np.mean(ics) if ics else np.nan
        return row

    # ═════════════════════════════════════════════════════════════════
    # Quantile Returns (uses pre-computed forward returns)
    # ═════════════════════════════════════════════════════════════════

    def _compute_quantile_parallel(
        self, factors: list[str], data_fwd: pd.DataFrame,
        horizon: int, n_quantiles: int,
    ) -> dict:
        fwd_col = f"fwd_ret_{horizon}d"
        dates = sorted(data_fwd["date"].unique())
        results = {}
        for f in factors:
            if f not in data_fwd.columns or fwd_col not in data_fwd.columns:
                results[f] = pd.DataFrame()
                continue
            try:
                qrows = {}
                for d in dates:
                    sub = data_fwd[data_fwd["date"] == d].dropna(subset=[f, fwd_col])
                    if len(sub) < n_quantiles * 2:
                        continue
                    sub = sub.copy()
                    sub["q"] = pd.qcut(sub[f], n_quantiles, labels=False, duplicates="drop")
                    means = sub.groupby("q")[fwd_col].mean()
                    for q, v in means.items():
                        qrows.setdefault(d, {})[f"q{int(q)+1}"] = v
                results[f] = pd.DataFrame(qrows).T
            except Exception:
                results[f] = pd.DataFrame()
        return results

    # ═════════════════════════════════════════════════════════════════
    # Turnover
    # ═════════════════════════════════════════════════════════════════

    def _compute_all_turnover(self, factors: list[str]) -> pd.Series:
        vals = {}
        for f in factors:
            col = self.data[f]
            if col.isna().all():
                vals[f] = np.nan
                continue
            # 每日截面 rank，计算 rank 的日间变化
            df = self.data[["date", "code"]].copy()
            df["_val"] = col
            df = df.drop_duplicates(subset=["date", "code"])
            df["_rank"] = df.groupby("date")["_val"].rank(pct=True)
            ranked = df.pivot(index="date", columns="code", values="_rank")
            changes = ranked.diff().abs().mean(axis=1)
            vals[f] = float(changes.mean()) if len(changes) > 0 else np.nan
        return pd.Series(vals)

    # ═════════════════════════════════════════════════════════════════
    # Stability (autocorrelation)
    # ═════════════════════════════════════════════════════════════════

    def _compute_all_stability(self, factors: list[str]) -> pd.Series:
        vals = {}
        for f in factors:
            ranked = self._daily_cross_sectional_rank(f)
            if ranked.empty:
                vals[f] = np.nan
                continue
            ac = ranked.autocorr(lag=20)
            vals[f] = float(ac) if not np.isnan(ac) else np.nan
        return pd.Series(vals)

    def _daily_cross_sectional_rank(self, factor: str) -> pd.Series:
        """返回每日截面 rank 均值序列。"""
        df = self.data[["date", factor]].dropna()
        if df.empty:
            return pd.Series(dtype=float)
        df["_rank"] = df.groupby("date")[factor].rank(pct=True)
        return df.groupby("date")["_rank"].mean()

    # ═════════════════════════════════════════════════════════════════
    # Monotonicity
    # ═════════════════════════════════════════════════════════════════

    def _compute_all_monotonicity(self, quantile_results: dict) -> pd.Series:
        vals = {}
        for f, qdf in quantile_results.items():
            if qdf is None or qdf.empty or len(qdf.columns) < 2:
                vals[f] = np.nan
                continue
            mean_per_q = qdf.mean()
            q_ranks = np.arange(1, len(mean_per_q) + 1)
            if len(mean_per_q.dropna()) < 2:
                vals[f] = np.nan
                continue
            try:
                r, _ = spearmanr(q_ranks, mean_per_q.values)
                vals[f] = float(r)
            except Exception:
                vals[f] = np.nan
        return pd.Series(vals)

    # ═════════════════════════════════════════════════════════════════
    # Correlation Clusters
    # ═════════════════════════════════════════════════════════════════

    def _compute_correlation_clusters(self, corr_matrix: pd.DataFrame) -> dict:
        if corr_matrix.empty or len(corr_matrix) < 2:
            return {}
        d = 1.0 - corr_matrix.abs().values
        d = (d + d.T) / 2.0
        np.fill_diagonal(d, 0.0)
        try:
            link = sch.linkage(squareform(d), method="ward")
            labels = sch.fcluster(link, t=0.35, criterion="distance")
            return {f: int(c) for f, c in zip(corr_matrix.index, labels)}
        except Exception:
            return {f: 0 for f in corr_matrix.index}

    # ═════════════════════════════════════════════════════════════════
    # Assemble factor_report_df
    # ═════════════════════════════════════════════════════════════════

    def _assemble_report(
        self, factors, basic, rolling_ic, ic_decay,
        quantile_results, regime_ic,
        turnover, stability, monotonicity, clusters,
        corr_matrix=None,
    ) -> pd.DataFrame:
        rows = []
        for f in factors:
            info = get_factor_info(f) or {}
            row = {
                "factor": f,
                "group": info.get("group", ""),
                "sign": int(info.get("sign", 0) or 0),
                "name": info.get("name", f),
                "primary": bool(info.get("primary", False)),
            }
            # Basic stats
            bs = basic.get(f, {})
            row["mean_value"] = bs.get("mean_value", np.nan)
            row["std_value"] = bs.get("std_value", np.nan)
            row["coverage_pct"] = bs.get("coverage_pct", 0.0)

            # Rolling IC summary
            ic_col = f"ic_{f}"
            if ic_col in (rolling_ic.columns if rolling_ic is not None else []):
                ic_series = rolling_ic[ic_col].dropna()
                row["mean_ic"] = float(ic_series.mean()) if len(ic_series) > 0 else np.nan
                row["std_ic"] = float(ic_series.std()) if len(ic_series) > 0 else np.nan
                row["icir"] = float(row["mean_ic"] / (row["std_ic"] + 1e-12))
                row["ic_tstat"] = float(
                    row["mean_ic"] / (row["std_ic"] / max(np.sqrt(len(ic_series)), 1) + 1e-12)
                )
                row["n_obs_dates"] = len(ic_series)
            else:
                row["mean_ic"] = np.nan
                row["std_ic"] = np.nan
                row["icir"] = np.nan
                row["ic_tstat"] = np.nan
                row["n_obs_dates"] = 0

            # IC Decay
            if ic_decay is not None and not ic_decay.empty and f in ic_decay["factor"].values:
                decay_row = ic_decay[ic_decay["factor"] == f].iloc[0]
                for h in ["h1", "h5", "h10", "h20"]:
                    row[f"ic_{h}"] = decay_row.get(h, np.nan)
                h_vals = [decay_row.get(f"h{h}", np.nan) for h in [1, 5, 10, 20]]
                h_valid = [v for v in h_vals if not np.isnan(v)]
                if len(h_valid) >= 2:
                    log_h = np.log([1, 5, 10, 20])
                    valid_idx = [i for i, v in enumerate(h_vals) if not np.isnan(v)]
                    y = np.array([h_vals[i] for i in valid_idx])
                    x = log_h[valid_idx]
                    if len(x) >= 2:
                        row["ic_decay_slope"] = float(np.polyfit(x, y, 1)[0])
                    else:
                        row["ic_decay_slope"] = np.nan
                else:
                    row["ic_decay_slope"] = np.nan
            else:
                for h in ["h1", "h5", "h10", "h20"]:
                    row[f"ic_{h}"] = np.nan
                row["ic_decay_slope"] = np.nan

            # Quantile
            qdf = quantile_results.get(f)
            if qdf is not None and not qdf.empty:
                q_means = qdf.mean()
                q_cols = sorted(q_means.index.tolist())
                if q_cols:
                    row["q1_return"] = float(q_means.get(q_cols[0], np.nan))
                    row["q5_return"] = float(q_means.get(q_cols[-1], np.nan))
                    row["q5_minus_q1"] = float(row["q5_return"] - row["q1_return"])
            else:
                row["q1_return"] = np.nan
                row["q5_return"] = np.nan
                row["q5_minus_q1"] = np.nan
            row["monotonicity"] = monotonicity.get(f, np.nan)

            # Regime IC
            if regime_ic is not None and not regime_ic.empty and f in regime_ic["factor"].values:
                rr = regime_ic[regime_ic["factor"] == f].iloc[0]
                regime_vals = []
                for r in ["bull", "bear", "sideways", "panic", "bull_volatile"]:
                    v = rr.get(r, np.nan)
                    row[f"{r}_ic"] = v
                    if not np.isnan(v):
                        regime_vals.append(v)
                if regime_vals:
                    row["regime_consistency"] = float(
                        1.0 - np.std(regime_vals) / (np.mean(regime_vals) + 1e-12)
                    )
                else:
                    row["regime_consistency"] = np.nan
            else:
                for r in ["bull", "bear", "sideways", "panic", "bull_volatile"]:
                    row[f"{r}_ic"] = np.nan
                row["regime_consistency"] = np.nan

            # Turnover / Stability
            row["turnover"] = turnover.get(f, np.nan)
            row["autocorr_20"] = stability.get(f, np.nan)

            # Correlation clusters
            row["corr_cluster"] = clusters.get(f, 0)
            row["max_pairwise_corr"] = np.nan
            row["max_corr_with"] = ""

            # Average ETFs per date
            row["n_obs_pool"] = 0

            rows.append(row)

        report = pd.DataFrame(rows)

        # Compute max_pairwise_corr from corr_matrix
        if corr_matrix is not None and not corr_matrix.empty:
            for f in factors:
                if f in corr_matrix.index:
                    abs_corr = corr_matrix.loc[f].drop(f, errors="ignore").abs()
                    valid = abs_corr.dropna()
                    if len(valid) > 0:
                        max_idx = valid.idxmax()
                        max_val = valid.max()
                        mask = report["factor"] == f
                        report.loc[mask, "max_pairwise_corr"] = max_val
                        report.loc[mask, "max_corr_with"] = max_idx

        # n_obs_pool: average number of unique codes per date
        avg_n = self.data.groupby("date")["code"].nunique().mean()
        report["n_obs_pool"] = int(avg_n) if not np.isnan(avg_n) else 0

        return report
