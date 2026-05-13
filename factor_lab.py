"""
Factor Lab — 因子研究平台。

提供：
  1. Rolling RankIC — 因子滚动 IC 序列
  2. IC Decay — 不同未来周期的 IC 衰减
  3. Quantile Return — 因子分位数组合收益
  4. Factor Correlation — 因子截面相关性矩阵
  5. Regime IC — 不同市场状态下的因子有效性
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from factor_monitor import FACTOR_GROUPS, GROUP_REPRESENTATIVES


class FactorLab:
    """因子研究与分析引擎。"""

    def __init__(self, data: pd.DataFrame):
        """
        data: long-form DataFrame with [date, code, close] + factor columns.
        """
        self.data = data.sort_values(["code", "date"]).copy()
        self._ensure_returns()

    def _ensure_returns(self):
        """确保有 ret 和 fwd_ret 列。"""
        if "ret" not in self.data.columns and "close" in self.data.columns:
            self.data["ret"] = self.data.groupby("code")["close"].transform(
                lambda x: x.pct_change()
            )

    def _add_forward_ret(self, df: pd.DataFrame, horizon: int) -> pd.DataFrame:
        col = f"fwd_ret_{horizon}d"
        if col not in df.columns:
            df = df.copy()
            df[col] = df.groupby("code")["close"].transform(
                lambda x: x.shift(-horizon) / x - 1.0
            )
        return df

    # ── Rolling RankIC ────────────────────────────────────────
    def compute_rolling_ic(
        self, factors: list[str] | None = None, horizon: int = 5, window: int = 60
    ) -> pd.DataFrame:
        """
        每个 factor 的滚动 RankIC。
        Returns: date | ic_{factor}  (每列一个因子)
        """
        if factors is None:
            factors = list(GROUP_REPRESENTATIVES.values())
        factors = [f for f in factors if f in self.data.columns]
        if not factors:
            return pd.DataFrame()

        df = self._add_forward_ret(self.data, horizon)
        fwd_col = f"fwd_ret_{horizon}d"

        all_ics = {}
        dates = sorted(df["date"].unique())

        for factor in factors:
            ic_series = {}
            for i in range(horizon, len(dates)):
                t = dates[i]
                t_lag = dates[i - horizon]
                sub_f = df[df["date"] == t_lag].set_index("code")[factor].dropna()
                sub_r = df[df["date"] == t].set_index("code")[fwd_col].dropna()
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

        result = pd.DataFrame({"date": dates})
        for k, v in all_ics.items():
            result = result.merge(
                pd.DataFrame({"date": v.index, k: v.values}), on="date", how="left"
            )
        return result

    # ── IC Decay ──────────────────────────────────────────────
    def compute_ic_decay(
        self, factors: list[str] | None = None, horizons: list[int] | None = None
    ) -> pd.DataFrame:
        """
        不同 forward horizon 下的平均 |IC|。
        Returns: factor | h1 | h5 | h10 | h20
        """
        if factors is None:
            factors = list(GROUP_REPRESENTATIVES.values())
        if horizons is None:
            horizons = [1, 5, 10, 20]
        factors = [f for f in factors if f in self.data.columns]
        rows = []
        for f in factors:
            row = {"factor": f}
            for h in horizons:
                df = self._add_forward_ret(self.data, h)
                fwd_col = f"fwd_ret_{h}d"
                ics = []
                dates = sorted(df["date"].unique())
                for i in range(h, len(dates)):
                    t = dates[i]
                    t_lag = dates[i - h]
                    sub_f = df[df["date"] == t_lag].set_index("code")[f].dropna()
                    sub_r = df[df["date"] == t].set_index("code")[fwd_col].dropna()
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
            rows.append(row)
        return pd.DataFrame(rows)

    # ── Quantile Return ──────────────────────────────────────
    def compute_quantile_returns(
        self, factor: str, horizon: int = 5, n_quantiles: int = 5
    ) -> pd.DataFrame:
        """
        按因子分位数分组，计算每组平均未来收益。
        Returns: date | q1 | q2 | ... | q{n}
        """
        df = self._add_forward_ret(self.data, horizon)
        fwd_col = f"fwd_ret_{horizon}d"
        dates = sorted(df["date"].unique())
        results = {}
        for d in dates:
            sub = df[df["date"] == d].dropna(subset=[factor, fwd_col])
            if len(sub) < n_quantiles * 2:
                continue
            sub["q"] = pd.qcut(sub[factor], n_quantiles, labels=False, duplicates="drop")
            means = sub.groupby("q")[fwd_col].mean()
            row = {}
            for q, v in means.items():
                row[f"q{int(q)+1}"] = v
            results[d] = row
        return pd.DataFrame(results).T

    # ── Factor Correlation ────────────────────────────────────
    def compute_factor_correlation(
        self, factors: list[str] | None = None
    ) -> pd.DataFrame:
        """
        每日横截面因子值之间的平均相关性矩阵。
        """
        if factors is None:
            factors = list(GROUP_REPRESENTATIVES.values())
        factors = [f for f in factors if f in self.data.columns]
        dates = sorted(self.data["date"].unique())
        corr_mats = []
        for d in dates:
            sub = self.data[self.data["date"] == d][factors].dropna()
            if len(sub) < 5:
                continue
            corr_mats.append(sub.corr().values)
        if not corr_mats:
            return pd.DataFrame(index=factors, columns=factors)
        avg_corr = np.mean(corr_mats, axis=0)
        return pd.DataFrame(avg_corr, index=factors, columns=factors)

    # ── Regime IC ─────────────────────────────────────────────
    def compute_regime_ic(
        self, factors: list[str] | None = None, horizon: int = 5
    ) -> pd.DataFrame:
        """
        在不同 regime 下的平均 RankIC。
        Returns: factor | bull | bear | sideways | panic | bull_volatile
        """
        if factors is None:
            factors = list(GROUP_REPRESENTATIVES.values())
        factors = [f for f in factors if f in self.data.columns]
        df = self._add_forward_ret(self.data, horizon)
        fwd_col = f"fwd_ret_{horizon}d"
        dates = sorted(df["date"].unique())

        regimes = ["bull", "bear", "sideways", "panic", "bull_volatile"]
        result = {f: {r: [] for r in regimes} for f in factors}

        for i in range(horizon, len(dates)):
            t = dates[i]
            t_lag = dates[i - horizon]
            sub_lag = df[df["date"] == t_lag]
            regime_val = sub_lag["regime"].iloc[0] if "regime" in sub_lag.columns and len(sub_lag) > 0 else None
            if regime_val is None or regime_val not in regimes:
                continue
            for f in factors:
                sub_f = sub_lag.set_index("code")[f].dropna()
                sub_r = df[df["date"] == t].set_index("code")[fwd_col].dropna()
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
            for r in regimes:
                vals = result[f][r]
                row[r] = np.mean(vals) if vals else np.nan
            rows.append(row)
        return pd.DataFrame(rows)
