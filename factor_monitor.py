"""
Factor Monitor — Dynamic IC-based Factor Weight System.

Computes rolling RankIC and ICIR for each factor, then produces per-date
dynamic weights that replace the static regime weight matrix.

Algorithm:
  1. Per-factor, per-date cross-sectional RankIC vs 5d forward return
     (lag-adjusted: IC at t uses factor[t-5] vs ret[t-5 → t])
  2. Rolling 60-day ICIR = mean(IC) / std(IC)
  3. Dynamic weight = clip(ICIR, 0, 2), then L1-normalize across factors
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


# Factor groups matching the strategy builder's FACTOR_GROUPS
FACTOR_GROUPS = {
    "trend":    ["roc20", "roc60", "breakout20", "relative_strength_vs_hs300"],
    "risk":     ["volatility20", "downside_volatility", "max_drawdown20"],
    "volume":   ["turnover_change", "vol_ratio20"],
    "structure": ["barra_beta", "corr_hs300_60"],
    "mean_reversion": ["ma_distance"],
}

# Representative factor per group (used for IC computation)
GROUP_REPRESENTATIVES = {
    "trend":    "roc20",
    "risk":     "volatility20",
    "volume":   "vol_ratio20",
    "structure": "barra_beta",
    "mean_reversion": "ma_distance",
}

# Column naming: prefix for dynamic weight columns
DW_PREFIX = "dw_"


class FactorMonitor:
    """Computes rolling RankIC, ICIR, and dynamic factor weights."""

    def __init__(self, data: pd.DataFrame):
        """
        data: long-form DataFrame with columns [date, code, ...factor columns...]
              Must contain 'close' and all factor columns listed in FACTOR_GROUPS.
        """
        self.data = data.sort_values(["code", "date"]).copy()

    # ── forward return ──────────────────────────────────────────
    def _add_forward_return(self, df: pd.DataFrame, fwd_days: int = 5) -> pd.DataFrame:
        """Add fwd_ret_{fwd_days}d column per code (no look-ahead in IC calc)."""
        col = f"fwd_ret_{fwd_days}d"
        df = df.copy()
        df[col] = df.groupby("code")["close"].transform(
            lambda x: x.shift(-fwd_days) / x - 1.0
        )
        return df

    # ── lag-adjusted RankIC ─────────────────────────────────────
    def _rankic_lag_adjusted(
        self, df: pd.DataFrame, factor_col: str, fwd_days: int = 5
    ) -> pd.Series:
        """
        RankIC at date t: spearmanr( factor[t - fwd_days], ret[t-fwd_days → t] ).
        Both sides are known at time t — no look-ahead.
        """
        fwd_col = f"fwd_ret_{fwd_days}d"
        results = {}
        dates = sorted(df["date"].unique())

        # Build lookup: for each date, factor values indexed by code
        factor_by_date = {}
        fwd_by_date = {}
        for d in dates:
            sub = df[df["date"] == d]
            factor_by_date[d] = sub.set_index("code")[factor_col].dropna()
            if fwd_col in sub.columns:
                fwd_by_date[d] = sub.set_index("code")[fwd_col].dropna()

        # At each date t, correlate factor[t - fwd_days] with ret[t-fwd_days → t]
        min_idx = fwd_days  # need at least fwd_days history
        for i in range(min_idx, len(dates)):
            t = dates[i]
            t_lag = dates[i - fwd_days]

            f_lag = factor_by_date.get(t_lag)
            r_now = fwd_by_date.get(t)  # this is return from t_lag → t

            if f_lag is None or r_now is None:
                results[t] = np.nan
                continue

            common = f_lag.index.intersection(r_now.index)
            if len(common) < 5:
                results[t] = np.nan
                continue

            try:
                ic, _ = spearmanr(f_lag.loc[common], r_now.loc[common])
                # Use absolute IC — direction is handled by expression sign (+/-)
                results[t] = abs(ic) if not np.isnan(ic) else np.nan
            except Exception:
                results[t] = np.nan

        return pd.Series(results, name=f"ic_{factor_col}")

    # ── compute IC for all group representatives ─────────────────
    def compute_all_ic(self, fwd_days: int = 5) -> pd.DataFrame:
        """
        Compute lag-adjusted RankIC for each factor group's representative.
        Returns wide DataFrame: date | ic_trend | ic_risk | ic_volume | ic_structure | ic_meanrev
        """
        df = self._add_forward_return(self.data, fwd_days)
        ic_series = {}
        for group, rep in GROUP_REPRESENTATIVES.items():
            if rep in df.columns:
                s = self._rankic_lag_adjusted(df, rep, fwd_days)
                ic_series[f"ic_{group}"] = s
        ic_df = pd.DataFrame(ic_series).reset_index().rename(columns={"index": "date"})
        return ic_df.dropna(subset=[c for c in ic_df.columns if c != "date"], how="all")

    # ── rolling ICIR ────────────────────────────────────────────
    def compute_icir(self, ic_df: pd.DataFrame, window: int = 60) -> pd.DataFrame:
        """
        Rolling ICIR = rolling_mean(IC) / rolling_std(IC).
        Minimum 20 observations before computing.
        """
        ic_cols = [c for c in ic_df.columns if c.startswith("ic_")]
        result = ic_df[["date"]].copy()
        for col in ic_cols:
            roll_mean = ic_df[col].rolling(window, min_periods=20).mean()
            roll_std = ic_df[col].rolling(window, min_periods=20).std()
            ir = roll_mean / roll_std.replace(0, np.nan)
            result[col.replace("ic_", "ir_")] = ir
        return result

    # ── dynamic weights ─────────────────────────────────────────
    def compute_dynamic_weights(
        self, ir_df: pd.DataFrame, clip_range: tuple = (0.0, 2.0)
    ) -> pd.DataFrame:
        """
        Clip ICIR to [clip_low, clip_high], then L1-normalize across groups.
        Returns: date | dw_trend | dw_risk | dw_volume | dw_structure | dw_meanrev
        """
        ir_cols = [c for c in ir_df.columns if c.startswith("ir_")]
        result = ir_df[["date"]].copy()

        # Extract IR values
        ir_data = {}
        for col in ir_cols:
            group = col.replace("ir_", "")
            ir_data[group] = ir_df[col].values

        # Clip and normalize per row
        n = len(ir_df)
        dw = {}
        for group in ir_data:
            dw[group] = np.full(n, np.nan)

        clip_low, clip_high = clip_range
        for i in range(n):
            row_ir = {}
            for group in ir_data:
                val = ir_data[group][i]
                if not np.isnan(val):
                    row_ir[group] = np.clip(val, clip_low, clip_high)
            total = sum(row_ir.values())
            if total > 0:
                for group, v in row_ir.items():
                    dw[group][i] = v / total

        for group in dw:
            result[f"{DW_PREFIX}{group}"] = dw[group]

        return result

    # ── convenience: full pipeline ───────────────────────────────
    def compute_weights(self, fwd_days: int = 5, ic_window: int = 60) -> pd.DataFrame:
        """Run the full pipeline: IC → ICIR → dynamic weights."""
        ic_df = self.compute_all_ic(fwd_days=fwd_days)
        if ic_df.empty or len(ic_df.columns) < 2:
            return pd.DataFrame(columns=["date"])
        ir_df = self.compute_icir(ic_df, window=ic_window)
        return self.compute_dynamic_weights(ir_df)
