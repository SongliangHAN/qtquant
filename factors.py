import numpy as np
import pandas as pd

from scipy.stats import spearmanr


class FactorEngine:
    """
    因子引擎（横截面因子计算）。
    所有因子结果直接返回DataFrame，不再写入SQLite。
    ResearchDataService负责合并到research parquet中。
    """

    def __init__(self):
        # 不再需要SQLite连接
        pass

    # ==========================================
    # 横截面动量（向量化，比逐日循环快20~100x）
    # ==========================================

    def cross_section_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        对roc12做每日横截面0~1百分位排名。
        返回长格式：date, code, factor_name, factor_value
        """
        if "roc12" not in df.columns:
            return pd.DataFrame()

        # 向量化：groupby("date").rank(pct=True)，一次性完成
        df = df.copy()
        df["_v"] = df.groupby("date")["roc12"].rank(pct=True)

        result = df[df["_v"].notna()].copy()
        result["factor_name"] = "cross_momentum"
        result = result.rename(columns={"_v": "factor_value"})
        return result[["date", "code", "factor_name", "factor_value"]].reset_index(drop=True)

    # ==========================================
    # 横截面波动率（向量化）
    # ==========================================

    def cross_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        if "volatility20" not in df.columns:
            return pd.DataFrame()

        df = df.copy()
        df["_v"] = df.groupby("date")["volatility20"].rank(pct=True)

        result = df[df["_v"].notna()].copy()
        result["factor_name"] = "cross_volatility"
        result = result.rename(columns={"_v": "factor_value"})
        return result[["date", "code", "factor_name", "factor_value"]].reset_index(drop=True)

    # ==========================================
    # ETF-HS300相关性（风险暴露维度）
    # ==========================================

    def corr_hs300(self, df, hs300_ret: pd.Series | None = None) -> pd.DataFrame:
        """计算每只ETF与沪深300的60日滚动相关系数"""
        result = []

        if hs300_ret is None or hs300_ret.empty:
            bench = df[df["code"] == "510300"]
            if bench.empty:
                return pd.DataFrame()
            bench = bench.drop_duplicates("date").sort_values("date")
            hs300_ret = bench.set_index("date")["ret"]

        for code in df["code"].unique():
            sub = df[df["code"] == code].drop_duplicates("date").set_index("date").sort_index()
            ret_etf = sub["ret"]
            if ret_etf.isna().sum() > len(ret_etf) * 0.5:
                continue

            common = ret_etf.index.intersection(hs300_ret.index)
            aligned = pd.DataFrame({
                "etf": ret_etf.reindex(common),
                "hs300": hs300_ret.reindex(common),
            }).dropna()

            if len(aligned) < 60:
                continue

            corr_series = aligned["etf"].rolling(60).corr(aligned["hs300"])
            for d, v in corr_series.dropna().items():
                result.append({
                    "date": str(d), "code": code,
                    "factor_name": "corr_hs300_60", "factor_value": float(v),
                })

        return pd.DataFrame(result)

    # ==========================================
    # 滚动Beta（协方差公式，去掉sklearn，快10x+）
    # ==========================================

    def barra_beta(self, df, hs300_df: pd.DataFrame | None = None) -> pd.DataFrame:
        """
        Beta = Cov(ret_etf, ret_hs300) / Var(ret_hs300)
        直接用numpy协方差，不再逐日实例化LinearRegression。
        """
        result = []

        df = df.drop_duplicates(subset=["date", "code"], keep="last")

        # 市场收益
        if hs300_df is not None and not hs300_df.empty:
            market = hs300_df.set_index("date")["ret"]
        else:
            bench = df[df["code"] == "510300"]
            if not bench.empty:
                bench = bench.drop_duplicates("date")
                market = bench.set_index("date")["ret"]
            else:
                pivot = df.pivot(index="date", columns="code", values="ret")
                market = pivot.mean(axis=1)

        pivot = df.pivot(index="date", columns="code", values="ret")

        for code in pivot.columns:
            try:
                y = pivot[code].values
                dates_arr = pivot.index.values
                mkt_vals = market.reindex(pivot.index).values

                for i in range(60, len(y)):
                    window_y = y[i-60:i]
                    window_x = mkt_vals[i-60:i]
                    mask = ~np.isnan(window_y) & ~np.isnan(window_x)
                    if mask.sum() < 30:
                        continue
                    wy = window_y[mask]
                    wx = window_x[mask]

                    # Beta = Cov(x,y) / Var(x)
                    cov = np.cov(wx, wy, ddof=1)[0, 1]
                    var_x = np.var(wx, ddof=1)
                    if var_x < 1e-12:
                        continue
                    beta = cov / var_x

                    result.append({
                        "date": str(dates_arr[i]),
                        "code": code,
                        "factor_name": "barra_beta",
                        "factor_value": float(beta),
                    })
            except Exception:
                pass

        return pd.DataFrame(result)

    # ==========================================
    # IC
    # ==========================================

    def calc_ic(self, factor, future_ret):
        x = factor.dropna()
        y = future_ret.loc[x.index]
        if len(x) < 5:
            return np.nan
        return x.corr(y)

    # ==========================================
    # RankIC
    # ==========================================

    def calc_rank_ic(self, factor, future_ret):
        x = factor.dropna()
        y = future_ret.loc[x.index]
        if len(x) < 5:
            return np.nan
        return spearmanr(x, y)[0]

    # ==========================================
    # 保存（保留兼容，但已不再使用SQLite）
    # ==========================================

    def save(self, df):
        """Research parquet接管后，此方法变为no-op"""
        pass

    def load_all(self):
        """Research parquet接管后，不再从SQLite加载"""
        return pd.DataFrame()

    def load(self, code):
        """Research parquet接管后，不再从SQLite加载"""
        return pd.DataFrame()
