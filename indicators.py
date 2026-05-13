import numpy as np
import pandas as pd


# ==========================================
# 指标引擎（时间序列因子）
# ==========================================

class IndicatorEngine:

    def __init__(self):
        # 不再需要SQLite
        pass

    # ==========================================
    # 计算全部指标
    # ==========================================

    def calculate(self, df):
        """计算全部时间序列指标。对单ETF DataFrame调用。"""
        df = df.copy()
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        # ═══════ 缓存常用滚动窗口 ═══════
        _r5 = close.rolling(5)
        _r9 = close.rolling(9)
        _r20 = close.rolling(20)
        _r60 = close.rolling(60)

        # =====================================
        # 趋势
        # =====================================

        df["ma5"] = _r5.mean()
        df["ma10"] = close.rolling(10).mean()
        df["ma20"] = _r20.mean()
        df["ma60"] = _r60.mean()

        df["ema12"] = close.ewm(span=12).mean()
        df["ema26"] = close.ewm(span=26).mean()

        # =====================================
        # MACD
        # =====================================
        df["macd_dif"] = df["ema12"] - df["ema26"]
        df["macd_dea"] = df["macd_dif"].ewm(span=9).mean()
        df["macd_hist"] = (df["macd_dif"] - df["macd_dea"]) * 2

        # =====================================
        # KDJ
        # =====================================
        low_n = low.rolling(9).min()
        high_n = high.rolling(9).max()
        rsv = (close - low_n) / (high_n - low_n) * 100
        df["kdj_k"] = rsv.ewm(com=2).mean()
        df["kdj_d"] = df["kdj_k"].ewm(com=2).mean()
        df["kdj_j"] = 3 * df["kdj_k"] - 2 * df["kdj_d"]

        # =====================================
        # BOLL（复用 _r20 mean/std）
        # =====================================
        mid = _r20.mean()
        std = _r20.std()
        df["boll_mid"] = mid
        df["boll_up"] = mid + 2 * std
        df["boll_low"] = mid - 2 * std

        # =====================================
        # RSI
        # =====================================
        diff = close.diff()
        up = diff.clip(lower=0)
        down = -diff.clip(upper=0)
        for n in [6, 12, 24]:
            avg_up = up.rolling(n).mean()
            avg_down = down.rolling(n).mean()
            rs = avg_up / avg_down
            df[f"rsi{n}"] = 100 - 100 / (1 + rs)

        # =====================================
        # 动量
        # =====================================
        df["roc12"] = (close / close.shift(12) - 1) * 100
        df["ret5"] = (close / close.shift(5) - 1) * 100
        df["ret20"] = (close / close.shift(20) - 1) * 100
        df["ma_ratio_5_20"] = df["ma5"] / df["ma20"] - 1

        # =====================================
        # 扩展动量因子（复用 ret20 已算的 20日收益）
        # =====================================
        df["roc20"] = df["ret20"]
        df["roc60"] = (close / close.shift(60) - 1) * 100
        df["roc120"] = (close / close.shift(120) - 1) * 100

        # =====================================
        # 扩展趋势因子
        # =====================================
        df["ma_distance"] = (close / df["ma20"] - 1) * 100
        high20 = high.rolling(20).max()
        df["breakout20"] = (close / high20 - 1) * 100

        # =====================================
        # 波动率（复用 _r20 std）
        # =====================================
        ret = close.pct_change()
        df["volatility20"] = _r20.std() * np.sqrt(252)
        df["vol_ratio20"] = volume / (volume.rolling(20).mean() + 1e-12)

        # 52周低点（趋势位置参考）
        df["low52w"] = close / (close.rolling(252).min() + 1e-12) - 1

        # =====================================
        # 扩展波动率因子
        # =====================================
        neg_ret = ret.clip(upper=0)
        df["downside_volatility"] = neg_ret.rolling(20).std() * np.sqrt(252)

        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df["atr14"] = tr.rolling(14).mean()
        df["atr20"] = tr.rolling(20).mean()

        # =====================================
        # 扩展资金流因子
        # =====================================
        vol_ma5 = volume.rolling(5).mean()
        vol_ma20 = volume.rolling(20).mean()
        df["turnover_change"] = (vol_ma5 / (vol_ma20 + 1e-12) - 1) * 100

        # =====================================
        # 相对强弱因子（占位，横截面阶段计算）
        # =====================================
        df["relative_strength_vs_hs300"] = np.nan

        # =====================================
        # 回撤因子
        # =====================================
        roll_max = close.rolling(20).max()
        drawdown = close / roll_max - 1
        df["max_drawdown20"] = drawdown.rolling(20).min()

        # =====================================
        # OBV
        # =====================================
        direction = np.where(close > close.shift(1), 1,
                     np.where(close < close.shift(1), -1, 0))
        df["obv"] = (direction * volume).cumsum()

        # =====================================
        # CCI
        # =====================================
        tp = (high + low + close) / 3
        ma = tp.rolling(14).mean()
        md = (tp - ma).abs().rolling(14).mean()
        df["cci14"] = (tp - ma) / (0.015 * md)

        # =====================================
        # 扩展因子（正交 / 结构性）
        # =====================================
        df["ret"] = close.pct_change()

        vol120 = ret.rolling(120).std() * np.sqrt(252)
        df["volatility_ratio"] = df["volatility20"] / (vol120 + 1e-12)

        cons = (
            (df["roc20"].fillna(0) > 0).astype(int)
            + (df["roc60"].fillna(0) > 0).astype(int)
            + (df["roc120"].fillna(0) > 0).astype(int)
        )
        df["trend_consistency"] = cons

        return df

    # ==========================================
    # get_conn / save_factors（保留兼容桩，不再使用SQLite）
    # ==========================================

    def get_conn(self, asset_type):
        return None

    def save_factors(self, df, asset_type):
        """Research parquet接管后不再写入SQLite"""
        pass

    # ==========================================
    # 加载因子
    # ==========================================

    def load_factors(self, code, asset_type):

        conn = self.get_conn(asset_type)

        sql = """
        SELECT *
        FROM factor_daily
        WHERE code=?
        """

        return pd.read_sql(
            sql,
            conn,
            params=(code,)
        )
