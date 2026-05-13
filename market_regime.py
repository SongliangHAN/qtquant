import pandas as pd
import numpy as np


class MarketRegimeDetector:

    """
    市场状态识别器（长周期 + 短周期双系统）

    长周期（基于 MA50/MA200）：
      - bull: 牛市（价格 > MA200, MA50 > MA200, 低波动）
      - bull_volatile: 牛市高波动
      - sideways: 震荡（均线缠绕）
      - bear: 熊市（价格 < MA200, MA50 < MA200, 低波动）
      - panic: 恐慌（价格 < MA200, MA50 < MA200, 高波动）
      - unknown: 数据不足

    短周期（基于 market_breadth + MA20）：
      - breadth: 上涨ETF占比（0~1），快速反映市场宽度变化
      - 比MA200更早捕捉市场转弱信号
    """

    def __init__(self,
                 vol_threshold: float = 0.30,
                 ma_short: int = 50,
                 ma_long: int = 200):
        self.vol_threshold = vol_threshold
        self.ma_short = ma_short
        self.ma_long = ma_long

    def detect(self, benchmark_df: pd.DataFrame) -> pd.DataFrame:
        """
        原始detect（不做滞后，语义不变）。
        输入：benchmark_df 需包含 date, close 列
        输出：DataFrame(date, regime)
        """
        df = benchmark_df.copy()

        close = df["close"]

        ma_short = close.rolling(self.ma_short).mean()
        ma_long = close.rolling(self.ma_long).mean()

        ret = close.pct_change()

        vol20 = ret.rolling(20).std() * np.sqrt(252)

        # 短周期信号：价格距离MA20
        ma20 = close.rolling(20).mean()
        df["ma20_distance"] = close / ma20 - 1  # 正数=MA20上方

        regime = []

        for i in range(len(df)):

            if pd.isna(ma_long.iloc[i]):
                regime.append("unknown")
                continue

            price = close.iloc[i]

            bull = (
                price > ma_long.iloc[i]
                and ma_short.iloc[i] > ma_long.iloc[i]
            )

            bear = (
                price < ma_long.iloc[i]
                and ma_short.iloc[i] < ma_long.iloc[i]
            )

            high_vol = (
                pd.notna(vol20.iloc[i])
                and vol20.iloc[i] > self.vol_threshold
            )

            if bear:
                if high_vol:
                    regime.append("panic")
                else:
                    regime.append("bear")

            elif bull:
                if high_vol:
                    regime.append("bull_volatile")
                else:
                    regime.append("bull")

            else:
                regime.append("sideways")

        df["regime"] = regime

        return df[["date", "regime", "ma20_distance"]]

    def detect_smooth(self, benchmark_df: pd.DataFrame,
                       persistence: int = 5) -> pd.DataFrame:
        """
        带滞后平滑的市场状态检测（防止频繁抖动切换）。

        persistence: 连续N天满足新状态条件后才正式切换。
                     只在原始状态持续≥persistence天时才发出切换信号。

        输出的 regime 列不会有"日频抖动"，回测时更接近真实决策逻辑。
        """
        df = self.detect(benchmark_df)  # 先算原始regime
        regimes = df["regime"].values
        smoothed = regimes.copy()
        n = len(regimes)

        current = regimes[0]
        current_start = 0
        pending = None
        pending_start = 0

        for i in range(1, n):
            r = regimes[i]
            if r == current:
                # 持续当前状态
                pending = None
            else:
                # 出现新候选状态
                if pending is None:
                    pending = r
                    pending_start = i
                elif r == pending:
                    # 候选状态持续中
                    if i - pending_start + 1 >= persistence:
                        # 确认切换：只回填 pending 期间的观测值（避免引入未来信息）
                        for j in range(pending_start, i + 1):
                            smoothed[j] = pending
                        current = pending
                        current_start = pending_start
                        pending = None
                else:
                    # 又变了，重置候选
                    pending = r
                    pending_start = i
            # 填入当前状态
            smoothed[i] = current

        df["regime_raw"] = df["regime"]  # 保留原始值
        df["regime"] = smoothed
        return df[["date", "regime", "regime_raw", "ma20_distance"]]

    def calc_market_breadth(self, all_etf_df: pd.DataFrame, core_codes: list[str] | None = None) -> pd.DataFrame:
        """
        计算市场宽度：每日上涨ETF占比

        输入：all_etf_df 需包含 date, code, ret (日收益率)
        core_codes: 可选，固定核心ETF池（如宽基+一级行业，~30-40只）。
                    传入后只使用这些code计算breadth，避免ETF池变化导致历史不可比。
        输出：DataFrame(date, breadth, n_etfs)
        """
        df = all_etf_df.copy()

        # 固定核心池：breadth可跨时间比较
        if core_codes:
            df = df[df["code"].astype(str).isin(core_codes)]

        # 标记上涨ETF
        df["is_up"] = (df["ret"] > 0).astype(int)

        # 每日统计
        breadth = df.groupby("date").agg(
            breadth=("is_up", "mean"),
            n_etfs=("code", "nunique")
        ).reset_index()

        # 平滑：5日均值
        breadth["breadth_smooth"] = (
            breadth["breadth"]
            .rolling(5, min_periods=1)
            .mean()
        )

        return breadth[["date", "breadth", "breadth_smooth", "n_etfs"]]

    def calc_ma20_breadth(self, all_etf_df: pd.DataFrame, core_codes: list[str] | None = None) -> pd.DataFrame:
        """计算MA20宽度：close > ma20 的ETF占比"""
        df = all_etf_df.copy()
        if core_codes:
            df = df[df["code"].astype(str).isin(core_codes)]
        if "ma20" not in df.columns:
            raise KeyError("all_etf_df 缺少 ma20 列，请先计算 IndicatorEngine")
        df["above_ma20"] = (df["close"] > df["ma20"]).astype(int)
        b = df.groupby("date").agg(breadth_ma20=("above_ma20", "mean")).reset_index()
        b["breadth_ma20_smooth"] = b["breadth_ma20"].rolling(5, min_periods=1).mean()
        return b[["date", "breadth_ma20", "breadth_ma20_smooth"]]

    def calc_roc20_breadth(self, all_etf_df: pd.DataFrame, core_codes: list[str] | None = None) -> pd.DataFrame:
        """计算ROC20宽度：roc20 > 0 的ETF占比"""
        df = all_etf_df.copy()
        if core_codes:
            df = df[df["code"].astype(str).isin(core_codes)]
        if "roc20" not in df.columns:
            raise KeyError("all_etf_df 缺少 roc20 列")
        df["roc20_up"] = (df["roc20"] > 0).astype(int)
        b = df.groupby("date").agg(breadth_roc20=("roc20_up", "mean")).reset_index()
        b["breadth_roc20_smooth"] = b["breadth_roc20"].rolling(5, min_periods=1).mean()
        return b[["date", "breadth_roc20", "breadth_roc20_smooth"]]

    def calc_newhigh_breadth(self, all_etf_df: pd.DataFrame, core_codes: list[str] | None = None) -> pd.DataFrame:
        """计算新高宽度：breakout20 > 0（接近20日高点）的ETF占比"""
        df = all_etf_df.copy()
        if core_codes:
            df = df[df["code"].astype(str).isin(core_codes)]
        if "breakout20" not in df.columns:
            raise KeyError("all_etf_df 缺少 breakout20 列")
        df["is_newhigh"] = (df["breakout20"] > 0).astype(int)
        b = df.groupby("date").agg(breadth_newhigh=("is_newhigh", "mean")).reset_index()
        b["breadth_newhigh_smooth"] = b["breadth_newhigh"].rolling(5, min_periods=1).mean()
        return b[["date", "breadth_newhigh", "breadth_newhigh_smooth"]]

    def calc_dispersion(self, all_etf_df: pd.DataFrame, core_codes: list[str] | None = None) -> pd.DataFrame:
        """计算横截面收益离散度：ret20 的横截面标准差"""
        df = all_etf_df.copy()
        if core_codes:
            df = df[df["code"].astype(str).isin(core_codes)]
        if "ret20" not in df.columns:
            raise KeyError("all_etf_df 缺少 ret20 列")
        d = df.groupby("date")["ret20"].std().reset_index()
        d.columns = ["date", "dispersion"]
        d["dispersion_smooth"] = d["dispersion"].rolling(5, min_periods=1).mean()
        return d[["date", "dispersion", "dispersion_smooth"]]

    def calc_market_score(self, all_etf_df: pd.DataFrame, core_codes: list[str] | None = None) -> pd.DataFrame:
        """
        计算综合市场评分 market_score (0~1)。

        固定权重（不可优化，市场结构认知）：
          market_score = 0.35 * breadth_ma20 + 0.35 * breadth_roc20
                       + 0.15 * breadth_up + 0.15 * breadth_newhigh
                       - 0.10 * dispersion_norm

        其中 dispersion 做 expanding min-max 归一化到 0~1（无未来数据泄露）。
        """
        b_up = self.calc_market_breadth(all_etf_df, core_codes)
        b_ma20 = self.calc_ma20_breadth(all_etf_df, core_codes)
        b_roc20 = self.calc_roc20_breadth(all_etf_df, core_codes)
        b_newhigh = self.calc_newhigh_breadth(all_etf_df, core_codes)
        b_disp = self.calc_dispersion(all_etf_df, core_codes)

        m = b_up[["date"]].copy()
        m["breadth_up"] = b_up["breadth_smooth"]
        m["breadth_ma20"] = b_ma20["breadth_ma20_smooth"]
        m["breadth_roc20"] = b_roc20["breadth_roc20_smooth"]
        m["breadth_newhigh"] = b_newhigh["breadth_newhigh_smooth"]
        m["breadth_dispersion"] = b_disp["dispersion_smooth"]

        # dispersion 归一化到 0~1（expanding window：每期只用当期及之前的数据，无未来泄露）
        m["d_expand_min"] = m["breadth_dispersion"].expanding().min()
        m["d_expand_max"] = m["breadth_dispersion"].expanding().max()
        gap = m["d_expand_max"] - m["d_expand_min"]
        mask = gap > 0
        m["dispersion_norm"] = 0.5  # 默认值：数据不足时用 0.5
        m.loc[mask, "dispersion_norm"] = (
            (m.loc[mask, "breadth_dispersion"] - m.loc[mask, "d_expand_min"])
            / gap[mask]
        )

        m["market_score"] = (
            0.35 * m["breadth_ma20"]
            + 0.35 * m["breadth_roc20"]
            + 0.15 * m["breadth_up"]
            + 0.15 * m["breadth_newhigh"]
            - 0.10 * m["dispersion_norm"]
        )

        return m[["date", "market_score", "breadth_up", "breadth_ma20",
                   "breadth_roc20", "breadth_newhigh", "breadth_dispersion"]]
