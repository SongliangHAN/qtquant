import math
from dataclasses import dataclass
from typing import Callable

import pandas as pd
import numpy as np
from scipy.cluster import hierarchy as sch
from scipy.spatial.distance import squareform
from factor_expression import ExpressionEngine


@dataclass
class Position:
    code: str
    shares: float
    entry_date: str
    entry_idx: int
    entry_price: float


class BacktestEngine:

    """
    交易规则（核心约束）：
    - 信号仅使用 n-1 日收盘前可得数据（即按 decision_date 的横截面因子/指标排序）
    - 在 n 日开盘成交买入（exec_date=open）
    - 最早在 n+1 日开盘卖出（min_hold_days=1 强制约束）
    """

    def __init__(self):
        self.expr = ExpressionEngine()

    @staticmethod
    def _safe_scalar(s, default=np.nan):
        """
        从 Series/DataFrame/scalar 中安全提取标量值。
        避免 .loc[] 在重复 index 时返回 Series 导致的 ambiguity 错误。
        """
        if isinstance(s, pd.DataFrame):
            if s.shape[0] > 0:
                return float(s.iloc[0, 0])
            return default
        if isinstance(s, pd.Series):
            if len(s) > 0:
                return float(s.iloc[0])
            return default
        if isinstance(s, (np.bool_, bool)):
            return s
        try:
            return float(s)
        except (ValueError, TypeError):
            return default

    @staticmethod
    def _has_code_safe(day: pd.DataFrame, code: str) -> bool:
        """检查 code 是否在 day 中，容忍重复索引"""
        return code in day.index

    # =====================================
    # HRP 分层风险平价权重
    # =====================================

    def _hrp_weights(self, returns_df: pd.DataFrame) -> dict:
        """
        Hierarchical Risk Parity 权重分配。

        returns_df: T x N DataFrame，列为 code，值为日收益率
        返回: {code: weight}，权重和为 1
        """
        codes = list(returns_df.columns)
        n = len(codes)
        if n <= 1:
            return {c: 1.0 / n for c in codes} if n > 0 else {}
        if n == 2:
            # 两个标的时直接按逆波动率分配
            vols = returns_df.std()
            inv_vol = 1.0 / vols.replace(0, 0.2)
            w = inv_vol / inv_vol.sum()
            return {c: float(w[c]) for c in codes}

        corr = returns_df.corr().values
        # 距离矩阵
        dist = np.sqrt(0.5 * (1.0 - corr))
        # 确保对称且对角线为0
        dist = (dist + dist.T) / 2.0
        np.fill_diagonal(dist, 0.0)

        try:
            link = sch.linkage(squareform(dist), method="ward")
        except Exception:
            # 回退到逆波动率
            vols = returns_df.std().clip(lower=0.01)
            inv_vol = 1.0 / vols
            w = inv_vol / inv_vol.sum()
            return {c: float(w[c]) for c in codes}

        # 递归二分分配
        n_items = len(codes)
        cluster_items = [[i] for i in range(n_items)]

        def _get_cluster_var(indices):
            if len(indices) == 1:
                return np.var(returns_df.iloc[:, indices[0]])
            sub_cov = returns_df.iloc[:, indices].cov().values
            w_eq = np.ones(len(indices)) / len(indices)
            return float(w_eq @ sub_cov @ w_eq)

        def _bisect(cluster_idx):
            items = cluster_items[cluster_idx]
            if len(items) <= 1:
                return {items[0]: 1.0}
            # 在 linkage 中找到该簇的二分
            left, right = None, None
            for l_idx in range(len(link)):
                merged = int(link[l_idx, 0]), int(link[l_idx, 1])
                merged_set = set()
                for m in merged:
                    if m < n_items:
                        merged_set.update(cluster_items[m])
                    else:
                        merged_set.update(cluster_items[m])
                if merged_set == set(items):
                    left_raw = int(link[l_idx, 0])
                    right_raw = int(link[l_idx, 1])
                    left = cluster_items[left_raw] if left_raw < n_items else cluster_items[left_raw]
                    right = cluster_items[right_raw] if right_raw < n_items else cluster_items[right_raw]
                    break
            if left is None or right is None:
                # 均分
                return {i: 1.0 / len(items) for i in items}

            var_left = _get_cluster_var(left)
            var_right = _get_cluster_var(right)
            alloc_left = (1.0 / var_left) if var_left > 0 else 1.0
            alloc_right = (1.0 / var_right) if var_right > 0 else 1.0
            total_alloc = alloc_left + alloc_right
            w_left = alloc_left / total_alloc
            w_right = alloc_right / total_alloc

            result = {}
            left_weights = _bisect(left_raw) if left_raw >= n_items else {left[0]: 1.0}
            right_weights = _bisect(right_raw) if right_raw >= n_items else {right[0]: 1.0}
            for i, w in left_weights.items():
                result[i] = w * w_left
            for i, w in right_weights.items():
                result[i] = w * w_right

            # 重归一化
            total = sum(result.values())
            if total > 0:
                result = {k: v / total for k, v in result.items()}
            return result

        # 根簇索引为 len(link) + n_items - 1 (linkage 返回的最后一个合并)
        root_cluster_items = list(range(n_items))
        weights = _bisect(len(link) + n_items - 1)
        return {codes[i]: weights.get(i, 0.0) for i in range(n_items)}

    # =====================================
    # 计算策略在某个 decision_date 的选股结果
    # =====================================

    def select_on_date(self, strategy: dict, sub: pd.DataFrame) -> list[str]:
        """
        sub: 某一个 decision_date 的横截面数据（多只标的行）
        return: 按优先级排序的 code 列表
        """
        buy_rule = strategy.get("buy_rule", {}) or {}
        stype = strategy.get("type", "rule_based")

        # 兼容：rule_based 只有一个 factor
        if stype in ("rule_based", "search_generated"):
            # 如果提供了表达式，则走表达式路径
            if buy_rule.get("score_expr"):
                stype = "expr"
            else:
                factor = buy_rule.get("factor")
                if not factor or factor not in sub.columns:
                    return []
                ascending = bool(buy_rule.get("ascending", False))
                ranked = sub.sort_values(factor, ascending=ascending)
                return ranked["code"].astype(str).tolist()

        # 表达式选股（score_expr + 可选 filter_expr）
        if stype in ("expr", "expr_strategy"):
            score_expr = buy_rule.get("score_expr", "")
            filt_expr = buy_rule.get("filter_expr", "")
            ascending = bool(buy_rule.get("ascending", False))

            # 市场状态过滤（同一天所有ETF的regime一致，取首个即可）
            market_filter = buy_rule.get("market_filter") or {}
            if market_filter.get("enabled"):
                allow_regimes = market_filter.get("allow_regimes", [])
                if allow_regimes and "regime" in sub.columns and len(sub) > 0:
                    regime = sub["regime"].iloc[0]
                    if pd.notna(regime) and regime not in allow_regimes:
                        # 当前市场状态不允许买入 → 返回防御资产（按低波动+相对强弱排序）
                        pos_rule = strategy.get("position", {}) or {}
                        defensive = pos_rule.get("defensive_assets", [])
                        if defensive:
                            defensive_in_pool = [c for c in defensive if c in sub["code"].astype(str).tolist()]
                            if defensive_in_pool:
                                # 防御资产也需打分排序，避免每次固定顺序偏向首个资产
                                def_sub = sub[sub["code"].astype(str).isin(defensive_in_pool)].copy()
                                score_vol = -def_sub["volatility20"].rank(pct=True) if "volatility20" in def_sub.columns else pd.Series(0, index=def_sub.index)
                                score_rs = def_sub["relative_strength_vs_hs300"].rank(pct=True) if "relative_strength_vs_hs300" in def_sub.columns else pd.Series(0, index=def_sub.index)
                                def_sub["_def_score"] = 0.5 * score_vol + 0.5 * score_rs
                                ranked = def_sub.sort_values("_def_score", ascending=False)
                                return ranked["code"].astype(str).tolist()
                            return []
                        return []

            df = sub.copy()

            # ══════════════════════════════════════════
            # 流动性过滤：20日均成交额 > 3000万
            # ══════════════════════════════════════════
            liquidity_min = float(buy_rule.get("liquidity_min_amount", 0) or 0)
            if liquidity_min > 0 and "amount_ma20" in df.columns:
                df = df[df["amount_ma20"] >= liquidity_min].copy()
            if df.empty:
                return []

            try:
                score = self.expr.eval(score_expr, df)
            except Exception as e:
                print(f"[select_on_date] score_expr 评估失败: {score_expr!r}, 错误: {e}")
                return []
            if score is None:
                return []
            if isinstance(score, pd.DataFrame):
                # 兜底：取第一列
                score = score.iloc[:, 0] if score.shape[1] > 0 else pd.Series(dtype=float)
            if not isinstance(score, pd.Series):
                score = pd.Series([float(score)] * len(df), index=df.index)
            df["_score"] = score.astype(float)

            if filt_expr:
                try:
                    cond = self.expr.eval(filt_expr, df)
                except Exception as e:
                    print(f"[select_on_date] filt_expr 评估失败: {filt_expr!r}, 错误: {e}")
                    cond = None
                if isinstance(cond, pd.Series):
                    df = df[cond.fillna(False)]

            if df.empty:
                return []

            # 信号强度阈值：top1 vs top5 的差距（横截面gap，不受ETF数量变化影响）
            # gap = (best_score - score_5th) / std，默认阈值0.5表示最强信号显著拉开差距
            signal_threshold = buy_rule.get("signal_strength_threshold")
            if signal_threshold is not None and len(df) >= 5:
                scores = df["_score"].sort_values(ascending=False)
                top1 = scores.iloc[0]
                top5_idx = min(4, len(scores) - 1)
                top5 = scores.iloc[top5_idx]
                std_s = scores.std(ddof=0)
                if std_s > 1e-12:
                    gap = (top1 - top5) / std_s
                    if gap < float(signal_threshold):
                        return []  # 信号不够突出，空仓
                else:
                    return []  # 所有ETF得分相同
            elif signal_threshold is not None and len(df) < 5:
                pass  # 样本太少，不设阈值

            ranked = df.sort_values("_score", ascending=ascending)
            return ranked["code"].astype(str).tolist()

            factor = buy_rule.get("factor")
            if not factor or factor not in sub.columns:
                return []
            ascending = bool(buy_rule.get("ascending", False))
            ranked = sub.sort_values(factor, ascending=ascending)
            return ranked["code"].astype(str).tolist()

        # 多因子加权：score = Σ w * zscore(factor)
        if stype == "composite_factor":
            factors = buy_rule.get("factors", [])
            if not factors:
                return []

            score = pd.Series(0.0, index=sub.index)
            used = 0
            for item in factors:
                name = item.get("name")
                w = float(item.get("weight", 1.0))
                if not name or name not in sub.columns:
                    continue
                s = sub[name].astype(float)
                if s.isna().all():
                    continue
                # 横截面标准化，避免量纲差异
                z = (s - s.mean()) / (s.std(ddof=0) + 1e-12)
                score = score + w * z.fillna(0)
                used += 1

            if used == 0:
                return []

            ranked = sub.assign(_score=score).sort_values("_score", ascending=False)
            return ranked["code"].astype(str).tolist()

        # 强化学习/其他类型：暂不支持
        return []

    # =====================================
    # 回测
    # =====================================

    def run(
        self,
        strategy: dict,
        data: pd.DataFrame,
        *,
        benchmark: pd.DataFrame | None = None,
        base_point: float = 1000.0,
        commission_bps: float = 1.0,
        slippage_bps: float = 2.0,
        progress_cb: Callable | None = None,
    ):
        """
        data: long-form 数据，至少包含：
          date, code, open, close + 策略所需因子列
        benchmark: DataFrame(date, close) - 可选
        """
        if data is None or data.empty:
            return {
                "trades": pd.DataFrame(),
                "equity": pd.DataFrame(),
                "benchmark": pd.DataFrame(),
                "metrics": {},
            }

        data = data.copy()
        data["code"] = data["code"].astype(str)

        # 交易成本
        commission = commission_bps / 10000.0
        base_slippage = slippage_bps / 10000.0

        def _dynamic_slippage(code, day_df, notional=0.0):
            """
            动态滑点：基础 + 波动率加成 + sqrt 冲击模型。
            impact = impact_k * sqrt(notional / daily_amount)
            """
            slip = base_slippage
            try:
                row = day_df[day_df["code"] == code]
                if len(row) > 0:
                    vol20 = self._safe_scalar(row["volatility20"].iloc[0]) if "volatility20" in row.columns else np.nan
                    if pd.notna(vol20):
                        slip += 0.5 * abs(vol20)
                    # sqrt 冲击模型
                    if notional > 0:
                        daily_amount = self._safe_scalar(row["amount"].iloc[0]) if "amount" in row.columns else np.nan
                        if pd.notna(daily_amount) and daily_amount > 0:
                            participation = notional / daily_amount
                            impact = impact_k * np.sqrt(participation)
                            slip += impact
            except (KeyError, ValueError, TypeError, IndexError):
                pass
            return max(slip, base_slippage * 0.5)

        # 策略参数
        buy_rule = strategy.get("buy_rule", {}) or {}
        sell_rule = strategy.get("sell_rule", {}) or {}
        pos_rule = strategy.get("position", {}) or {}
        impact_k = float(pos_rule.get("impact_k", 0.001))

        max_positions = int(pos_rule.get("max_positions", buy_rule.get("top_n", 1)) or 1)
        min_hold_days = max(1, int(strategy.get("execution", {}).get("min_hold_days", 1) or 1))

        stop_loss_cfg = sell_rule.get("stop_loss") or sell_rule.get("risk_exit") or {}
        stop_loss_pct = float(stop_loss_cfg.get("pct", 0) or 0)      # 0.05 表示 -5%

        signal_exit_cfg = sell_rule.get("signal_exit") or {}
        signal_exit_expr = (signal_exit_cfg.get("expr") or "").strip()

        reb_cfg = sell_rule.get("rebalance") or strategy.get("rebalance") or {}
        rebalance_enabled = bool(reb_cfg.get("enabled", False))
        # 注意：调仓应由“信号”驱动，而不是固定周期。
        # 这里将 rebalance 解释为：每个交易日都基于上一交易日信号生成 target_set，
        # 若持仓不在 target_set 则在当日开盘卖出，并按 target_set 补齐买入。
        # 为兼容旧策略，frequency 字段保留但不再作为“是否调仓”的硬条件。
        _ = reb_cfg.get("frequency", 1)

        # 构造日期序列
        dates = sorted(data["date"].unique())
        date_index = {d: i for i, d in enumerate(dates)}

        # 预先生成：decision_date -> exec_date(open) 的选股结果
        desired: dict[str, list[str]] = {}
        for i in range(0, len(dates) - 1):
            decision_date = dates[i]
            exec_date = dates[i + 1]
            sub = data[data["date"] == decision_date]
            codes_sorted = self.select_on_date(strategy, sub)
            if not codes_sorted:
                continue
            desired[exec_date] = codes_sorted

        cash = float(base_point)
        positions: dict[str, Position] = {}
        trades: list[dict] = []
        equity_rows: list[dict] = []

        def is_rebalance_day(t_idx: int) -> bool:
            # t_idx=0 无法交易（没有上一日决策），从 t_idx=1 起允许调仓
            if not rebalance_enabled:
                return False
            if t_idx < 1:
                return False
            return True

        # 逐日回测（按开盘成交、按收盘计净值）
        for t in dates:
            t_idx = date_index[t]
            day = data[data["date"] == t].drop_duplicates("code").set_index("code")
            # 预建 dict 加速后续标量查找（dict.get() 比 .loc[] 快）
            day_open = day["open"].to_dict()
            day_close = day["close"].to_dict() if "close" in day.columns else {}
            day_corr = day["corr_hs300_60"].to_dict() if "corr_hs300_60" in day.columns else {}
            day_vol = day["volatility20"].to_dict() if "volatility20" in day.columns else {}
            prev = None
            prev_close_dict: dict[str, float] = {}
            if t_idx >= 1:
                prev_day = dates[t_idx - 1]
                prev = data[data["date"] == prev_day].drop_duplicates("code").set_index("code")
                prev_close_dict = prev["close"].to_dict() if "close" in prev.columns else {}

            halted_codes = set()  # 记录当日停牌/无价格的ETF

            # 计算当前 rebalance 目标持仓（使用上一交易日信号，在执行日开盘调仓）
            target_set = None
            if rebalance_enabled and is_rebalance_day(t_idx) and (t in desired):
                target_list = desired[t][:max_positions]
                target_set = set(target_list)

            # ==============
            # 卖出：多规则叠加（严格使用 T-1 数据做卖出决策，在 T 开盘成交）
            # ==============
            to_close: list[str] = []
            close_reason: dict[str, str] = {}
            for code, p in positions.items():
                held = t_idx - p.entry_idx
                if held < min_hold_days:
                    continue

                # 1) 轮动调仓：不在目标集合则卖出
                if target_set is not None and code not in target_set:
                    to_close.append(code)
                    close_reason[code] = "rebalance"
                    continue

                # 2) 止损（用上一日收盘价判断，避免用到当日信息）
                if prev is not None and code in prev.index and stop_loss_pct:
                    prev_close = self._safe_scalar(prev_close_dict.get(code, np.nan))
                    if not pd.isna(prev_close) and p.entry_price:
                        pnl_pct = float(prev_close) / float(p.entry_price) - 1.0
                        if stop_loss_pct and pnl_pct <= -abs(stop_loss_pct):
                            to_close.append(code)
                            close_reason[code] = "risk_exit"
                            continue

                # 3) 信号卖出（表达式用上一日数据判断，避免泄露）
                if signal_exit_expr and prev is not None and code in prev.index:
                    one = prev.loc[[code]].copy()
                    try:
                        cond = self.expr.eval(signal_exit_expr, one)
                    except Exception as e:
                        print(f"[run] signal_exit_expr 评估失败: {signal_exit_expr!r}, 错误: {e}")
                        cond = None
                    if isinstance(cond, pd.Series) and len(cond) > 0:
                        v = bool(cond.iloc[0]) if pd.notna(cond.iloc[0]) else False
                    elif isinstance(cond, pd.DataFrame) and len(cond) > 0:
                        # DataFrame: 取第一行第一列
                        try:
                            v = bool(cond.iloc[0, 0])
                        except Exception:
                            v = False
                    elif isinstance(cond, (bool, np.bool_)):
                        v = bool(cond)
                    elif isinstance(cond, (int, float)):
                        v = bool(cond)
                    else:
                        v = False
                    if v:
                        to_close.append(code)
                        close_reason[code] = "signal_exit"
                        continue

            for code in to_close:
                p = positions.pop(code)
                open_price = self._safe_scalar(day_open.get(code, np.nan))
                if code not in day.index or pd.isna(open_price):
                    # 停牌/无价格：强制继续持有（回补）
                    halted_codes.add(code)
                    positions[code] = p
                    continue
                raw_price = float(open_price)
                est_notional = raw_price * p.shares
                dyn_slip = _dynamic_slippage(code, day, est_notional)
                sell_price = raw_price * (1 - dyn_slip)
                notional = sell_price * p.shares
                fee = notional * commission
                cash += notional - fee

                trades.append(
                    {
                        "code": code,
                        "buy_date": p.entry_date,
                        "buy_price": p.entry_price,
                        "sell_date": t,
                        "sell_price": sell_price,
                        "shares": p.shares,
                        "pnl": (sell_price - p.entry_price) * p.shares - fee,
                        "pnl_pct": (sell_price / p.entry_price - 1) if p.entry_price else np.nan,
                        "holding_days": t_idx - p.entry_idx,
                        "exit_reason": close_reason.get(code, ""),
                    }
                )

            # ==============
            # 买入：执行日开盘，使用上一交易日的信号
            # ==============
            trade_today = (t in desired) and (not rebalance_enabled or is_rebalance_day(t_idx))

            if trade_today:
                if target_set is not None:
                    want = [c for c in desired[t] if c in target_set]
                else:
                    want = desired[t][: max_positions * 5]  # 冗余：避免缺价/停牌导致买不满

                # 剔除已持仓（由rebalance卖出规则处理换仓）
                want = [c for c in want if c not in positions]

                # 筛选可交易标的
                tradable: dict[str, float] = {}
                for code in want:
                    if code not in day.index:
                        halted_codes.add(code)
                        continue
                    open_price = self._safe_scalar(day_open.get(code, np.nan))
                    if pd.isna(open_price):
                        halted_codes.add(code)
                        continue
                    raw_price = float(open_price)
                    est_notional = effective_cash * raw_wt.get(code, 0)
                    dyn_slip = _dynamic_slippage(code, day, est_notional)
                    buy_price = raw_price * (1 + dyn_slip)

                    # 流动性冻结：OHLC全相同且成交量极低 → 不可交易
                    try:
                        row = day[day["code"] == code]
                        if len(row) > 0:
                            o = self._safe_scalar(row["open"].iloc[0])
                            c = self._safe_scalar(row["close"].iloc[0])
                            h = self._safe_scalar(row["high"].iloc[0]) if "high" in row.columns else o
                            l = self._safe_scalar(row["low"].iloc[0]) if "low" in row.columns else o
                            vol = self._safe_scalar(row["volume"].iloc[0]) if "volume" in row.columns else 0
                            if (pd.notna(o) and pd.notna(c) and o == c == h == l
                                    and pd.notna(vol) and vol < 100):
                                continue  # 流动性冻结，跳过
                    except (KeyError, ValueError, TypeError, IndexError):
                        pass

                    if buy_price > 0:
                        tradable[code] = buy_price

                if tradable and cash > 0:
                    # ══════════════════════════════════════════
                    # 动态仓位暴露：market_score → 连续 exposure bucket
                    # market_score >= 0.60 → 100%
                    # market_score 0.40~0.60 → 60%
                    # market_score 0.20~0.40 → 30%
                    # market_score < 0.20 → 0%
                    # ══════════════════════════════════════════
                    ms_today = None
                    if "market_score" in day.columns and len(day) > 0:
                        try:
                            ms_today = float(day["market_score"].iloc[0])
                        except (ValueError, TypeError):
                            ms_today = None
                    if ms_today is not None and pd.notna(ms_today):
                        if ms_today >= 0.60:
                            exposure_mult = 1.0
                        elif ms_today >= 0.40:
                            exposure_mult = 0.60
                        elif ms_today >= 0.20:
                            exposure_mult = 0.30
                        else:
                            exposure_mult = 0.0
                    else:
                        exposure_mult = 0.60  # 无 market_score 时的默认值

                    # ══════════════════════════════════════════
                    # 组合层风控1：最大回撤保护
                    # 组合从峰值回撤超过阈值 → 强制降仓
                    # ══════════════════════════════════════════
                    dd_protect = pos_rule.get("max_portfolio_dd")
                    if dd_protect is not None and equity_rows:
                        peak = max(r["equity"] for r in equity_rows)
                        current_eq = equity_rows[-1]["equity"]
                        dd_from_peak = current_eq / peak - 1 if peak > 0 else 0
                        if dd_from_peak <= -abs(dd_protect):
                            exposure_mult = min(exposure_mult, 0.3)  # 强制降至30%

                    # ══════════════════════════════════════════
                    # 组合层风控2：Breadth/panic 兜底保护
                    # breadth<0.20 → 最大30%；panic → 强制空仓
                    # ══════════════════════════════════════════
                    breadth_today = None
                    if "breadth" in day.columns and len(day) > 0:
                        try:
                            breadth_today = float(day["breadth"].iloc[0])
                        except (ValueError, TypeError):
                            breadth_today = None
                    if breadth_today is not None and pd.notna(breadth_today) and breadth_today < 0.20:
                        exposure_mult = min(exposure_mult, 0.30)

                    current_regime = day["regime"].iloc[0] if "regime" in day.columns and len(day) > 0 else "unknown"
                    if str(current_regime) == "panic":
                        exposure_mult = 0.0  # 恐慌强制空仓

                    effective_cash = cash * exposure_mult

                    # ══════════════════════════════════════════
                    # 组合层风控3：相关性约束（防同质化持仓）
                    # 如果候选ETF与已持仓ETF的corr_hs300差距<0.1，视为同质→跳过
                    # ══════════════════════════════════════════
                    corr_constraint = pos_rule.get("max_corr_overlap")
                    if corr_constraint is not None and "corr_hs300_60" in day.columns:
                        held_corrs = {}
                        for held_code in positions:
                            try:
                                hc = self._safe_scalar(day_corr.get(held_code, np.nan))
                                if pd.notna(hc):
                                    held_corrs[held_code] = hc
                            except (KeyError, ValueError, TypeError):
                                pass
                        if held_corrs:
                            tradable_filtered = {}
                            for code, px in tradable.items():
                                try:
                                    tc = self._safe_scalar(day_corr.get(code, np.nan))
                                except (KeyError, ValueError, TypeError):
                                    tc = np.nan
                                if pd.isna(tc):
                                    tradable_filtered[code] = px
                                    continue
                                # 检查与已持仓的距离
                                too_close = any(
                                    abs(tc - hc) < corr_constraint
                                    for hc in held_corrs.values()
                                )
                                if not too_close:
                                    tradable_filtered[code] = px
                            if tradable_filtered:
                                tradable = tradable_filtered

                    # ══════════════════════════════════════════
                    # 仓位权重计算：逆波动率加权 / HRP
                    # ══════════════════════════════════════════
                    allocation = pos_rule.get("allocation", "inv_vol")
                    max_single_wt = pos_rule.get("max_single_weight")
                    hrp_lookback = int(pos_rule.get("hrp_lookback", 60))

                    raw_wt: dict[str, float] = {}
                    if allocation == "hrp" and len(tradable) >= 2:
                        try:
                            # 构建历史收益率矩阵（过去 hrp_lookback 天）
                            all_dates = sorted(data["date"].unique())
                            current_idx = all_dates.index(t) if t in all_dates else len(all_dates) - 1
                            start_idx = max(0, current_idx - hrp_lookback)
                            lookback_dates = all_dates[start_idx:current_idx + 1]
                            hist = data[data["date"].isin(lookback_dates)]
                            ret_pivot = hist.pivot_table(
                                index="date", columns="code", values="ret", aggfunc="first"
                            ).dropna(axis=1, how="all")
                            avail_codes = [c for c in tradable if c in ret_pivot.columns]
                            if len(avail_codes) >= 2:
                                hrp_wt = self._hrp_weights(ret_pivot[avail_codes])
                                raw_wt = {c: hrp_wt.get(c, 0.0) for c in tradable}
                                total_w = sum(raw_wt.values())
                                if total_w > 0:
                                    raw_wt = {c: w / total_w for c, w in raw_wt.items()}
                        except Exception:
                            allocation = "inv_vol"  # 回退

                    if not raw_wt:
                        vol_target = pos_rule.get("volatility_target")
                        if vol_target is not None and "volatility20" in day.columns:
                            inv_vol = {}
                            for code in tradable:
                                try:
                                    vol = self._safe_scalar(day_vol.get(code, np.nan))
                                except (KeyError, ValueError, TypeError):
                                    vol = np.nan
                                if pd.notna(vol) and vol > 0.01:
                                    inv_vol[code] = 1.0 / vol
                                else:
                                    inv_vol[code] = 1.0 / 0.20
                            total_inv = sum(inv_vol.values())
                            raw_wt = {c: v / total_inv for c, v in inv_vol.items()} if total_inv > 0 else {}
                    if not raw_wt:
                        # 等权兜底
                        raw_wt = {c: 1.0 / len(tradable) for c in tradable}

                    # 单标的上限裁剪 + 重归一化
                    if max_single_wt:
                        for code in raw_wt:
                            raw_wt[code] = min(raw_wt[code], max_single_wt)
                        total_wt = sum(raw_wt.values())
                        if total_wt > 0:
                            raw_wt = {c: w / total_wt for c, w in raw_wt.items()}

                    # 按权重执行买入（权重高者优先）
                    for code in sorted(tradable.keys(), key=lambda c: raw_wt.get(c, 0), reverse=True):
                        if code in positions:
                            continue
                        wt = raw_wt.get(code, 0)
                        budget = effective_cash * wt
                        buy_price = tradable[code]
                        if buy_price <= 0 or budget <= 0:
                            continue

                        shares = budget / buy_price
                        if shares <= 0:
                            continue

                        # 容量限制：不超过日成交额的5%
                        try:
                            row = day[day["code"] == code]
                            if len(row) > 0 and "amount" in row.columns:
                                daily_amount = self._safe_scalar(row["amount"].iloc[0])
                                if pd.notna(daily_amount) and daily_amount > 0:
                                    max_notional = daily_amount * 0.05
                                    if buy_price * shares > max_notional:
                                        shares = max_notional / buy_price
                        except (KeyError, ValueError, TypeError, IndexError):
                            pass

                        notional = buy_price * shares
                        fee = notional * commission
                        total_cost = notional + fee
                        if total_cost > cash:
                            scale = cash / total_cost
                            shares = shares * scale
                            notional = buy_price * shares
                            fee = notional * commission
                            total_cost = notional + fee
                        if shares <= 0 or total_cost <= 0 or total_cost > cash * 1.01:
                            continue

                        cash -= total_cost
                        positions[code] = Position(
                            code=code,
                            shares=float(shares),
                            entry_date=t,
                            entry_idx=t_idx,
                            entry_price=float(buy_price),
                        )

            # ==============
            # 计算当日收盘权益
            # ==============
            holding_value = 0.0
            for code, p in positions.items():
                if code not in day.index:
                    continue
                close_price = self._safe_scalar(day_close.get(code, np.nan))
                if pd.isna(close_price):
                    continue
                holding_value += close_price * p.shares

            total_equity = cash + holding_value
            equity_rows.append(
                {
                    "date": t,
                    "equity": total_equity,
                    "cash": cash,
                    "holding_value": holding_value,
                    "positions": len(positions),
                    "halted": len(halted_codes),
                }
            )
            if progress_cb and t_idx % 100 == 0:
                progress_cb(t_idx, len(dates))

        equity_df = pd.DataFrame(equity_rows)
        if equity_df.empty:
            bench_df = pd.DataFrame()
            return {
                "trades": pd.DataFrame(trades),
                "equity": equity_df,
                "benchmark": bench_df,
                "metrics": {},
            }

        # 归一化：基点=base_point
        equity_df["nav"] = equity_df["equity"] / float(equity_df["equity"].iloc[0]) * base_point

        bench_df = pd.DataFrame()
        if benchmark is not None and not benchmark.empty:
            bench_df = benchmark.copy()
            bench_df = bench_df.drop_duplicates("date").sort_values("date")
            bench_df["nav"] = bench_df["close"] / float(bench_df["close"].iloc[0]) * base_point

        metrics = self.calc_metrics(equity_df, trades)

        return {
            "trades": pd.DataFrame(trades),
            "equity": equity_df,
            "benchmark": bench_df,
            "metrics": metrics,
        }

    # =====================================
    # 指标统计
    # =====================================

    def calc_metrics(self, equity_df: pd.DataFrame, trades: list[dict]) -> dict:
        if equity_df is None or equity_df.empty:
            return {}

        nav = equity_df["nav"].astype(float)
        ret = nav.pct_change().fillna(0.0)

        total_return = nav.iloc[-1] / nav.iloc[0] - 1
        ann_return = (1 + total_return) ** (252 / max(1, len(nav) - 1)) - 1
        ann_vol = float(ret.std(ddof=0) * math.sqrt(252))
        sharpe = float(ann_return / (ann_vol + 1e-12))

        # 最大回撤
        cummax = nav.cummax()
        dd = nav / cummax - 1
        max_dd = float(dd.min())

        # Sortino: 下行波动率（仅用负收益）
        neg_ret = ret[ret < 0]
        if len(neg_ret) > 2:
            down_vol = float(neg_ret.std(ddof=0) * math.sqrt(252))
            sortino = float(ann_return / (down_vol + 1e-12))
        else:
            sortino = float(sharpe)  # 兜底

        # Calmar: 年化收益/最大回撤
        calmar = float(ann_return / (abs(max_dd) + 1e-12))

        # 交易胜率
        trades_df = pd.DataFrame(trades)
        win_rate = np.nan
        if not trades_df.empty and "pnl" in trades_df.columns:
            win_rate = float((trades_df["pnl"] > 0).mean())

        # 真实换手率：Σ(sell_notional) / avg_equity / years
        years = max(len(equity_df) / 252.0, 0.5)
        avg_equity = float(equity_df["equity"].mean()) if "equity" in equity_df.columns else float(nav.mean())
        turnover = 0.0
        if not trades_df.empty and avg_equity > 0:
            sell_notionals = trades_df["sell_price"].astype(float) * trades_df["shares"].astype(float)
            total_sell_notional = float(sell_notionals.sum())
            turnover = (total_sell_notional / avg_equity) / years

        return {
            "total_return": float(total_return),
            "annual_return": float(ann_return),
            "annual_vol": float(ann_vol),
            "sharpe": float(sharpe),
            "sortino": sortino,
            "calmar": calmar,
            "max_drawdown": float(max_dd),
            "trades": int(len(trades)),
            "win_rate": win_rate,
            "turnover": float(turnover),
            "years": float(years),
        }
