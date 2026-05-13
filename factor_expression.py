import ast
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

# 运行时标记：确认使用的是新版 factor_expression
def _to_1d_series(s: Any) -> pd.Series:
    """强制转换为 1D float Series，处理 DataFrame/ndarray/scalar 等所有输入"""
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0] if s.shape[1] > 0 else pd.Series(dtype=float)
    if isinstance(s, np.ndarray):
        s = pd.Series(s.ravel())
    if not isinstance(s, pd.Series):
        s = pd.Series([float(s)] if np.isscalar(s) else list(s) if hasattr(s, '__iter__') else [float(s)])
    return s.astype(float)


def _rank(s) -> pd.Series:
    """安全 rank：处理 DataFrame / Series / scalar / ndarray 等所有输入"""
    s = _to_1d_series(s)
    return s.rank(pct=True)


def _zscore(s) -> pd.Series:
    """安全 zscore：处理所有输入类型"""
    s = _to_1d_series(s)
    return (s - s.mean()) / (s.std(ddof=0) + 1e-12)


def _clip(s, low: float | None = None, high: float | None = None) -> pd.Series:
    s = _to_1d_series(s)
    return s.clip(lower=low, upper=high)


def _where(cond: Any, x: Any, y: Any) -> pd.Series:
    """
    安全的三元 where，完全在 pandas 内操作，避免 numpy/pandas 边界歧义。
    
    关键防护：
    1. 所有输入统一为同 index 的 1D float Series
    2. 标量输入自动广播到 Series 的 index
    3. 返回始终为 float Series
    """
    # Step 1: 确定目标 index（cond > x > y 优先）
    if isinstance(cond, pd.Series) and len(cond) > 0:
        target_idx = cond.index
    elif isinstance(x, pd.Series) and len(x) > 0:
        target_idx = x.index
    elif isinstance(y, pd.Series) and len(y) > 0:
        target_idx = y.index
    else:
        target_idx = pd.RangeIndex(1)
    
    # Step 2: 统一 cond 为目标 index 的 Series
    if isinstance(cond, pd.Series):
        cond_s = cond.reindex(target_idx).fillna(False)
    else:
        cond_s = pd.Series(bool(cond), index=target_idx)
    
    # Step 3: 统一 x 和 y（标量广播到 target_idx）
    if isinstance(x, pd.Series):
        x_s = x.reindex(target_idx)
    else:
        x_s = pd.Series(float(x), index=target_idx)
    
    if isinstance(y, pd.Series):
        y_s = y.reindex(target_idx)
    else:
        y_s = pd.Series(float(y), index=target_idx)
    
    # Step 4: 使用 pandas .where() 做向量化选择
    result = x_s.where(cond_s.astype(bool), y_s)
    return result.astype(float)


def _sigmoid(s, k=1.0) -> pd.Series:
    """Sigmoid: 1/(1+exp(-k*s))。自然截断到 0-1，适合做概率化因子"""
    s = _to_1d_series(s)
    k = float(k)
    return 1.0 / (1.0 + np.exp(-k * s))


def _interaction(a, b) -> pd.Series:
    """因子交互: sqrt(rank(a) * rank(b))。
    仅当两个因子同时高时得分高，单一维度高不会拉分。
    内部固定使用 rank 防止量纲爆炸。"""
    a = _rank(a)
    b = _rank(b)
    return np.sqrt((a * b).clip(lower=0))


def _group_mean(s) -> pd.Series:
    """横截面均值：返回标量均值广播到原 index"""
    s = _to_1d_series(s)
    mean_val = s.mean()
    return pd.Series(mean_val, index=s.index)


def _ts_rank(s, period=20) -> pd.Series:
    """
    时间序列百分位排名：当前值在过去 period 内的分位数。
    ⚠ 需要多日时间序列数据。在 select_on_date（逐日横截面）中不可用。
    应用于预计算列或 indicators 引擎中。
    """
    s = _to_1d_series(s)
    p = int(period)
    return s.rolling(p, min_periods=1).apply(
        lambda x: (x.iloc[-1] >= x).mean(), raw=False
    )


def _rolling_std(s, period=20) -> pd.Series:
    """滚动标准差。
    ⚠ 需要多日时间序列数据。在 select_on_date（逐日横截面）中不可用。"""
    s = _to_1d_series(s)
    return s.rolling(int(period), min_periods=1).std()


def _rolling_mean(s, period=20) -> pd.Series:
    """滚动均值。
    ⚠ 需要多日时间序列数据。在 select_on_date（逐日横截面）中不可用。"""
    s = _to_1d_series(s)
    return s.rolling(int(period), min_periods=1).mean()


ALLOWED_FUNCS = {
    "rank": _rank,
    "zscore": _zscore,
    "clip": _clip,
    "abs": np.abs,
    "log": np.log,
    "sqrt": np.sqrt,
    "where": _where,
    "sigmoid": _sigmoid,
    "interaction": _interaction,
    "group_mean": _group_mean,
    "ts_rank": _ts_rank,
    "rolling_std": _rolling_std,
    "rolling_mean": _rolling_mean,
}


ALLOWED_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Call,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.Compare,
    # 注意：不支持 ast.BoolOp/ast.And/ast.Or（Python 的 and/or）
    # 因为 pandas Series 不能直接用 and/or，会触发 "truth value ambiguous" 错误
    # 请使用 & (BitAnd) 和 | (BitOr) 代替，并加括号：
    #   错误: a > 0 and b < 0
    #   正确: (a > 0) & (b < 0)
    ast.BitAnd,
    ast.BitOr,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.USub,
    ast.UAdd,
    ast.Gt,
    ast.GtE,
    ast.Lt,
    ast.LtE,
    ast.Eq,
    ast.NotEq,
)


class ExpressionEngine:
    """
    安全表达式执行器（用于横截面/单行信号计算）

    支持：
      - 四则运算、括号、比较
      - 逻辑运算：& (与)、| (或)，必须加括号
        例如：(a > 0.5) & (b < 0.3) 或 (a > 0.5) | (b < 0.3)
        注意：不支持 Python 的 and/or，因为 pandas Series 不能直接使用
      - 函数：rank(x)、zscore(x)、clip(x,low,high)、abs/log/sqrt/where
      - 变量：必须是 df 的列名（如 roc12、volatility20、cross_momentum 等）
    """

    def __init__(self):
        pass

    def __init__(self):
        self._compile_cache: dict[str, Any] = {}

    def _validate(self, node: ast.AST):
        for n in ast.walk(node):
            if not isinstance(n, ALLOWED_NODES):
                raise ValueError(f"不支持的表达式节点: {type(n).__name__}")
            if isinstance(n, ast.Call):
                if not isinstance(n.func, ast.Name):
                    raise ValueError("函数调用仅允许直接函数名")
                if n.func.id not in ALLOWED_FUNCS:
                    raise ValueError(f"不支持的函数: {n.func.id}")

    def _compile_expr(self, expr: str):
        """解析+验证+编译表达式，结果缓存以跳过重复的 AST 解析"""
        if expr in self._compile_cache:
            return self._compile_cache[expr]
        node = ast.parse(expr, mode="eval")
        self._validate(node)
        code = compile(node, expr, "eval")
        # 限制缓存大小，防止无限增长
        if len(self._compile_cache) > 512:
            self._compile_cache.clear()
        self._compile_cache[expr] = code
        return code

    def eval(self, expr: str, df: pd.DataFrame):
        """安全评估表达式。始终返回 1D Series 或 None。"""
        expr = (expr or "").strip()
        if not expr:
            return None

        code = self._compile_expr(expr)

        env = dict(ALLOWED_FUNCS)
        for c in df.columns:
            if c.isidentifier():
                env[c] = df[c]

        result = eval(code, {"__builtins__": {}}, env)

        if result is None:
            return None
        if isinstance(result, pd.DataFrame):
            result = result.iloc[:, 0] if result.shape[1] > 0 else pd.Series(dtype=float)
        if not isinstance(result, pd.Series):
            result = pd.Series(float(result), index=df.index)
        return result
