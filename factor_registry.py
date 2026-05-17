"""
统一因子注册中心 — 所有因子的唯一定义来源。

IndicatorEngine / FactorLab / FactorMonitor / Strategy Builder 均从此文件读取因子定义。
"""

from __future__ import annotations

# ══════════════════════════════════════════════════════════════════════════════
# 因子组定义
# ══════════════════════════════════════════════════════════════════════════════

FACTOR_GROUPS: dict[str, dict] = {
    "trend":           {"name": "趋势",       "sign_default": +1},
    "risk":            {"name": "风险",       "sign_default": -1},
    "volume":          {"name": "资金",       "sign_default": +1},
    "structure":       {"name": "结构",       "sign_default": None},
    "consistency":     {"name": "一致性",     "sign_default": +1},
    "mean_reversion":  {"name": "均值回复",   "sign_default": None},
    "auxiliary":       {"name": "辅助指标",   "sign_default": None},
}

# ══════════════════════════════════════════════════════════════════════════════
# 因子注册表 — 单一定义源
# ══════════════════════════════════════════════════════════════════════════════

FACTOR_REGISTRY: dict[str, dict] = {
    # ── Trend / Momentum ──────────────────────────────────────────────────
    "roc20": {
        "name": "ROC20",
        "group": "trend",
        "type": "time_series",
        "column": "roc20",
        "sign": +1,
        "description": "20日涨跌幅（动量因子）",
        "primary": True,
    },
    "roc60": {
        "name": "ROC60",
        "group": "trend",
        "type": "time_series",
        "column": "roc60",
        "sign": +1,
        "description": "60日涨跌幅（中期动量）",
        "primary": True,
    },
    "roc120": {
        "name": "ROC120",
        "group": "trend",
        "type": "time_series",
        "column": "roc120",
        "sign": +1,
        "description": "120日涨跌幅（长期动量，低换手）",
        "primary": False,
    },
    "roc12": {
        "name": "ROC12",
        "group": "trend",
        "type": "time_series",
        "column": "roc12",
        "sign": +1,
        "description": "12日涨跌幅（短期动量，用于 cross_momentum 计算）",
        "primary": False,
    },
    "ret5": {
        "name": "Return5",
        "group": "trend",
        "type": "time_series",
        "column": "ret5",
        "sign": +1,
        "description": "5日涨跌幅（短期动量）",
        "primary": False,
    },
    "ret20": {
        "name": "Return20",
        "group": "trend",
        "type": "time_series",
        "column": "ret20",
        "sign": +1,
        "description": "20日涨跌幅（= roc20）",
        "primary": False,
    },
    "ma_ratio_5_20": {
        "name": "MA5/MA20 Ratio",
        "group": "trend",
        "type": "time_series",
        "column": "ma_ratio_5_20",
        "sign": +1,
        "description": "MA5 / MA20 - 1（短期趋势强度）",
        "primary": False,
    },
    "breakout20": {
        "name": "Breakout20",
        "group": "trend",
        "type": "time_series",
        "column": "breakout20",
        "sign": +1,
        "description": "收盘价 / 20日最高价 - 1（突破强度）",
        "primary": True,
    },
    "relative_strength_vs_hs300": {
        "name": "RS vs HS300",
        "group": "trend",
        "type": "derived",
        "column": "relative_strength_vs_hs300",
        "sign": +1,
        "description": "ret20 - HS300 ret20（相对强度）",
        "primary": True,
    },
    "cross_momentum": {
        "name": "Cross Momentum",
        "group": "trend",
        "type": "cross_sectional",
        "column": "cross_momentum",
        "sign": +1,
        "description": "roc12 的横截面0-1百分位排名",
        "primary": False,
    },
    "macd_dif": {
        "name": "MACD DIF",
        "group": "trend",
        "type": "time_series",
        "column": "macd_dif",
        "sign": +1,
        "description": "EMA12 - EMA26（快慢均线差值）",
        "primary": False,
    },
    "macd_hist": {
        "name": "MACD Histogram",
        "group": "trend",
        "type": "time_series",
        "column": "macd_hist",
        "sign": +1,
        "description": "(DIF - DEA) × 2（MACD柱）",
        "primary": False,
    },
    "low52w": {
        "name": "52周低位",
        "group": "trend",
        "type": "time_series",
        "column": "low52w",
        "sign": None,
        "description": "收盘价 / 52周最低价 - 1（趋势位置参考）",
        "primary": False,
    },

    # ── Risk / Volatility ─────────────────────────────────────────────────
    "volatility20": {
        "name": "Volatility20",
        "group": "risk",
        "type": "time_series",
        "column": "volatility20",
        "sign": -1,
        "description": "20日年化波动率（越低越好）",
        "primary": True,
    },
    "downside_volatility": {
        "name": "Downside Volatility",
        "group": "risk",
        "type": "time_series",
        "column": "downside_volatility",
        "sign": -1,
        "description": "下侧波动率（仅计入负收益，20日年化）",
        "primary": True,
    },
    "max_drawdown20": {
        "name": "Max Drawdown20",
        "group": "risk",
        "type": "time_series",
        "column": "max_drawdown20",
        "sign": -1,
        "description": "20日最大回撤（负值，越小/绝对值越大越差）",
        "primary": True,
    },
    "atr20": {
        "name": "ATR20",
        "group": "risk",
        "type": "time_series",
        "column": "atr20",
        "sign": -1,
        "description": "20日平均真实波幅（波动性指标）",
        "primary": True,
    },
    "atr14": {
        "name": "ATR14",
        "group": "risk",
        "type": "time_series",
        "column": "atr14",
        "sign": -1,
        "description": "14日平均真实波幅",
        "primary": False,
    },
    "volatility_ratio": {
        "name": "Volatility Ratio",
        "group": "risk",
        "type": "time_series",
        "column": "volatility_ratio",
        "sign": None,
        "description": "volatility20 / volatility120（波动率状态切换信号）",
        "primary": False,
    },
    "cross_volatility": {
        "name": "Cross Volatility",
        "group": "risk",
        "type": "cross_sectional",
        "column": "cross_volatility",
        "sign": -1,
        "description": "volatility20 的横截面0-1百分位排名",
        "primary": False,
    },

    # ── Volume / Liquidity ────────────────────────────────────────────────
    "turnover_change": {
        "name": "Turnover Change",
        "group": "volume",
        "type": "time_series",
        "column": "turnover_change",
        "sign": +1,
        "description": "(MA5成交量 / MA20成交量 - 1) × 100（资金流入信号）",
        "primary": True,
    },
    "vol_ratio20": {
        "name": "Volume Ratio20",
        "group": "volume",
        "type": "time_series",
        "column": "vol_ratio20",
        "sign": +1,
        "description": "当日成交量 / 20日均量（放量倍数）",
        "primary": True,
    },
    "obv": {
        "name": "OBV",
        "group": "volume",
        "type": "time_series",
        "column": "obv",
        "sign": +1,
        "description": "On-Balance Volume（累积资金流）",
        "primary": False,
    },

    # ── Structure / Correlation ───────────────────────────────────────────
    "barra_beta": {
        "name": "Barra Beta",
        "group": "structure",
        "type": "cross_sectional",
        "column": "barra_beta",
        "sign": None,
        "description": "60日滚动 Beta vs 沪深300（牛市正相关，熊市需惩罚偏离β=1）",
        "primary": True,
        "transform_bear": "beta_penalty",
    },
    "corr_hs300_60": {
        "name": "Corr HS300 60D",
        "group": "structure",
        "type": "cross_sectional",
        "column": "corr_hs300_60",
        "sign": None,
        "description": "60日滚动相关系数 vs 沪深300",
        "primary": True,
    },

    # ── Consistency ───────────────────────────────────────────────────────
    "trend_consistency": {
        "name": "Trend Consistency",
        "group": "consistency",
        "type": "time_series",
        "column": "trend_consistency",
        "sign": +1,
        "description": "roc20/60/120 中 >0 的个数（0-3，趋势一致性）",
        "primary": True,
    },

    # ── Mean Reversion ────────────────────────────────────────────────────
    "ma_distance": {
        "name": "MA Distance",
        "group": "mean_reversion",
        "type": "time_series",
        "column": "ma_distance",
        "sign": None,
        "description": "(收盘价 / MA20 - 1) × 100（偏离均线程度，震荡市均值回复信号）",
        "primary": True,
    },
    "rsi12": {
        "name": "RSI12",
        "group": "mean_reversion",
        "type": "time_series",
        "column": "rsi12",
        "sign": None,
        "description": "12日相对强弱指数（>70 超买，<30 超卖）",
        "primary": False,
    },
    "rsi24": {
        "name": "RSI24",
        "group": "mean_reversion",
        "type": "time_series",
        "column": "rsi24",
        "sign": None,
        "description": "24日相对强弱指数（慢速RSI）",
        "primary": False,
    },
    "rsi6": {
        "name": "RSI6",
        "group": "mean_reversion",
        "type": "time_series",
        "column": "rsi6",
        "sign": None,
        "description": "6日相对强弱指数（快速RSI）",
        "primary": False,
    },
    "cci14": {
        "name": "CCI14",
        "group": "mean_reversion",
        "type": "time_series",
        "column": "cci14",
        "sign": None,
        "description": "14日商品通道指数（均值回复信号）",
        "primary": False,
    },

    # ── Auxiliary / Reference ─────────────────────────────────────────────
    "kdj_j": {
        "name": "KDJ-J",
        "group": "auxiliary",
        "type": "time_series",
        "column": "kdj_j",
        "sign": None,
        "description": "KDJ J值（>100超买，<0超卖）",
        "primary": False,
    },
    "kdj_k": {
        "name": "KDJ-K",
        "group": "auxiliary",
        "type": "time_series",
        "column": "kdj_k",
        "sign": None,
        "description": "KDJ K值",
        "primary": False,
    },
    "kdj_d": {
        "name": "KDJ-D",
        "group": "auxiliary",
        "type": "time_series",
        "column": "kdj_d",
        "sign": None,
        "description": "KDJ D值",
        "primary": False,
    },
}

# 确保 registry key 与 column 一致
for _k, _v in FACTOR_REGISTRY.items():
    assert _k == _v["column"], f"Registry key '{_k}' != column '{_v['column']}'"


# ══════════════════════════════════════════════════════════════════════════════
# 各组的代表因子（用于 FactorMonitor IC 计算）
# ══════════════════════════════════════════════════════════════════════════════

GROUP_REPRESENTATIVES: dict[str, str] = {
    "trend":          "roc20",
    "risk":           "volatility20",
    "volume":         "vol_ratio20",
    "structure":      "barra_beta",
    "consistency":    "trend_consistency",
    "mean_reversion": "ma_distance",
}


# ══════════════════════════════════════════════════════════════════════════════
# 查询函数
# ══════════════════════════════════════════════════════════════════════════════

def get_all_factor_names() -> list[str]:
    """返回所有注册因子的 key 列表。"""
    return list(FACTOR_REGISTRY.keys())


def get_primary_factors() -> list[str]:
    """返回 primary=True 的因子（当前活跃因子）。"""
    return [k for k, v in FACTOR_REGISTRY.items() if v.get("primary")]


def get_factors_by_group() -> dict[str, list[str]]:
    """返回 {group_name: [factor_keys]}。"""
    result: dict[str, list[str]] = {}
    for k, v in FACTOR_REGISTRY.items():
        g = v["group"]
        result.setdefault(g, []).append(k)
    return result


def get_factor_groups() -> list[str]:
    """返回所有因子组名（排除 auxiliary）。"""
    return [g for g in FACTOR_GROUPS if g != "auxiliary"]


def get_group_representative(group: str) -> str | None:
    """返回某组的代表因子名。"""
    return GROUP_REPRESENTATIVES.get(group)


def get_factor_info(name: str) -> dict | None:
    """返回单个因子的完整信息。"""
    return FACTOR_REGISTRY.get(name)


def get_active_factors_for_strategy() -> dict[str, list[str]]:
    """
    返回策略构建用的 {group: [active_factors]}。
    目前返回 primary 因子，后续可从 approved_factors.yaml 覆盖。
    """
    result: dict[str, list[str]] = {}
    for g in get_factor_groups():
        factors = [
            k for k, v in FACTOR_REGISTRY.items()
            if v.get("group") == g and v.get("primary")
        ]
        if factors:
            result[g] = factors
    return result


# ══════════════════════════════════════════════════════════════════════════════
# FactorApproval — 人工审批配置管理
# ══════════════════════════════════════════════════════════════════════════════

import json
from pathlib import Path
from datetime import datetime

_DEFAULT_APPROVAL_PATH = Path("data/research/approved_factors.json")


class FactorApproval:
    """管理人工审批后的因子配置。"""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else _DEFAULT_APPROVAL_PATH
        self.data: dict = {}

    def load(self) -> bool:
        if not self.path.exists():
            return False
        try:
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
            return True
        except (json.JSONDecodeError, KeyError):
            return False

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(self.data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        tmp.replace(self.path)

    def build_defaults(self, research_metrics_path: str = ""):
        groups_data = {}
        for g in get_factor_groups():
            all_in_group = [
                k for k, v in FACTOR_REGISTRY.items()
                if v.get("group") == g
            ]
            primary_in_group = [
                k for k, v in FACTOR_REGISTRY.items()
                if v.get("group") == g and v.get("primary")
            ]
            groups_data[g] = {
                "active_factors": primary_in_group,
                "excluded_factors": [k for k in all_in_group if k not in primary_in_group],
                "reason": "",
            }

        regime_map = {
            "bull": {"include_groups": ["trend", "volume", "structure"], "exclude_groups": []},
            "bull_volatile": {"include_groups": ["trend", "volume"], "exclude_groups": []},
            "bear": {"include_groups": ["risk", "volume"], "exclude_groups": [], "structure_transform": "beta_penalty"},
            "panic": {"include_groups": ["risk", "volume"], "exclude_groups": [], "structure_transform": "beta_penalty"},
            "sideways": {"include_groups": ["trend", "volume", "structure", "mean_reversion"], "exclude_groups": []},
        }

        self.data = {
            "version": 1,
            "last_modified": datetime.now().isoformat(),
            "research_basis": {"metrics_parquet": research_metrics_path},
            "groups": groups_data,
            "regime_factor_map": regime_map,
        }

    def apply_research_suggestions(self, report: pd.DataFrame):
        """根据因子研究指标自动建议启用/排除。

        规则：
          - coverage_pct < 50%  →  建议排除（数据不足）
          - abs(icir) < 0.15    →  建议排除（IC不显著）
          - 以上都满足          →  建议启用
        """
        if report is None or report.empty:
            return
        groups_data = self.data.get("groups", {})
        if not groups_data:
            return
        report_idx = report.set_index("factor")
        for g_name, g_data in groups_data.items():
            all_factors = g_data.get("active_factors", []) + g_data.get("excluded_factors", [])
            suggested_active = []
            suggested_excluded = []
            reasons = []
            for f_key in all_factors:
                if f_key not in report_idx.index:
                    suggested_active.append(f_key)
                    continue
                row = report_idx.loc[f_key]
                coverage = float(row.get("coverage_pct", 0) or 0)
                icir = float(row.get("icir", 0) or 0)
                mean_ic = float(row.get("mean_ic", 0) or 0)
                issues = []
                if coverage < 50:
                    issues.append(f"低覆盖率({coverage:.0f}%)")
                if abs(icir) < 0.15:
                    issues.append(f"低ICIR({icir:.3f})")
                if issues:
                    suggested_excluded.append(f_key)
                    info = get_factor_info(f_key) or {}
                    reasons.append(f"{info.get('name', f_key)}: {'; '.join(issues)}")
                else:
                    suggested_active.append(f_key)
                    info = get_factor_info(f_key) or {}
                    reasons.append(f"{info.get('name', f_key)}: IC={mean_ic:.3f}, ICIR={icir:.3f}, Cov={coverage:.0f}%")
            g_data["active_factors"] = suggested_active
            g_data["excluded_factors"] = suggested_excluded
            g_data["reason"] = " | ".join(reasons) if reasons else ""
        self.data["last_modified"] = datetime.now().isoformat()

    def get_active_factors(self, group: str | None = None) -> list[str]:
        groups = self.data.get("groups", {})
        if group:
            return groups.get(group, {}).get("active_factors", [])
        result = []
        for g_data in groups.values():
            result.extend(g_data.get("active_factors", []))
        return result

    def get_excluded_factors(self) -> list[str]:
        result = []
        for g_data in self.data.get("groups", {}).values():
            result.extend(g_data.get("excluded_factors", []))
        return result

    def get_regime_groups(self, regime: str) -> dict:
        return self.data.get("regime_factor_map", {}).get(regime, {})

    def get_active_factors_for_regime(self, regime: str) -> dict[str, list[str]]:
        rmap = self.get_regime_groups(regime)
        include = rmap.get("include_groups", [])
        exclude = rmap.get("exclude_groups", [])
        groups = self.data.get("groups", {})
        result = {}
        for g in include:
            if g not in exclude:
                factors = groups.get(g, {}).get("active_factors", [])
                if factors:
                    result[g] = factors
        return result

    def validate(self) -> list[str]:
        errors = []
        all_registered = get_all_factor_names()
        for g, g_data in self.data.get("groups", {}).items():
            for f in g_data.get("active_factors", []):
                if f not in all_registered:
                    errors.append(f"Group '{g}': active factor '{f}' not in registry")
            for f in g_data.get("excluded_factors", []):
                if f not in all_registered:
                    errors.append(f"Group '{g}': excluded factor '{f}' not in registry")
        for regime, r_data in self.data.get("regime_factor_map", {}).items():
            for g in r_data.get("include_groups", []) + r_data.get("exclude_groups", []):
                if g not in FACTOR_GROUPS and g != "auxiliary":
                    errors.append(f"Regime '{regime}': unknown group '{g}'")
        return errors
