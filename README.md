# QtQuant — 研究驱动型量化交易框架

基于 ETF 多因子轮动的量化交易系统，核心原则：**研究结果沉淀，决策权保留给研究者**。

---

## 目录

- [架构概览](#架构概览)
- [因子研究系统](#因子研究系统)
  - [1. 因子注册中心](#1-因子注册中心-factor_registry)
  - [2. 因子计算管线](#2-因子计算管线)
  - [3. 因子研究引擎](#3-因子研究引擎-factor_research_engine)
  - [4. 研究报告（三类输出）](#4-研究报告三类输出)
  - [5. 缓存策略](#5-缓存策略-factor_cache)
  - [6. 人工审批](#6-人工审批-factorapproval)
- [策略构建系统](#策略构建系统)
  - [7. 动态因子权重](#7-动态因子权重-factormonitor)
  - [8. 市场状态检测](#8-市场状态检测-market_regime)
  - [9. 表达式引擎](#9-表达式引擎-factor_expression)
  - [10. 策略生成](#10-策略生成)
  - [11. 回测执行](#11-回测执行-backtest_engine)
  - [12. 策略优化](#12-策略优化)
- [完整数据流](#完整数据流)
- [文件索引](#文件索引)

---

## 架构概览

```
┌─────────────────────────────────────────────────────────────────┐
│                     FACTOR REGISTRY                              │
│         所有因子定义的唯一来源 (factor_registry.py)                │
│         · 34个因子 · 7个分类 · sign / type / group              │
└──────────────────────┬──────────────────────────────────────────┘
                       │
        ┌──────────────┴──────────────┐
        ▼                             ▼
┌───────────────┐             ┌───────────────────┐
│  FACTOR       │             │  STRATEGY         │
│  RESEARCH     │             │  BUILDING         │
│               │             │                   │
│ 研究全量因子   │             │ 读取审批配置       │
│ → 三类报告     │             │ → 生成 regime-    │
│ → 缓存到磁盘   │             │   conditional     │
│ → 人工审批     │             │   表达式          │
│               │             │ → 回测验证         │
└───────────────┘             └───────────────────┘
        │                             │
        └──────────────┬──────────────┘
                       ▼
              ┌─────────────────┐
              │  APPROVED CONFIG │
              │  (人工决策层)     │
              │                  │
              │ · 因子白名单      │
              │ · regime 映射     │
              │ · 排除重复因子    │
              └─────────────────┘
```

**核心设计决策**：

1. **研究与策略联动但不耦合**：研究结果保存为 `factor_metrics.parquet`，策略构建从 `approved_factors.json` 读取，中间由人工决策
2. **全量研究而非选择性研究**：对 34 个注册因子进行系统分析，不预设哪些有效，让数据说话
3. **所有参数均可解释**：regime 权重来自市场结构认知（不可优化），信号权重来自滚动 ICIR（自适应的），持仓权重来自 HRP（风险分散）

---

# 因子研究系统

## 1. 因子注册中心 (`factor_registry`)

### 设计动机

改造前，因子定义散落在 4 个文件：

| 文件 | 定义内容 |
|---|---|
| `indicators.py` | 时间序列因子计算（隐式，通过 `calculate()` 方法） |
| `factors.py` | 横截面因子（4 个） |
| `factor_monitor.py` | 因子分组（5 组） |
| `main_window.py` | 策略因子分组（6 组，与 factor_monitor 不完全一致） |

新增因子需要修改多处，容易出现不一致。`factor_registry.py` 作为单一真相来源解决了这个问题。

### 数据结构

每个因子在 `FACTOR_REGISTRY` 中的定义：

```python
"roc20": {
    "name": "ROC20",          # 显示名称
    "group": "trend",          # 所属分组 (7组之一)
    "type": "time_series",     # time_series | cross_sectional | derived
    "column": "roc20",         # 对应 parquet 列名
    "sign": +1,                # +1 正向 / -1 反向 / None 视市场状态而定
    "description": "...",     # 因子说明
    "primary": True,           # 是否默认纳入策略构建
}
```

**因子分类（7组）**：

| 组 | 经济含义 | sign 默认 | 代表因子 | 数量 |
|---|---|---|---|---|
| `trend` | 趋势/动量 | +1 | roc20 | 13 |
| `risk` | 波动/风险 | -1 | volatility20 | 7 |
| `volume` | 资金/成交量 | +1 | vol_ratio20 | 3 |
| `structure` | 结构/相关性 | None | barra_beta | 2 |
| `consistency` | 趋势一致性 | +1 | trend_consistency | 1 |
| `mean_reversion` | 均值回复 | None | ma_distance | 5 |
| `auxiliary` | 辅助指标 | None | — | 3 |

**`type` 字段含义**：

- `time_series`：由 `IndicatorEngine.calculate()` 直接计算，每只 ETF 独立
- `cross_sectional`：由 `FactorEngine` 计算，需要全市场 ETF 横截面
- `derived`：需要额外处理（如 `relative_strength_vs_hs300` = ret20 - hs300_ret20）

### 查询 API

```python
get_all_factor_names()              # → 34 个因子名
get_primary_factors()               # → 14 个默认活跃因子
get_factors_by_group()              # → {group: [factors]}
get_active_factors_for_strategy()   # → 策略构建用 {group: [primary_factors]}
get_group_representative(group)     # → 每组代表因子（用于 IC 计算）
get_factor_info(name)               # → 单因子完整信息
```

---

## 2. 因子计算管线

### 时间序列层：`IndicatorEngine`

对每只 ETF 独立执行，输入 OHLCV，输出 ~35 个技术指标列。

```
OHLCV DataFrame (单 ETF)
    │
    ▼
IndicatorEngine.calculate(df)
    │
    ├── 移动平均: ma5, ma10, ma20, ma60, ema12, ema26
    ├── MACD:      macd_dif, macd_dea, macd_hist
    ├── KDJ:       kdj_k, kdj_d, kdj_j
    ├── Bollinger: boll_mid, boll_up, boll_low
    ├── RSI:       rsi6, rsi12, rsi24
    ├── 动量:      roc12, ret5, ret20(=roc20), roc60, roc120, ma_ratio_5_20
    ├── 趋势:      ma_distance, breakout20, low52w
    ├── 波动率:    volatility20, vol_ratio20, downside_volatility, atr14, atr20
    ├── 资金:      turnover_change
    ├── 回撤:      max_drawdown20
    └── 其他:      obv, cci14, ret, volatility_ratio, trend_consistency
    │
    ▼
含 ~35 个指标列的 DataFrame
```

### 横截面层：`FactorEngine`

在全部 ETF 上执行，输出长格式 `[date, code, factor_name, factor_value]`：

| 方法 | 输出列 | 说明 |
|---|---|---|
| `cross_section_momentum()` | `cross_momentum` | roc12 的每日横截面百分位排名 |
| `cross_volatility()` | `cross_volatility` | volatility20 的每日横截面百分位排名 |
| `corr_hs300()` | `corr_hs300_60` | 60日滚动 Pearson 相关性 vs HS300 |
| `barra_beta()` | `barra_beta` | 60日滚动 Cov/var Beta vs HS300 |

### 数据组装：`ResearchDataService.build_all()`

```
对于每只 ETF:
  load_quotes(code)              →  OHLCV raw data
    ↓
  IndicatorEngine.calculate()    →  +35 time-series indicators
    ↓
  合并 cross_factors (long format) → 横截面因子
    ↓
  合并 regime_df                  → 市场状态
    ↓
  合并 breadth_df                 → 市场宽度 / market_score
    ↓
  save → data/research/{code}.parquet
```

最终每只 ETF 的 research parquet 含 ~50 列。

---

## 3. 因子研究引擎 (`FactorResearchEngine`)

### 设计原则

1. **全量研究**：对 `FACTOR_REGISTRY` 中所有在数据中可用的因子进行研究，不预设取舍
2. **增量构建**：包装已有 `FactorLab` 的 5 个核心方法，新增 turnover/stability/monotonicity/clustering
3. **并行加速**：IC Decay 和 Quantile Return 按因子并行（`ThreadPoolExecutor`，max_workers=8）
4. **进度反馈**：7 个阶段逐个 emit `stage` 和 `progress` 信号

### 研究流程（7 阶段）

```
Stage 1: 基础统计   → mean_value, std_value, coverage_pct
Stage 2: Rolling IC → 60日滚动 |RankIC| 序列 (每个因子一列)
Stage 3: IC Decay   → horizon=1/5/10/20 的平均 |IC|（并行）
Stage 4: 分位数收益 → 按因子值分5组，计算各组平均未来收益
Stage 5: 因子相关性 → 每日横截面相关矩阵的均值
Stage 6: Regime IC  → 5种市场状态下分别的 |IC|
Stage 7: Turnover / Stability / Monotonicity / Clustering
```

### 输出：`factor_report_df`

每个因子一行，共 30+ 列指标：

| 类别 | 列名 | 含义 |
|---|---|---|
| 基础信息 | `factor, group, sign, name, primary` | 因子元信息 |
| 基础统计 | `mean_value, std_value, coverage_pct` | 均值/标准差/覆盖率 |
| IC 指标 | `mean_ic, std_ic, icir, ic_tstat` | 滚动 IC 均值/波动/信息比率/t值 |
| IC 衰减 | `ic_h1, ic_h5, ic_h10, ic_h20, ic_decay_slope` | 各 horizon 的 IC 及衰减斜率 |
| 分位数 | `q1_return, q5_return, q5_minus_q1, monotonicity` | 多空收益差及单调性 |
| Regime IC | `bull_ic, bear_ic, sideways_ic, panic_ic, bull_volatile_ic, regime_consistency` | 各市场状态的 IC |
| 稳定性 | `turnover, autocorr_20` | 截面排序变动率/20日自相关 |
| 聚类 | `corr_cluster, max_pairwise_corr, max_corr_with` | 聚类ID/最高相关因子 |

### RankIC 计算的核心细节

```
RankIC(t) = spearmanr( factor_value[t - 5], return[t-5 → t] )

为什么用 t-5 的因子值？
  → 避免 look-ahead bias：因子值在 t-5 已知，收益在 t 已知
  → 两者不存在同时性偏差（simultaneity bias）

为什么用绝对值 |IC|？
  → 方向由因子 sign 在表达式中确定
  → 动态权重计算只关心预测强度，不关心方向
```

---

## 4. 研究报告（三类输出）

### 4.1 Summary Report（总览表）

`factor_report_df` 显示在 `QTableView` 中，支持排序/筛选。

| factor | group | mean_ic | icir | bull_ic | bear_ic | turnover | monotonicity | corr_cluster |
|---|---|---|---|---|---|---|---|---|
| roc20 | trend | 0.045 | 1.2 | 0.052 | 0.018 | 0.08 | 0.92 | 1 |
| volatility20 | risk | 0.038 | 0.95 | 0.035 | 0.042 | 0.05 | 0.85 | 2 |

**使用场景**：快速排序找 top ICIR 因子；对比 bull vs bear 有效性差异；识别高 turnover 因子（换手成本高）。

### 4.2 Factor Detail Report（单因子画像）

每个因子独立展示，双击 Summary 行跳转：

- **Rolling IC 曲线**：IC 随时间变化，识别因子失效期
- **Quantile Return**：按因子值分5组，验证多空收益是否单调递增
- **IC Decay**：IC 随 horizon 衰减速度，判断适合短周期还是长周期
- **Regime IC**：各市场状态下 IC 柱状图

**使用场景**：确认因子逻辑是否符合预期（如 ROC20 在牛市中 IC 应显著高于熊市）。

### 4.3 Correlation / Clustering Report

- 热力图：所有因子两两相关性的可视化
- 聚类树状图：ward 层次聚类 (`distance = 1 - |corr|`, threshold=0.35)
- 高相关对列表：`|corr| > 0.7` 的因子对

**使用场景**：识别冗余因子。例如 `roc20` 与 `roc60` 相关性 0.87 → 保留 roc20，排除 roc60，避免"假多因子"。

**聚类算法**：
```
distance = 1 - |correlation_matrix|
linkage = ward(distance)
clusters = fcluster(linkage, t=0.35, criterion='distance')
```

---

## 5. 缓存策略 (`FactorCache`)

### 缓存文件布局

```
data/research/
  factor_metrics.parquet          ← factor_report_df（一键加载 Summary）
  factor_cache/
    cache_meta.json               ← {cache_key, date_range, n_codes, created_at}
    rolling_ic.parquet            ← date × ic_{factor} 宽表
    ic_decay.parquet              ← factor × h1/h5/h10/h20
    corr_matrix.parquet           ← 因子相关性矩阵
    regime_ic.parquet            ← factor × bull/bear/sideways/panic/bull_volatile
    turnover.parquet             ← factor × turnover
```

### 缓存校验

```python
cache_key = sha256(sorted(codes) + start_date + end_date)[:16]
```

当 codes 列表、起始日期、结束日期任一变化时，自动失效。用户可手动点击"重新计算"强制失效。

### 性能收益

| 场景 | 无缓存 | 有缓存 |
|---|---|---|
| 首次运行 | 60-120s（全量计算） | 60-120s |
| 二次打开 | 60-120s（重算） | ~0.5s（读 parquet） |
| 新增 ETF | 60-120s | 60-120s（缓存失效，需重算） |

---

## 6. 人工审批 (`FactorApproval`)

### 审批配置

`data/research/approved_factors.json`：

```json
{
  "version": 1,
  "groups": {
    "trend": {
      "active_factors": ["roc20", "breakout20", "relative_strength_vs_hs300"],
      "excluded_factors": ["roc60", "roc120"],
      "reason": "roc60 与 roc20 高度相关 (0.87)，roc120 换手过低"
    }
  },
  "regime_factor_map": {
    "bull":   {"include_groups": ["trend", "volume", "structure"]},
    "bear":   {"include_groups": ["risk", "volume"], "structure_transform": "beta_penalty"},
    "sideways": {"include_groups": ["trend", "volume", "structure", "mean_reversion"]}
  }
}
```

### 审批 UI

审批 Tab 提供：
- **树形勾选**：Group → Factor，勾选=启用，取消=排除
- **Regime 映射表**：每个 regime 勾选使用的因子组
- **保存/加载/重置**：一键持久化，支持回滚

---

# 策略构建系统

## 7. 动态因子权重 (`FactorMonitor`)

### 设计动机

传统做法给每个因子分配固定权重（如 momentum=0.4, vol=0.3）。问题：因子有效性随时间变化，固定权重在因子失效时仍"投票"。

### 算法

```
Step 1: 每日计算每个因子组代表的 RankIC(t)
         IC(t) = |spearmanr(factor[t-5], ret[t-5→t])|

Step 2: 60日滚动 ICIR(t)
         ICIR(t) = mean(IC[t-60:t]) / std(IC[t-60:t])
         min_periods=20，不足时 ICIR=0

Step 3: 截断 → [0, 2]，L1 归一化
         weight_i = clip(ICIR_i, 0, 2) / Σ clip(ICIR_j, 0, 2)
```

### 输出

每个交易日为每个因子组生成一个动态权重列：

```
dw_trend, dw_risk, dw_volume, dw_structure, dw_meanrev
```

**权重的语义**：
- `dw_trend = 0.4`：趋势因子近期预测能力强，占 40% 权重
- `dw_risk = 0.1`：波动率因子近期预测能力弱，仅 10%
- 总和恒为 1.0

### 在策略表达式中

```python
score = dw_trend * trend_score + dw_volume * volume_score
      + dw_structure * structure_score - dw_risk * risk_score
```

这种设计下，趋势因子有效时 dw_trend 自动变大，失效时自动变小，策略自适应市场变化。

---

## 8. 市场状态检测 (`MarketRegimeDetector`)

### 两个系统

**长周期 regime（MA50/MA200 交叉，基于 HS300 基准）**：

```
bull:           price > MA200, MA50 > MA200, vol20 < 30%
bull_volatile:  bull 条件 + vol20 > 30%
sideways:       既不 bull 也不 bear
bear:           price < MA200, MA50 < MA200, vol20 < 30%
panic:          bear 条件 + vol20 > 30%
```

**短周期市场评分（ETF 池广度 + 加速度）**：

```
market_score = 0.35 × breadth_ma20     (% ETF with close > MA20)
             + 0.35 × breadth_roc20     (% ETF with roc20 > 0)
             + 0.15 × breadth_up        (% ETF with positive daily return)
             + 0.15 × breadth_newhigh   (% ETF near 20-day high)
             - 0.10 × dispersion_norm   (横截面收益离散度，expanding归一化)
             + thrust_bonus             (广度加速度信号, ±0.12)

market_score ∈ [0, 1]
```

### 在策略中的应用

```
breadth < 0.25  → 强制使用 bear_expr（无论 regime 是什么）
regime == panic → 强制卖出
regime ∉ [bull, bull_volatile, sideways] → 空仓，持有防御资产
```

---

## 9. 表达式引擎 (`ExpressionEngine`)

### 安全模型

利用 Python AST 白名单的白盒安全，禁止任意代码执行：

```python
ALLOWED_NODES = {
    BinOp, UnaryOp, Call, Name, Constant, Compare,
    Add, Sub, Mult, Div, Pow, Gt, GtE, Lt, LtE, Eq, NotEq
}
# 明确禁止: BoolOp, And, Or, Lambda, Attribute, Subscript 等
```

**为什么禁止 `and`/`or`？** pandas Series 不支持 Python 布尔运算符，必须用 `&`/`|` 并按优先级加括号。

### 可用函数

| 函数 | 含义 | 示例 |
|---|---|---|
| `rank(x)` | 截面 0-1 百分位排名 | `rank(roc20)` |
| `zscore(x)` | (x - mean) / std | `zscore(volatility20)` |
| `clip(x, lo, hi)` | 截断 | `clip(rank(x), 0.1, 0.9)` |
| `abs(x)` | 绝对值 | `abs(ma_distance)` |
| `log(x)` | 自然对数 | `log(amount)` |
| `sqrt(x)` | 平方根 | `sqrt(corr_hs300_60)` |
| `where(c, a, b)` | 三元条件分支 | `where(regime=='bull', bull_expr, bear_expr)` |
| `sigmoid(x, k)` | S形压缩到 0-1 | `sigmoid(score, 5)` |
| `interaction(a, b)` | 两因子交互项 | `sqrt(rank(a) × rank(b))` |

### 执行流程

```python
# 表达式是字符串，但执行有严格限制
expr = "dw_trend * (rank(roc20) + rank(breakout20)) / 2 - dw_risk * rank(volatility20)"

# 1. ast.parse(mode="eval") → 语法树
# 2. 遍历所有节点，验证每个都在 ALLOWED_NODES 中
# 3. compile() → Python code object
# 4. eval(code, {"__builtins__": {}}, env)
#    ↑ 空 builtins 禁用所有内置函数
#    env = {**ALLOWED_FUNCS, **{col: df[col] for col in df.columns}}
```

---

## 10. 策略生成

### 表达式结构

策略的核心是一个嵌套的 `where()` 表达式：

```
score = where(breadth < threshold,
              bear_expr,
              where(regime == 'bull',        bull_expr,
              where(regime == 'bull_volatile', bull_expr,
              where(regime == 'bear',        bear_expr,
              where(regime == 'panic',       bear_expr,
              sideways_expr)))))
```

### 各 regime 的表达式

```
bull_expr = dw_trend × trend_score
          + dw_volume × volume_score
          + dw_structure × structure_score
          - dw_risk × risk_score

bear_expr = bull_expr 但 structure_score 替换为 _z_beta_penalty
           (_z_beta_penalty = zscore(-abs(beta - 1)))
           熊市中偏离 β=1 的 ETF 受惩罚

sideways_expr = bull_expr + dw_meanrev × _z_ma_distance
               震荡市加入均值回复因子
```

### 组内得分

每个 group 的得分是组内因子 zscore 的等权均值：

```python
trend_score = (_z_roc20 + _z_roc60 + _z_breakout20 + _z_relative_strength_vs_hs300) / 4
```

**为什么用 zscore 而非 rank？** rank 将分布压缩到 [0,1]，丢失厚尾信息。zscore 保留了离散度，当少数 ETF 在因子上极端值时，能给足够区分度。

### 审批配置的集成

```
策略生成时检查 approved_factors.json 是否存在：
  ├── 否 → 使用 FACTOR_REGISTRY 中的 primary 因子（默认14个）
  └── 是 → 使用审批后的 active_factors
          · active_factors → 构建组内得分表达式
          · excluded_factors → 排除冗余因子
          · regime_factor_map → 构建 regime-conditional 嵌套
```

---

## 11. 回测执行 (`BacktestEngine`)

### 每个交易日

```
1. select_on_date(date, strategy)
   ├── 市场过滤：regime 不在 allow_regimes → 防御资产列表
   ├── 流动性过滤：amount_ma20 < 30M → 排除
   ├── 计算得分：ExpressionEngine.eval(score_expr, df)
   ├── 过滤器：  ExpressionEngine.eval(filter_expr, df)
   ├── 排序取 top_n
   └── 返回入选 ETF 列表

2. 持仓优化
   ├── 波动率目标：仓位缩放至 target_vol=15%
   ├── HRP 分配：层次风险平价（60日相关性矩阵）
   ├── 单票上限：max_single_weight=30%
   └── 相关性约束：exclude 与已持仓 |corr| > max_corr_overlap 的 ETF

3. 卖出检查
   ├── 信号恶化：rank(score) < exit_threshold | trend_consistency < 2 | regime == panic
   ├── 止损：drawdown > stop_loss
   └── 再平衡：定期调仓
```

### 交易成本

sqrt 冲击模型：`impact = k × sqrt(trade_value / daily_volume)`，`k=0.001`。

---

## 12. 策略优化

### 搜索空间（仅 5 个参数）

```python
{
    "top_n":               [1, 2, 3, 4, 5],                    # 持仓数量
    "exit_rank_threshold": [0, 0.1, 0.2, 0.3, 0.4],            # 卖出信号阈值
    "breadth_threshold":   [0.15, 0.20, 0.25, 0.30, 0.35, 0.40], # 广度阈值
    "stop_loss":           [0, 0.03, 0.05, 0.08, 0.12],        # 止损线
    "max_corr_overlap":    [0.05, 0.10, 0.15, 0.20],           # 持仓相关性上限
}
```

**为什么只优化这些参数？**
- 它们是组合层面的风控参数，不涉及因子选择
- 因子权重由 `FactorMonitor` 动态确定（ICIR 自适应）
- regime 映射来自市场结构认知（bull 用趋势，bear 用风险——这是经济学原理，不是拟合目标）
- 仅 5 个参数，降低过拟合风险

### 三阶段优化器 (`StrategyOptimizer`)

```
Stage 1: 硬过滤 → 剔除明显无效的参数组合
Stage 2: Pareto 前沿搜索 → 多目标（收益 / 回撤 / Sharpe）
Stage 3: 参数稳定性验证 → 最优参数在附近区域的稳定性
```

---

## 完整数据流

```
[DATA ACQUISITION - 数据采集]
  DataService.pytdx_daily() / eastmoney_daily()
    → data/raw/etf/{code}.parquet  (OHLCV only, ~105 ETFs)

[PREPROCESSING - 数据准备]
  ResearchDataService.build_all()
    ├── IndicatorEngine.calculate()          → +35 time-series factors
    ├── FactorEngine cross computations       → +4 cross-sectional factors
    ├── MarketRegimeDetector.detect_smooth()  → regime + regime_raw
    └── MarketRegimeDetector.calc_market_score() → breadth + market_score
    → data/research/{code}.parquet  (OHLCV + ~45 indicators)

[FACTOR RESEARCH - 因子研究]
  FactorResearchWorker (QRunnable, 后台线程)
    ├── load_many(codes) → long-format DataFrame
    ├── FactorResearchEngine.run_full_research() → 7-stage study
    └── save → factor_metrics.parquet + factor_cache/*.parquet
  → UI: Summary | Detail | Correlation | Approval tabs

[HUMAN REVIEW - 人工决策]
  FactorApproval
    ├── 审批：哪些因子启用 / 排除
    ├── 审批：regime → group 映射
    └── save → data/research/approved_factors.json

[STRATEGY BUILDING - 策略构建]
  on_optimize_strategy()
    ├── read approved_factors.json  (如果有)
    ├── build_backtest_data(codes)
    │     ├── load research parquets
    │     ├── _z_* cross-sectional zscores   (pre-computed once)
    │     ├── _z_beta_penalty               (bear-market structure)
    │     └── FactorMonitor.compute_weights() → dw_* columns
    ├── build_strategy(params)
    │     ├── generate group score expressions
    │     ├── generate regime-conditional score_expr
    │     └── generate explain_groups metadata
    └── StrategyOptimizer → Pareto search → best strategy JSON

[BACKTEST - 回测]
  BacktestEngine.run(strategy, data)
    ├── daily: ExpressionEngine.eval(score_expr, df_subset)
    ├── position optimization (HRP with volatility target)
    ├── exit signal check
    └── output → equity curve + trade logs + performance metrics
```

---

## 文件索引

| 文件 | 职责 | 关键类/函数 |
|---|---|---|
| `factor_registry.py` | 因子唯一定义源 + 审批管理 | `FACTOR_REGISTRY`, `FactorApproval` |
| `indicators.py` | 时间序列因子计算（30+ 指标） | `IndicatorEngine.calculate()` |
| `factors.py` | 横截面因子计算（4 个因子） | `FactorEngine` |
| `factor_lab.py` | 因子分析基础引擎（5 个分析方法） | `FactorLab` |
| `factor_research_engine.py` | 全量因子研究引擎（7 阶段） | `FactorResearchEngine.run_full_research()` |
| `factor_cache.py` | 研究结果缓存与校验 | `FactorCache` |
| `factor_monitor.py` | 动态 IC 权重（IC→ICIR→dw_*） | `FactorMonitor.compute_weights()` |
| `market_regime.py` | 市场状态检测 + 市场评分 | `MarketRegimeDetector` |
| `factor_expression.py` | 安全表达式求值引擎 | `ExpressionEngine.eval()` |
| `data_service.py` | 数据下载 / 存储 / 研究管线 | `DataService`, `ResearchDataService` |
| `backtest_engine.py` | 回测引擎（选股+持仓+风控） | `BacktestEngine.run()` |
| `strategy_engine.py` | 策略 CRUD（JSON 序列化） | `StrategyEngine` |
| `strategy_optimizer.py` | 策略参数优化器 | `StrategyOptimizer` |
| `main_window.py` | GUI 主界面 | `MainWindow`, `FactorResearchWorker` |
