from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import QDesktopServices, QColor
from PySide6.QtCore import QUrl

import os
import re
import shutil
import pandas as pd
import numpy as np
import pyqtgraph as pg
from datetime import datetime, timedelta

from data_service import DataService
from indicators import IndicatorEngine
from factors import FactorEngine
from downloader import DownloadWorker

from chart_widget import ChartWidget
from strategy_engine import StrategyEngine
from backtest_engine import BacktestEngine
from backtest_widget import BacktestChartWidget, BacktestWorker
from utils import now_str
from strategy_optimizer import staged_optimization
from optimizer_worker import OptimizerWorker
from market_regime import MarketRegimeDetector
from factor_lab import FactorLab


# ==========================================
# 指标 / 因子说明
# ==========================================

FACTOR_EXPLAIN = {

    # ━━━━━━━━━━━━━━━━ 趋势类 ━━━━━━━━━━━━━━━━
    "MA":
        "【移动平均线 MA】\n"
        "计算方法：过去N日收盘价的算术平均值。MA5/MA10/MA20/MA60。\n"
        "  MA5 > MA20 > MA60 → 多头排列，趋势向上\n"
        "  MA5 < MA20 < MA60 → 空头排列，趋势向下\n"
        "  均线金叉（短线上穿长线）→ 买入信号\n"
        "  价格运行在MA20上方 → 中期趋势偏多\n"
        "  价格运行在MA20下方 → 中期趋势偏空\n"
        "  均线斜率越大 → 趋势越强\n"
        "实践中：MA20是最常用的趋势判断基准线，牛市回调不破MA20。",

    "MACD":
        "【指数平滑异同移动平均线 MACD】\n"
        "计算方法：DIF=EMA12-EMA26, DEA=DIF的9日EMA, 柱=(DIF-DEA)×2。\n"
        "  DIF>0且DEA>0 → 多头主导\n"
        "  DIF<0且DEA<0 → 空头主导\n"
        "  柱线由负转正(金叉) → 买入信号\n"
        "  柱线由正转负(死叉) → 卖出信号\n"
        "  柱线高度持续放大 → 趋势加速\n"
        "  柱线萎缩/背离 → 趋势衰竭预警\n"
        "实践中：MACD是动量+趋势复合指标，适合中长线判断，\n"
        "短线来回穿越时信号噪音多。",

    "BOLL":
        "【布林带 BOLL】\n"
        "计算方法：中轨=MA20, 上轨=中轨+2σ, 下轨=中轨-2σ。\n"
        "  价格触上轨 → 短线偏高，可能回落\n"
        "  价格触下轨 → 短线偏低，可能反弹\n"
        "  带宽收窄(BOLL squeeze) → 蓄势待发，即将选择方向\n"
        "  带宽放大 → 波动加剧，趋势行情启动\n"
        "  价格沿上轨上行 → 强势多头（顺势不摸顶）\n"
        "  价格沿下轨下行 → 强势空头\n"
        "实践中：缩口+放量是最经典的突破信号，\n"
        "不要简单地'触上轨就卖'，强势行情会贴着上轨走。",

    # ━━━━━━━━━━━━━━━━ 动量类 ━━━━━━━━━━━━━━━━
    "ROC":
        "【变动率 ROC (Rate of Change)】\n"
        "计算方法：ROC=(今日收盘价/N日前收盘价-1)×100。\n"
        "系统计算ROC12/ROC20/ROC60/ROC120四个周期。\n"
        "  ROC>0 → 当前价格高于N日前，处于上涨\n"
        "  ROC<0 → 当前价格低于N日前，处于下跌\n"
        "  ROC20>ROC60 → 中期动量加速（短期强于中期）\n"
        "  ROC持续为正且扩大 → 趋势走强\n"
        "  ROC从高位回落 → 动量衰减\n"
        "实践中：ROC是ETF轮动最核心的因子之一。\n"
        "ROC20适合判断月线级别方向，ROC60/120用于判断长趋势。\n"
        "注意：高ROC也意味着追高风险，需结合波动率使用。",

    "ma_ratio_5_20":
        "【均线乖离率 MA Ratio 5/20】\n"
        "计算方法：MA5/MA20 - 1，即5日均线与20日均线的比值偏移。\n"
        "  >0 → 短期均线在中期均线上方，近期走势偏强\n"
        "  <0 → 短期均线在中期均线下方，近期走势偏弱\n"
        "  数值越大 → 短期偏离中期越远\n"
        "实践中：此因子捕捉短期趋势相对中期趋势的偏离程度，\n"
        "配合ROC和ma_distance使用可判断趋势结构和力度。",

    # ━━━━━━━━━━━━━━━━ 收益率类 ━━━━━━━━━━━━━━━━
    "ret":
        "【日收益率 Return】\n"
        "计算方法：ret=今日收盘价/昨日收盘价-1（小数表示）。\n"
        "  ret5 = 过去5日累计收益率\n"
        "  ret20 = 过去20日累计收益率\n"
        "  ret>0 → 价格上涨\n"
        "  ret<0 → 价格下跌\n"
        "实践中：ret是计算波动率、相关性等衍生因子的基础变量。\n"
        "ret20用于计算相对沪深300强弱，ret5用于捕捉超短期动量。",

    # ━━━━━━━━━━━━━━━━ 震荡/超买超卖类 ━━━━━━━━━━━━━━━━
    "KDJ":
        "【随机指标 KDJ】\n"
        "计算方法：RSV=(C-L9)/(H9-L9)×100, K/D为其平滑, J=3K-2D。\n"
        "  K>80 → 超买区域，不宜追高\n"
        "  K<20 → 超卖区域，不宜杀跌\n"
        "  J>100 → 严重超买，回调概率高\n"
        "  J<0 → 严重超卖，反弹概率高\n"
        "  K上穿D(金叉) → 买入信号\n"
        "  K下穿D(死叉) → 卖出信号\n"
        "实践中：KDJ是短线指标，震荡市中金叉/死叉较有效，\n"
        "单边趋势中容易钝化失效，需结合趋势指标使用。",

    "RSI":
        "【相对强弱指数 RSI】\n"
        "计算方法：RSI=100-100/(1+RS), RS=N日平均涨幅/N日平均跌幅。\n"
        "系统计算RSI6(短期)、RSI12(中期)、RSI24(长期)。\n"
        "  >70 → 超买，短期可能回调\n"
        "  <30 → 超卖，短期可能反弹\n"
        "  =50 → 多空均衡\n"
        "  RSI6上穿RSI24 → 短期转强信号\n"
        "  底背离(价格新低+RSI不新低) → 见底信号\n"
        "  顶背离(价格新高+RSI不新高) → 见顶信号\n"
        "实践中：RSI在震荡行情中效果好，趋势行情中容易长期超买/超卖。\n"
        "与KDJ互补：RSI偏重价格强度，KDJ偏重价格位置。",

    "CCI":
        "【顺势指标 CCI】\n"
        "计算方法：TP=(H+L+C)/3, CCI=(TP-MA14)/(0.015×MD14)。\n"
        "  >+100 → 价格极端偏高，特殊强势\n"
        "  <-100 → 价格极端偏低，特殊弱势\n"
        "  从-100下方上穿-100 → 买入信号\n"
        "  从+100上方下穿+100 → 卖出信号\n"
        "  绝对值越大 → 价格偏离统计均值越远\n"
        "实践中：CCI没有上下限(理论上可到±∞)，比RSI更敏感。\n"
        "适合于捕捉极端行情和趋势加速信号。",

    # ━━━━━━━━━━━━━━━━ 波动率类 ━━━━━━━━━━━━━━━━
    "ATR":
        "【平均真实波幅 ATR】\n"
        "计算方法：TR=max(H-L,|H-昨收|,|L-昨收|), ATR=TR的N日均值。\n"
        "系统计算ATR14和ATR20。\n"
        "  ATR越大 → 日间波动幅度越大，风险越高\n"
        "  ATR越低 → 市场波动平淡，可能处于盘整\n"
        "  ATR突然跳升 → 可能即将出现大行情\n"
        "  ATR持续下降 → 波动收窄蓄势\n"
        "实践中：ATR是最实用的波动率指标，常用于：\n"
        "①设置止损（如：2×ATR移动止损）②判断市场活跃度\n"
        "③仓位大小调整(ATR大→仓位小)。ATR是绝对值，跨标不直接可比。",

    "VOLATILITY":
        "【历史波动率 Volatility】\n"
        "计算方法：Volatility20=日收益率20日标准差×√252(年化)。\n"
        "  volatility20越大 → 近期价格波动越剧烈\n"
        "  volatility20越小 → 价格越稳定\n"
        "  牛市中volatility20往往较低(稳步上涨)\n"
        "  恐慌/见顶时volatility20往往飙高\n"
        "实践中：在ETF轮动中，低波动策略是有效因子。\n"
        "熊市重低波，牛市低波反而意味没有进攻性。",

    "downside_volatility":
        "【下行波动率 Downside Volatility】\n"
        "计算方法：仅取日收益率<0的部分，20日标准差×√252(年化)。\n"
        "  >volatility20 → 近期下跌波动大于上涨波动（偏空）\n"
        "  <volatility20 → 近期上涨波动更多（偏正）\n"
        "  越大 → 近期下跌风险越大\n"
        "实践中：相比总波动率，下行波动率更专注风险侧。\n"
        "熊市中downside_volatility大幅高于volatility20是危险信号。\n"
        "典型用法：downside_vol/volatility20的比值>0.7说明风险不对称。",

    # ━━━━━━━━━━━━━━━━ 量价类 ━━━━━━━━━━━━━━━━
    "OBV":
        "【能量潮 OBV】\n"
        "计算方法：上涨日累加成交量，下跌日减去成交量。\n"
        "  价升OBV升 → 量价配合良好，趋势健康\n"
        "  价升OBV降 → 量价背离，上涨动力不足\n"
        "  价跌OBV升 → 可能有资金暗中吸筹\n"
        "  价跌OBV降 → 量价配合下跌，趋势延续\n"
        "实践中：OBV本身是绝对数值不适合跨标比，\n"
        "但OBV方向与价格方向的背离/配合是有效的量价验证信号。",

    # ━━━━━━━━━━━━━━━━ 资金流类 ━━━━━━━━━━━━━━━━
    "volume_ratio":
        "【成交量比 Volume Ratio / vol_ratio20】\n"
        "计算方法：当日成交量÷20日平均成交量。\n"
        "  >1 → 成交量高于平均水平，交易活跃\n"
        "  >2 → 显著放量，可能有重要资金进出\n"
        "  <0.5 → 极度缩量，市场冷清\n"
        "  价涨+vol_ratio>1.5 → 放量上涨=趋势确认\n"
        "  价跌+vol_ratio>1.5 → 放量下跌=恐慌抛售\n"
        "  价涨+vol_ratio<0.5 → 缩量上涨=追高意愿不足\n"
        "实践中：vol_ratio是判断突破有效性的关键指标。\n"
        "真正的突破必须伴随放量(vol_ratio>1.5)，否则可能是假突破。",

    "turnover_change":
        "【换手率变化 Turnover Change】\n"
        "计算方法：(5日均量÷20日均量-1)×100。\n"
        "  >0 → 近期成交量趋势上升，资金参与度提升\n"
        "  <0 → 近期成交量萎缩，资金参与度下降\n"
        "  持续>20% → 明显有增量资金入场\n"
        "  持续<-20% → 市场交投清淡\n"
        "实践中：适用于判断资金流向趋势而非绝对活跃度。\n"
        "配合vol_ratio使用：turnover_change看趋势，vol_ratio看当日。",

    # ━━━━━━━━━━━━━━━━ 趋势位置类 ━━━━━━━━━━━━━━━━
    "ma_distance":
        "【均线偏离度 MA Distance】\n"
        "计算方法：(收盘价÷MA20-1)×100，百分比表示。\n"
        "  >0且较大 → 涨幅已大，短期获利盘压力\n"
        "  <0且较大 → 超跌状态，可能技术反弹\n"
        "  ~0 → 价格在均线附近震荡\n"
        "  |ma_distance|>5% → 严重偏离均线，回归概率增大\n"
        "实践中：均线引力是A股特征之一。\n"
        "但强势行情ma_distance可以持续>5%，不要机械等待回归。\n"
        "震荡市中ma_distance在±3%范围内是常态。",

    "breakout20":
        "【20日突破度 Breakout20】\n"
        "计算方法：(收盘价÷20日最高价-1)×100。\n"
        "  =0 → 价格正好等于20日高点\n"
        "  >0 → 创20日新高，突破发生\n"
        "  <0 → 距20日高点仍有距离\n"
        "  越大 → 突破力度越强，上升空间打开\n"
        "实践中：breakout20>0配合vol_ratio>1.5是有效突破。\n"
        "连续多日>0且数值扩大是强势上涨特征。",

    "low52w":
        "【52周低点距离 Low52w】\n"
        "计算方法：(收盘价÷252日最低价-1)。\n"
        "  值=0.1 → 距52周低点上涨10%，已经脱离低点\n"
        "  值接近0 → 价格接近52周低点\n"
        "实践中：low52w<5%意味着标的正接近年内低点，\n"
        "可能是超跌反弹机会，也可能是弱势延续。需结合趋势判断。",

    # ━━━━━━━━━━━━━━━━ 相对强弱类 ━━━━━━━━━━━━━━━━
    "relative_strength_vs_hs300":
        "【相对沪深300强弱 RS vs HS300】\n"
        "计算方法：(标的ret20-沪深300(510300)的ret20)×100。\n"
        "  >0 → 20日表现优于沪深300，相对强势\n"
        "  <0 → 20日表现弱于沪深300，相对弱势\n"
        "  持续>3% → 标的显著跑赢大盘\n"
        "  持续<-3% → 标的显著跑输大盘\n"
        "实践中：这是ETF轮动中判断板块强弱的核心因子之一。\n"
        "熊市中选RS>0的板块更能抗跌，牛市中选RS持续扩大的板块\n"
        "能抓住领涨品种。注意：RS为正不代表绝对赚钱。",

    # ━━━━━━━━━━━━━━━━ 市场宽度类 ━━━━━━━━━━━━━━━━
    "breadth":
        "【市场宽度 Market Breadth】\n"
        "计算方法：当日上涨ETF数÷总ETF数，取5日平滑。\n"
        "  >0.6 → 六成以上ETF上涨，市场整体乐观\n"
        "  <0.4 → 不足四成上涨，市场整体偏弱\n"
        "  从>0.7快速回落 → 上涨动能耗尽，警惕拐点\n"
        "  持续<0.3 → 普跌行情，宜空仓或防御\n"
        "实践中：breadth是比MA200更灵敏的短周期市场温度计。\n"
        "breadth先于指数转弱是常见的顶部信号(指数不跌但宽度收窄)。\n"
        "在regime系统中与长周期MA判断互补使用。",

    "ma20_distance":
        "【大盘MA20偏离度(基准检测)】\n"
        "计算方法：(沪深300收盘价÷沪深300MA20-1)×100。\n"
        "  >0 → 大盘在MA20上方，短期趋势健康\n"
        "  <0 → 大盘在MA20下方，短期偏弱\n"
        "  此指标用于market_regime模块的短周期信号检测，\n"
        "配合长周期MA50/MA200判断市场状态。\n"
        "实践中：当ma20_distance<0时需提高警惕，\n"
        "连续多日<0且扩大说明短期下跌趋势在加强。",

    # ━━━━━━━━━━━━━━━━ 回撤类 ━━━━━━━━━━━━━━━━
    "max_drawdown20":
        "【20日最大回撤 Max Drawdown20】\n"
        "计算方法：20日内(收盘价÷期间最高价-1)的最小值。\n"
        "  =0 → 期间持续创新高，无回撤（最强）\n"
        "  =-5% → 期间最大回调幅度5%\n"
        "  =-15% → 期间大幅回撤，风险高\n"
        "  越接近0 → 价格越稳定，风险越小（越好）\n"
        "  负值越大 → 回撤越深，风险越大（越差）\n"
        "实践中：熊市区重点使用，rank(-abs(max_drawdown20))选出\n"
        "回撤最小的ETF作为防御持仓。不要直接用rank(max_drawdown20)，\n"
        "那会选出回撤最大的ETF（逻辑反了！）。",

    # ━━━━━━━━━━━━━━━━ 横截面因子类 ━━━━━━━━━━━━━━━━
    "Cross Momentum":
        "【横截面动量 Cross Momentum】\n"
        "计算方法：每日对全部ETF的ROC12进行0~1百分位排名。\n"
        "  排名>0.8 → 动量在全市场中处于前20%\n"
        "  排名<0.2 → 动量处于后20%\n"
        "实践中：Cross Momentum是ETF轮动最经典的横截面因子，\n"
        "在A股市场中期动量效应明显，1-3个月周期效果最好。",

    "cross_volatility":
        "【横截面波动率 Cross Volatility】\n"
        "计算方法：每日对全部ETF的volatility20进行0~1百分位排名。\n"
        "  排名>0.8 → 高波动品种(波动率处于全市场前20%)\n"
        "  排名<0.2 → 低波动品种\n"
        "实践中：低波动因子在A股ETF中有显著的防御效果，\n"
        "熊市中选低cross_volatility的ETF能有效控制回撤。",

    "Barra Beta":
        "【Barra Beta 市场敏感度】\n"
        "计算方法：过去60日标的收益率vs市场收益率线性回归的斜率。\n"
        "  >1 → 波动大于市场（进攻型，牛市有利）\n"
        "  <1 → 波动小于市场（防御型，熊市有利）\n"
        "  <0 → 与市场反向（对冲属性）\n"
        "实践中：Beta本身不是选股因子，而是风险暴露度量。\n"
        "牛市选高Beta，熊市选低Beta是经典的风格切换逻辑。",

    # ━━━━━━━━━━━━━━━━ 扩展因子（正交/结构性） ━━━━━━━━━━━━━━━━
    "corr_hs300_60":
        "【沪深300相关性 Corr HS300 60】\n"
        "计算方法：ETF日收益率与沪深300(510300)日收益率的60日滚动相关系数。\n"
        "  →1.0 → 与大盘高度同步，Beta型ETF\n"
        "  →0.2 → 与大盘弱相关，独立行情ETF\n"
        "  为正 → 市场多空方向一致\n"
        "  为负 → 市场反向（如债券/黄金ETF）\n"
        "实践中：ETF轮动的本质是风险暴露切换。\n"
        "牛市选高corr(进攻)，熊市选低corr(防御)，震荡市选中间。",

    "volatility_ratio":
        "【波动率压缩比 Volatility Ratio】\n"
        "计算方法：volatility20 ÷ volatility120。\n"
        "  <0.7 → 波动率急剧收窄，蓄势待发（常见于大行情前）\n"
        "  >1.3 → 波动率急剧放大，可能进入恐慌或趋势加速\n"
        "  0.8~1.2 → 波动模式正常\n"
        "实践中：vol_ratio是vol regime切换信号，\n"
        "缩量+波动压缩+突破是最强信号组合。",

    "trend_consistency":
        "【趋势一致性 Trend Consistency】\n"
        "计算方法：roc20>0 + roc60>0 + roc120>0，取值0~3。\n"
        "  3 → 三个周期全部上涨，趋势高度一致（最稳定）\n"
        "  2 → 两多一空，趋势有裂痕\n"
        "  1 → 一多两空，趋势接近反转\n"
        "  0 → 全空，熊市特征明显\n"
        "实践中：trend_consistency=3时持有胜率最高，\n"
        "比单一ROC更能过滤假突破。配合regime使用效果更好。",
}


# ==========================================
# 因子正交分类系统
# 每类因子经济含义不同，优化器按类别约束搜索，避免同质因子重复加杠杆
# ==========================================
FACTOR_GROUPS = {
    "trend": {      # 趋势/动量
        "name": "趋势",
        "factors": ["roc20", "roc60", "breakout20", "relative_strength_vs_hs300"],
        "sign": +1,  # 牛市正向、熊市惩罚
    },
    "risk": {       # 波动/回撤（通常为惩罚项）
        "name": "风险",
        "factors": ["volatility20", "downside_volatility", "max_drawdown20", "atr20"],
        "sign": -1,
    },
    "volume": {     # 资金/成交量
        "name": "资金",
        "factors": ["turnover_change", "vol_ratio20"],
        "sign": +1,
    },
    "structure": {  # 结构/相关性
        "name": "结构",
        "factors": ["barra_beta", "corr_hs300_60"],
        "sign": None,  # 牛市偏好高beta；熊市偏好beta≈1
    },
    "consistency": {  # 趋势一致性
        "name": "一致性",
        "factors": ["trend_consistency"],
        "sign": +1,
    },
    "mean_reversion": {  # 均值回复（震荡市专用）
        "name": "均值回复",
        "factors": ["ma_distance"],
        "sign": None,  # 震荡市偏好负偏离（超跌反弹）
    },
}


class MainWindow(QMainWindow):

    def __init__(self):

        super().__init__()

        self.setWindowTitle("Quant Qt")

        self.resize(1800, 1000)

        self.ds = DataService()

        # ResearchDataService：预计算研究数据（回测即插即用）
        from data_service import ResearchDataService
        self.research = ResearchDataService(self.ds)

        # 回测数据缓存（避免每次都重新构建）
        self._bt_data_cache: pd.DataFrame | None = None
        self._bt_data_cache_key: str = ""

        self.ind = IndicatorEngine()
        self.factor = FactorEngine()

        self.stg = StrategyEngine()

        self.bt = BacktestEngine()

        self.pool = QThreadPool()

        self.worker = None

        # 保持对后台优化任务的引用，避免 QRunnable 被GC导致 finished 信号丢失
        self.opt_worker = None

        self.init_ui()

        self.set_dark_theme()

    # ==========================================
    # UI
    # ==========================================

    def init_ui(self):

        root = QWidget()

        self.setCentralWidget(root)

        layout = QVBoxLayout(root)

        # =====================================
        # 顶栏
        # =====================================

        top = QWidget()

        top.setFixedHeight(60)

        top_layout = QHBoxLayout(top)

        title = QLabel("Quant Qt")

        title.setStyleSheet("""
            font-size:28px;
            font-weight:bold;
            color:#40c4ff;
        """)

        top_layout.addWidget(title)

        top_layout.addStretch()

        layout.addWidget(top)

        # =====================================
        # 分页
        # =====================================

        self.tabs = QTabWidget()

        layout.addWidget(self.tabs)

        # 数据下载
        self.tabs.addTab(
            self.build_download_page(),
            "数据下载"
        )

        # 指标计算
        self.tabs.addTab(
            self.build_indicator_page(),
            "指标 / 因子计算"
        )

        # 可视化
        self.tabs.addTab(
            self.build_chart_page(),
            "可视化分析"
        )

        # 策略构建
        self.tabs.addTab(
            self.build_strategy_page(),
            "策略构建"
        )

        # 策略回测
        self.tabs.addTab(
            self.build_backtest_page(),
            "策略回测"
        )

        # 因子研究
        self.tabs.addTab(
            self.build_factor_lab_page(),
            "因子研究"
        )


    # ==========================================
    # 因子研究页面
    # ==========================================

    def build_factor_lab_page(self):

        page = QWidget()
        layout = QVBoxLayout(page)

        title = QLabel("因子研究 — 滚动RankIC · IC衰减 · 分位数收益 · 相关性 · 状态IC")
        title.setStyleSheet("font-size:20px;font-weight:bold;")
        layout.addWidget(title)

        # ── 控制栏 ──
        ctrl = QHBoxLayout()

        ctrl.addWidget(QLabel("因子："))
        self.fl_factor_checks = {}
        from factor_monitor import GROUP_REPRESENTATIVES
        for group, rep in GROUP_REPRESENTATIVES.items():
            cb = QCheckBox(f"{group}({rep})")
            cb.setChecked(True)
            self.fl_factor_checks[rep] = cb
            ctrl.addWidget(cb)

        ctrl.addSpacing(16)
        ctrl.addWidget(QLabel("IC前瞻期："))
        self.fl_horizon = QSpinBox()
        self.fl_horizon.setRange(1, 60)
        self.fl_horizon.setValue(5)
        ctrl.addWidget(self.fl_horizon)

        ctrl.addWidget(QLabel("滚动窗口："))
        self.fl_window = QSpinBox()
        self.fl_window.setRange(20, 252)
        self.fl_window.setValue(60)
        ctrl.addWidget(self.fl_window)

        ctrl.addWidget(QLabel("分位数："))
        self.fl_n_quantiles = QSpinBox()
        self.fl_n_quantiles.setRange(3, 10)
        self.fl_n_quantiles.setValue(5)
        ctrl.addWidget(self.fl_n_quantiles)

        self.fl_compute_btn = QPushButton("开始计算")
        self.fl_compute_btn.clicked.connect(self._on_fl_compute)
        ctrl.addWidget(self.fl_compute_btn)

        ctrl.addStretch()
        layout.addLayout(ctrl)

        # ── 图表子选项卡 ──
        self.fl_chart_tabs = QTabWidget()
        layout.addWidget(self.fl_chart_tabs, stretch=1)

        # Rolling IC
        self.fl_ic_plot = pg.PlotWidget(title="Rolling RankIC")
        self.fl_ic_plot.setLabel("left", "|IC|")
        self.fl_ic_plot.setLabel("bottom", "日期")
        self.fl_ic_plot.addLegend()
        self.fl_ic_plot.showGrid(x=True, y=True, alpha=0.3)
        self.fl_chart_tabs.addTab(self.fl_ic_plot, "Rolling IC")

        # IC Decay
        self.fl_decay_plot = pg.PlotWidget(title="IC Decay")
        self.fl_decay_plot.setLabel("left", "Mean |IC|")
        self.fl_decay_plot.setLabel("bottom", "Horizon (d)")
        self.fl_decay_plot.addLegend()
        self.fl_decay_plot.showGrid(x=True, y=True, alpha=0.3)
        self.fl_chart_tabs.addTab(self.fl_decay_plot, "IC Decay")

        # Quantile Return
        self.fl_quantile_plot = pg.PlotWidget(title="Quantile Forward Returns")
        self.fl_quantile_plot.setLabel("left", "Mean Fwd Return")
        self.fl_quantile_plot.setLabel("bottom", "日期")
        self.fl_quantile_plot.addLegend()
        self.fl_quantile_plot.showGrid(x=True, y=True, alpha=0.3)
        self.fl_chart_tabs.addTab(self.fl_quantile_plot, "分位数收益")

        # Factor Correlation Heatmap
        self.fl_corr_plot = pg.PlotWidget(title="因子截面相关性矩阵")
        self.fl_corr_plot.setLabel("left", "")
        self.fl_corr_plot.setLabel("bottom", "")
        self.fl_chart_tabs.addTab(self.fl_corr_plot, "因子相关性")

        # Regime IC
        self.fl_regime_plot = pg.PlotWidget(title="Regime IC")
        self.fl_regime_plot.setLabel("left", "Mean |IC|")
        self.fl_regime_plot.setLabel("bottom", "Regime")
        self.fl_regime_plot.addLegend()
        self.fl_regime_plot.showGrid(x=True, y=True, alpha=0.3)
        self.fl_chart_tabs.addTab(self.fl_regime_plot, "Regime IC")

        # ── 汇总表格 ──
        self.fl_summary_table = QTableWidget()
        self.fl_summary_table.setMinimumHeight(120)
        layout.addWidget(self.fl_summary_table)

        return page

    def _on_fl_compute(self):
        """执行因子研究计算并绘制结果。"""
        selected = [f for f, cb in self.fl_factor_checks.items() if cb.isChecked()]
        if not selected:
            QMessageBox.warning(self, "提示", "请至少选择一个因子")
            return

        horizon = self.fl_horizon.value()
        window = self.fl_window.value()
        n_q = self.fl_n_quantiles.value()
        self.fl_compute_btn.setEnabled(False)
        self.fl_compute_btn.setText("计算中...")
        QApplication.processEvents()

        try:
            # 加载数据
            codes = [x["code"] for x in self.ds.get_all_etf()]
            data = self.research.load_many(codes, start="2015-01-01")
            if data.empty:
                QMessageBox.warning(self, "提示", "没有可用的研究数据")
                return

            lab = FactorLab(data)

            # ── Rolling IC ──
            ic_df = lab.compute_rolling_ic(factors=selected, horizon=horizon, window=window)
            self.fl_ic_plot.clear()
            self.fl_ic_plot.addLegend()
            if not ic_df.empty:
                colors = ["#40c4ff", "#ffd54f", "#00e676", "#ff4081", "#ffab40"]
                for idx, col in enumerate([c for c in ic_df.columns if c != "date"]):
                    y = ic_df[col].values
                    x = ic_df["date"].values.astype("datetime64[s]").astype(int)
                    pen = pg.mkPen(color=colors[idx % len(colors)], width=1.5)
                    self.fl_ic_plot.plot(x, y, pen=pen, name=col.replace("ic_", ""))

            # ── IC Decay ──
            decay_df = lab.compute_ic_decay(factors=selected, horizons=[1, 5, 10, 20])
            self.fl_decay_plot.clear()
            self.fl_decay_plot.addLegend()
            if not decay_df.empty:
                colors = ["#40c4ff", "#ffd54f", "#00e676", "#ff4081", "#ffab40"]
                for idx, row in decay_df.iterrows():
                    factor = row["factor"]
                    x = [1, 5, 10, 20]
                    y = [row.get(f"h{h}", np.nan) for h in x]
                    pen = pg.mkPen(color=colors[idx % len(colors)], width=2)
                    sym = pg.ScatterPlotItem(x=x, y=y, pen=pen, brush=colors[idx % len(colors)],
                                              size=10, name=factor)
                    self.fl_decay_plot.addItem(sym)
                    self.fl_decay_plot.plot(x, y, pen=pen)

            # ── Quantile Return ──
            if selected:
                q_df = lab.compute_quantile_returns(factor=selected[0], horizon=horizon, n_quantiles=n_q)
                self.fl_quantile_plot.clear()
                self.fl_quantile_plot.addLegend()
                if not q_df.empty:
                    cmap = ["#2d8cf0", "#40c4ff", "#69f0ae", "#ffd54f", "#ff4081"]
                    for idx, col in enumerate(sorted(q_df.columns)):
                        if col.startswith("q"):
                            y = q_df[col].values
                            x = q_df.index.values.astype("datetime64[s]").astype(int)
                            pen = pg.mkPen(color=cmap[idx % len(cmap)], width=1.5)
                            self.fl_quantile_plot.plot(x, y, pen=pen, name=f"{selected[0]} {col}")

            # ── Factor Correlation ──
            corr_df = lab.compute_factor_correlation(factors=selected)
            self.fl_corr_plot.clear()
            if not corr_df.empty:
                n = len(corr_df)
                img = pg.ImageItem()
                img.setImage(corr_df.values)
                # Center the image at row/column indices
                img.setRect(-0.5, -0.5, n, n)
                # Custom color map: blue to white to red
                from pyqtgraph import ColorMap
                cmap = pg.colormap.get("coolwarm")
                if cmap is None:
                    lut_vals = []
                    for i in range(256):
                        r = int(255 * (i / 255.0))
                        g = int(255 * (1.0 - abs(i - 127.5) / 127.5))
                        b = int(255 * (1.0 - i / 255.0))
                        lut_vals.append([r, g, b])
                    cmap = ColorMap(np.array(lut_vals, dtype=np.ubyte))
                img.setLookupTable(cmap.getLookupTable())
                self.fl_corr_plot.addItem(img)

                # Labels
                tick_positions = [(i, name) for i, name in enumerate(corr_df.index)]
                ax = self.fl_corr_plot.getAxis("bottom")
                ax.setTicks([tick_positions])
                ay = self.fl_corr_plot.getAxis("left")
                ay.setTicks([tick_positions])

            # ── Regime IC ──
            regime_df = lab.compute_regime_ic(factors=selected, horizon=horizon)
            self.fl_regime_plot.clear()
            self.fl_regime_plot.addLegend()
            if not regime_df.empty:
                regimes_list = regime_df.columns.drop("factor", errors="ignore").tolist()
                bar_width = 0.15
                colors = ["#40c4ff", "#ffd54f", "#00e676", "#ff4081", "#ffab40"]
                for r_idx, regime_name in enumerate(regimes_list):
                    for f_idx, row in regime_df.iterrows():
                        factor_name = row["factor"]
                        val = row.get(regime_name, np.nan)
                        if not np.isnan(val):
                            x_pos = f_idx + r_idx * bar_width
                            bar = pg.BarGraphItem(
                                x=[x_pos], height=[val], width=bar_width,
                                brush=colors[r_idx % len(colors)], name=regime_name
                            )
                            self.fl_regime_plot.addItem(bar)

                # X-axis ticks at factor positions
                x_ticks = [(i + bar_width * (len(regimes_list) - 1) / 2,
                            regime_df["factor"].iloc[i]) for i in range(len(regime_df))]
                self.fl_regime_plot.getAxis("bottom").setTicks([x_ticks])

            # ── 汇总表格 ──
            self.fl_summary_table.clear()
            if not ic_df.empty:
                ic_cols = [c for c in ic_df.columns if c != "date"]
                # 最近的滚动IC值
                last_row = ic_df.dropna().iloc[-1] if len(ic_df.dropna()) > 0 else None
                if last_row is not None:
                    rows = len(ic_cols)
                    self.fl_summary_table.setRowCount(rows)
                    self.fl_summary_table.setColumnCount(4)
                    self.fl_summary_table.setHorizontalHeaderLabels(["因子", "最新RollingIC", "ICIR", "Mean IC (Decay h5)"])
                    for i, col in enumerate(ic_cols):
                        factor_name = col.replace("ic_", "")
                        ic_val = last_row.get(col, np.nan)
                        ic_series = ic_df[col].dropna()
                        icir = ic_series.mean() / ic_series.std() if len(ic_series) > 1 and ic_series.std() > 0 else np.nan
                        decay_h5 = np.nan
                        if not decay_df.empty:
                            dr = decay_df[decay_df["factor"] == factor_name]
                            if len(dr) > 0:
                                decay_h5 = dr["h5"].iloc[0]

                        self.fl_summary_table.setItem(i, 0, QTableWidgetItem(factor_name))
                        self.fl_summary_table.setItem(i, 1,
                            QTableWidgetItem(f"{ic_val:.4f}" if not np.isnan(ic_val) else "N/A"))
                        self.fl_summary_table.setItem(i, 2,
                            QTableWidgetItem(f"{icir:.4f}" if not np.isnan(icir) else "N/A"))
                        self.fl_summary_table.setItem(i, 3,
                            QTableWidgetItem(f"{decay_h5:.4f}" if not np.isnan(decay_h5) else "N/A"))

        except Exception as e:
            QMessageBox.warning(self, "计算错误", str(e))
        finally:
            self.fl_compute_btn.setEnabled(True)
            self.fl_compute_btn.setText("开始计算")

    # ==========================================
    # 数据下载页面
    # ==========================================

    def build_download_page(self):

        page = QWidget()

        layout = QVBoxLayout(page)

        title = QLabel("ETF历史数据下载")

        title.setStyleSheet("""
            font-size:24px;
            font-weight:bold;
        """)

        layout.addWidget(title)

        # =====================================
        # 控制栏
        # =====================================

        form = QGridLayout()

        self.start_edit = QLineEdit("20000101")

        self.end_edit = QLineEdit("20990101")

        form.addWidget(QLabel("开始日期"), 0, 0)

        form.addWidget(self.start_edit, 0, 1)

        form.addWidget(QLabel("结束日期"), 1, 0)

        form.addWidget(self.end_edit, 1, 1)

        layout.addLayout(form)

        # =====================================
        # ETF分类过滤
        # =====================================

        filter_layout = QHBoxLayout()
        
        filter_layout.addWidget(QLabel("分类筛选:"))
        
        self.category_combo = QComboBox()
        self.category_combo.addItem("全部")
        self.category_combo.addItems([
            "宽基指数", "科技/TMT", "新能源/碳中和", "医药/医疗",
            "消费", "金融", "工业/制造", "周期/资源", "红利/价值",
            "海外市场", "商品/另类", "债券", "主题/概念"
        ])
        self.category_combo.currentTextChanged.connect(self.filter_etf_by_category)
        
        filter_layout.addWidget(self.category_combo)
        
        # 全选/取消全选按钮
        self.select_all_btn = QPushButton("全选")
        self.select_all_btn.clicked.connect(self.select_all_etf)
        self.deselect_all_btn = QPushButton("取消全选")
        self.deselect_all_btn.clicked.connect(self.deselect_all_etf)
        
        filter_layout.addWidget(self.select_all_btn)
        filter_layout.addWidget(self.deselect_all_btn)
        filter_layout.addStretch()
        
        layout.addLayout(filter_layout)

        # =====================================
        # ETF多列表格
        # =====================================

        self.etf_table = QTableWidget()
        self.etf_table.setColumnCount(4)
        self.etf_table.setHorizontalHeaderLabels(["选择", "代码", "名称", "分类"])
        self.etf_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.etf_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.etf_table.setAlternatingRowColors(False)  # 关闭交替行颜色
        
        # 加载ETF数据
        self.all_etf_data = []
        self.etf_categories = {
            "宽基指数": [],
            "科技/TMT": [],
            "新能源/碳中和": [],
            "医药/医疗": [],
            "消费": [],
            "金融": [],
            "工业/制造": [],
            "周期/资源": [],
            "红利/价值": [],
            "海外市场": [],
            "商品/另类": [],
            "债券": [],
            "主题/概念": [],
        }
        
        self._load_etf_list()
        
        layout.addWidget(self.etf_table)

        # =====================================
        # 按钮行1：下载控制
        # =====================================

        btn_layout = QHBoxLayout()

        self.download_btn = QPushButton("开始下载")

        self.pause_btn = QPushButton("暂停")

        self.resume_btn = QPushButton("继续")

        self.stop_btn = QPushButton("停止")

        btn_layout.addWidget(self.download_btn)

        btn_layout.addWidget(self.pause_btn)

        btn_layout.addWidget(self.resume_btn)

        btn_layout.addWidget(self.stop_btn)

        layout.addLayout(btn_layout)

        # =====================================
        # 按钮行2：增量更新和定时更新
        # =====================================

        update_layout = QHBoxLayout()
        
        # 增量更新
        self.incremental_btn = QPushButton("增量更新最新数据")
        self.incremental_btn.setStyleSheet("background:#67c23a;")
        self.incremental_btn.setToolTip("只下载已有ETF的最新数据（从最后日期到今天）")
        
        # 定时更新设置
        update_layout.addWidget(self.incremental_btn)
        update_layout.addWidget(QLabel(" | "))
        
        update_layout.addWidget(QLabel("自动更新:"))
        
        self.auto_update_check = QCheckBox("启用")
        self.auto_update_check.setChecked(False)
        update_layout.addWidget(self.auto_update_check)
        
        update_layout.addWidget(QLabel("更新时间:"))
        
        self.update_time_edit = QTimeEdit()
        self.update_time_edit.setDisplayFormat("HH:mm")
        self.update_time_edit.setTime(QTime(18, 0))  # 默认18:00
        update_layout.addWidget(self.update_time_edit)
        
        update_layout.addWidget(QLabel("更新周期:"))
        
        self.update_interval_combo = QComboBox()
        self.update_interval_combo.addItems(["每日", "每周", "每月"])
        update_layout.addWidget(self.update_interval_combo)
        
        # 启动定时更新按钮
        self.start_timer_btn = QPushButton("启动定时")
        self.start_timer_btn.setCheckable(True)
        self.start_timer_btn.setStyleSheet("background:#e6a23c;")
        update_layout.addWidget(self.start_timer_btn)
        
        # 定时器状态显示
        self.timer_status_label = QLabel("定时器: 未启动")
        self.timer_status_label.setStyleSheet("color:#909399;")
        update_layout.addWidget(self.timer_status_label)
        
        update_layout.addStretch()
        
        layout.addLayout(update_layout)

        # =====================================
        # 进度条
        # =====================================

        self.progress = QProgressBar()

        layout.addWidget(self.progress)

        # =====================================
        # 日志
        # =====================================

        self.download_log = QTextEdit()

        self.download_log.setReadOnly(True)

        self.download_log.setMaximumHeight(150)

        layout.addWidget(self.download_log)

        # =====================================
        # 定时器
        # =====================================
        
        self.auto_update_timer = QTimer()
        self.auto_update_timer.timeout.connect(self.check_auto_update)
        self.auto_update_timer.setInterval(60000)  # 每分钟检查一次
        self.last_update_date = None
        
        # =====================================
        # 信号
        # =====================================

        self.download_btn.clicked.connect(
            self.download_data
        )

        self.pause_btn.clicked.connect(
            self.pause_download
        )

        self.resume_btn.clicked.connect(
            self.resume_download
        )

        self.stop_btn.clicked.connect(
            self.stop_download
        )
        
        self.incremental_btn.clicked.connect(
            self.incremental_update
        )
        
        self.start_timer_btn.clicked.connect(
            self.toggle_auto_update_timer
        )

        return page

    # ==========================================
    # 加载ETF列表到表格
    # ==========================================
    
    def _load_etf_list(self):
        """加载ETF列表到表格，支持分类"""
        etfs = self.ds.get_all_etf()
        self.all_etf_data = etfs
        
        # 分类ETF
        current_category = None
        category_idx = 0
        categories = list(self.etf_categories.keys())
        
        for item in etfs:
            code = item["code"]
            name = item["name"]
            
            # 根据代码判断分类
            cat = self._get_etf_category(code, name)
            item["category"] = cat
            
            if cat in self.etf_categories:
                self.etf_categories[cat].append(item)
        
        # 显示所有ETF
        self._populate_table(etfs)
    
    def _get_etf_category(self, code, name):
        """根据ETF代码和名称判断分类"""
        name_upper = name.upper()
        
        # 宽基指数
        if any(k in name for k in ["沪深300", "上证50", "中证500", "中证1000", 
                                    "创业板", "科创50", "深证100", "双创50"]):
            return "宽基指数"
        
        # 科技/TMT
        if any(k in name for k in ["芯片", "半导体", "AI", "科技", "通信", "传媒", 
                                    "电子", "云计算", "信创", "人工智能", "大数据", "计算机"]):
            return "科技/TMT"
        
        # 新能源/碳中和
        if any(k in name for k in ["光伏", "新能源", "储能", "电池", "碳中和", 
                                    "新能源车", "锂电"]):
            return "新能源/碳中和"
        
        # 医药/医疗
        if any(k in name for k in ["医药", "医疗", "创新药", "生物", "器械", "中药"]):
            return "医药/医疗"
        
        # 消费
        if any(k in name for k in ["酒", "食品", "消费", "家电", "旅游", "餐饮", "房地产"]):
            return "消费"
        
        # 金融
        if any(k in name for k in ["证券", "银行", "非银", "券商", "保险"]):
            return "金融"
        
        # 工业/制造
        if any(k in name for k in ["基建", "军工", "装备", "工业", "环保", "建筑"]):
            return "工业/制造"
        
        # 周期/资源
        if any(k in name for k in ["有色", "稀土", "钢铁", "煤炭", "能源", "资源", "化工", "建材"]):
            return "周期/资源"
        
        # 红利/价值
        if any(k in name for k in ["红利", "央企", "价值"]):
            return "红利/价值"
        
        # 海外市场
        if any(k in name for k in ["纳指", "标普", "恒生", "港股", "德国", "日经", "法国"]):
            return "海外市场"
        
        # 商品/另类
        if any(k in name for k in ["黄金", "有色"]):
            return "商品/另类"
        
        # 债券
        if any(k in name for k in ["国债", "城投", "公司债", "债券"]):
            return "债券"
        
        return "主题/概念"
    
    def _populate_table(self, etfs):
        """填充表格数据"""
        self.etf_table.setRowCount(len(etfs))
        
        # 分类颜色配置（柔和色调，仅用于分类标签列）
        cat_colors = {
            "宽基指数": ("#E3F2FD", "#1976D2"),      # 浅蓝背景，深蓝文字
            "科技/TMT": ("#E8F5E9", "#388E3C"),      # 浅绿背景，深绿文字
            "新能源/碳中和": ("#FFF3E0", "#F57C00"),  # 浅橙背景，深橙文字
            "医药/医疗": ("#FCE4EC", "#C2185B"),      # 浅粉背景，深粉文字
            "消费": ("#F3E5F5", "#7B1FA2"),          # 浅紫背景，深紫文字
            "金融": ("#E0F7FA", "#0097A7"),          # 浅青背景，深青文字
            "工业/制造": ("#E8EAF6", "#303F9F"),      # 浅靛蓝背景，深靛蓝文字
            "周期/资源": ("#FFF8E1", "#FFA000"),      # 浅琥珀背景，深琥珀文字
            "红利/价值": ("#E0F2F1", "#00796B"),      # 浅青绿背景，深青绿文字
            "海外市场": ("#E1F5FE", "#0288D1"),      # 浅天蓝背景，深天蓝文字
            "商品/另类": ("#FBE9E7", "#D84315"),      # 浅深橙背景，深深橙文字
            "债券": ("#F1F8E9", "#689F38"),          # 浅浅绿背景，深浅绿文字
            "主题/概念": ("#ECEFF1", "#455A64"),      # 浅蓝灰背景，深蓝灰文字
        }
        
        for row, item in enumerate(etfs):
            code = item["code"]
            name = item["name"]
            category = item.get("category", "其他")
            
            # 复选框
            check_item = QTableWidgetItem()
            check_item.setCheckState(Qt.Checked)
            check_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable | Qt.ItemIsSelectable)
            self.etf_table.setItem(row, 0, check_item)
            
            # 代码
            code_item = QTableWidgetItem(code)
            code_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.etf_table.setItem(row, 1, code_item)
            
            # 名称
            name_item = QTableWidgetItem(name)
            name_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.etf_table.setItem(row, 2, name_item)
            
            # 分类（使用柔和配色）
            cat_item = QTableWidgetItem(category)
            cat_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            bg_color, text_color = cat_colors.get(category, ("#F5F5F5", "#616161"))
            cat_item.setBackground(QColor(bg_color))
            cat_item.setForeground(QColor(text_color))
            self.etf_table.setItem(row, 3, cat_item)
    
    def filter_etf_by_category(self, category):
        """按分类筛选ETF"""
        if category == "全部":
            self._populate_table(self.all_etf_data)
        else:
            filtered = [item for item in self.all_etf_data if item.get("category") == category]
            self._populate_table(filtered)
    
    def select_all_etf(self):
        """全选"""
        for row in range(self.etf_table.rowCount()):
            item = self.etf_table.item(row, 0)
            if item:
                item.setCheckState(Qt.Checked)
    
    def deselect_all_etf(self):
        """取消全选"""
        for row in range(self.etf_table.rowCount()):
            item = self.etf_table.item(row, 0)
            if item:
                item.setCheckState(Qt.Unchecked)

    # ==========================================
    # 增量更新
    # ==========================================
    
    def incremental_update(self):
        """增量更新：只下载最新数据"""
        codes = self._get_selected_etf_codes()
        
        if not codes:
            QMessageBox.warning(self, "提示", "请选择要更新的ETF")
            return
        
        # 使用今天的日期作为结束日期
        today = datetime.now().strftime("%Y%m%d")
        
        self.download_log.append(f"开始增量更新 {len(codes)} 个ETF...")
        self.download_log.append(f"结束日期: {today}")
        
        self.worker = DownloadWorker(
            self.ds,
            codes,
            "ETF",
            datetime.now().strftime("%Y%m%d"),  # 开始日期不重要，会自动增量
            today
        )
        
        self.worker.signals.log.connect(self.download_log.append)
        self.worker.signals.failed.connect(self.download_log.append)
        self.worker.signals.progress.connect(self.update_progress)
        self.worker.signals.finished.connect(self._on_incremental_finished)
        self.worker.signals.updated.connect(self._on_codes_updated)
        
        self.pool.start(self.worker)
    
    def _on_incremental_finished(self):
        """增量更新完成"""
        self.download_log.append("增量更新完成！")
        self.last_update_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if self.timer_status_label:
            self.timer_status_label.setText(f"上次更新: {self.last_update_date}")
    
    def _get_selected_etf_codes(self):
        """获取选中的ETF代码列表"""
        codes = []
        for row in range(self.etf_table.rowCount()):
            check_item = self.etf_table.item(row, 0)
            if check_item and check_item.checkState() == Qt.Checked:
                code_item = self.etf_table.item(row, 1)
                name_item = self.etf_table.item(row, 2)
                if code_item:
                    codes.append({
                        "code": code_item.text(),
                        "name": name_item.text() if name_item else ""
                    })
        return codes

    # ==========================================
    # 自动定时更新
    # ==========================================
    
    def toggle_auto_update_timer(self):
        """切换定时器开关"""
        if self.start_timer_btn.isChecked():
            self.start_auto_update_timer()
        else:
            self.stop_auto_update_timer()
    
    def start_auto_update_timer(self):
        """启动定时更新"""
        if not self.auto_update_check.isChecked():
            QMessageBox.warning(self, "提示", "请先勾选'启用'自动更新")
            self.start_timer_btn.setChecked(False)
            return
        
        self.auto_update_timer.start()
        update_time = self.update_time_edit.time().toString("HH:mm")
        interval = self.update_interval_combo.currentText()
        self.timer_status_label.setText(f"定时器: 运行中 ({update_time} {interval})")
        self.timer_status_label.setStyleSheet("color:#67c23a;")
        self.start_timer_btn.setText("停止定时")
        self.start_timer_btn.setStyleSheet("background:#f56c6c;")
        self.download_log.append(f"定时更新已启动: {update_time} {interval}")
    
    def stop_auto_update_timer(self):
        """停止定时更新"""
        self.auto_update_timer.stop()
        self.timer_status_label.setText("定时器: 已停止")
        self.timer_status_label.setStyleSheet("color:#909399;")
        self.start_timer_btn.setText("启动定时")
        self.start_timer_btn.setStyleSheet("background:#e6a23c;")
        self.download_log.append("定时更新已停止")
    
    def check_auto_update(self):
        """检查是否需要执行自动更新"""
        now = datetime.now()
        update_time = self.update_time_edit.time()
        target_hour = update_time.hour()
        target_minute = update_time.minute()
        
        # 检查时间是否匹配（允许1分钟误差）
        if now.hour == target_hour and abs(now.minute - target_minute) <= 1:
            # 检查是否在正确的更新日
            interval = self.update_interval_combo.currentText()
            should_update = False
            
            if interval == "每日":
                should_update = True
            elif interval == "每周" and now.weekday() == 4:  # 周五
                should_update = True
            elif interval == "每月" and now.day == 1:  # 每月1号
                should_update = True
            
            if should_update:
                # 避免同一分钟内重复更新
                check_key = now.strftime("%Y%m%d%H%M")
                if not hasattr(self, '_last_check_key') or self._last_check_key != check_key:
                    self._last_check_key = check_key
                    self.download_log.append(f"触发自动更新: {now.strftime('%Y-%m-%d %H:%M')}")
                    self.incremental_update()

    # ==========================================
    # 指标页面
    # ==========================================

    def build_indicator_page(self):

        page = QWidget()

        layout = QVBoxLayout(page)

        title = QLabel("批量指标 / 因子计算")

        title.setStyleSheet("""
            font-size:24px;
            font-weight:bold;
        """)

        layout.addWidget(title)

        # =====================================
        # 指标说明
        # =====================================

        explain = QTextEdit()

        explain.setReadOnly(True)

        txt = ""

        for k, v in FACTOR_EXPLAIN.items():

            txt += f"【{k}】\n"

            txt += f"{v}\n\n"

        explain.setText(txt)

        layout.addWidget(explain)

        # =====================================
        # 计算按钮
        # =====================================

        self.calc_btn = QPushButton(
            "开始计算全部指标和因子"
        )

        self.calc_btn.clicked.connect(
            self.calculate_all_factors
        )

        layout.addWidget(self.calc_btn)

        self.research_btn = QPushButton("构建研究数据 (parquet，回测即插即用)")
        self.research_btn.clicked.connect(self.build_research_data)
        layout.addWidget(self.research_btn)

        # =====================================
        # 日志
        # =====================================

        self.factor_log = QTextEdit()

        layout.addWidget(self.factor_log)

        return page

    # ==========================================
    # 图表页面
    # ==========================================

    def build_chart_page(self):

        page = QWidget()

        layout = QVBoxLayout(page)

        top = QHBoxLayout()

        top.addWidget(QLabel("ETF"))

        self.etf_combo = QComboBox()

        etfs = self.ds.get_all_etf()

        self.etf_map = {}

        for item in etfs:

            code = item["code"]

            name = item["name"]

            txt = f"{code} {name}"

            self.etf_combo.addItem(txt)

            self.etf_map[txt] = code

        self.chart_btn = QPushButton("加载图表")

        self.chart_btn.clicked.connect(
            self.load_chart
        )

        top.addWidget(self.etf_combo)

        top.addWidget(self.chart_btn)

        top.addStretch()

        layout.addLayout(top)

        self.chart = ChartWidget()

        layout.addWidget(self.chart)

        return page

    def build_strategy_page(self):

        page = QWidget()

        layout = QVBoxLayout(page)

        title = QLabel("策略构建")
        title.setStyleSheet("font-size:20px;font-weight:bold;")
        layout.addWidget(title)

        form = QGridLayout()

        self.builder_combo = QComboBox()
        self.builder_combo.addItems([
            "结构化搜索-因子权重",
        ])

        self.strategy_name_edit = QLineEdit("MyStrategy")
        form.addWidget(QLabel("构建方法"), 0, 0)
        form.addWidget(self.builder_combo, 0, 1)
        form.addWidget(QLabel("策略名称"), 1, 0)
        form.addWidget(self.strategy_name_edit, 1, 1)

        layout.addLayout(form)

        btn_row = QHBoxLayout()
        self.optimize_btn = QPushButton("开始参数搜索优化")
        self.cancel_optimize_btn = QPushButton("停止优化")
        self.cancel_optimize_btn.setStyleSheet("background:#f56c6c; color:#fff;")
        self.cancel_optimize_btn.setEnabled(False)
        self.cancel_optimize_btn.hide()
        self.open_strategy_dir_btn = QPushButton("打开策略文件夹")
        btn_row.addWidget(self.optimize_btn)
        btn_row.addWidget(self.cancel_optimize_btn)
        btn_row.addWidget(self.open_strategy_dir_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        # Walk-forward 优化参数
        opt_box = QGroupBox("走样本外优化参数（训练2年/测试1年；每段测试需进入TopK；2025年后不参与搜索）")
        opt_layout = QGridLayout(opt_box)
        self.opt_train = QSpinBox()
        self.opt_train.setRange(200, 2000)
        self.opt_train.setValue(504)  # 约2年交易日
        self.opt_train.setEnabled(False)
        self.opt_test = QSpinBox()
        self.opt_test.setRange(100, 1000)
        self.opt_test.setValue(252)   # 约1年交易日
        self.opt_test.setEnabled(False)
        self.opt_trials = QSpinBox()
        self.opt_trials.setRange(5, 2000)
        self.opt_trials.setValue(300)
        self.opt_topk = QSpinBox()
        self.opt_topk.setRange(1, 200)
        self.opt_topk.setValue(10)
        opt_layout.addWidget(QLabel("训练窗口(天)"), 0, 0)
        opt_layout.addWidget(self.opt_train, 0, 1)
        opt_layout.addWidget(QLabel("测试窗口(天)"), 0, 2)
        opt_layout.addWidget(self.opt_test, 0, 3)
        opt_layout.addWidget(QLabel("搜索次数"), 1, 0)
        opt_layout.addWidget(self.opt_trials, 1, 1)
        opt_layout.addWidget(QLabel("每段TopK"), 1, 2)
        opt_layout.addWidget(self.opt_topk, 1, 3)
        opt_layout.addWidget(QLabel("搜索截止日期"), 2, 0)
        self.opt_cutoff = QLineEdit("2025-12-31")
        opt_layout.addWidget(self.opt_cutoff, 2, 3)
        layout.addWidget(opt_box)

        # ── 状态栏 ──
        self.strategy_status = QLabel("就绪")
        self.strategy_status.setStyleSheet("""
            QLabel {
                background: #2a2a2a;
                border: 1px solid #444;
                padding: 8px 12px;
                font-size: 13px;
                color: #b0b0b0;
                border-radius: 4px;
            }
        """)
        self.strategy_status.setWordWrap(True)
        self.strategy_status.setMinimumHeight(40)
        layout.addWidget(self.strategy_status)

        # ── 进度条 ──
        self.strategy_progress = QProgressBar()
        self.strategy_progress.setRange(0, 100)
        self.strategy_progress.setValue(0)
        self.strategy_progress.setTextVisible(True)
        self.strategy_progress.setFormat("")
        layout.addWidget(self.strategy_progress)

        self.strategy_build_log = QTextEdit()
        self.strategy_build_log.setReadOnly(True)
        layout.addWidget(self.strategy_build_log)

        # 事件
        self.builder_combo.currentTextChanged.connect(self.on_builder_changed)
        self.optimize_btn.clicked.connect(self.on_optimize_strategy)
        self.cancel_optimize_btn.clicked.connect(self.on_cancel_optimize)
        self.open_strategy_dir_btn.clicked.connect(self.open_strategy_dir)
        self.on_builder_changed(self.builder_combo.currentText())

        return page

    def build_backtest_page(self):

        page = QWidget()

        layout = QVBoxLayout(page)

        title = QLabel("策略回测")
        title.setStyleSheet("font-size:20px;font-weight:bold;")
        layout.addWidget(title)

        top = QGridLayout()

        self.strategy_combo = QComboBox()
        self.refresh_strategies_btn = QPushButton("刷新策略列表")

        self.benchmark_combo = QComboBox()
        self.benchmark_combo.addItem("无基准")
        for item in self.ds.get_all_etf():
            self.benchmark_combo.addItem(f"{item['code']} {item['name']}")

        self.bt_start = QLineEdit("")
        self.bt_end = QLineEdit("")

        self.comm_spin = QDoubleSpinBox()
        self.comm_spin.setRange(0, 100)
        self.comm_spin.setDecimals(2)
        self.comm_spin.setValue(1.0)
        self.comm_spin.setSuffix(" bps")

        self.slip_spin = QDoubleSpinBox()
        self.slip_spin.setRange(0, 100)
        self.slip_spin.setDecimals(2)
        self.slip_spin.setValue(2.0)
        self.slip_spin.setSuffix(" bps")

        top.addWidget(QLabel("本地策略"), 0, 0)
        top.addWidget(self.strategy_combo, 0, 1)
        top.addWidget(self.refresh_strategies_btn, 0, 2)
        top.addWidget(QLabel("基准指数/ETF"), 1, 0)
        top.addWidget(self.benchmark_combo, 1, 1, 1, 2)
        top.addWidget(QLabel("开始日期(可空)"), 2, 0)
        top.addWidget(self.bt_start, 2, 1)
        top.addWidget(QLabel("结束日期(可空)"), 2, 2)
        top.addWidget(self.bt_end, 2, 3)
        top.addWidget(QLabel("手续费"), 3, 0)
        top.addWidget(self.comm_spin, 3, 1)
        top.addWidget(QLabel("滑点"), 3, 2)
        top.addWidget(self.slip_spin, 3, 3)

        layout.addLayout(top)

        self.run_backtest_btn = QPushButton("开始回测")
        layout.addWidget(self.run_backtest_btn)

        self.bt_metrics = QLabel("")
        self.bt_metrics.setWordWrap(True)
        layout.addWidget(self.bt_metrics)

        self.bt_chart = BacktestChartWidget()
        layout.addWidget(self.bt_chart, stretch=1)

        self.backtest_log = QTextEdit()
        self.backtest_log.setReadOnly(True)
        layout.addWidget(self.backtest_log)

        # 策略解释器面板
        layout.addWidget(QLabel("策略解释器（回测完成后显示最后一日的Top-5持仓分析）"))
        self.strategy_explain = QTextEdit()
        self.strategy_explain.setReadOnly(True)
        self.strategy_explain.setMaximumHeight(200)
        self.strategy_explain.setStyleSheet("QTextEdit { font-family: 'Consolas', 'monospace'; font-size: 12px; }")
        layout.addWidget(self.strategy_explain)

        # 事件
        self.refresh_strategies_btn.clicked.connect(self.refresh_strategy_list)
        self.run_backtest_btn.clicked.connect(self.run_backtest)
        self.refresh_strategy_list()

        return page

    # ==========================================
    # 策略构建 / 回测辅助
    # ==========================================

    def get_factor_candidates(self, group: str = ""):
        """获取因子候选列表。group 为空时返回全部；指定 group 只返回该类因子"""
        if group and group in FACTOR_GROUPS:
            return [f for f in FACTOR_GROUPS[group]["factors"] if f.isidentifier()]
        # 返回全部可用的列名因子（去重）
        all_factors = []
        seen = set()
        for g in FACTOR_GROUPS.values():
            for f in g["factors"]:
                if f not in seen and f.isidentifier():
                    all_factors.append(f)
                    seen.add(f)
        return all_factors

    def on_builder_changed(self, text: str):
        self.strategy_build_log.setText(
            "结构化搜索-因子权重：固定因子分组 + 固定Regime权重 + Pareto多目标优化。\n"
            "趋势组：roc20/roc60/breakout20/relative_strength_vs_hs300\n"
            "风险组：volatility20/downside_volatility/max_drawdown20\n"
            "资金组：turnover_change/vol_ratio20\n"
            "结构组：barra_beta/corr_hs300_60\n"
            "均值回复：ma_distance (仅Sideways)\n"
            "可优化：top_n, entry/exit_rank_threshold, breadth_threshold, stop_loss\n"
            "固定：因子权重、Regime权重、Breadth权重 — 市场结构认知，不拟合"
        )

    def open_strategy_dir(self):
        path = os.path.abspath(self.stg.strategy_dir)
        QDesktopServices.openUrl(QUrl.fromLocalFile(path))

    def on_optimize_strategy(self):
        method = self.builder_combo.currentText().strip()

        # 防止卡UI：后台线程执行
        self.optimize_btn.setEnabled(False)
        self.cancel_optimize_btn.setEnabled(True)
        self.cancel_optimize_btn.show()
        self.cancel_optimize_btn.setText("停止优化")
        self.strategy_build_log.clear()
        self.strategy_status.setText("")
        self.strategy_progress.setValue(0)
        self.strategy_progress.setFormat("准备中...")

        factors = self.get_factor_candidates()
        cutoff = self.normalize_date_input(self.opt_cutoff.text()) or "2025-12-31"

        train_days = int(self.opt_train.value())
        test_days = int(self.opt_test.value())
        n_trials = int(self.opt_trials.value())
        top_k = int(self.opt_topk.value())

        # 搜索空间（仅可优化参数，固定权重不参与搜索）
        space = {
            "top_n":                  [1, 2, 3, 4, 5],
            "entry_rank_threshold":   [0, 0.5, 0.6, 0.7, 0.8],
            "exit_rank_threshold":    [0, 0.1, 0.2, 0.3, 0.4],
            "breadth_threshold":      [0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
            "stop_loss":              [0, 0.03, 0.05, 0.08, 0.12],
            "max_corr_overlap":       [0.05, 0.10, 0.15, 0.20],
        }

        def build_strategy(params: dict):
            top_n = int(params.get("top_n", 3))
            entry_q = float(params.get("entry_rank_threshold", 0) or 0)
            exit_q = float(params.get("exit_rank_threshold", 0) or 0)
            stop_loss = float(params.get("stop_loss", 0) or 0)

            # ── 固定因子分组得分（组内 rank 均值）──
            # trend_score  = mean(rank(roc20), rank(roc60), rank(breakout20), rank(relative_strength_vs_hs300))
            # risk_score   = mean(rank(volatility20), rank(downside_volatility), rank(max_drawdown20))
            # volume_score = mean(rank(turnover_change), rank(vol_ratio20))
            # structure_score = mean(rank(barra_beta), rank(corr_hs300_60))
            # mean_reversion_score = rank(ma_distance)

            _trend_raw = ["roc20", "roc60", "breakout20", "relative_strength_vs_hs300"]
            _risk_raw  = ["volatility20", "downside_volatility", "max_drawdown20"]
            _vol_raw   = ["turnover_change", "vol_ratio20"]
            _struct_raw = ["barra_beta", "corr_hs300_60"]

            def _group_mean(factors):
                if len(factors) == 1:
                    return f"_r_{factors[0]}"
                n = len(factors)
                return "(" + " + ".join(f"_r_{f}" for f in factors) + f") / {n}"

            trend_score    = _group_mean(_trend_raw)
            risk_score     = _group_mean(_risk_raw)
            volume_score   = _group_mean(_vol_raw)
            structure_score = _group_mean(_struct_raw)
            mean_rev_score = "_r_ma_distance"

            # ── 固定 Regime 权重（不可优化）──
            # ── 动态因子权重（由 FactorMonitor 预计算为 dw_* 列）──
            # 方向由表达式中 +/- 确定，dw_* 值为非负（基于 |ICIR|）
            # 若 dw_* 列缺失则回退到默认等权 0.2

            # Bear/panic 中 structure 改为 beta 偏离惩罚
            struct_bear_expr = f"( _r_beta_penalty + _r_corr_hs300_60 ) / 2"

            bull_expr = (
                f"dw_trend * {trend_score} + "
                f"dw_volume * {volume_score} + "
                f"dw_structure * {structure_score} - "
                f"dw_risk * {risk_score}"
            )
            bear_expr = (
                f"dw_trend * {trend_score} + "
                f"dw_volume * {volume_score} + "
                f"dw_structure * {struct_bear_expr} - "
                f"dw_risk * {risk_score}"
            )
            sideways_expr = (
                f"dw_trend * {trend_score} + "
                f"dw_volume * {volume_score} + "
                f"dw_structure * {structure_score} - "
                f"dw_risk * {risk_score} + "
                f"dw_meanrev * {mean_rev_score}"
            )

            regime_expr = (
                "where(regime == 'bull', " + bull_expr + ", "
                "where(regime == 'bull_volatile', " + bull_expr + ", "
                "where(regime == 'bear', " + bear_expr + ", "
                "where(regime == 'panic', " + bear_expr + ", "
                + sideways_expr + "))))"
            )
            bt = float(params.get("breadth_threshold", 0.30))
            score_expr = f"where(breadth < {bt:.2f}, " + bear_expr + ", " + regime_expr + ")"
            filter_expr = f"rank({score_expr}) > {entry_q}" if entry_q else ""
            ascending = False

            # ===== 卖出：rebalance + signal_deterioration + risk_exit =====
            sell_rule = {
                "rebalance": {"enabled": True},
            }
            if stop_loss:
                sell_rule["risk_exit"] = {"pct": abs(stop_loss)}
            if exit_q:
                # signal_deterioration: rank(final_score) < threshold | trend_consistency < 2 | panic
                sell_rule["signal_exit"] = {
                    "expr": f"(rank({score_expr}) < {exit_q}) | (trend_consistency < 2) | (regime == 'panic')"
                }

            return {
                "version": 2,
                "name": "WF_Optimized",
                "type": "expr_strategy",
                "created_at": now_str(),
                "description": "固定Regime权重 + market_score暴露 + Pareto稳定性优化",
                "buy_rule": {
                    "score_expr": score_expr,
                    "filter_expr": filter_expr,
                    "top_n": top_n,
                    "ascending": ascending,
                    "signal_strength_threshold": 0.6,
                    "liquidity_min_amount": 30000000,
                    "market_filter": {
                        "enabled": True,
                        "allow_regimes": [
                            "bull",
                            "bull_volatile",
                            "sideways"
                        ]
                    },
                },
                # 分组定义：用于策略解释器精确分解得分
                "explain_groups": {
                    "trend":    {"factors": _trend_raw,   "regimes": ["bull", "bull_volatile"]},
                    "risk":     {"factors": _risk_raw,    "regimes": ["bull", "bull_volatile", "bear", "panic", "sideways"]},
                    "volume":   {"factors": _vol_raw,     "regimes": ["bull", "bull_volatile", "bear", "panic", "sideways"]},
                    "structure":{"factors": _struct_raw,  "regimes": ["bull", "bull_volatile", "sideways"]},
                    "structure_bear": {"factors": ["barra_beta", "corr_hs300_60"], "regimes": ["bear", "panic"], "transform": "beta_penalty"},
                    "mean_reversion": {"factors": ["ma_distance"], "regimes": ["sideways"]},
                },
                "sell_rule": sell_rule,
                "execution": {
                    "decision_at": "close",
                    "buy_at": "open_next",
                    "sell_at": "open",
                    "min_hold_days": 1
                },
                "position": {
                    "max_positions": top_n,
                    "allocation": "hrp",          # "inv_vol" 或 "hrp"
                    "hrp_lookback": 60,
                    "impact_k": 0.001,             # sqrt 冲击模型系数
                    "volatility_target": 0.15,
                    "max_single_weight": 0.30,
                    "max_portfolio_dd": 0.15,
                    "max_corr_overlap": float(params.get("max_corr_overlap", 0.10)),
                    "cash_when_no_signal": True,
                    "defensive_assets": [
                        "511260",
                        "518880",
                        "510880"
                    ],
                },
            }

        def build_data_fn():
            codes = [x["code"] for x in self.ds.get_all_etf()]
            data = self.build_backtest_data(codes)
            if data is None or data.empty:
                return data
            # 加速：先过滤一遍（避免构造太多fold）
            data = data[data["date"] <= cutoff].copy()

            # 预计算逐日截面排名，避免每次 trial 都重复 rank()
            skip_cols = {"code", "date", "regime", "breadth", "amount",
                         "close", "open", "high", "low", "volume", "amount_ma20"}
            for col in data.columns:
                if col in skip_cols or not col.isidentifier():
                    continue
                try:
                    data[f"_r_{col}"] = data.groupby("date")[col].transform(
                        lambda x: x.rank(pct=True))
                except Exception:
                    pass

            # 预计算 Bear 模式的 beta 偏离惩罚 rank
            try:
                data["_r_beta_penalty"] = data.groupby("date")["barra_beta"].transform(
                    lambda x: (-abs(x - 1.0)).rank(pct=True))
            except Exception:
                pass

            # 预计算动态因子权重（Rolling RankIC → ICIR → 归一化权重）
            try:
                from factor_monitor import FactorMonitor
                fm = FactorMonitor(data)
                dw_df = fm.compute_weights(fwd_days=5, ic_window=60)
                if not dw_df.empty:
                    dw_cols = [c for c in dw_df.columns if c.startswith("dw_")]
                    data = data.merge(dw_df[["date"] + dw_cols], on="date", how="left")
                    # 前向填充缺失值（初始窗口期 IC 不足）
                    for c in dw_cols:
                        if c in data.columns:
                            data[c] = data[c].fillna(data[c].mean() if data[c].notna().any() else 0.2)
            except Exception:
                pass

            return data

        self.strategy_build_log.append(
            f"开始三阶段搜索：次数={n_trials} | TopK={top_k} | 截止={cutoff}"
        )

        worker = OptimizerWorker(
            staged_optimization,
            build_data_fn=build_data_fn,
            build_strategy_fn=build_strategy,
            space=space,
            config={
                "bt_engine": self.bt,
                "n_trials": n_trials,
                "seed": 42,
                "train_days": train_days,
                "test_days": test_days,
                "top_k": top_k,
                "max_search_date": cutoff,
                "commission_bps": float(self.comm_spin.value()),
                "slippage_bps": float(self.slip_spin.value()),
            },
        )
        # IMPORTANT: 保留引用，避免任务运行中被GC
        self.opt_worker = worker

        def on_log(msg: str):
            self.strategy_build_log.append(msg)

        def on_progress(cur: int, total: int, best: float):
            self.strategy_build_log.append(f"进度 {cur}/{total} | 当前最优均值得分 {best:.4f}")

        def on_failed(msg: str):
            self.optimize_btn.setEnabled(True)
            self.cancel_optimize_btn.setEnabled(False)
            self.cancel_optimize_btn.hide()
            self.opt_worker = None
            self.strategy_status.setText("优化失败")
            self.strategy_progress.setFormat("失败")
            QMessageBox.warning(self, "优化失败", msg)

        def on_finished(payload: dict):
            self.optimize_btn.setEnabled(True)
            self.cancel_optimize_btn.setEnabled(False)
            self.cancel_optimize_btn.hide()
            self.opt_worker = None
            self.strategy_status.setText("优化完成")
            self.strategy_progress.setValue(100)
            self.strategy_progress.setFormat("完成")
            stg = payload.get("strategy") or {}
            best_score = float(payload.get("best_score", 0.0))
            best_params = payload.get("best_params") or {}
            detail = payload.get("detail") or []

            self.strategy_build_log.append(f"优化完成：best_score={best_score:.4f}")
            self.strategy_build_log.append(f"best_params={best_params}")

            # 输出分段摘要（若有）
            if detail and isinstance(detail[0], dict):
                s = detail[0].get("summary") or {}
                if isinstance(s, dict):
                    folds = s.get("folds")
                    if folds:
                        self.strategy_build_log.append(f"分段数量：{len(folds)}")
                    if s.get("warning"):
                        self.strategy_build_log.append(f"警告：{s.get('warning')}")
                    if s.get("candidates") is not None:
                        self.strategy_build_log.append(f"满足每段TopK的候选数：{s.get('candidates')}")
                # 如果优化器返回了硬错误，直接提示（此时不弹保存框）
                if "error" in detail[0]:
                    QMessageBox.warning(self, "没有可用策略", str(detail[0]["error"]))
                    return

            # 保存最终策略：
            # 1) 先自动保存到策略库（防止文件对话框被遮挡/用户误取消）
            name = (self.strategy_name_edit.text().strip() or "MyStrategy") + "_opt"
            stg["name"] = name
            local_path = self.stg.save_strategy(stg)
            self.strategy_build_log.append(f"已自动保存到策略库：{local_path}")

            # 2) 再弹窗导出到任意位置（可取消）
            self.strategy_build_log.append("准备导出策略文件（将弹出保存对话框）...")
            default_path = os.path.join(self.stg.strategy_dir, f"{name}.json")
            path, _ = QFileDialog.getSaveFileName(
                self,
                "保存优化后的策略",
                default_path,
                "JSON (*.json)"
            )
            if not path:
                self.strategy_build_log.append("已取消导出（策略已保存在策略库，可在回测页直接使用）")
                self.refresh_strategy_list()
                return

            self.stg.save_strategy_as(stg, path)
            self.strategy_build_log.append(f"已导出：{path}")

            self.refresh_strategy_list()

        worker.signals.log.connect(on_log, Qt.QueuedConnection)
        worker.signals.progress.connect(on_progress, Qt.QueuedConnection)
        worker.signals.failed.connect(on_failed, Qt.QueuedConnection)
        worker.signals.finished.connect(on_finished, Qt.QueuedConnection)
        worker.signals.phase.connect(self._on_opt_phase, Qt.QueuedConnection)
        worker.signals.status.connect(self._on_opt_status, Qt.QueuedConnection)
        worker.signals.cancelled.connect(self._on_opt_cancelled, Qt.QueuedConnection)

        self.pool.start(worker)

    def refresh_strategy_list(self):
        self.strategy_combo.clear()
        for n in self.stg.get_all_strategies():
            self.strategy_combo.addItem(n)

    def on_cancel_optimize(self):
        """用户点击停止优化"""
        self.cancel_optimize_btn.setEnabled(False)
        self.cancel_optimize_btn.setText("停止中...")
        self.strategy_build_log.append("正在停止优化...")
        if self.opt_worker:
            self.opt_worker.cancel()

    def _on_opt_phase(self, phase_name: str):
        """阶段切换"""
        self.strategy_build_log.append(f"[{phase_name}] 进行中...")

    def _on_opt_status(self, d: dict):
        """实时状态栏更新"""
        phase = d.get("phase", "")
        trial = d.get("trial", 0)
        n_trials = d.get("n_trials", 0)
        fold = d.get("fold", 0)
        n_folds = d.get("n_folds", 0)
        best = d.get("best", 0.0)
        elapsed = d.get("elapsed_sec", 0.0)
        eta = d.get("eta_sec", 0.0)

        def _fmt(sec):
            if sec < 60:
                return f"{sec:.0f}s"
            m, s = divmod(int(sec), 60)
            if m < 60:
                return f"{m}m{s:02d}s"
            h, m = divmod(m, 60)
            return f"{h}h{m:02d}m"

        parts = [f"[{phase}]"]
        if n_trials > 0:
            parts.append(f"试验 {trial}/{n_trials}")
        if n_folds > 0:
            parts.append(f"折 {fold}/{n_folds}")
        parts.append(f"最优 {best:.4f}")
        parts.append(f"已用 {_fmt(elapsed)}")
        if eta > 0:
            parts.append(f"剩余 {_fmt(eta)}")

        self.strategy_status.setText(" | ".join(parts))

        if n_trials > 0 and phase == "Trial搜索":
            pct = int(trial / n_trials * 100)
            self.strategy_progress.setValue(pct)
            self.strategy_progress.setFormat(f"{pct}% ({trial}/{n_trials})")

    def _on_opt_cancelled(self, partial: dict):
        """优化被取消，展示部分结果"""
        self.optimize_btn.setEnabled(True)
        self.cancel_optimize_btn.setEnabled(False)
        self.cancel_optimize_btn.hide()
        self.opt_worker = None
        self.strategy_status.setText("已停止")
        self.strategy_progress.setFormat("已停止")
        self.strategy_build_log.append("优化已取消")

        best_params = partial.get("best_params") or {}
        best_score = partial.get("best_score", 0.0)
        if best_params:
            self.strategy_build_log.append(
                f"部分结果: best_score={best_score:.4f}, best_params={best_params}"
            )
        else:
            self.strategy_build_log.append("无可用部分结果")

    def normalize_date_input(self, s: str):
        s = (s or "").strip()
        if not s:
            return ""
        # 支持 20200101
        if re.fullmatch(r"\d{8}", s):
            return f"{s[:4]}-{s[4:6]}-{s[6:]}"
        # 支持 2020-01-01
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
            return s
        return s

    def build_backtest_data(self, codes: list[str], start: str = "", end: str = ""):
        start = self.normalize_date_input(start)
        end = self.normalize_date_input(end)

        # ══════════════════════════════════════════
        # 统一使用预计算 research parquet
        # ══════════════════════════════════════════
        codes_available = [c for c in codes if self.research.exists(c)]
        if len(codes_available) < max(1, len(codes) // 2):
            return pd.DataFrame()  # 数据不足，由调用方提示用户

        data = self.research.load_many(codes, start=start, end=end)
        if data.empty:
            return pd.DataFrame()

        # 补充流动性指标
        if "amount" in data.columns:
            data["amount_ma20"] = (
                data.groupby("code")["amount"]
                .transform(lambda x: x.rolling(20, min_periods=5).mean())
            )

        return data

    def _build_backtest_data_fallback(self, codes, start, end):
        """旧路径（保留但不推荐，仅用于没有research parquet时手动调用）"""
        all_df = []
        listing_dates: dict[str, str] = {}

        # 批量加载横截面因子（一次性，性能优化）
        all_cross = self.factor.load_all()

        for idx, code in enumerate(codes):
            df = self.ds.load_quotes(code)
            if df.empty:
                continue
            if start:
                df = df[df["date"] >= start]
            if end:
                df = df[df["date"] <= end]
            if df.empty:
                continue

            # 记录该ETF的上市日期（最早有数据的日期）
            first_date = str(df["date"].min())
            listing_dates[code] = first_date

            df = self.ind.calculate(df)

            # 每5个code刷新UI防止卡顿
            if idx % 5 == 0:
                QApplication.processEvents()

            # 横截面因子（批量加载后按code筛选）
            if not all_cross.empty:
                cross = all_cross[all_cross["code"] == code]
                if not cross.empty:
                    pivot = cross.pivot(
                        index="date",
                        columns="factor_name",
                        values="factor_value"
                    ).reset_index()
                    df = pd.merge(df, pivot, on="date", how="left")

            all_df.append(df)


        if not all_df:
            return pd.DataFrame()

        data = pd.concat(all_df, ignore_index=True)

        # ══════════════════════════════════════════
        # 幸存者偏差修复：每只ETF只能在其上市日期之后参与回测
        # ══════════════════════════════════════════
        for code, first_d in listing_dates.items():
            mask = (data["code"] == code) & (data["date"] < first_d)
            data = data[~mask]
        # 确保正确排序
        data = data.sort_values(["date", "code"]).reset_index(drop=True)

        # ══════════════════════════════════════════
        # 流动性过滤：20日均成交额 > 3000万（避免冷门ETF实盘买不动）
        # ══════════════════════════════════════════
        if "amount" in data.columns:
            data["amount_ma20"] = (
                data.groupby("code")["amount"]
                .transform(lambda x: x.rolling(20, min_periods=5).mean())
            )

        # ==============================
        # 市场状态（基于沪深300）
        # ==============================

        bench = self.ds.load_quotes("510300")

        if not bench.empty:

            regime_detector = MarketRegimeDetector()

            regime_df = regime_detector.detect(bench)

            data = pd.merge(
                data,
                regime_df,
                on="date",
                how="left"
            )

            # 市场宽度（基于固定核心ETF池，确保历史可比）
            # 核心池：宽基 + 一级行业代表，~30只，跨时间稳定
            _core_breadth = [
                "510300", "510050", "510500", "512100", "159915", "588000", "159949",
                "510180", "159901", "159905", "510880", "512890", "512040",
                "512480", "515880", "512980", "515790", "516260",
                "512170", "159881", "516110", "512290",
                "516160", "562500", "159819", "515050",
            ]
            breadth_df = regime_detector.calc_market_breadth(data, core_codes=_core_breadth)
            if not breadth_df.empty:
                data = pd.merge(
                    data,
                    breadth_df[["date", "breadth_smooth"]],
                    on="date",
                    how="left"
                )
                # 重命名便于表达式使用
                data = data.rename(columns={"breadth_smooth": "breadth"})

        return data

    def build_research_data(self):
        """一键构建全部ETF的研究数据parquet（下载→指标→因子→regime）"""
        codes = [x["code"] for x in self.ds.get_all_etf()]
        self.backtest_log.clear()
        self.backtest_log.append(f"开始构建研究数据: {len(codes)} 只ETF...")
        QApplication.processEvents()

        def _log(msg):
            self.backtest_log.append(msg)
            QApplication.processEvents()

        built = self.research.build_all(
            codes,
            indicator_engine=self.ind,
            factor_engine=self.factor,
            log_cb=_log,
        )
        self.backtest_log.append(f"完成！共构建 {built}/{len(codes)} 只ETF研究数据")
        QMessageBox.information(self, "完成", f"研究数据构建完成: {built} 只ETF")

    def run_backtest(self):
        name = self.strategy_combo.currentText().strip()
        if not name:
            QMessageBox.warning(self, "提示", "请先选择策略")
            return

        stg = self.stg.load_strategy(name)
        if stg.get("type") in ("rl_placeholder", "search_placeholder"):
            QMessageBox.information(
                self,
                "提示",
                "该策略类型目前为占位，回测将不会产生交易信号。\n"
                "建议先使用「因子排名/多因子加权」策略。"
            )

        self.backtest_log.clear()
        self.backtest_log.append(f"加载策略：{name}")

        # 默认：全部精选ETF作为交易池
        codes = [x["code"] for x in self.ds.get_all_etf()]
        start_date = self.bt_start.text()
        end_date = self.bt_end.text()

        # 缓存：策略/代码/日期范围不变时复用已构建数据
        cache_key = f"{','.join(sorted(codes))}|{start_date}|{end_date}"
        if self._bt_data_cache_key == cache_key and self._bt_data_cache is not None:
            data = self._bt_data_cache.copy()
        else:
            data = self.build_backtest_data(codes, start=start_date, end=end_date)
            if not data.empty:
                self._bt_data_cache = data.copy()
                self._bt_data_cache_key = cache_key

        if data.empty:
            QMessageBox.warning(self, "提示", "没有可用于回测的数据（请先下载数据，并计算指标/因子）")
            self.run_backtest_btn.setEnabled(True)
            return

        # 基准
        bench_df = pd.DataFrame()
        btxt = self.benchmark_combo.currentText()
        if btxt and btxt != "无基准":
            bcode = btxt.split()[0]
            bq = self.ds.load_quotes(bcode)
            if not bq.empty:
                start = self.normalize_date_input(self.bt_start.text())
                end = self.normalize_date_input(self.bt_end.text())
                if start:
                    bq = bq[bq["date"] >= start]
                if end:
                    bq = bq[bq["date"] <= end]
                bench_df = bq[["date", "close"]].copy()

        self.backtest_log.append("回测计算已提交到后台线程...")

        # 禁用按钮，防止重复点击
        self.run_backtest_btn.setEnabled(False)
        self.run_backtest_btn.setText("计算中...")

        worker = BacktestWorker(
            self.bt,
            stg,
            data,
            benchmark=bench_df,
            base_point=1000.0,
            commission_bps=float(self.comm_spin.value()),
            slippage_bps=float(self.slip_spin.value()),
        )
        worker.signals.log.connect(self._on_bt_log, Qt.QueuedConnection)
        worker.signals.finished.connect(self._on_bt_finished, Qt.QueuedConnection)
        worker.signals.failed.connect(self._on_bt_failed, Qt.QueuedConnection)
        QThreadPool.globalInstance().start(worker)

    def _on_bt_log(self, msg: str):
        self.backtest_log.append(msg)

    def _on_bt_finished(self, result: dict):
        self.run_backtest_btn.setEnabled(True)
        self.run_backtest_btn.setText("开始回测")

        eq = result.get("equity", pd.DataFrame())
        trades = result.get("trades", pd.DataFrame())
        metrics = result.get("metrics", {})
        bench_df = result.get("benchmark", pd.DataFrame())

        self.bt_chart.draw(eq, bench_df, trades, metrics)

        if metrics:
            self.bt_metrics.setText(
                " | ".join([
                    f"总收益 {metrics.get('total_return', 0) * 100:.2f}%",
                    f"年化 {metrics.get('annual_return', 0) * 100:.2f}%",
                    f"波动 {metrics.get('annual_vol', 0) * 100:.2f}%",
                    f"夏普 {metrics.get('sharpe', 0):.2f}",
                    f"Sortino {metrics.get('sortino', 0):.2f}",
                    f"Calmar {metrics.get('calmar', 0):.2f}",
                    f"最大回撤 {metrics.get('max_drawdown', 0) * 100:.2f}%",
                    f"交易数 {metrics.get('trades', 0)}",
                    (
                        f"胜率 {metrics.get('win_rate', 0) * 100:.1f}%"
                        if metrics.get("win_rate") == metrics.get("win_rate")
                        else "胜率 NA"
                    ),
                ])
            )
        else:
            self.bt_metrics.setText("")

        if not eq.empty:
            self.backtest_log.append(f"回测完成：净值 {float(eq['nav'].iloc[-1]):.2f}")
        self.backtest_log.append(f"交易数量：{len(trades)}")
        if not trades.empty:
            self.backtest_log.append("最近10笔交易：")
            show = trades.tail(10).copy()
            for _, r in show.iterrows():
                self.backtest_log.append(
                    f"{r['code']} {r['buy_date']}->{r['sell_date']} "
                    f"收益 {r['pnl_pct'] * 100:.2f}%"
                )

        # 策略解释器
        self._explain_last_day(data, stg)

    def _explain_last_day(self, data: pd.DataFrame, stg: dict):
        """解释最后一日的持仓选择：用策略真实分组定义，精确分解 Top-5 ETF 得分"""
        try:
            dates = sorted(data["date"].unique())
            if not dates:
                return
            last_date = dates[-1]
            last_df = data[data["date"] == last_date].copy()
            if last_df.empty:
                return

            score_expr = stg.get("buy_rule", {}).get("score_expr", "")
            if not score_expr:
                self.strategy_explain.setText("(无评分表达式)")
                return

            # 评估真实最终得分
            from factor_expression import ExpressionEngine
            ee = ExpressionEngine()
            scores = ee.eval(score_expr, last_df)
            if scores is None:
                self.strategy_explain.setText("(表达式评估失败)")
                return

            last_df = last_df.copy()
            last_df["_score"] = scores
            last_df = last_df.sort_values("_score", ascending=False)

            # 从策略中读取真实分组定义
            explain_groups = stg.get("explain_groups", {})
            if not explain_groups:
                # 兜底：旧策略无 explain_groups，用默认定义
                explain_groups = {
                    "trend":    {"factors": ["roc20", "roc60", "breakout20", "relative_strength_vs_hs300"]},
                    "risk":     {"factors": ["volatility20", "downside_volatility", "max_drawdown20"]},
                    "volume":   {"factors": ["turnover_change", "vol_ratio20"]},
                    "structure":{"factors": ["barra_beta", "corr_hs300_60"]},
                }

            lines = [f"=== 策略解释器 [{last_date}] ==="]

            # regime & market_score
            if "regime" in last_df.columns:
                current_regime = str(last_df["regime"].iloc[0])
                lines.append(f"Regime: {current_regime}")
            else:
                current_regime = ""
            if "market_score" in last_df.columns:
                ms = float(last_df["market_score"].iloc[0])
                lines.append(f"market_score: {ms:.3f}")
            if "breadth" in last_df.columns:
                lines.append(f"breadth: {float(last_df['breadth'].iloc[0]):.3f}")

            lines.append(f"\nTop-5 ETF 分解得分（真实分组 rank 均值）:")
            top5 = last_df.head(5)
            for _, row in top5.iterrows():
                code = str(row["code"])
                name = self._get_etf_name(code)
                final = float(row["_score"])
                lines.append(f"\n  {code} {name}: final={final:.4f}")

                for cat_name, cat_def in explain_groups.items():
                    factors = cat_def.get("factors", [])
                    avail = [f for f in factors if f in last_df.columns]
                    if not avail:
                        continue

                    transform = cat_def.get("transform", "")
                    if transform == "beta_penalty":
                        # Bear 中 structure：用 -abs(beta-1) 代替 barra_beta
                        group_ranks = []
                        for f in avail:
                            if f == "barra_beta":
                                penalty = -abs(last_df["barra_beta"] - 1.0)
                                group_ranks.append(penalty.rank(pct=True))
                            else:
                                group_ranks.append(last_df[f].rank(pct=True))
                        cat_val = float(np.mean([r.loc[row.name] for r in group_ranks]))
                    else:
                        ranks = [last_df[f].rank(pct=True) for f in avail]
                        cat_val = float(np.mean([r.loc[row.name] for r in ranks]))

                    lines.append(f"    {cat_name}_score: {cat_val:.4f}")

            self.strategy_explain.setText("\n".join(lines))
        except Exception as e:
            import traceback
            self.strategy_explain.setText(f"(解释器异常: {e}\n{traceback.format_exc()})")

    def _get_etf_name(self, code: str) -> str:
        """获取ETF名称"""
        try:
            all_etfs = self.ds.get_all_etf()
            for x in all_etfs:
                if str(x.get("code", "")) == code:
                    return x.get("name", "")
        except Exception:
            pass
        return ""

    def _on_bt_failed(self, msg: str):
        self.run_backtest_btn.setEnabled(True)
        self.run_backtest_btn.setText("开始回测")
        self.backtest_log.append(f"回测失败:\n{msg}")
        QMessageBox.critical(self, "回测异常", msg)
    
    # ==========================================
    # 下载数据
    # ==========================================

    def download_data(self):

        codes = self._get_selected_etf_codes()

        if not codes:

            QMessageBox.warning(
                self,
                "提示",
                "请选择ETF"
            )

            return

        beg = self.start_edit.text()

        end = self.end_edit.text()

        self.worker = DownloadWorker(

            self.ds,

            codes,

            "ETF",

            beg,

            end
        )

        self.worker.signals.log.connect(
            self.download_log.append
        )

        self.worker.signals.failed.connect(
            self.download_log.append
        )

        self.worker.signals.progress.connect(
            self.update_progress
        )

        self.worker.signals.finished.connect(
            self.download_finished
        )

        self.worker.signals.updated.connect(self._on_codes_updated)

        self.pool.start(self.worker)

    # ==========================================
    # 更新进度
    # ==========================================

    def update_progress(self, cur, total):

        v = int(cur / total * 100)

        self.progress.setValue(v)

    # ==========================================
    # 下载完成
    # ==========================================

    def download_finished(self):

        self.download_log.append(
            "全部下载完成 — 自动重建研究数据..."
        )

    # ==========================================
    # 下载后自动重建 research
    # ==========================================

    def _on_codes_updated(self, codes: list):
        """下载成功后自动增量重建research parquet"""
        if not codes:
            return
        # 横截面因子依赖全量数据，直接全量重建（后台线程不阻塞UI）
        self.download_log.append(f"正在重建 {len(codes)} 只ETF的研究数据...")
        self._rebuild_research_in_background()

    def _rebuild_research_in_background(self):
        """后台线程全量重建research parquet"""
        from PySide6.QtCore import QThreadPool, QRunnable, QTimer

        all_codes = [x["code"] for x in self.ds.get_all_etf()]
        result = {"built": 0, "total": len(all_codes), "done": False, "logs": []}

        class _RebuildWorker(QRunnable):
            def __init__(self, research, ind, factor, codes, result):
                super().__init__()
                self.r = research; self.ind = ind; self.factor = factor
                self.codes = codes; self._r = result

            def run(self):
                built = self.r.build_all(self.codes, indicator_engine=self.ind,
                                         factor_engine=self.factor,
                                         log_cb=lambda m: self._r["logs"].append(m))
                self._r["built"] = built; self._r["done"] = True

        worker = _RebuildWorker(self.research, self.ind, self.factor, all_codes, result)
        QThreadPool.globalInstance().start(worker)

        def _poll():
            # drain logs
            while result["logs"]:
                self.download_log.append(result["logs"].pop(0))
            if result["done"]:
                self.download_log.append(f"研究数据重建完成: {result['built']}/{result['total']} 只ETF")
                if hasattr(self, "_rebuild_timer"):
                    self._rebuild_timer.stop()
            elif hasattr(self, "_rebuild_timer"):
                self._rebuild_timer.start(300)  # keep polling

        timer = QTimer(self)
        timer.setSingleShot(True)
        timer.timeout.connect(_poll)
        self._rebuild_timer = timer
        timer.start(100)

    # ==========================================
    # 暂停
    # ==========================================

    def pause_download(self):

        if self.worker:

            self.worker.pause()

            self.download_log.append(
                "下载已暂停"
            )

    # ==========================================
    # 继续
    # ==========================================

    def resume_download(self):

        if self.worker:

            self.worker.resume()

            self.download_log.append(
                "继续下载"
            )

    # ==========================================
    # 停止
    # ==========================================

    def stop_download(self):

        if self.worker:

            self.worker.stop()

            self.download_log.append(
                "下载已停止"
            )

    # ==========================================
    # 批量计算
    # ==========================================

    def calculate_all_factors(self):
        """一键计算全部指标+横截面因子，保存为research parquet"""
        codes = [x["code"] for x in self.ds.get_all_etf()]
        self.factor_log.append(f"开始构建研究数据: {len(codes)} 只ETF...")

        def _log(msg):
            self.factor_log.append(msg)
            QApplication.processEvents()

        built = self.research.build_all(codes,
            indicator_engine=self.ind,
            factor_engine=self.factor,
            log_cb=_log)
        self.factor_log.append(f"完成！共构建 {built}/{len(codes)} 只ETF研究数据")

    # ==========================================
    # 以下方法已由 research.build_all 接管，保留空桩
    # ==========================================

    # ==========================================
    # 加载图表
    # ==========================================

    def load_chart(self):

        txt = self.etf_combo.currentText()

        code = self.etf_map[txt]

        # 优先从 research parquet 直接读（10x加速，不再calculate）
        df = self.research.load(code)

        if df.empty:
            QMessageBox.warning(self, "提示", "请先\"构建研究数据\"")
            return

        self.chart.draw(df)

    # ==========================================
    # 深色主题
    # ==========================================

    def set_dark_theme(self):

        self.setStyleSheet("""

        QWidget{
            background:#1e1e1e;
            color:#dddddd;
            font-size:14px;
        }

        QTabWidget::pane{
            border:1px solid #333;
        }

        QPushButton{
            background:#2d8cf0;
            border:none;
            padding:10px;
            border-radius:5px;
        }

        QPushButton:hover{
            background:#4da3ff;
        }

        QTextEdit,
        QListWidget,
        QComboBox,
        QLineEdit{

            background:#111111;

            border:1px solid #444;

            padding:5px;
        }

        QProgressBar{

            background:#111111;

            border:1px solid #444;

            height:20px;

            text-align:center;
        }

        QProgressBar::chunk{

            background:#2d8cf0;
        }

        """)
