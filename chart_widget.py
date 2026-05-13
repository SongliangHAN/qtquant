import numpy as np
import pyqtgraph as pg

from PySide6.QtWidgets import *
from PySide6.QtCore import *

# ==========================================
# 指标配置
# ==========================================

INDICATOR_CONFIG = {
    # ======================================
    # 横截面因子
    # ======================================
    "Cross Momentum": {
        "type": "single", "cols": ["cross_momentum"],
        "colors": ["#00e676"]
    },
    "Cross Volatility": {
        "type": "single", "cols": ["cross_volatility"],
        "colors": ["#40c4ff"]
    },
    "Barra Beta": {
        "type": "single", "cols": ["barra_beta"],
        "colors": ["#ff4081"]
    },
    "Corr HS300(60d)": {
        "type": "single", "cols": ["corr_hs300_60"],
        "colors": ["#ffab40"]
    },

    # ======================================
    # Rel Strength
    # ======================================
    "Rel Strength vs HS300": {
        "type": "single", "cols": ["relative_strength_vs_hs300"],
        "colors": ["#ffea00"]
    },

    # ======================================
    # MACD
    # ======================================
    "MACD": {
        "type": "multi",
        "cols": ["macd_dif", "macd_dea", "macd_hist"],
        "colors": ["#ffffff", "#ffd54f", "#ff4d4f"]
    },

    # ======================================
    # KDJ
    # ======================================
    "KDJ": {
        "type": "multi",
        "cols": ["kdj_k", "kdj_d", "kdj_j"],
        "colors": ["#ffd54f", "#40c4ff", "#ff4081"]
    },

    # ======================================
    # BOLL
    # ======================================
    "BOLL": {
        "type": "multi",
        "cols": ["close", "boll_up", "boll_mid", "boll_low"],
        "colors": ["#ffffff", "#ff4d4f", "#ffd54f", "#00c853"]
    },

    # ======================================
    # RSI
    # ======================================
    "RSI": {
        "type": "multi",
        "cols": ["rsi6", "rsi12", "rsi24"],
        "colors": ["#ffd54f", "#40c4ff", "#ff4081"]
    },

    # ======================================
    # 趋势动量
    # ======================================
    "ROC20": {"type": "single", "cols": ["roc20"], "colors": ["#00e676"]},
    "ROC60": {"type": "single", "cols": ["roc60"], "colors": ["#69f0ae"]},
    "ROC120": {"type": "single", "cols": ["roc120"], "colors": ["#b9f6ca"]},
    "Breakout20": {"type": "single", "cols": ["breakout20"], "colors": ["#40c4ff"]},
    "MA Distance": {"type": "single", "cols": ["ma_distance"], "colors": ["#7c4dff"]},

    # ======================================
     # ROC (legacy)
     # ======================================
     "ROC12": {"type": "single", "cols": ["roc12"], "colors": ["#40c4ff"]},

     # ======================================
     # 波动率
     # ======================================
     "Volatility20": {"type": "single", "cols": ["volatility20"], "colors": ["#9c27b0"]},
    "Downside Vol": {"type": "single", "cols": ["downside_volatility"], "colors": ["#ce93d8"]},
    "ATR14": {"type": "single", "cols": ["atr14"], "colors": ["#ffd54f"]},
    "ATR20": {"type": "single", "cols": ["atr20"], "colors": ["#ffe082"]},
    "MaxDD20": {"type": "single", "cols": ["max_drawdown20"], "colors": ["#ff5252"]},

    # ======================================
    # 资金流
    # ======================================
    "Turnover Chg": {"type": "single", "cols": ["turnover_change"], "colors": ["#ff9800"]},
    "Vol Ratio20": {"type": "single", "cols": ["vol_ratio20"], "colors": ["#ffab40"]},

    # ======================================
    # OBV / CCI
    # ======================================
    "OBV": {"type": "single", "cols": ["obv"], "colors": ["#00c853"]},
    "CCI14": {"type": "single", "cols": ["cci14"], "colors": ["#ff9800"]},

    # ======================================
    # 趋势一致性
    # ======================================
    "Trend Consistency": {"type": "single", "cols": ["trend_consistency"], "colors": ["#18ffff"]},

    # ======================================
    # Breadth
    # ======================================
    "Breadth": {"type": "single", "cols": ["breadth"], "colors": ["#ea80fc"]},
}
# ==========================================
# 日期轴
# ==========================================

class DateAxis(pg.AxisItem):

    def __init__(self, dates, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.dates = dates

    def tickStrings(self, values, scale, spacing):

        strings = []

        for v in values:

            v = int(v)

            if 0 <= v < len(self.dates):

                strings.append(
                    str(self.dates[v])[:10]
                )

            else:

                strings.append("")

        return strings


# ==========================================
# K线图元
# ==========================================

class CandlestickItem(pg.GraphicsObject):

    def __init__(self, data):

        super().__init__()

        self.data = data

        self.generatePicture()

    def generatePicture(self):

        self.picture = pg.QtGui.QPicture()

        p = pg.QtGui.QPainter(self.picture)

        w = 0.35

        for (t, open_, close_, low_, high_) in self.data:

            # 红涨绿跌
            if close_ >= open_:

                color = pg.mkColor("#ff4d4f")

            else:

                color = pg.mkColor("#00c853")

            p.setPen(pg.mkPen(color))

            # 上下影线
            p.drawLine(
                pg.QtCore.QPointF(t, low_),
                pg.QtCore.QPointF(t, high_)
            )

            rect = pg.QtCore.QRectF(
                t - w,
                open_,
                w * 2,
                close_ - open_
            )

            p.fillRect(rect, color)

            p.drawRect(rect)

        p.end()

    def paint(self, painter, option, widget):

        painter.drawPicture(0, 0, self.picture)

    def boundingRect(self):

        return pg.QtCore.QRectF(
            self.picture.boundingRect()
        )


# ==========================================
# 金融ViewBox
# ==========================================

class FinanceViewBox(pg.ViewBox):

    def __init__(self):

        super().__init__()

        # 框选缩放
        self.setMouseMode(self.RectMode)

    # 鼠标滚轮
    # 只缩放X轴
    def wheelEvent(self, ev, axis=None):

        if ev.delta() > 0:

            scale = 0.9

        else:

            scale = 1.1

        self.scaleBy((scale, 1))

        ev.accept()


# ==========================================
# 主图表
# ==========================================

class ChartWidget(QWidget):

    def __init__(self):

        super().__init__()

        self.df = None

        layout = QVBoxLayout(self)

        # =====================================
        # 工具栏
        # =====================================

        toolbar = QHBoxLayout()

        self.btn_reset = QPushButton("复位")

        self.btn_reset.clicked.connect(
            self.reset_view
        )

        toolbar.addWidget(self.btn_reset)

        # =====================================
        # 指标切换
        # =====================================

        toolbar.addWidget(QLabel("图3"))

        self.indicator3 = QComboBox()

        self.indicator3.addItems(
            list(INDICATOR_CONFIG.keys())
        )

        toolbar.addWidget(self.indicator3)

        toolbar.addWidget(QLabel("图4"))

        self.indicator4 = QComboBox()

        self.indicator4.addItems(
            list(INDICATOR_CONFIG.keys())
        )

        toolbar.addWidget(self.indicator4)

        self.indicator3.currentTextChanged.connect(
            self.redraw_indicator
        )

        self.indicator4.currentTextChanged.connect(
            self.redraw_indicator
        )

        toolbar.addStretch()

        layout.addLayout(toolbar)

        # =====================================
        # 图形
        # =====================================

        self.graph = pg.GraphicsLayoutWidget()

        layout.addWidget(self.graph)

        pg.setConfigOptions(

            background="#1e1e1e",

            foreground="#dcdcdc",

            antialias=True
        )

        # =====================================
        # 日期轴
        # =====================================

        self.axis1 = DateAxis([], orientation='bottom')
        self.axis2 = DateAxis([], orientation='bottom')
        self.axis3 = DateAxis([], orientation='bottom')
        self.axis4 = DateAxis([], orientation='bottom')

        # =====================================
        # 四联图
        # =====================================

        self.p1 = self.graph.addPlot(

            row=0,

            col=0,

            axisItems={
                "bottom": self.axis1
            },

            viewBox=FinanceViewBox()
        )

        self.p2 = self.graph.addPlot(

            row=1,

            col=0,

            axisItems={
                "bottom": self.axis2
            },

            viewBox=FinanceViewBox()
        )

        self.p3 = self.graph.addPlot(

            row=2,

            col=0,

            axisItems={
                "bottom": self.axis3
            },

            viewBox=FinanceViewBox()
        )

        self.p4 = self.graph.addPlot(

            row=3,

            col=0,

            axisItems={
                "bottom": self.axis4
            },

            viewBox=FinanceViewBox()
        )

        # =====================================
        # 联动
        # =====================================

        self.p2.setXLink(self.p1)
        self.p3.setXLink(self.p1)
        self.p4.setXLink(self.p1)

        # =====================================
        # 网格
        # =====================================

        for p in [

            self.p1,
            self.p2,
            self.p3,
            self.p4
        ]:

            p.showGrid(x=True, y=True)

            p.setMenuEnabled(False)

            p.enableAutoRange()

        # =====================================
        # 十字光标
        # =====================================

        self.vline = pg.InfiniteLine(
            angle=90,
            movable=False,
            pen=pg.mkPen("#888")
        )

        self.hline = pg.InfiniteLine(
            angle=0,
            movable=False,
            pen=pg.mkPen("#888")
        )

        self.p1.addItem(
            self.vline,
            ignoreBounds=True
        )

        self.p1.addItem(
            self.hline,
            ignoreBounds=True
        )

        # =====================================
        # OHLC 信息框
        # =====================================

        self.info = pg.TextItem(
            anchor=(0, 0)
        )

        self.p1.addItem(self.info)

        # =====================================
        # 鼠标事件
        # =====================================

        self.proxy = pg.SignalProxy(

            self.p1.scene().sigMouseMoved,

            rateLimit=60,

            slot=self.mouse_moved
        )

        # =====================================
        # X轴变化
        # 动态Y轴
        # =====================================

        self.p1.sigXRangeChanged.connect(
            self.update_y_range
        )

    # ==========================================
    # 鼠标移动
    # ==========================================

    def mouse_moved(self, evt):

        if self.df is None:
            return

        pos = evt[0]

        vb = self.p1.vb

        mousePoint = vb.mapSceneToView(pos)

        x = int(mousePoint.x())

        if x < 0 or x >= len(self.df):
            return

        row = self.df.iloc[x]

        self.vline.setPos(x)

        self.hline.setPos(mousePoint.y())

        txt = (

            f"日期: {row['date']}<br>"

            f"开: {row['open']:.2f}<br>"

            f"高: {row['high']:.2f}<br>"

            f"低: {row['low']:.2f}<br>"

            f"收: {row['close']:.2f}"
        )

        self.info.setHtml(txt)

        self.info.setPos(
            x,
            row["high"]
        )

    # ==========================================
    # 动态Y轴
    # ==========================================

    def update_y_range(self):

        if self.df is None:
            return

        xmin, xmax = self.p1.viewRange()[0]

        xmin = max(int(xmin), 0)

        xmax = min(int(xmax), len(self.df)-1)

        if xmax <= xmin:
            return

        sub = self.df.iloc[xmin:xmax]

        # =====================================
        # K线
        # =====================================

        low = sub["low"].min()

        high = sub["high"].max()

        pad = (high - low) * 0.05

        self.p1.setYRange(
            low - pad,
            high + pad
        )

        # =====================================
        # 成交量
        # =====================================

        vmax = sub["volume"].max()

        self.p2.setYRange(
            0,
            vmax * 1.1
        )

        # =====================================
        # 图3
        # =====================================

        self.auto_range_indicator(
            self.p3,
            sub,
            self.indicator3.currentText()
        )

        # =====================================
        # 图4
        # =====================================

        self.auto_range_indicator(
            self.p4,
            sub,
            self.indicator4.currentText()
        )

    # ==========================================
    # 指标Y轴
    # ==========================================

    def auto_range_indicator(
            self,
            plot,
            sub,
            name
    ):

        if name not in INDICATOR_CONFIG:
            return

        cols = INDICATOR_CONFIG[name]["cols"]

        vals = []

        for c in cols:

            if c not in sub.columns:
                continue

            vals.extend(
                sub[c]
                .dropna()
                .tolist()
            )

        if not vals:
            return

        low = min(vals)

        high = max(vals)

        pad = (high - low) * 0.1

        if pad == 0:
            pad = 1

        plot.setYRange(
            low - pad,
            high + pad
        )

    # ==========================================
    # 重绘指标
    # ==========================================

    def redraw_indicator(self):

        if self.df is None:
            return

        self.draw(self.df)

    # ==========================================
    # 复位
    # ==========================================

    def reset_view(self):

        if self.df is None:
            return

        self.p1.setXRange(
            max(0, len(self.df) - 200),
            len(self.df)
        )

    # ==========================================
    # 绘制指标
    # ==========================================

    def draw_indicator(
            self,
            plot,
            x,
            df,
            name
    ):

        plot.clear()

        if name not in INDICATOR_CONFIG:
            return

        cfg = INDICATOR_CONFIG[name]

        cols = cfg["cols"]

        colors = cfg["colors"]

        # =====================================
        # MACD柱
        # =====================================

        if name == "MACD":

            # DIF
            plot.plot(
                x,
                df["macd_dif"],
                pen=pg.mkPen(colors[0])
            )

            # DEA
            plot.plot(
                x,
                df["macd_dea"],
                pen=pg.mkPen(colors[1])
            )

            # HIST — 批量绘制（避免逐根addItem导致的n次scene更新）
            hist_vals = df["macd_hist"].fillna(0).values
            colors = ["#ff4d4f" if v >= 0 else "#00c853" for v in hist_vals]
            bg = pg.BarGraphItem(
                x=np.arange(len(hist_vals)),
                height=hist_vals,
                width=0.6,
                brushes=colors,
            )
            plot.addItem(bg)

            return

        # =====================================
        # 通用指标
        # =====================================

        for col, color in zip(cols, colors):

            if col not in df.columns:
                continue

            plot.plot(

                x,

                df[col],

                pen=pg.mkPen(
                    color,
                    width=1.2
                ),

                name=col
            )
    # ==========================================
    # 绘图
    # ==========================================

    def draw(self, df):

        self.df = df

        dates = df["date"].tolist()

        self.axis1.dates = dates
        self.axis2.dates = dates
        self.axis3.dates = dates
        self.axis4.dates = dates

        self.p1.clear()
        self.p2.clear()
        self.p3.clear()
        self.p4.clear()

        # 重新添加十字线
        self.p1.addItem(self.vline)

        self.p1.addItem(self.hline)

        self.p1.addItem(self.info)

        x = np.arange(len(df))

        # =====================================
        # K线
        # =====================================

        candle_data = []

        for i, row in df.iterrows():

            candle_data.append((
                i,
                row["open"],
                row["close"],
                row["low"],
                row["high"]
            ))

        item = CandlestickItem(candle_data)

        self.p1.addItem(item)

        # MA
        self.p1.plot(
            x,
            df["ma5"],
            pen=pg.mkPen("#ffd54f")
        )

        self.p1.plot(
            x,
            df["ma20"],
            pen=pg.mkPen("#40c4ff")
        )

        # =====================================
        # 成交量（批量 BarGraphItem）
        # =====================================
        vol_vals = df["volume"].fillna(0).values
        vol_colors = [
            "#ff4d4f" if row["close"] >= row["open"] else "#00c853"
            for _, row in df.iterrows()
        ]
        vol_bars = pg.BarGraphItem(
            x=np.arange(len(vol_vals)),
            height=vol_vals,
            width=0.6,
            brushes=vol_colors,
        )
        self.p2.addItem(vol_bars)

        # =====================================
        # 动态指标
        # =====================================

        self.draw_indicator(
            self.p3,
            x,
            df,
            self.indicator3.currentText()
        )

        self.draw_indicator(
            self.p4,
            x,
            df,
            self.indicator4.currentText()
        )

        # =====================================
        # 初始视图
        # =====================================

        self.reset_view()

        self.update_y_range()
