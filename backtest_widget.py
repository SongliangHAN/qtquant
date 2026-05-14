import numpy as np
import pyqtgraph as pg
import pandas as pd

from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import QColor

from chart_widget import DateAxis


class BacktestSignals(QObject):
    """回测线程信号"""
    log = Signal(str)
    progress = Signal(int, int)  # (current, total)
    finished = Signal(dict)   # {"equity": df, "benchmark": df, "trades": df, "metrics": dict}
    failed = Signal(str)


class BacktestWorker(QRunnable):
    """后台执行回测，避免界面卡死"""

    def __init__(self, bt_engine, stg, data, *, benchmark=None, base_point=1000.0,
                 commission_bps=1.0, slippage_bps=2.0):
        super().__init__()
        self.signals = BacktestSignals()
        self.bt_engine = bt_engine
        self.stg = stg
        self.data = data
        self.benchmark = benchmark
        self.base_point = base_point
        self.commission_bps = commission_bps
        self.slippage_bps = slippage_bps

    def run(self):
        try:
            self.signals.log.emit("正在执行回测...")
            res = self.bt_engine.run(
                self.stg,
                self.data,
                benchmark=self.benchmark,
                base_point=self.base_point,
                commission_bps=self.commission_bps,
                slippage_bps=self.slippage_bps,
                progress_cb=lambda cur, total: self.signals.progress.emit(cur, total),
            )
            self.signals.finished.emit({
                "equity": res.get("equity", pd.DataFrame()),
                "benchmark": res.get("benchmark", pd.DataFrame()),
                "trades": res.get("trades", pd.DataFrame()),
                "metrics": res.get("metrics", {}),
                "filter_chain": res.get("filter_chain", {}),
            })
        except Exception:
            import traceback
            self.signals.failed.emit(traceback.format_exc())


class BacktestChartWidget(QWidget):
    """
    回测可视化：
    - 上图：策略净值 vs 基准（标注调仓买卖点）
    - 下表：持仓明细表 + 分阶段收益率表
    """

    def __init__(self):
        super().__init__()

        self._results = {}

        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # ── 图表 ──
        self.graph = pg.GraphicsLayoutWidget()
        layout.addWidget(self.graph, stretch=3)

        pg.setConfigOptions(background="#1e1e1e", foreground="#dcdcdc", antialias=True)

        self.axis_nav = DateAxis([], orientation="bottom")
        self.p1 = self.graph.addPlot(
            row=0, col=0, axisItems={"bottom": self.axis_nav}
        )
        self.p1.showGrid(x=True, y=True)
        self.p1.setMenuEnabled(False)
        self.p1.addLegend()

        # ── 持仓明细 ──
        self.holdings_table = QTableWidget()
        self.holdings_table.setColumnCount(4)
        self.holdings_table.setHorizontalHeaderLabels(["开始日期", "结束日期", "持仓标的", "期间收益率"])
        self.holdings_table.horizontalHeader().setStretchLastSection(True)
        self.holdings_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.holdings_table.setAlternatingRowColors(True)
        self.holdings_table.setStyleSheet("QTableWidget { font-size: 11px; }")
        layout.addWidget(self.holdings_table, stretch=2)

    # ───── 绘制 ─────
    def draw(self, equity_df, benchmark_df=None, trades=None, metrics=None):
        self._results = {
            "equity": equity_df,
            "benchmark": benchmark_df,
            "trades": trades,
            "metrics": metrics or {},
        }
        self.p1.clear()
        self.p1.addLegend()

        if equity_df is None or equity_df.empty:
            return

        eq = equity_df.sort_values("date").copy()
        dates = eq["date"].tolist()
        x = np.arange(len(eq))
        nav = eq["nav"].astype(float)

        self.axis_nav.dates = dates

        # 净值曲线
        self.p1.plot(x, nav, pen=pg.mkPen("#40c4ff", width=2.5), name="策略")

        # 基准
        if benchmark_df is not None and not benchmark_df.empty:
            bench = benchmark_df.drop_duplicates("date").sort_values("date")
            bench = bench[bench["date"].isin(set(dates))]
            if not bench.empty:
                bench = bench.set_index("date").reindex(dates).reset_index()
                bnav = bench["nav"].astype(float)
                self.p1.plot(x, bnav, pen=pg.mkPen("#ffd54f", width=2), name="基准")

        # ── 调仓买卖点标注 ──
        if trades is not None and not trades.empty:
            b_dates = set()
            s_dates = set()
            for _, r in trades.iterrows():
                b_dates.add(str(r.get("buy_date", "")))
                s_dates.add(str(r.get("sell_date", "")))

            buy_x = [i for i, d in enumerate(dates) if str(d) in b_dates]
            sell_x = [i for i, d in enumerate(dates) if str(d) in s_dates]

            if buy_x:
                buy_y = [float(nav.iloc[i]) for i in buy_x if i < len(nav)]
                scatter_buy = pg.ScatterPlotItem(
                    x=buy_x[: len(buy_y)],
                    y=buy_y,
                    size=8,
                    pen=pg.mkPen("#00e676", width=1.2),
                    brush=pg.mkBrush("#00e676"),
                    symbol="t1",
                    name="买入",
                )
                self.p1.addItem(scatter_buy)

            if sell_x:
                sell_y = [float(nav.iloc[i]) for i in sell_x if i < len(nav)]
                scatter_sell = pg.ScatterPlotItem(
                    x=sell_x[: len(sell_y)],
                    y=sell_y,
                    size=8,
                    pen=pg.mkPen("#ff5252", width=1.2),
                    brush=pg.mkBrush("#ff5252"),
                    symbol="t",
                    name="卖出",
                )
                self.p1.addItem(scatter_sell)

        # ── 持仓表 ──
        self._build_holdings_table(equity_df, trades)

    # ───── 持仓明细表 ─────
    def _build_holdings_table(self, equity_df, trades):
        self.holdings_table.setRowCount(0)

        if trades is None or trades.empty:
            self.holdings_table.setRowCount(1)
            self.holdings_table.setItem(0, 0, QTableWidgetItem("无交易"))
            return

        rows = []
        for _, r in trades.iterrows():
            buy_date = str(r.get("buy_date", ""))
            sell_date = str(r.get("sell_date", ""))
            code = str(r.get("code", ""))
            pnl_pct = r.get("pnl_pct", 0)
            pnl_str = f"{pnl_pct * 100:.2f}%" if not np.isnan(pnl_pct) else "NA"
            rows.append((buy_date, sell_date, code, pnl_str))

        # 按买入日期排序
        rows.sort(key=lambda x: x[0])

        self.holdings_table.setRowCount(len(rows))
        for i, (b, s, c, p) in enumerate(rows):
            self.holdings_table.setItem(i, 0, QTableWidgetItem(b))
            self.holdings_table.setItem(i, 1, QTableWidgetItem(s))
            self.holdings_table.setItem(i, 2, QTableWidgetItem(c))
            item = QTableWidgetItem(p)
            if p != "NA":
                try:
                    if float(p.replace("%", "")) >= 0:
                        item.setForeground(QColor("#00e676"))
                    else:
                        item.setForeground(QColor("#ff5252"))
                except ValueError:
                    pass
            self.holdings_table.setItem(i, 3, item)

        self.holdings_table.resizeColumnsToContents()
