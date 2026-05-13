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

        # ── 表格区 ──
        table_splitter = QSplitter(Qt.Horizontal)

        # 持仓明细
        self.holdings_table = QTableWidget()
        self.holdings_table.setColumnCount(4)
        self.holdings_table.setHorizontalHeaderLabels(["开始日期", "结束日期", "持仓标的", "期间收益率"])
        self.holdings_table.horizontalHeader().setStretchLastSection(True)
        self.holdings_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.holdings_table.setAlternatingRowColors(True)
        self.holdings_table.setStyleSheet("QTableWidget { font-size: 11px; }")
        table_splitter.addWidget(self.holdings_table)

        # 分阶段收益
        self.return_table = QTableWidget()
        self.return_table.setColumnCount(4)
        self.return_table.setHorizontalHeaderLabels(["阶段", "开始-结束", "策略收益", "基准收益"])
        self.return_table.horizontalHeader().setStretchLastSection(True)
        self.return_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.return_table.setAlternatingRowColors(True)
        self.return_table.setStyleSheet("QTableWidget { font-size: 11px; }")
        table_splitter.addWidget(self.return_table)

        layout.addWidget(table_splitter, stretch=2)

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

        # ── 收益率表 ──
        self._build_return_table(equity_df, benchmark_df)

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

    # ───── 分阶段收益率表 ─────
    def _build_return_table(self, equity_df, benchmark_df=None):
        self.return_table.setRowCount(0)

        if equity_df is None or equity_df.empty:
            self.return_table.setRowCount(1)
            self.return_table.setItem(0, 0, QTableWidgetItem("无数据"))
            return

        eq = equity_df.sort_values("date")
        nav = eq["nav"].astype(float)

        # 分阶段：月度、季度、年度
        eq2 = eq.copy()
        eq2["date"] = pd.to_datetime(eq2["date"])
        eq2["nav"] = nav

        periods = []

        # 月度
        monthly = eq2.resample("ME", on="date").agg({"nav": "last", "date": "last"}).dropna()
        monthly["ret"] = monthly["nav"].pct_change()
        for i in range(1, len(monthly)):
            label = f"月度{i}"
            s_date = str(monthly["date"].iloc[i - 1])[:10]
            e_date = str(monthly["date"].iloc[i])[:10]
            r = monthly["ret"].iloc[i]
            periods.append((label, f"{s_date} ~ {e_date}", r))

        # 年度
        yearly = eq2.resample("YE", on="date").agg({"nav": "last", "date": "last"}).dropna()
        yearly["ret"] = yearly["nav"].pct_change()
        for i in range(1, len(yearly)):
            label = f"年度{i}"
            s_date = str(yearly["date"].iloc[i - 1])[:10]
            e_date = str(yearly["date"].iloc[i])[:10]
            r = yearly["ret"].iloc[i]
            periods.append((label, f"{s_date} ~ {e_date}", r))

        # 全期间
        total_ret = nav.iloc[-1] / nav.iloc[0] - 1
        periods.append((
            "全期间",
            f"{str(eq['date'].iloc[0])[:10]} ~ {str(eq['date'].iloc[-1])[:10]}",
            total_ret,
        ))

        # 基准收益率（对齐）
        bench_rets = {}
        if benchmark_df is not None and not benchmark_df.empty:
            bm = benchmark_df.drop_duplicates("date").sort_values("date").copy()
            bm["date"] = pd.to_datetime(bm["date"])
            bnav = bm["nav"].astype(float)
            # 月度基准
            bm_m = bm.resample("ME", on="date").agg({"nav": "last", "date": "last"}).dropna()
            bm_m["ret"] = bm_m["nav"].pct_change()
            for i in range(1, len(bm_m)):
                key = f"月度{i}"
                bench_rets[key] = bm_m["ret"].iloc[i]
            # 年度基准
            bm_y = bm.resample("YE", on="date").agg({"nav": "last", "date": "last"}).dropna()
            bm_y["ret"] = bm_y["nav"].pct_change()
            for i in range(1, len(bm_y)):
                key = f"年度{i}"
                bench_rets[key] = bm_y["ret"].iloc[i]
            # 全期间基准
            if len(bnav) >= 2:
                bench_rets["全期间"] = bnav.iloc[-1] / bnav.iloc[0] - 1

        self.return_table.setRowCount(len(periods))
        for i, (label, dr, r) in enumerate(periods):
            self.return_table.setItem(i, 0, QTableWidgetItem(label))
            self.return_table.setItem(i, 1, QTableWidgetItem(dr))

            r_str = f"{r * 100:.2f}%"
            item = QTableWidgetItem(r_str)
            if r >= 0:
                item.setForeground(QColor("#00e676"))
            else:
                item.setForeground(QColor("#ff5252"))
            self.return_table.setItem(i, 2, item)

            # 基准
            br = bench_rets.get(label)
            if br is not None:
                br_str = f"{br * 100:.2f}%"
                bitem = QTableWidgetItem(br_str)
                if br >= 0:
                    bitem.setForeground(QColor("#00e676"))
                else:
                    bitem.setForeground(QColor("#ff5252"))
                self.return_table.setItem(i, 3, bitem)
            else:
                self.return_table.setItem(i, 3, QTableWidgetItem("-"))

        self.return_table.resizeColumnsToContents()
