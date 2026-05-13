from PySide6.QtCore import QObject, Signal, QRunnable


class OptimizerSignals(QObject):
    log = Signal(str)
    progress = Signal(int, int, float)  # cur, total, best
    finished = Signal(dict)            # {"strategy": dict, "best_score": float, "best_params": dict, "detail": list}
    failed = Signal(str)


class OptimizerWorker(QRunnable):
    def __init__(self, optimize_fn, *, build_data_fn, build_strategy_fn, space: dict, config: dict):
        super().__init__()
        self.signals = OptimizerSignals()
        self.optimize_fn = optimize_fn
        self.build_data_fn = build_data_fn
        self.build_strategy_fn = build_strategy_fn
        self.space = space
        self.config = config

    def run(self):
        try:
            self.signals.log.emit("开始构建数据...")

            # build_data_fn 在 worker 线程中运行，不应创建任何 QWidget
            data = self.build_data_fn()
            if data is None or data.empty:
                self.signals.failed.emit("没有可用于优化的数据（请先下载数据，并构建研究数据）")
                return

            self.signals.log.emit(f"数据准备完成：{len(data)} 行，开始搜索...")

            def cb(cur, total, best):
                self.signals.progress.emit(int(cur), int(total), float(best))

            res = self.optimize_fn(
                self.config["bt_engine"],
                self.build_strategy_fn,
                data,
                space=self.space,
                n_trials=self.config["n_trials"],
                seed=self.config.get("seed", 42),
                train_days=self.config["train_days"],
                test_days=self.config["test_days"],
                top_k=self.config["top_k"],
                max_search_date=self.config["max_search_date"],
                commission_bps=self.config["commission_bps"],
                slippage_bps=self.config["slippage_bps"],
                progress_cb=cb,
            )

            self.signals.log.emit("搜索计算结束，准备生成策略并发送结果...")

            best_params = res.best_params or {}
            stg = self.build_strategy_fn(best_params)
            self.signals.finished.emit(
                {
                    "strategy": stg,
                    "best_score": float(res.best_score),
                    "best_params": best_params,
                    "detail": res.detail,
                }
            )
            self.signals.log.emit("结果已发送（finished）")
        except Exception as e:
            import traceback
            full_tb = traceback.format_exc()
            print(full_tb)
            self.signals.failed.emit(full_tb)
