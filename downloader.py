import traceback
import time
import random
import pandas as pd
from PySide6.QtCore import *


class DownloadSignals(QObject):

    progress = Signal(int, int)

    log = Signal(str)

    finished = Signal()

    failed = Signal(str)

    # 下载成功后发出，携带成功更新的 code 列表（用于自动重建research）
    updated = Signal(list)


class DownloadWorker(QRunnable):

    def __init__(
            self,
            ds,
            codes,
            asset_type,
            beg,
            end
    ):

        super().__init__()

        self.ds = ds

        self.codes = codes

        self.asset_type = asset_type

        self.beg = beg

        self.end = end

        self.signals = DownloadSignals()

        self.is_paused = False

        self.is_running = True

        self.failed_list = []

        self.updated_codes = []  # 成功更新的code列表

    # =====================================
    # 暂停
    # =====================================

    def pause(self):

        self.is_paused = True

    # =====================================
    # 继续
    # =====================================

    def resume(self):

        self.is_paused = False

    # =====================================
    # 停止
    # =====================================

    def stop(self):

        self.is_running = False

    # =====================================
    # run
    # =====================================

    def run(self):

        total = len(self.codes)

        for i, item in enumerate(self.codes):

            if not self.is_running:
                break

            while self.is_paused:

                QThread.msleep(200)

            code = item["code"]

            name = item.get("name", "")

            try:

                self.signals.log.emit(
                    f"[{i+1}/{total}] "
                    f"下载 {code} {name}"
                )

                # =========================
                # 增量更新
                # =========================

                latest = self.ds.get_latest_date(
                    code,
                    self.asset_type
                )

                beg = self.beg

                if latest:

                    beg = latest.replace("-", "")

                # 下载（重试3次 + 随机等待限速）
                df = pd.DataFrame()
                for attempt in range(3):
                    time.sleep(random.uniform(0.2, 0.8))
                    df = self.ds.download_daily(code, beg, self.end)
                    if not df.empty:
                        break
                    if attempt < 2:
                        self.signals.log.emit(f"{code} 第{attempt+1}次重试...")
                        time.sleep(random.uniform(1.0, 3.0))

                # =========================
                # 保存
                # =========================

                self.ds.save_quotes(
                    df,
                    self.asset_type
                )

                self.signals.log.emit(
                    f"{code} 完成 "
                    f"{len(df)} 条"
                )

                self.updated_codes.append(code)

            except Exception as e:

                self.failed_list.append(code)

                self.signals.failed.emit(
                    f"{code} 失败: {e}"
                )

                print(traceback.format_exc())

            self.signals.progress.emit(
                i + 1,
                total
            )

        # =============================
        # 失败列表
        # =============================

        if self.failed_list:

            self.signals.log.emit(
                "失败列表:"
            )

            for code in self.failed_list:

                self.signals.log.emit(code)

        # 发出更新列表（用于自动重建 research）
        if self.updated_codes:
            self.signals.updated.emit(self.updated_codes)

        self.signals.finished.emit()