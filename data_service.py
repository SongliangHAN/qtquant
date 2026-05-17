import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import duckdb
import pandas as pd

from pytdx.hq import TdxHq_API
from curl_cffi import requests


# ═══════════════════════════════════════════
# 数据目录结构
# ═══════════════════════════════════════════
# data/
#   raw/etf/510300.parquet   ← 每只ETF一个parquet（替代daily_quote表）
#   raw/stock/               ← 股票同理
#   research/510300.parquet  ← 预计算研究数据（指标+因子+regime，回测即插即用）
#   quant.duckdb             ← DuckDB查询引擎（聚合/批量查询）
# ═══════════════════════════════════════════


class DataService:

    def __init__(self):

        # ── 目录 ──
        self.base = Path("data")
        self.raw_etf = self.base / "raw" / "etf"
        self.raw_stock = self.base / "raw" / "stock"
        self.raw_etf.mkdir(parents=True, exist_ok=True)
        self.raw_stock.mkdir(parents=True, exist_ok=True)

        # ── DuckDB 连接（查询引擎，不是数据仓库）──
        self.conn = duckdb.connect("data/quant.duckdb")

    # ══════════════════════════════════════════
    # 内部：获取parquet路径
    # ══════════════════════════════════════════

    def _parquet_path(self, code: str, asset_type: str = "ETF") -> Path:
        folder = self.raw_etf if asset_type == "ETF" else self.raw_stock
        return folder / f"{code}.parquet"

    # ══════════════════════════════════════════
    # get_conn（保留兼容）
    # ══════════════════════════════════════════

    def get_conn(self, asset_type: str):
        """保留兼容（indicators.py save_factors 仍用SQLite）。返回路径信息。"""
        return {"type": "parquet", "dir": self.raw_etf if asset_type == "ETF" else self.raw_stock}

    # ══════════════════════════════════════════
    # 保存行情 → parquet
    # ══════════════════════════════════════════

    def save_quotes(self, df: pd.DataFrame, asset_type: str = "ETF"):
        if df.empty:
            return

        code = str(df["code"].iloc[0])
        path = self._parquet_path(code, asset_type)

        df = df.copy()
        # 统一 date 格式
        df["date"] = df["date"].astype(str).str[:10]

        cols = ["date", "open", "high", "low", "close", "volume", "amount"]
        df = df[cols].sort_values("date").drop_duplicates("date")

        if path.exists():
            old = pd.read_parquet(path)
            old["date"] = old["date"].astype(str).str[:10]
            df = pd.concat([old, df], ignore_index=True)
            df = df.drop_duplicates("date").sort_values("date")

        df.to_parquet(path, index=False, compression="zstd")

    # ══════════════════════════════════════════
    # 读取行情 → 从parquet
    # ══════════════════════════════════════════

    def load_quotes(self, code: str) -> pd.DataFrame:
        """保持不变接口，内部从parquet读取。5~20x加速。"""
        code = str(code)

        # parquet路径
        for folder in (self.raw_etf, self.raw_stock):
            path = folder / f"{code}.parquet"
            if path.exists():
                df = pd.read_parquet(path)
                df["code"] = code
                df["date"] = df["date"].astype(str).str[:10]
                return df

        return pd.DataFrame()

    # ══════════════════════════════════════════
    # 获取最新日期
    # ══════════════════════════════════════════

    def get_latest_date(self, code: str, asset_type: str = "ETF") -> str | None:
        path = self._parquet_path(code, asset_type)
        if not path.exists():
            return None
        df = pd.read_parquet(path, columns=["date"])
        if df.empty:
            return None
        return df["date"].max().replace("-", "")

    # ══════════════════════════════════════════
    # DuckDB 聚合查询（批量横截面）
    # ══════════════════════════════════════════

    def query_parquet(self, sql: str) -> pd.DataFrame:
        """直接对 parquet 执行 SQL（自动并行+列裁剪+谓词下推）"""
        return self.conn.execute(sql).df()

    def load_all_etf_wide(self, start: str = "", end: str = "") -> pd.DataFrame:
        """
        一次性加载全部ETF行情（long-form），用于计算横截面因子。
        用 DuckDB 的 read_parquet glob，比逐个load+concat快10x+。
        """
        etf_dir = str(self.raw_etf / "*.parquet")
        conditions = []
        if start:
            conditions.append(f"date >= '{start}'")
        if end:
            conditions.append(f"date <= '{end}'")
        where = " AND ".join(conditions) if conditions else "1=1"

        sql = f"""
        SELECT * FROM read_parquet('{etf_dir}', filename=true)
        WHERE {where}
        ORDER BY date
        """
        return self.query_parquet(sql)

    # ══════════════════════════════════════════
    # 以下不变：下载相关
    # ══════════════════════════════════════════

    def create_tables(self):
        """兼容调用，parquet不需要建表"""
        pass

    def get_all_etf(self):
        etfs = [

            # ==================== 宽基指数 ====================
            ("510300", "沪深300ETF"),
            ("510050", "上证50ETF"),
            ("510500", "中证500ETF"),
            ("512100", "中证1000ETF"),
            ("159915", "创业板ETF"),
            ("588000", "科创50ETF"),
            ("159949", "创业板50ETF"),
            ("510180", "上证180ETF"),
            ("159901", "深证100ETF"),
            ("159905", "深证红利ETF"),

            # ==================== 风格因子 / Smart Beta ====================
            ("510880", "红利ETF"),
            ("512890", "红利低波ETF"),
            ("563280", "央企红利ETF"),
            ("512040", "价值ETF"),
            ("159581", "红利质量ETF"),
            ("560170", "央企现代能源ETF"),
            ("515180", "银行红利ETF"),
            ("560050", "央企科技ETF"),

            # ==================== 科技 / TMT ====================
            ("512480", "半导体ETF"),
            ("515880", "通信ETF"),
            ("159851", "云计算ETF"),
            ("516160", "信创ETF"),
            ("512980", "传媒ETF"),
            ("159819", "AI智能ETF"),
            ("159825", "大数据ETF"),
            ("588200", "科创芯片ETF"),
            ("159870", "5GETF"),
            ("562500", "机器人ETF"),
            ("159770", "游戏ETF"),
            ("516220", "软件ETF"),
            ("561120", "算力ETF"),
            ("560800", "数据要素ETF"),

            # ==================== 新能源 / 碳中和 ====================
            ("515790", "光伏ETF"),
            ("159875", "电池ETF"),
            ("516260", "新能源车ETF"),
            ("516850", "储能ETF"),
            ("159863", "碳中和ETF"),
            ("561810", "锂电ETF"),
            ("159642", "新能源龙头ETF"),
            ("516390", "绿色电力ETF"),

            # ==================== 医药 / 医疗 ====================
            ("512170", "医疗ETF"),
            ("159881", "创新药ETF"),
            ("516110", "中药ETF"),
            ("159882", "医疗器械ETF"),
            ("512290", "生物医药ETF"),
            ("159857", "医疗服务ETF"),
            ("159892", "恒生医药ETF"),
            ("513060", "恒生医疗ETF"),

            # ==================== 消费 ====================
            ("512690", "酒ETF"),
            ("515170", "食品饮料ETF"),
            ("516530", "家电ETF"),
            ("159766", "旅游ETF"),
            ("159928", "消费ETF"),
            ("512360", "食品ETF"),
            ("516630", "餐饮旅游ETF"),
            ("159736", "消费电子ETF"),

            # ==================== 金融地产 ====================
            ("512880", "证券ETF"),
            ("512800", "银行ETF"),
            ("512070", "非银金融ETF"),
            ("512200", "房地产ETF"),
            ("159707", "地产ETF"),
            ("512000", "券商ETF龙头"),

            # ==================== 工业 / 制造 / 军工 ====================
            ("512680", "军工ETF"),
            ("159887", "工业母机ETF"),
            ("516760", "高端装备ETF"),
            ("516950", "基建ETF"),
            ("512580", "环保ETF"),
            ("159745", "建筑材料ETF"),
            ("512670", "国防ETF"),
            ("516800", "智能制造ETF"),

            # ==================== 周期 / 资源 ====================
            ("512400", "有色金属ETF"),
            ("516150", "煤炭ETF"),
            ("159861", "化工ETF"),
            ("159667", "钢铁ETF"),
            ("516780", "稀土ETF"),
            ("159985", "能源化工ETF"),
            ("516970", "建材ETF"),
            ("516220", "煤化工ETF"),

            # ==================== 农业 / 公用事业 ====================
            ("159825", "农业ETF"),
            ("516880", "畜牧ETF"),
            ("560610", "水利ETF"),
            ("560980", "电力ETF"),

            # ==================== 海外市场 ====================
            ("513100", "纳指ETF"),
            ("513500", "标普500ETF"),
            ("159920", "恒生ETF"),
            ("513130", "恒生科技ETF"),
            ("513050", "港股科技ETF"),
            ("513520", "日经ETF"),
            ("513030", "德国ETF"),
            ("513080", "法国ETF"),
            ("159605", "中概互联ETF"),
            ("513090", "港股通ETF"),
            ("513880", "日经225ETF"),
            ("513660", "港股通红利ETF"),

            # ==================== 商品 / 另类资产 ====================
            ("518880", "黄金ETF"),
            ("518800", "黄金股ETF"),
            ("159980", "有色商品ETF"),
            ("159981", "能源化工ETF商品"),
            ("159937", "黄金现货ETF"),

            # ==================== 债券 ====================
            ("511260", "十年国债ETF"),
            ("511270", "三十年国债ETF"),
            ("511030", "公司债ETF"),
            ("511580", "政金债ETF"),
            ("511220", "城投债ETF"),
            ("159816", "地方债ETF"),
            ("511010", "国债ETF"),

            # ==================== REITs / 另类收益 ====================
            ("508000", "REITsETF"),
            ("508056", "产业园REIT"),
            ("508099", "仓储物流REIT"),

        ]
        result = []
        for code, name in etfs:
            result.append({"code": code, "name": name})
        return result

    def get_all_a_stocks(self):
        result = []
        for i in range(600000, 605000):
            result.append({"code": str(i), "name": ""})
        for i in range(1, 4000):
            result.append({"code": str(i).zfill(6), "name": ""})
        for i in range(300000, 301500):
            result.append({"code": str(i), "name": ""})
        return result

    def get_market(self, code):
        code = str(code)
        if code.startswith(("5", "6", "9")):
            return 1
        return 0

    def pytdx_daily(self, code):
        api = TdxHq_API()
        hosts = [
            ("119.147.212.81", 7709),
            ("119.147.212.83", 7709),
            ("60.191.117.167", 7709),
            ("218.108.47.69", 7709),
            ("119.147.171.206", 443),
        ]
        for host, port in hosts:
            try:
                ok = api.connect(host, port)
                if not ok:
                    continue
                market = self.get_market(code)
                all_data = []
                for start in range(0, 10000, 800):
                    bars = api.get_security_bars(9, market, code, start, 800)
                    if bars is None:
                        break
                    if len(bars) == 0:
                        break
                    data = api.to_df(bars)
                    if data.empty:
                        break
                    all_data.append(data)
                    if len(data) < 800:
                        break
                api.disconnect()
                if not all_data:
                    continue
                data = pd.concat(all_data, ignore_index=True)
                data = data.rename(columns={"datetime": "date", "vol": "volume"})
                data["date"] = data["date"].astype(str).str[:10]
                data["code"] = code
                data = data.drop_duplicates(subset=["date"]).sort_values("date")
                return data[["code", "date", "open", "high", "low", "close", "volume", "amount"]]
            except Exception as e:
                print(f"pytdx失败 {host}:{port}", e)
        return pd.DataFrame()

    def eastmoney_daily(self, code, beg, end):
        try:
            secid = f"{self.get_market(code)}.{code}"
            url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
            params = {
                "fields1": "f1,f2,f3,f4,f5,f6",
                "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
                "beg": beg, "end": end, "rtntype": 6,
                "secid": secid, "klt": 101, "fqt": 1,  # 前复权
            }
            r = requests.get(url, params=params, impersonate="chrome124", timeout=20)
            js = r.json()
            if not js.get("data"):
                return pd.DataFrame()
            rows = []
            for item in js["data"]["klines"]:
                arr = item.split(",")
                rows.append({
                    "code": code, "date": arr[0],
                    "open": float(arr[1]), "close": float(arr[2]),
                    "high": float(arr[3]), "low": float(arr[4]),
                    "volume": float(arr[5]), "amount": float(arr[6]),
                })
            return pd.DataFrame(rows)
        except Exception as e:
            print("东财失败:", e)
            return pd.DataFrame()

    def tencent_realtime(self, code):
        try:
            symbol = f"sh{code}" if code.startswith(("6", "5")) else f"sz{code}"
            url = f"https://qt.gtimg.cn/q={symbol}"
            r = requests.get(url, impersonate="chrome124", timeout=10)
            return r.text
        except Exception as e:
            print("腾讯失败:", e)
            return ""

    def download_daily(self, code, beg, end):
        try:
            df = self.pytdx_daily(code)
            if not df.empty:
                return df
        except Exception as e:
            print(f"{code} pytdx失败:", e)
        try:
            df = self.eastmoney_daily(code, beg, end)
            if not df.empty:
                return df
        except Exception as e:
            print(f"{code} 东财失败:", e)
        try:
            self.tencent_realtime(code)
        except Exception:
            pass
        return pd.DataFrame()


# ═══════════════════════════════════════════
# ResearchDataService：预计算研究数据
# ═══════════════════════════════════════════

class ResearchDataService:
    """
    预计算"回测即插即用"parquet文件。
    每只ETF一个research parquet，包含：
      date, open, close, volume, amount,
      + 全部时间序列指标（ma, macd, roc, volatility...）
      + 横截面因子（cross_momentum, cross_volatility, barra_beta, corr_hs300_60）
      + 市场状态（regime, breadth）
    build_backtest_data直接读这个，无需重复计算/merge。
    """

    def __init__(self, ds: DataService):
        self.ds = ds
        self.base = Path("data") / "research"
        self.base.mkdir(parents=True, exist_ok=True)

    def path_for(self, code: str) -> Path:
        return self.base / f"{code}.parquet"

    def exists(self, code: str) -> bool:
        return self.path_for(code).exists()

    def load(self, code: str) -> pd.DataFrame:
        """加载单只ETF的研究数据（已含全部指标和因子）"""
        p = self.path_for(code)
        if not p.exists():
            return pd.DataFrame()
        return pd.read_parquet(p)

    def load_many(self, codes: list[str], start: str = "",
                  end: str = "", max_workers: int = 8) -> pd.DataFrame:
        """批量并行加载 parquet 文件"""
        def _load_one(code):
            p = self.path_for(code)
            if not p.exists():
                return None
            df = pd.read_parquet(p)
            if start:
                df = df[df["date"] >= start]
            if end:
                df = df[df["date"] <= end]
            if df.empty:
                return None
            df["code"] = code
            return df

        frames = []
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(_load_one, code): code for code in codes}
            for f in futures:
                df = f.result()
                if df is not None:
                    frames.append(df)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True).sort_values(["date", "code"]).reset_index(drop=True)

    def build(self, code: str,
              indicator_engine,
              cross_factors: pd.DataFrame | None = None,
              regime_df: pd.DataFrame | None = None,
              breadth_df: pd.DataFrame | None = None) -> bool:
        """
        为单只ETF构建research parquet。
        cross_factors: 预加载的全部横截面因子（含corr_hs300_60等）
        """
        df = self.ds.load_quotes(code)
        if df.empty:
            return False

        # 时间序列指标
        df = indicator_engine.calculate(df)

        # 移除占位列（由横截面阶段填充真实值）
        if "relative_strength_vs_hs300" in df.columns:
            df = df.drop(columns=["relative_strength_vs_hs300"])

        # 横截面因子 merge
        if cross_factors is not None and not cross_factors.empty:
            cross = cross_factors[cross_factors["code"] == code]
            if not cross.empty:
                # 去重：防止 vectorized cross factor 中同(date,factor_name)重复行
                cross = cross.drop_duplicates(["date", "factor_name"])
                pivot = cross.pivot(
                    index="date", columns="factor_name", values="factor_value"
                ).reset_index()
                df = pd.merge(df, pivot, on="date", how="left")

        # 市场状态 merge
        if regime_df is not None and not regime_df.empty:
            df = pd.merge(df, regime_df, on="date", how="left")

        if breadth_df is not None and not breadth_df.empty:
            df = pd.merge(df, breadth_df, on="date", how="left")
            # 兼容旧字段：breadth_smooth → breadth
            if "breadth_smooth" in df.columns:
                df = df.rename(columns={"breadth_smooth": "breadth"})
            # market_score 的 breadth_up 也可充当 breadth 列（兼容旧逻辑）
            if "breadth" not in df.columns and "breadth_up" in df.columns:
                df["breadth"] = df["breadth_up"]
            if "market_score" not in df.columns and "breadth_up" in df.columns:
                # 兜底market_score（expanding dispersion norm，无未来泄露）
                d_raw = df.get("breadth_dispersion", pd.Series([0.5] * len(df)))
                if isinstance(d_raw, pd.Series) and len(d_raw) > 1:
                    d_expand_min = d_raw.expanding().min()
                    d_expand_max = d_raw.expanding().max()
                    gap = d_expand_max - d_expand_min
                    d_norm = pd.Series(0.5, index=d_raw.index)
                    d_norm[gap > 0] = (d_raw[gap > 0] - d_expand_min[gap > 0]) / gap[gap > 0]
                else:
                    d_norm = 0.5
                nh = df.get("breadth_newhigh", pd.Series([0.5] * len(df)))
                df["market_score"] = 0.35 * df.get("breadth_ma20", 0.5) + \
                                      0.35 * df.get("breadth_roc20", 0.5) + \
                                      0.15 * df.get("breadth_up", 0.5) + \
                                      0.15 * nh - \
                                      0.10 * d_norm

        # 去重+排序+去掉重复列名
        df = df.loc[:, ~df.columns.duplicated()].copy()
        df = df.drop_duplicates("date").sort_values("date")

        # 写入 parquet（原子写入：先写临时文件再rename，防止读脏）
        out = self.path_for(code)
        tmp = out.with_suffix(".tmp")
        df.to_parquet(tmp, index=False, compression="zstd")
        tmp.replace(out)
        return True

    def build_all(self, codes: list[str],
                  indicator_engine,
                  factor_engine,
                  log_cb=None) -> int:
        """
        批量构建全部ETF的research parquet。
        这是"下载完成 → 一键预处理"的核心入口。
        """
        from market_regime import MarketRegimeDetector

        built = 0
        all_data = []

        # 第一步：加载全部行情 + 计算指标
        for code in codes:
            df = self.ds.load_quotes(code)
            if df.empty:
                continue
            df = indicator_engine.calculate(df)
            all_data.append(df)
            if log_cb:
                log_cb(f"指标计算 {code}")

        if not all_data:
            return 0

        full = pd.concat(all_data, ignore_index=True)
        if log_cb:
            log_cb(f"全部指标完成，共 {len(full)} 行，开始横截面因子...")

        # 第二步：HS300 relative strength
        bench = full[full["code"] == "510300"][["date", "ret20"]]
        if not bench.empty:
            bench = bench.rename(columns={"ret20": "hs300_ret20"})
            full = pd.merge(full, bench, on="date", how="left")
            full["relative_strength_vs_hs300"] = full["ret20"] - full["hs300_ret20"]

            # 转换为横截面格式以通过 build() → parquet 持久化
            rel_str = full[["date", "code"]].copy()
            rel_str["factor_name"] = "relative_strength_vs_hs300"
            rel_str["factor_value"] = full["relative_strength_vs_hs300"]
            rel_str = rel_str.dropna(subset=["factor_value"])

        # 第三步：横截面因子
        cross_mom = factor_engine.cross_section_momentum(full)
        cross_vol = factor_engine.cross_volatility(full)

        # corr_hs300
        hs300_data = full[full["code"] == "510300"][["date", "ret"]].drop_duplicates("date")
        hs300_ret = hs300_data.set_index("date")["ret"] if not hs300_data.empty else None
        corr = factor_engine.corr_hs300(full, hs300_ret=hs300_ret)

        # barra_beta
        beta = factor_engine.barra_beta(full, hs300_df=hs300_data)

        # 合并全部横截面
        cross_all = pd.concat([cross_mom, cross_vol, corr, beta, rel_str], ignore_index=True)
        if log_cb:
            log_cb(f"横截面因子完成，共 {len(cross_all)} 条")

        # 第四步：regime + breadth + market_score
        bench_raw = self.ds.load_quotes("510300")
        regime_df = pd.DataFrame()
        breadth_df = pd.DataFrame()
        if not bench_raw.empty:
            detector = MarketRegimeDetector()
            regime_df = detector.detect_smooth(bench_raw, persistence=5)
            # 计算 market_score（含5个breadth子指标）
            breadth_df = detector.calc_market_score(full)

        # 第五步：逐code写入
        for code in codes:
            success = self.build(
                code, indicator_engine,
                cross_factors=cross_all,
                regime_df=regime_df,
                breadth_df=breadth_df,
            )
            if success:
                built += 1
                if log_cb:
                    log_cb(f"研究数据已保存 {code}")

        return built
