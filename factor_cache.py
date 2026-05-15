"""
因子研究缓存层 — 避免重复计算。

将 FactorResearchEngine 的完整计算结果持久化到 parquet，
下次加载时直接读取缓存。当 codes 列表或日期范围变化时自动失效。
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from datetime import datetime

import pandas as pd


class FactorCache:
    """管理因子研究结果的磁盘缓存。"""

    def __init__(self, base_dir: str | Path = "data/research"):
        self.base_dir = Path(base_dir)
        self.cache_dir = self.base_dir / "factor_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.cache_dir / "cache_meta.json"
        self.metrics_path = self.base_dir / "factor_metrics.parquet"

    # ── 缓存元数据 ─────────────────────────────────────────────────────

    def _compute_cache_key(self, codes: list[str], start: str, end: str) -> str:
        raw = f"{sorted(codes)}|{start}|{end}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _load_meta(self) -> dict:
        if self.meta_path.exists():
            return json.loads(self.meta_path.read_text(encoding="utf-8"))
        return {}

    def _save_meta(self, meta: dict):
        self.meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    # ── 缓存校验 ─────────────────────────────────────────────────────

    def is_valid(self, codes: list[str], start: str, end: str) -> bool:
        """检查缓存是否对当前参数有效。"""
        meta = self._load_meta()
        if not meta:
            return False
        key = self._compute_cache_key(codes, start, end)
        if meta.get("cache_key") != key:
            return False
        # 检查关键文件是否存在
        for fname in meta.get("files", []):
            if not (self.cache_dir / fname).exists():
                return False
        return self.metrics_path.exists()

    def mark_valid(self, codes: list[str], start: str, end: str, files: list[str]):
        """标记缓存为有效。"""
        meta = {
            "cache_key": self._compute_cache_key(codes, start, end),
            "date_range": f"{start} → {end}",
            "n_codes": len(codes),
            "created_at": datetime.now().isoformat(),
            "files": files,
        }
        self._save_meta(meta)

    # ── factor_report_df ──────────────────────────────────────────────

    def load_metrics(self) -> pd.DataFrame | None:
        """加载缓存的 factor_report_df。"""
        if self.metrics_path.exists():
            return pd.read_parquet(self.metrics_path)
        return None

    def save_metrics(self, df: pd.DataFrame):
        """原子写入 factor_report_df。"""
        tmp = self.metrics_path.with_suffix(".parquet.tmp")
        df.to_parquet(tmp, index=False, compression="zstd")
        tmp.replace(self.metrics_path)

    # ── 其他计算结果 ──────────────────────────────────────────────────

    def load_cached(self, name: str) -> pd.DataFrame | None:
        """加载单个缓存结果。"""
        p = self.cache_dir / f"{name}.parquet"
        if p.exists():
            return pd.read_parquet(p)
        return None

    def save_cached(self, name: str, df: pd.DataFrame):
        """保存单个计算结果。"""
        p = self.cache_dir / f"{name}.parquet"
        tmp = p.with_suffix(".parquet.tmp")
        df.to_parquet(tmp, index=False, compression="zstd")
        tmp.replace(p)

    # ── 工具 ──────────────────────────────────────────────────────────

    def invalidate(self):
        """清除所有缓存。"""
        meta = self._load_meta()
        for fname in meta.get("files", []):
            fp = self.cache_dir / fname
            if fp.exists():
                fp.unlink()
        if self.metrics_path.exists():
            self.metrics_path.unlink()
        if self.meta_path.exists():
            self.meta_path.unlink()

    def cached_file_names(self) -> list[str]:
        """返回已缓存的 parquet 文件名。"""
        return sorted(p.name for p in self.cache_dir.glob("*.parquet") if not p.name.endswith(".tmp"))
