"""
Phase 2 · 召回 · 热门兜底
========================
最朴素的一路:近期(train 尾部 days 天)最高频的 k 个商品。
用途:候选不满 20 时补齐,救活冷启动 / 超短 session。它不该抢真候选的位置,
所以在 candidates.py 里排在真候选之后。
"""
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
TRAIN_PARQUET = DATA / "parquet" / "train" / "*.parquet"
DAY_MS = 86_400_000


def build_popularity(days: int = 7, k: int = 20) -> list[int]:
    """返回近期最热的 k 个 aid(按出现次数)。"""
    lf = pl.scan_parquet(TRAIN_PARQUET)
    max_ts = lf.select(pl.col("ts").max()).collect().item()
    boundary = max_ts - days * DAY_MS
    top = (lf.filter(pl.col("ts") >= boundary)
             .group_by("aid").agg(pl.len().alias("cnt"))
             .sort("cnt", descending=True)
             .head(k)
             .collect())
    return top["aid"].to_list()
