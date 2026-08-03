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


def build_popularity(days: int = 7, k: int = 20, val_days: int = 7) -> list[int]:
    """返回近期最热的 k 个 aid(按出现次数)。

    防泄漏:只统计验证窗口边界**之前**的 days 天,不数进验证未来事件
    (否则被大量下单/加购的答案商品会虚高地进热门兜底,轻微抬分)。
    """
    lf = pl.scan_parquet(TRAIN_PARQUET)
    max_ts = lf.select(pl.col("ts").max()).collect().item()
    boundary = max_ts - val_days * DAY_MS         # 验证窗口起点 = 预测时刻
    lo = boundary - days * DAY_MS                 # 只看预测时刻之前的 days 天
    top = (lf.filter((pl.col("ts") >= lo) & (pl.col("ts") < boundary))
             .group_by("aid").agg(pl.len().alias("cnt"))
             .sort("cnt", descending=True)
             .head(k)
             .collect())
    return top["aid"].to_list()
