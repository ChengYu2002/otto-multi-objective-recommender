"""
Phase 1 · 数据准备:嵌套 JSONL -> 高效 long-format parquet
==========================================================
原始格式(每行一个 session):
    {"session": 0, "events": [{"aid":123,"ts":1659..,"type":"clicks"}, ...]}

转成"一行一个事件"的长表,列 = [session, aid, ts, type],type 编码 0/1/2。
好处:体积骤降、读取飞快,后面召回/特征/评估都基于它。

关键工程点:11GB JSON 不能一次性读进内存 —— 按行分块(chunk),
每块 read_ndjson -> explode -> 写一个 parquet part,通过分块处理与列式存储
控制峰值内存占用。

跑法:
    conda activate otto
    python src/data_prep.py            # 转 train + test 全量
    python src/data_prep.py --sample   # 只转前 2 万行,快速验证
"""
import sys
import time
from pathlib import Path

import polars as pl

TYPE_MAP = {"clicks": 0, "carts": 1, "orders": 2}
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def _flush(lines: list[bytes], out_dir: Path, part: int) -> int:
    """把一批 JSONL 行转成长表并写成一个 parquet part,返回事件行数。"""
    df = (
        pl.read_ndjson(b"".join(lines))
        .explode("events")          # 每个 event 拆成一行
        .unnest("events")           # 展开 struct -> aid / ts / type 三列
        .with_columns(
            pl.col("session").cast(pl.Int32),
            pl.col("aid").cast(pl.Int32),
            pl.col("ts").cast(pl.Int64),                       # 毫秒时间戳
            pl.col("type").replace_strict(TYPE_MAP, return_dtype=pl.Int8),
        )
    )
    df.write_parquet(out_dir / f"part_{part:04d}.parquet")
    return df.height


def convert(name: str, chunk_lines: int = 250_000, limit: int | None = None) -> None:
    src = DATA / f"{name}.jsonl"
    out_dir = DATA / "parquet" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.parquet"):      # 清掉上次的 part,避免混淆
        old.unlink()

    t0 = time.time()
    total_ev = total_sess = part = read = 0
    buf: list[bytes] = []
    with open(src, "rb") as f:
        for line in f:
            buf.append(line)
            read += 1
            if len(buf) >= chunk_lines:
                total_ev += _flush(buf, out_dir, part)
                total_sess += len(buf)
                part += 1
                buf = []
                print(f"[{name}] part {part} | sessions {total_sess:,} "
                      f"| events {total_ev:,} | {time.time()-t0:.0f}s", flush=True)
            if limit and read >= limit:
                break
        if buf:
            total_ev += _flush(buf, out_dir, part)
            total_sess += len(buf)
            part += 1

    print(f"[{name}] ✅ 完成:{total_sess:,} sessions, {total_ev:,} events, "
          f"{part} parts, 用时 {time.time()-t0:.0f}s", flush=True)


def show_stats(name: str) -> None:
    """扫一遍 parquet,打印数据全貌 —— 设计验证集前必须先了解数据。"""
    lf = pl.scan_parquet(DATA / "parquet" / name / "*.parquet")
    stats = lf.select(
        n_events=pl.len(),
        n_sessions=pl.col("session").n_unique(),
        n_aids=pl.col("aid").n_unique(),
        ts_min=pl.col("ts").min(),
        ts_max=pl.col("ts").max(),
    ).collect()
    row = stats.row(0, named=True)
    span_days = (row["ts_max"] - row["ts_min"]) / 1000 / 86400
    print(f"\n===== {name} 数据全貌 =====")
    print(f"事件数   : {row['n_events']:,}")
    print(f"session数: {row['n_sessions']:,}")
    print(f"商品数   : {row['n_aids']:,}")
    print(f"时间跨度 : {span_days:.1f} 天  "
          f"({pl.from_epoch(pl.Series([row['ts_min']]), 'ms')[0]} ~ "
          f"{pl.from_epoch(pl.Series([row['ts_max']]), 'ms')[0]})")
    dist = (lf.group_by("type").agg(pl.len().alias("cnt"))
              .sort("type").collect())
    inv = {v: k for k, v in TYPE_MAP.items()}
    for r in dist.iter_rows(named=True):
        print(f"  {inv[r['type']]:<7}: {r['cnt']:,}")


if __name__ == "__main__":
    sample = "--sample" in sys.argv
    lim = 20_000 if sample else None
    for nm in ("test", "train"):        # 先转小的 test,快;train 最后
        convert(nm, limit=lim)
        show_stats(nm)
