"""
Phase 2 · 召回 · co-visitation 矩阵(Step 1 只做 click 这一路)
===========================================================
本质:数一遍"谁和谁常在同一 session 里一起出现",再对每个商品只保留最强的
top-N 邻居。不训练参数,纯统计。

工程关键(面试的"几个 G 内存不够怎么办"):
  全量 train 两两成对会爆内存 → 两招压住:
    1) 截断 session:每个 session 只留最近 ~30 个事件;
    2) 按 session 分块(session % n_chunks):一块块算,每块先聚合成
       [aid_x, aid_y, wgt] 再落盘,最后合并所有块 + 截 top-N。
  这样任一时刻内存里只有"一块的两两对",不是全量。

跑法:
    python src/recall/run_recall.py            # 会自动建 click 矩阵
    # 或单独建:python -c "from src.recall.covis import build_covis; build_covis()"
"""
import shutil
import time
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]          # src/recall/covis.py -> OTTO/
DATA = ROOT / "data"
TRAIN_PARQUET = DATA / "parquet" / "train" / "*.parquet"
VAL_INPUT = DATA / "parquet" / "val" / "input.parquet"
COVIS_DIR = DATA / "parquet" / "covis"
CORPUS_PARQUET = DATA / "parquet" / "covis_corpus.parquet"
DAY_MS = 86_400_000
VAL_DAYS = 7                                        # 必须与 make_validation_set 的 val_days 一致

# 三张矩阵的差异全在:时间窗口 / 事件过滤 / 邻居数 / 加权(见 _weight_expr)
# Step 1 只启用 click;buy_weighted 和 buy2buy 留到 Step 2。
COVIS_CONFIG = {
    "click": {"window": 60 * 60 * 1000, "types": None, "top_n": 20},   # 1 小时,不过滤
    # "buy_weighted": {"window": 60 * 60 * 1000, "types": None,   "top_n": 20},
    # "buy2buy":      {"window": 14 * DAY_MS,     "types": [1, 2], "top_n": 20},
}


def _weight_expr(kind: str, tmin: int, tmax: int) -> pl.Expr:
    """每个共现对的权重。click:近期加权(邻居事件越新权重越高,1~4)。"""
    if kind == "click":
        return 1 + 3 * (pl.col("ts_y") - tmin) / (tmax - tmin)
    raise NotImplementedError(f"Step 2 再实现 {kind} 的加权")


def build_corpus(force: bool = False) -> Path:
    """
    建 co-vis 语料 —— 防验证泄漏的关键一步。
    ----------------------------------------------------------------
    语料 = 历史(ts < 窗口边界)⊕ val_input(验证 session 的可见前半段),
    刻意**排除所有验证"未来"事件**(即 val_labels 那部分)。

    为什么:若直接读原始 train,某个验证 session 的答案(cutoff 之后的事件)
    会和它的输入一起被数进共现矩阵,等于"用答案预测答案"——target leakage。
    只保留每个 session 在预测时刻能看到的事件,泄漏就不存在了。
    """
    if CORPUS_PARQUET.exists() and not force:
        return CORPUS_PARQUET
    lf = pl.scan_parquet(TRAIN_PARQUET)
    max_ts = lf.select(pl.col("ts").max()).collect().item()
    boundary = max_ts - VAL_DAYS * DAY_MS
    history = lf.filter(pl.col("ts") < boundary)        # 窗口前的全部历史(无未来)
    visible = pl.scan_parquet(VAL_INPUT)                # 验证 session 的可见事件(已剔除未来)
    CORPUS_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    pl.concat([history, visible]).sink_parquet(CORPUS_PARQUET)   # 流式写,省内存
    print(f"[corpus] ✅ 防泄漏语料已建(历史 + val_input,排除未来)"
          f"-> {CORPUS_PARQUET.name}", flush=True)
    return CORPUS_PARQUET


def build_covis(kind: str = "click", n_chunks: int = 30,
                max_chunks: int | None = None, session_cap: int = 30,
                rebuild_corpus: bool = False):
    """建一张 co-vis 矩阵并落盘为 data/parquet/covis/{kind}.parquet。

    n_chunks       : 按 session 分成多少块(越多越省内存、越慢)
    max_chunks     : 只处理前几块(采样,快速验证用);None = 全部
    session_cap    : 每个 session 只保留最近多少个事件
    rebuild_corpus : 强制重建防泄漏语料(改了 val 切分后需要)
    """
    cfg = COVIS_CONFIG[kind]
    t0 = time.time()
    build_corpus(force=rebuild_corpus)              # ← 用防泄漏语料,不是原始 train
    lf = pl.scan_parquet(CORPUS_PARQUET)
    if cfg["types"] is not None:                    # buy2buy 只看加购/下单
        lf = lf.filter(pl.col("type").is_in(cfg["types"]))
    tmin, tmax = lf.select(pl.col("ts").min().alias("mn"),
                           pl.col("ts").max().alias("mx")).collect().row(0)

    tmp = COVIS_DIR / f"_{kind}_parts"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    n_use = max_chunks or n_chunks
    for c in range(n_use):
        df = lf.filter(pl.col("session") % n_chunks == c).collect()
        if df.is_empty():
            continue

        # ① 截断:每 session 只留最近 session_cap 个事件
        df = (df.sort(["session", "ts"])
                .with_columns(_n=pl.len().over("session"),
                              _i=pl.int_range(pl.len()).over("session"))
                .filter(pl.col("_i") >= pl.col("_n") - session_cap)
                .drop(["_n", "_i"]))

        # ② session 内两两成对,过滤自己 & 时间窗口外
        pairs = (df.join(df, on="session", suffix="_y")
                   .filter((pl.col("aid") != pl.col("aid_y")) &
                           ((pl.col("ts") - pl.col("ts_y")).abs() <= cfg["window"])))

        # ③ 加权 + chunk 内先聚合(把体积从"对"降到"唯一对")
        agg = (pairs.with_columns(w=_weight_expr(kind, tmin, tmax))
                    .group_by(["aid", "aid_y"]).agg(pl.col("w").sum().alias("wgt")))
        agg.write_parquet(tmp / f"part_{c:03d}.parquet")
        print(f"[covis:{kind}] chunk {c + 1}/{n_use} | 唯一对 {agg.height:,} "
              f"| {time.time() - t0:.0f}s", flush=True)

    # ④ 合并所有块 → 再求和 → 每个 aid_x 只留 top_n 邻居
    COVIS_DIR.mkdir(parents=True, exist_ok=True)
    final = (pl.scan_parquet(tmp / "*.parquet")
               .group_by(["aid", "aid_y"]).agg(pl.col("wgt").sum())
               .collect(engine="streaming"))       # 流式聚合,内存不随分块数膨胀
    final = (final.sort(["aid", "wgt"], descending=[False, True])
                  .with_columns(_r=pl.int_range(pl.len()).over("aid"))
                  .filter(pl.col("_r") < cfg["top_n"])
                  .drop("_r")
                  .rename({"aid": "aid_x"}))
    out = COVIS_DIR / f"{kind}.parquet"
    final.write_parquet(out)
    shutil.rmtree(tmp)
    print(f"[covis:{kind}] ✅ {final.height:,} 行 / {final['aid_x'].n_unique():,} 个源商品 "
          f"-> {out.name} | 用时 {time.time() - t0:.0f}s", flush=True)
    return final


def load_covis(kind: str = "click") -> pl.DataFrame:
    """读回矩阵:列 = [aid_x, aid_y, wgt]。"""
    return pl.read_parquet(COVIS_DIR / f"{kind}.parquet")
