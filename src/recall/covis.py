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
    "buy_weighted": {"window": 60 * 60 * 1000, "types": None,   "top_n": 20}, # 1 小时,不过滤
    "buy2buy":      {"window": 14 * DAY_MS,     "types": [1, 2], "top_n": 20}, 
    # 14 天, 只看加购/下单: type1,2; 购买跨度长(可能隔好几天),所以窗口从 1 小时放宽到 14 天
}


def _weight_expr(kind: str, tmin: int, tmax: int) -> pl.Expr:
    """返回“每一行共现商品对如何计算权重”的 Polars 表达式。

    注意返回值是 pl.Expr（列计算规则），不是已经算好的单个数字。调用方会在：
        pairs.with_columns(w=_weight_expr(...))
    中把这条规则应用到 pairs 的每一行，生成 w 列。

    self join 后一行商品对的关键列是：
        aid, ts          左表商品 x 及其事件时间
        aid_y, ts_y      右表邻居 y 及其事件时间（suffix="_y"）
    因而对于有向关系 x → y，ts_y 表示“邻居 y 这次事件发生的时间”。

    click 的权重把 ts_y 在全局训练时间范围 [tmin, tmax] 中的位置线性映射到
    [1, 4]：最早事件约为 1，时间中点为 2.5，最新事件约为 4。这样最近发生的
    共现关系会比很早以前的共现关系贡献更多票。

    这里衡量的是“这条关系在整个训练时间轴上有多新”；两次事件彼此相隔多久，
    由 build_covis() 中 abs(ts - ts_y) <= window 的过滤条件另外控制。
    """
    if kind == "click":
        # 第一步：(ts_y - tmin) / (tmax - tmin) 把时间归一化到 [0, 1]。
        # 第二步：乘 3 再加 1，把 [0, 1] 转换到 [1, 4]。
        # 例：ts_y 位于训练时间轴正中间 → 1 + 3 × 0.5 = 2.5。
        return 1 + 3 * (pl.col("ts_y") - tmin) / (tmax - tmin)

    if kind == "buy_weighted":
        # 按“邻居事件的类型”加权:加购/下单比点击值钱(不是按时间)
        return pl.col("type_y").replace_strict({0: 1.0, 1: 6.0, 2: 3.0}, 
                                            return_dtype=pl.Float64)
    if kind == "buy2buy":
        return pl.lit(1.0)     # 纯计数;“买→买”的共现, 不加权
    
                                            
    raise NotImplementedError(f"未知 co-vis 矩阵种类: {kind}")


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

    # 按 session 分块,每块独立处理,避免全量两两成对爆内存
    n_use = max_chunks or n_chunks
    for c in range(n_use):
        df = lf.filter(pl.col("session") % n_chunks == c).collect()
        if df.is_empty():
            continue

        # ① 截断：每个 session 最多只保留最近 session_cap 个事件。
        #
        # 先按 session 分组、ts 从早到晚排列。随后添加两个临时辅助列：
        #   _n = 该行所属 session 的事件总数。
        #        pl.len().over("session") 是窗口计算：按 session 算长度，
        #        但不把多行压成一行，所以同一 session 的每行都会重复保存 _n。
        #   _i = 该事件在 session 内按时间排列后的编号，从 0 开始。
        #        pl.int_range(pl.len()).over("session") 会为每个 session 分别生成
        #        0, 1, 2, ...；因为前面已按 ts 升序，_i 越大代表事件越新。
        #
        # 例如某 session 有 5 条事件、session_cap=3：
        #   _n = [5, 5, 5, 5, 5]
        #   _i = [0, 1, 2, 3, 4]
        #   过滤条件 _i >= _n-session_cap，即 _i >= 2，只保留编号 2/3/4，
        #   也就是最近 3 条。若事件总数小于 session_cap，右侧会是负数，
        #   所有 _i 都满足条件，因此原事件全部保留。
        #
        # 截断后 _n、_i 已经完成使命，用 drop 删除，避免污染后面的商品对表。
        df = (df.sort(["session", "ts"])
                .with_columns(
                    _n=pl.len().over("session"),
                    _i=pl.int_range(pl.len()).over("session"),
                )
                .filter(pl.col("_i") >= pl.col("_n") - session_cap)
                .drop(["_n", "_i"]))

        # ② session 内两两成对,过滤自己 & 时间窗口外
        pairs = (df.join(df, on="session", suffix="_y")
                   .filter((pl.col("aid") != pl.col("aid_y")) &
                           ((pl.col("ts") - pl.col("ts_y")).abs() <= cfg["window"])))

        # ③ 加权 + chunk 内先聚合：把大量重复的“商品对记录”压成
        #    “每个有向商品对一行”。注意 A→B 和 B→A 是两个不同的分组。
        #
        # pairs 中，同一个 A→B 可能因为出现在多个历史 session 而有很多行：
        #   session 1: A→B, w=1.5
        #   session 2: A→B, w=3.5
        #   session 3: A→B, w=3.1
        #
        # with_columns(w=...)：把 _weight_expr 返回的 Polars 表达式应用到
        # pairs 的每一行，生成该次共现贡献的临时权重列 w。
        # group_by(["aid", "aid_y"])：把起点和邻居都相同的有向对归为一组。
        # agg(sum(w).alias("wgt"))：把组内所有票相加，并将累计结果命名为 wgt。
        # 上面的三行 A→B 最终压成一行：A→B, wgt=1.5+3.5+3.1=8.1。
        #
        # 这里只聚合当前 chunk；不同 chunk 中的同一 A→B 会在步骤④再次求和。
        # 两级聚合能让原始 pairs 尽快缩小，避免所有共现记录同时占用内存。
        agg = (
            pairs.with_columns(w=_weight_expr(kind, tmin, tmax))
                 .group_by(["aid", "aid_y"])
                 .agg(pl.col("w").sum().alias("wgt"))
        )

        # 将当前 chunk 的唯一商品对写成临时 Parquet，使内存可以继续处理下一块。
        # {c:03d} 表示把编号补成 3 位：0→000、1→001、12→012。
        agg.write_parquet(tmp / f"part_{c:03d}.parquet")

        # agg.height 是当前 chunk 聚合后的“不同 aid→aid_y 数量”，
        # 不是原始事件数或 session 数。c+1 用于把程序内部的 0 起编号显示成
        # 人类习惯的 1 起进度；elapsed 是程序启动后的累计秒数。
        # flush=True 强制立即输出，长任务运行时能实时看到进度，而不等缓冲区刷新。
        print(f"[covis:{kind}] chunk {c + 1}/{n_use} | 唯一对 {agg.height:,} "
              f"| {time.time() - t0:.0f}s", flush=True)

    # ④ 合并所有块 → 再求和 → 每个 aid_x 只留 top_n 邻居
    COVIS_DIR.mkdir(parents=True, exist_ok=True)
    final = (pl.scan_parquet(tmp / "*.parquet")         # 流式读所有 chunk
               .group_by(["aid", "aid_y"]).agg(pl.col("wgt").sum()) # 跨 chunk 聚合, 根据 aid→aid_y 求和
               .collect(engine="streaming"))       # 流式聚合,内存不随分块数膨胀
    
    # ⑤ 排序 + 截 top_n
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
