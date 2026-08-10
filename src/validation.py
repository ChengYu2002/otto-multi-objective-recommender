"""
Phase 1 · 本地验证框架
======================
目标:在本地复刻 Kaggle LB 的加权 Recall@20,让"改了召回/特征 → 分数变没变"
可以离线快速验证,不依赖有限的 LB 提交次数。

三个产物:
  1. make_validation_set() : 从 train 切出本地验证集
       - val_input  : 每个验证 session 截断后的输入事件(喂给模型的那半段)
       - val_labels : 每个验证 session 藏起来的 ground truth(clicks/carts/orders)
  2. baseline_recent()     : 一个"傻 baseline"(推自己最近看过的商品),用来自测框架
  3. evaluate(preds, labels) -> dict : 加权 Recall@20

指标定义(必须和 LB 完全一致,micro 求和,不是逐 session 求平均!):
  对每个 type t ∈ {clicks, carts, orders}:
        Σ_s  | top20(s) ∩ gt_t(s) |
  R_t = ───────────────────────────      # 只统计"有该 type 答案"的 session
        Σ_s  min(20, | gt_t(s) |)
  weighted = 0.10*R_clicks + 0.30*R_carts + 0.60*R_orders

⚠️ 工程取舍:本实现只用"最后 val_days 天窗口内的事件"来切验证集,不去把跨界
   session 的完整早期历史拖出来。原因:OTTO 的 session 基本是"一次浏览",极少
   跨越多天,这样做结果几乎一样,却能避开对 2.16 亿行全量做重 join。想要更严谨
   (完整保留每个 session 历史 + 官方随机切点规则)的版本,参考官方切分脚本:
   https://github.com/otto-de/recsys-dataset (src/testset.py)。建议第一版跑通后,
   抽几行和 Radek 切好的 validation parquet 对一下数,确认 CV 和 LB 对得上。

跑法:
    conda activate otto
    python src/validation.py            # 全量验证集 + 自测
    python src/validation.py --sample   # 只抽 5 万 session,快速验证框架
"""
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
TRAIN_PARQUET = DATA / "parquet" / "train" / "*.parquet"
VAL_DIR = DATA / "parquet" / "val"          # 切好的验证集存这,供召回/精排复用

TYPE_MAP = {"clicks": 0, "carts": 1, "orders": 2}
INV_TYPE = {v: k for k, v in TYPE_MAP.items()}
WEIGHTS = {"clicks": 0.10, "carts": 0.30, "orders": 0.60}
K = 20
DAY_MS = 86_400_000                          # 一天的毫秒数(ts 是毫秒时间戳)


# ---------------------------------------------------------------------------
# Step 1 · 造本地验证集
# ---------------------------------------------------------------------------
def make_validation_set(val_days: int = 7, seed: int = 42,
                        sample_sessions: int | None = None):
    """
    从 data/parquet/train 切出本地验证集。

    两刀(见文件顶部说明):
      第一刀(挑 session):取时间上最后 val_days 天窗口内的事件;窗口里事件数 >= 2
                         的 session 才够格当验证 session(要能切出"输入+答案")。
      第二刀(切每个 session):随机选一个切点 cut,
                         idx < cut  的事件 = val_input(喂给模型)
                         idx >= cut 的事件 = 提炼成 labels(藏起来当答案)

    ground truth:
      clicks -> 切点后的"下一次点击"aid(只取紧接着 1 个)
      carts  -> 切点后所有加购 aid(去重)
      orders -> 切点后所有下单 aid(去重)

    参数:
      val_days        : 验证窗口天数(默认 7,对齐线上 test 的一周)
      seed            : 随机切点的种子,保证可复现
      sample_sessions : 只保留这么多个验证 session(调试/快速迭代用);None = 全量

    返回:
      val_input  : pl.DataFrame[session, aid, ts, type]
      val_labels : pl.DataFrame[session, type, ground_truth(list[int])]
    """
    lf = pl.scan_parquet(TRAIN_PARQUET)

    # --- 第一刀:确定时间界,取最后 val_days 天窗口内的事件 -------------------
    max_ts = lf.select(pl.col("ts").max()).collect().item()      # 最后一个事件时刻
    boundary = max_ts - val_days * DAY_MS                         # 窗口起点(毫秒)

    ev = (
        lf.filter(pl.col("ts") >= boundary)     # 只留窗口内事件 -> 大幅省内存
          .collect()
          .filter(pl.len().over("session") >= 2)  # 至少 2 个事件才切得动
    )

    # --- (可选)下采样验证 session,加快调试 --------------------------------
    # sort("session"):unique() 不保证输出顺序,不排序的话每次跑 session 顺序都可能变,
    # 后面按位置贴随机数(第 101 行)就会导致 session→切点 的映射漂移、seed 失效。
    # 排一下序,才能真正做到"同 seed → 同验证集",这是可复现迭代闭环的前提。
    all_ids = ev.select("session").unique().sort("session")
    if sample_sessions is not None and sample_sessions < all_ids.height:
        all_ids = all_ids.sample(n=sample_sessions, seed=seed)
        ev = ev.join(all_ids, on="session", how="semi")

    # --- 第二刀:给每个 session 分配一个随机切点 cut ------------------------
    # 给每个 session 摇一个 [0,1) 随机数,映射成切点 cut ∈ [1, n-1]:
    #   idx < cut  -> 输入(至少 1 个),idx >= cut -> 答案(至少 1 个)
    rng = np.random.default_rng(seed)
    rand = all_ids.with_columns(pl.Series("r", rng.random(all_ids.height)))

    ev = (
        ev.join(rand, on="session")
          .sort(["session", "ts"])                        # 保证 idx = 时间顺序
          .with_columns(
              idx=pl.int_range(pl.len()).over("session"),  # 组内第几个事件(0起)
              n=pl.len().over("session"),                  # 该 session 事件总数
          )
          .with_columns(
              # rand*(n-1) ∈ [0, n-1) -> floor ∈ [0, n-2] -> +1 ∈ [1, n-1]
              cut=(1 + (pl.col("r") * (pl.col("n") - 1)).floor()).cast(pl.Int32)
          )
    )

    val_input = ev.filter(pl.col("idx") < pl.col("cut")).select(
        "session", "aid", "ts", "type"
    )
    tail = ev.filter(pl.col("idx") >= pl.col("cut"))       # 切点之后 = 答案来源

    # --- 从 tail 提炼三种 ground truth ------------------------------------
    # clicks:切点后"第一次"点击(idx 最小的 click),head(1) 得到长度=1 的 list
    clicks = (
        tail.filter(pl.col("type") == TYPE_MAP["clicks"])
            .group_by("session")
            .agg(pl.col("aid").sort_by("idx").head(1).alias("ground_truth"))
            .with_columns(type=pl.lit(TYPE_MAP["clicks"], dtype=pl.Int8))
    )
    # carts / orders:切点后所有加购 / 下单 aid,去重
    carts = (
        tail.filter(pl.col("type") == TYPE_MAP["carts"])
            .group_by("session")
            .agg(pl.col("aid").unique().alias("ground_truth"))
            .with_columns(type=pl.lit(TYPE_MAP["carts"], dtype=pl.Int8))
    )
    orders = (
        tail.filter(pl.col("type") == TYPE_MAP["orders"])
            .group_by("session")
            .agg(pl.col("aid").unique().alias("ground_truth"))
            .with_columns(type=pl.lit(TYPE_MAP["orders"], dtype=pl.Int8))
    )
    val_labels = pl.concat([clicks, carts, orders]).select(
        "session", "type", "ground_truth"
    )

    # --- 落盘 + 打印全貌 --------------------------------------------------
    VAL_DIR.mkdir(parents=True, exist_ok=True)
    val_input.write_parquet(VAL_DIR / "input.parquet")
    val_labels.write_parquet(VAL_DIR / "labels.parquet")

    ts_str = pl.from_epoch(pl.Series([boundary]), "ms")[0]
    print(f"[验证集] 时间界 {ts_str} 之后为验证窗口(最后 {val_days} 天)")
    print(f"[验证集] 验证 session: {val_input['session'].n_unique():,} | "
          f"输入事件: {val_input.height:,}")
    for t in (0, 1, 2):
        cnt = val_labels.filter(pl.col("type") == t).height
        print(f"[验证集]   有 {INV_TYPE[t]:<7} 答案的 session: {cnt:,}")
    return val_input, val_labels


# ---------------------------------------------------------------------------
# Step 3 · 傻 baseline:每个 session 推它输入里"最近看过"的商品
#          用途:自测 —— 如果 evaluate 写对了,这个 baseline 应该跑出合理正分
# ---------------------------------------------------------------------------
def baseline_recent(val_input: pl.DataFrame) -> pl.DataFrame:
    """把每个 session 输入里最近交互的 K 个 aid(逆序去重)当预测,三种 type 共用。"""
    recent = (
        val_input
        .sort(["session", "ts"], descending=[False, True])   # 组内最近的排前面
        .group_by("session", maintain_order=True)
        .agg(pl.col("aid").unique(maintain_order=True)       # 去重、保留最近顺序
                 .head(K).alias("prediction"))
    )
    # 三个目标用同一份预测(clicks/carts/orders 各复制一份)
    return pl.concat([
        recent.with_columns(type=pl.lit(t, dtype=pl.Int8)) for t in (0, 1, 2)
    ]).select("session", "type", "prediction")


# ---------------------------------------------------------------------------
# Step 2 · 加权 Recall@20
# ---------------------------------------------------------------------------
def evaluate(preds: pl.DataFrame, labels: pl.DataFrame) -> dict:
    """
    计算加权 Recall@20。

    preds  : pl.DataFrame[session, type, prediction(list[int])]  每个目标的 top-K 预测
    labels : pl.DataFrame[session, type, ground_truth(list[int])] make_validation_set 产出
    返回   : {"clicks":.., "carts":.., "orders":.., "weighted":..}

    实现要点:micro 求和 —— 所有 session 的分子、分母分别加总,最后相除;
             绝不是逐 session 算 recall 再平均(那是另一个指标,对不上 LB)。
    """
    # 两边 list 内层 dtype 对齐,交集才算得对
    L = labels.with_columns(pl.col("ground_truth").cast(pl.List(pl.Int64)))
    P = preds.with_columns(pl.col("prediction").cast(pl.List(pl.Int64)))

    df = (
        L.join(P, on=["session", "type"], how="left")
         # 某 session/type 没给预测 -> 当空列表处理(命中 0)
         .with_columns(
             pl.when(pl.col("prediction").is_null())
               .then(pl.lit([], dtype=pl.List(pl.Int64)))
               .otherwise(pl.col("prediction"))
               .alias("prediction")
         )
         .with_columns(pl.col("prediction").list.head(K).alias("pred20"))
         .with_columns(
             # 分子:预测 top20 与答案的交集大小
             hits=pl.col("pred20").list.set_intersection(pl.col("ground_truth")).list.len(),
             # 分母:min(20, |答案|)
             denom=pl.min_horizontal(pl.lit(K), pl.col("ground_truth").list.len()),
         )
    )

    agg = df.group_by("type").agg(
        num=pl.col("hits").sum(),
        den=pl.col("denom").sum(),
    )

    res: dict[str, float] = {}
    for row in agg.iter_rows(named=True):
        name = INV_TYPE[row["type"]]
        res[name] = row["num"] / row["den"] if row["den"] else 0.0
    res["weighted"] = sum(WEIGHTS[k] * res.get(k, 0.0) for k in WEIGHTS)
    return res


def evaluate_at_ks(preds: pl.DataFrame, labels: pl.DataFrame,
                   ks=(20, 50, 100)) -> dict:
    """多个 K 的加权 Recall@K → {K: {clicks, carts, orders, weighted}}。

    recall@100 = Phase 3 精排的天花板:GBDT 只能在候选集里重排,
    没进 top-100 的正样本它也捞不回来。@20 是 LB 口径的正式分。
    """
    L = labels.with_columns(pl.col("ground_truth").cast(pl.List(pl.Int64)))
    P = preds.with_columns(pl.col("prediction").cast(pl.List(pl.Int64)))
    df = (L.join(P, on=["session", "type"], how="left")
           .with_columns(
               pl.when(pl.col("prediction").is_null())
                 .then(pl.lit([], dtype=pl.List(pl.Int64)))
                 .otherwise(pl.col("prediction")).alias("prediction")))

    out: dict = {}
    for k_ in ks:
        d = (df.with_columns(pl.col("prediction").list.head(k_).alias("predk"))
               .with_columns(
                   hits=pl.col("predk").list.set_intersection(pl.col("ground_truth")).list.len(),
                   denom=pl.min_horizontal(pl.lit(k_), pl.col("ground_truth").list.len()),
               ))
        agg = d.group_by("type").agg(num=pl.col("hits").sum(), den=pl.col("denom").sum())
        res: dict[str, float] = {}
        for row in agg.iter_rows(named=True):
            res[INV_TYPE[row["type"]]] = row["num"] / row["den"] if row["den"] else 0.0
        res["weighted"] = sum(WEIGHTS[t] * res.get(t, 0.0) for t in WEIGHTS)
        out[k_] = res
    return out


def popular_share(preds: pl.DataFrame, popular: list[int], k: int = 20) -> dict:
    """top-k 里有多少比例落在热门表 → {clicks, carts, orders}。
    偏高 = 召回不够力、靠热门凑数。注:是上界(真候选也可能恰好是热门)。
    """
    pop = pl.Series("p", popular, dtype=pl.Int64)
    d = (preds.with_columns(pl.col("prediction").cast(pl.List(pl.Int64)).list.head(k))
              .explode("prediction")
              .with_columns(is_pop=pl.col("prediction").is_in(pop))
              .group_by("type").agg(share=pl.col("is_pop").mean()))
    return {INV_TYPE[r["type"]]: r["share"] for r in d.iter_rows(named=True)}


# ---------------------------------------------------------------------------
# 自测入口
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    n = 50_000 if "--sample" in sys.argv else None
    val_input, val_labels = make_validation_set(sample_sessions=n)

    preds = baseline_recent(val_input)
    scores = evaluate(preds, val_labels)

    print("\n=== 自我召回 baseline(推最近看过的商品)===")
    for k in ("clicks", "carts", "orders", "weighted"):
        print(f"  {k:<8}: {scores[k]:.4f}")
    print("\n若四个分数都是合理正数(不为 0/1),说明验证框架跑通了。")
