"""
Phase 2 · 召回 · 候选生成(阶段②+③)
===================================
把"每个 session 的可见历史 + co-vis 矩阵 + 热门表"变成
preds[session, type, prediction]。Step 1 三个目标共用同一份候选,Step 2 再分目标。

三臂:
  自身臂  —— 种子(最近去重商品)本身,按 近期×类型 打分,再加大偏置 → 近似分层,
             保证复访候选永远排在纯 co-vis 前面。
  co-vis 臂 —— 每个种子给它的邻居加权投票:score = 种子权重 × co-vis 权重,累加。
  热门兜底 —— 排在真候选之后,只填空位。
"""
import polars as pl

# 类型权重:加购/下单比点击值钱(Chris Deotte 常用 1/6/3)
TYPE_W = {0: 1.0, 1: 6.0, 2: 3.0}
DECAY = 0.9                 # 种子近期衰减:最新种子权重 1,往前每位 ×0.9
# 分层偏置:自身臂 ord 从 0 起、co-vis 从 1000 起、热门从 2000 起 →
# 保证复访候选永远排在纯 co-vis 前,co-vis 只填自身臂用剩的空位。
TIER_COVIS = 1000
TIER_POP = 2000


def get_seeds(val_input: pl.DataFrame, max_seeds: int = 30) -> pl.DataFrame:
    """每个 session 的最近去重商品 + 权重(近期衰减 × 最近一次的类型权重)。"""
    # 同一个商品重复出现时，只保留它最近一次的信息
    # 按 session+aid 聚合,取最近一次 ts 和 type → 按 session+ts 排序 → 取前 max_seeds
    g = (val_input.group_by(["session", "aid"])
         .agg(last_ts=pl.col("ts").max(),
              last_type=pl.col("type").sort_by("ts").last()))
    
    # 在去重后的商品中，只选当前 session 最近的几个商品作为种子。
    # 按 session 排序,最近的种子排前面 → 取前 max_seeds 个
    g = (g.sort(["session", "last_ts"], descending=[False, True])
          .with_columns(rank=pl.int_range(pl.len()).over("session"))
          .filter(pl.col("rank") < max_seeds))
    
    # 计算种子权重 = 近期衰减 × 类型权重
    type_w = pl.col("last_type").replace_strict(TYPE_W, return_dtype=pl.Float64)
    # 近期衰减:最新种子权重 1,往前每位 ×0.9
    g = g.with_columns(seed_wgt=pl.lit(DECAY).pow(pl.col("rank")) * type_w)

    return g.select("session", "aid", "seed_wgt")


def generate_predictions(val_input: pl.DataFrame, covis: pl.DataFrame,
                         popular: list[int], k: int = 20,
                         max_seeds: int = 30) -> pl.DataFrame:
    """三臂分层融合 → 每 session top-k → 复制给三个目标。"""
    seeds = get_seeds(val_input, max_seeds)

    # 自身臂(tier 0):种子本身,按种子权重排,ord = session 内名次
    self_c = (seeds.sort(["session", "seed_wgt"], descending=[False, True])
                   .with_columns(ord=pl.int_range(pl.len()).over("session"))
                   .select("session", "aid", "ord"))

    # co-vis 臂(tier 1):种子 → 邻居 加权投票,按票数排,ord 从 1000 起
    votes = (seeds.join(covis, left_on="aid", right_on="aid_x")
                  .with_columns(score=pl.col("seed_wgt") * pl.col("wgt"))
                  .group_by(["session", "aid_y"]).agg(pl.col("score").sum())
                  .rename({"aid_y": "aid"})
                  .sort(["session", "score"], descending=[False, True])
                  .with_columns(ord=TIER_COVIS + pl.int_range(pl.len()).over("session"))
                  .select("session", "aid", "ord"))

    # 热门兜底(tier 2):ord 从 2000 起,永远排最后
    pop_df = (pl.DataFrame({"aid": pl.Series(popular, dtype=pl.Int32)})
                .with_row_index("pr")
                .with_columns(ord=TIER_POP + pl.col("pr").cast(pl.Int64))
                .select("aid", "ord"))
    pop_long = seeds.select("session").unique().join(pop_df, how="cross")

    # 三层合并 → 按 ord 排(自身臂 ord 最小、优先)→ 去重(同 aid 保留最小 ord)
    # → 取前 k → 收成列表
    pool = (pl.concat([self_c, votes, pop_long.select("session", "aid", "ord")])
              .sort(["session", "ord"])
              .unique(subset=["session", "aid"], keep="first", maintain_order=True)
              .with_columns(r=pl.int_range(pl.len()).over("session"))
              .filter(pl.col("r") < k)
              .group_by("session", maintain_order=True)
              .agg(pl.col("aid").alias("prediction")))

    # Step 1:三目标共用同一份候选(Step 2 再按 type 分开融合)
    return (pl.concat([pool.with_columns(type=pl.lit(t, dtype=pl.Int8))
                       for t in (0, 1, 2)])
              .select("session", "type", "prediction"))
