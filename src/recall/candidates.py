"""
Phase 2 · 召回 · 候选生成(阶段②+③)
===================================
把"每个 session 的可见历史 + 三张 co-vis 矩阵 + 热门表"变成
preds[session, type, prediction]。三个目标各用不同的矩阵组合(分目标融合)。

三臂(分层):
  自身臂  —— 种子(最近去重商品)本身,ord 从 0 起 → 复访候选永远排最前。
  co-vis 臂 —— 每个种子给邻居加权投票;一个目标用多张矩阵时,票相加合成一路。
  热门兜底 —— ord 最大,只填空位。
"""
import polars as pl

# 类型权重:加购/下单比点击值钱(Chris Deotte 常用 1/6/3)
TYPE_W = {0: 1.0, 1: 6.0, 2: 3.0}
DECAY = 0.9                 # 种子近期衰减:最新种子权重 1,往前每位 ×0.9
# 分层偏置:自身臂 ord 从 0 起、co-vis 从 1000 起、热门从 2000 起 →
# 保证复访候选永远排在纯 co-vis 前,co-vis 只填自身臂用剩的空位。
TIER_COVIS = 1000
TIER_POP = 2000

# 每个目标用哪几路 co-vis 矩阵(分目标融合)
TYPE_MATRIX = {
    0: ["click"],                     # clicks:只用点击共现
    1: ["buy_weighted", "buy2buy"],   # carts :用两路买信号
    2: ["buy_weighted", "buy2buy"],   # orders:同上
}


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


def _votes(seeds: pl.DataFrame, covis: pl.DataFrame) -> pl.DataFrame:
    """一张矩阵的加权投票 → [session, aid, score](原始票,不排序、不加 ord)。

    关键:这里只出"原始 score"。多张矩阵要在 _predict_one_type 里先把 score
    相加,才能体现"被多路共同看好"。若在这就排序加 ord,多矩阵就没法相加了。
    """
    return (seeds.join(covis, left_on="aid", right_on="aid_x")
                 .with_columns(score=pl.col("seed_wgt") * pl.col("wgt"))
                 .group_by(["session", "aid_y"]).agg(pl.col("score").sum())
                 .rename({"aid_y": "aid"}))


def _predict_one_type(seeds: pl.DataFrame, self_c: pl.DataFrame,
                      pop_long: pl.DataFrame, matrices: dict,
                      names: list[str], k: int) -> pl.DataFrame:
    """一个目标:合并 names 里所有矩阵的票 + 自身 + 兜底 → [session, prediction]。"""
    # ① 多张矩阵的票"相加"合成一路 co-vis(被多路共同看好的候选浮上来)
    covis_votes = (pl.concat([_votes(seeds, matrices[n]) for n in names])
                     .group_by(["session", "aid"]).agg(pl.col("score").sum()))

    # ② co-vis 臂:按总票数排,ord 从 TIER_COVIS(1000)起,排在自身臂之后
    covis_c = (covis_votes.sort(["session", "score"], descending=[False, True])
                          .with_columns(ord=TIER_COVIS + pl.int_range(pl.len()).over("session"))
                          .select("session", "aid", "ord"))

    # ③ 三层合并(自身 tier0 + co-vis tier1 + 热门 tier2)→ 去重保留最小 ord
    #    → 取前 k → 收成列表
    return (pl.concat([self_c, covis_c, pop_long])
              .sort(["session", "ord"])
              .unique(subset=["session", "aid"], keep="first", maintain_order=True)
              .with_columns(r=pl.int_range(pl.len()).over("session"))
              .filter(pl.col("r") < k)
              .group_by("session", maintain_order=True)
              .agg(pl.col("aid").alias("prediction")))


def generate_predictions(val_input: pl.DataFrame, matrices: dict,
                         popular: list[int], k: int = 20,
                         max_seeds: int = 30) -> pl.DataFrame:
    """分目标分层融合 → 每 session、每 type 各取 top-k。"""
    seeds = get_seeds(val_input, max_seeds)

    # 自身臂(tier0)、热门(tier2):和目标无关,只算一次,三目标复用
    self_c = (seeds.sort(["session", "seed_wgt"], descending=[False, True])
                   .with_columns(ord=pl.int_range(pl.len()).over("session"))
                   .select("session", "aid", "ord"))
    pop_df = (pl.DataFrame({"aid": pl.Series(popular, dtype=pl.Int32)})
                .with_row_index("pr")
                .with_columns(ord=TIER_POP + pl.col("pr").cast(pl.Int64))
                .select("aid", "ord"))
    pop_long = seeds.select("session").unique().join(pop_df, how="cross")

    # 按目标各跑一遍,只有 co-vis 那一路换矩阵
    out = [_predict_one_type(seeds, self_c, pop_long, matrices, names, k)
             .with_columns(type=pl.lit(t, dtype=pl.Int8))
           for t, names in TYPE_MATRIX.items()]
    return pl.concat(out).select("session", "type", "prediction")
