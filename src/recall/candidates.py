"""
Phase 2 · 召回 · 候选生成(阶段②+③)
===================================
把"每个 session 的可见历史 + 三张 co-vis 矩阵 + 热门表"变成
preds[session, type, prediction]。三个目标各用不同的矩阵组合(分目标融合),
且融合策略可按目标切换:硬分层(自身优先)或软融合(分数混排)。

三臂:
  自身臂  —— 种子(最近去重商品)本身,分数 = seed_wgt(近期×类型)。
  co-vis 臂 —— 每个种子给邻居加权投票;一个目标用多张矩阵时,票相加合成一路。
  热门兜底 —— ord 最大,只填空位。

融合(见 TYPE_BLEND):
  硬分层  —— 自身臂永远压 co-vis(clicks 复访主导,适用)。
  软融合  —— 两臂分数归一化后加权求和；已实验证伪，当前三个目标均使用硬分层。
"""
import polars as pl

# 类型权重:加购/下单比点击值钱(Chris Deotte 常用 1/6/3)
TYPE_W = {0: 1.0, 1: 6.0, 2: 3.0}
DECAY = 0.9                 # 种子近期衰减:最新种子权重 1,往前每位 ×0.9

# 融合策略:False=硬分层(自身永远优先),True=软融合(按分数混排)
TYPE_BLEND = {0: False, 1: False, 2: False}  # 硬分层:软融合实验证伪(见 experiments.md)
W_SELF, W_COVIS = 1.0, 1.0                   # 软融合两臂权重(要扫的 α)

# 硬分层偏置:自身臂 ord 从 0 起、co-vis 从 1000 起、热门从 2000 起 →
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
    # 每个 session 内给行编号， rank=0,1,2,...
    g = (val_input.group_by(["session", "aid"])
         .agg(last_ts=pl.col("ts").max(),
              last_type=pl.col("type").sort_by("ts").last()))

    # 在去重后的商品中，只选当前 session 最近的几个商品作为种子。
    # 按 session 排序,最近的种子排前面 → 取前 max_seeds 个
    g = (g.sort(["session", "last_ts", "aid"], descending=[False, True, False])  # aid 兜底可复现
          .with_columns(rank=pl.int_range(pl.len()).over("session"))
          .filter(pl.col("rank") < max_seeds))

    # 计算类型权重:加购/下单比点击值钱(Chris Deotte 常用 1/6/3)
    type_w = pl.col("last_type").replace_strict(TYPE_W, return_dtype=pl.Float64)

    # 计算种子权重 = 近期衰减 × 类型权重
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


def _hard_rank(self_scored: pl.DataFrame, covis_votes: pl.DataFrame) -> pl.DataFrame:
    """硬分层:自身 tier0 + co-vis tier1 → [session, aid, ord]。臂决定优先级。"""
    self_c = (self_scored.sort(["session", "score", "aid"], descending=[False, True, False])
                         .with_columns(ord=pl.int_range(pl.len()).over("session"))
                         .select("session", "aid", "ord"))
    covis_c = (covis_votes.sort(["session", "score", "aid"], descending=[False, True, False])
                          .with_columns(ord=TIER_COVIS + pl.int_range(pl.len()).over("session"))
                          .select("session", "aid", "ord"))
    return pl.concat([self_c, covis_c])


def _soft_rank(self_scored: pl.DataFrame, covis_votes: pl.DataFrame) -> pl.DataFrame:
    """软融合:两臂各自按 session 归一化 → 加权求和 → 排一次 → [session, aid, ord]。
    强度可跨臂竞争:高票 co-vis 新品能盖过弱自身候选。"""
    # 归一化:每个 session 内,score 归一化到 [0, 1]。避免不同 session 的 score 范围差异过大。
    def norm(df):
        lo = pl.col("score").min().over("session")
        hi = pl.col("score").max().over("session")
        return df.with_columns(n=(pl.col("score") - lo) / (hi - lo + 1e-9))
    
    # 两臂各自归一化,再按 session+aid join,缺失的填 0 → 计算总分 u → 排序 → 编 ord
    s = norm(self_scored).select("session", "aid", pl.col("n").alias("s_self"))
    c = norm(covis_votes).select("session", "aid", pl.col("n").alias("s_covis"))
    return (s.join(c, on=["session", "aid"], how="full", coalesce=True)
             .with_columns(u=W_SELF * pl.col("s_self").fill_null(0)
                             + W_COVIS * pl.col("s_covis").fill_null(0))
             .sort(["session", "u", "aid"], descending=[False, True, False])
             .with_columns(ord=pl.int_range(pl.len()).over("session"))
             .select("session", "aid", "ord"))


def _predict_one_type(seeds: pl.DataFrame, pop_long: pl.DataFrame,
                      matrices: dict, names: list[str], k: int,
                      blend: bool) -> pl.DataFrame:
    """两臂(带分数)按 blend 选硬/软合并 + 热门兜底 → [session, prediction]。"""
    # 1. 两臂,都带分数
    self_scored = seeds.select("session", "aid", score=pl.col("seed_wgt"))
    covis_votes = (pl.concat([_votes(seeds, matrices[n]) for n in names])
                     .group_by(["session", "aid"]).agg(pl.col("score").sum()))

    # 2. 分叉:只决定 real 怎么排
    real = (_soft_rank(self_scored, covis_votes) if blend
            else _hard_rank(self_scored, covis_votes))

    # 3. 共用尾巴:接热门 → 去重(留最小 ord)→ 取 k → 收列表
    # 某商品既是历史商品又被 co-vis 召回：保留历史版本；
    # 某商品既被 co-vis 召回又在热门榜：保留 co-vis 版本；
    #  热门榜只补还没有出现的商品。
    return (pl.concat([real, pop_long])
              .sort(["session", "ord", "aid"])
              .unique(subset=["session", "aid"], keep="first", maintain_order=True)
              .with_columns(r=pl.int_range(pl.len()).over("session"))
              .filter(pl.col("r") < k)
              .group_by("session", maintain_order=True)
              .agg(pl.col("aid").alias("prediction")))


def generate_predictions(val_input: pl.DataFrame, matrices: dict,
                         popular: list[int], k: int = 20,
                         max_seeds: int = 30) -> pl.DataFrame:
    """分目标融合(硬/软按 TYPE_BLEND)→ 每 session、每 type 取 top-k。"""
    seeds = get_seeds(val_input, max_seeds)

    # 热门(tier2):和目标无关,只算一次
    pop_df = (pl.DataFrame({"aid": pl.Series(popular, dtype=pl.Int32)})
                .with_row_index("pr")
                .with_columns(ord=TIER_POP + pl.col("pr").cast(pl.Int64))
                .select("aid", "ord"))
    pop_long = seeds.select("session").unique().join(pop_df, how="cross")

    # 按目标各跑一遍,只有 co-vis 那一路换矩阵、按 TYPE_BLEND 选硬/软
    out = [_predict_one_type(seeds, pop_long, matrices, names, k, blend=TYPE_BLEND[t])
             .with_columns(type=pl.lit(t, dtype=pl.Int8))
           for t, names in TYPE_MATRIX.items()]
    return pl.concat(out).select("session", "type", "prediction")


def generate_predictions_chunked(val_input: pl.DataFrame, matrices: dict,
                                 popular: list[int], k: int = 20,
                                 max_seeds: int = 30, n_chunks: int = 1) -> pl.DataFrame:
    """按 session 分 n_chunks 批,各调 generate_predictions 再拼 → 控内存/提速。

    session 之间互相独立(一个 session 的候选只看它自己的历史),所以按
    `session % n_chunks` 切开、各算各的、最后拼起来,结果和不分块**完全一样**
    —— 又一个 embarrassingly parallel。核心逻辑一行不改,只包一层批循环。
    n_chunks<=1 时直接走原函数。
    """
    if n_chunks <= 1:
        return generate_predictions(val_input, matrices, popular, k, max_seeds)
    parts = []
    for b in range(n_chunks):
        vb = val_input.filter(pl.col("session") % n_chunks == b)
        if vb.height == 0:
            continue
        parts.append(generate_predictions(vb, matrices, popular, k, max_seeds))
    return pl.concat(parts)
