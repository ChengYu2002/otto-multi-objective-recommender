"""
Phase 3 · 精排 · Step 0 · Block B:候选长表
================================================================
把每折的 co-vis(rank_artifacts)+ input 跑成"一行一个候选、带各路来源信号、贴 label"的长表。
  - 不截 20:留 K_LONG=100(Phase 3 精排的候选池);
  - 不压成 list:一行一个 (fold, session, type, aid);
  - 不压成单一总分:self / click / buy_weighted / buy2buy / pop 各自保留 flag/score/rank;
  - source_count 按具体来源计数;arm_count 按 Phase 2 的 self/co-vis/pop 三臂计数;
  - rule_rank:精确复现 Phase 2 分层顺序 → Gate A 能对上。

自检:
  - Gate A(自洽):本表 rule Top-20 的 recall@20 == 直接跑 generate_predictions 的 recall@20;
  - rule Recall@20/50/100 与 strict oracle@20 分开报告,不混淆指标分母;
  - zero-hit query 比例;
  - 候选唯一键 (session,type,aid) 无重复。

跑法:
    python src/rank/candidates_long.py --sample   # 读 folds/sample + rank_artifacts/sample

当前只允许 sample。full 约 22 亿候选行,在分块落盘实现前禁止误跑。

阅读抓手(不用逐行背 Polars):
  Block B = input 查 self/co-vis/pop → 合并去重 → 一行一个候选,保留各路 flag/score/rank。
  Block C = 候选按 (session,type) LEFT JOIN hidden GT → aid 在答案中为 1,否则为 0。
  LEFT JOIN 保留左侧已有候选,不会把右侧未召回的正确答案补进候选池。
  rank_train label 用于拟合;rank_valid label 只用于 early stopping/选型/评分。
  Block C 只贴完整标签,train 组内负采样属于下一步。
"""
import sys
import time
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.recall.candidates import (get_seeds, _votes, generate_predictions,   # noqa: E402
                                   TIER_COVIS, TIER_POP, TYPE_MATRIX)
from src.validation import evaluate, evaluate_at_ks                            # noqa: E402

DATA = ROOT / "data"
FOLDS_DIR = DATA / "parquet" / "folds"
RANK_ART = DATA / "parquet" / "rank_artifacts"
KINDS = ["click", "buy_weighted", "buy2buy"]
INV_TYPE = {0: "clicks", 1: "carts", 2: "orders"}
K_LONG = 100


# 读取当前 fold corpus
#         ↓
# 按 aid 分组
#         ↓
# 统计每个 aid 出现次数 → pop_score
#         ↓
# 出现次数从高到低排序
#         ↓
# 取 Top-100
#         ↓
# 位置编号 → pop_rank

def _fold_popularity_frame(corpus_path: Path, k: int = 100) -> pl.DataFrame:
    """本折热门 Top-k,同时保留频次(pop_score)与 0 起排名(pop_rank)。"""
    return (pl.scan_parquet(corpus_path).group_by("aid").agg(pl.len().alias("pop_score"))
              .sort(["pop_score", "aid"], descending=[True, False]).head(k).collect()
              .with_row_index("pop_rank")
              .with_columns(pop_rank=pl.col("pop_rank").cast(pl.Int32)))


def _fold_popularity(corpus_path: Path, k: int = 100) -> list[int]:
    """兼容 Phase 2 generate_predictions 所需的热门 aid list。"""
    return _fold_popularity_frame(corpus_path, k)["aid"].to_list()

# build_fold_covis 是 _one_type_long 的上游 (input为matrices的 seed);


def _one_type_long(seeds: pl.DataFrame, pop_df: pl.DataFrame, matrices: dict,
                   names: list[str], k: int) -> pl.DataFrame:
    """一个目标的候选长表:各来源 flag/score/rank + rule_rank + 两种 count。"""
    self_c = (seeds.select("session", "aid", self_score=pl.col("seed_wgt"))
                    .sort(["session", "self_score", "aid"], descending=[False, True, False])
                    .with_columns(self_rank=pl.int_range(pl.len()).over("session").cast(pl.Int32)))

    # 每张矩阵独立保留原始票与组内排名;合并排序仍严格沿用 Phase 2 的票数求和。
    votes = {}
    for m in names:
        votes[m] = (_votes(seeds, matrices[m])
                      .sort(["session", "score", "aid"], descending=[False, True, False])
                      .with_columns(rank=pl.int_range(pl.len()).over("session").cast(pl.Int32)))
    covis_comb = (pl.concat([votes[m].select("session", "aid", "score") for m in names])
                    .group_by(["session", "aid"]).agg(covis_score=pl.col("score").sum()))

    # 分层 ord(复现 Phase 2):自身 tier0 → co-vis(按合并分)tier1 → 热门 tier2
    self_ord = self_c.select("session", "aid", ord=pl.col("self_rank").cast(pl.Int64))
    covis_ord = (covis_comb.sort(["session", "covis_score", "aid"], descending=[False, True, False])
                           .with_columns(ord=TIER_COVIS + pl.int_range(pl.len()).over("session"))
                           .select("session", "aid", "ord"))
    pop_long = seeds.select("session").unique().join(pop_df.select("aid", "ord"), how="cross")

    pool = (pl.concat([self_ord, covis_ord, pop_long])
              .sort(["session", "ord", "aid"])
              .unique(subset=["session", "aid"], keep="first", maintain_order=True)
              .with_columns(rule_rank=pl.int_range(pl.len()).over("session"))
              # Polars filter 只保留条件为 True 的行:这里留下 rank 0..k-1。
              .filter(pl.col("rule_rank") < k)
              .select("session", "aid", "rule_rank"))

    # 各路信号 join 回来(缺失 = null)
    out = pool.join(self_c, on=["session", "aid"], how="left")
    for m in names:
        out = out.join(
            votes[m].rename({"score": f"{m}_score", "rank": f"{m}_rank"}),
            on=["session", "aid"], how="left",
        )
    out = out.join(pop_df.select("aid", "pop_score", "pop_rank"), on="aid", how="left")

    # 补齐本 type 没用到的矩阵列,让三个目标 schema 一致。
    for m in KINDS:
        if f"{m}_score" not in out.columns:
            out = out.with_columns(
                pl.lit(None, dtype=pl.Float64).alias(f"{m}_score"),
                pl.lit(None, dtype=pl.Int32).alias(f"{m}_rank"),
            )

    # flag 明确落盘,避免下游反复从 null 推导;两种 count 的语义不再混用。
    out = out.with_columns(
        self_flag=pl.col("self_score").is_not_null().cast(pl.Int8),
        pop_flag=pl.col("pop_rank").is_not_null().cast(pl.Int8),
        **{f"{m}_flag": pl.col(f"{m}_score").is_not_null().cast(pl.Int8) for m in KINDS},
    )
    covis_arm = pl.any_horizontal([pl.col(f"{m}_flag") == 1 for m in KINDS]).cast(pl.Int8)
    out = out.with_columns(
        source_count=pl.sum_horizontal(
            [pl.col("self_flag"), pl.col("pop_flag")]
            + [pl.col(f"{m}_flag") for m in KINDS]
        ).cast(pl.Int8),
        arm_count=(pl.col("self_flag") + covis_arm + pl.col("pop_flag")).cast(pl.Int8),
    )
    return out


# 1. 读取 fold input
# 2. 读取 fold labels
# 3. 读取三张 co-vis
# 4. 计算当前 fold 热门榜
# 5. 从 input 提取近期商品 seeds
# 6. 分别为 clicks/carts/orders 生成 Top-100 候选
# 7. 合并三个 type 的候选
# 8. LEFT JOIN hidden GT
# 9. 候选在 GT 中 → label=1，否则 0
# 10. 写入 candidates.parquet
def build_fold_long(mode: str, fold: str) -> pl.DataFrame:
    fold_dir = FOLDS_DIR / mode / fold
    covis_dir = RANK_ART / mode / fold / "covis"
    fold_input = pl.read_parquet(fold_dir / "input.parquet")
    labels = pl.read_parquet(fold_dir / "labels.parquet")
    matrices = {m: pl.read_parquet(covis_dir / f"{m}.parquet") for m in KINDS}
    pop_df = (_fold_popularity_frame(RANK_ART / mode / fold / "corpus.parquet", k=100)
                .with_columns(ord=(TIER_POP + pl.col("pop_rank")).cast(pl.Int64)))

    seeds = get_seeds(fold_input)
    canon = [
        "session", "aid", "rule_rank",
        "self_flag", "self_score", "self_rank",
        "click_flag", "click_score", "click_rank",
        "buy_weighted_flag", "buy_weighted_score", "buy_weighted_rank",
        "buy2buy_flag", "buy2buy_score", "buy2buy_rank",
        "pop_flag", "pop_score", "pop_rank",
        "source_count", "arm_count", "type",
    ]
    # Step 6:遍历 TYPE_MATRIX,分别生成 clicks/carts/orders 三张 Top-100 候选子表。
    parts = [_one_type_long(seeds, pop_df, matrices, names, K_LONG)
                 .with_columns(type=pl.lit(t, dtype=pl.Int8)).select(canon)   # 统一列序好 concat
             for t, names in TYPE_MATRIX.items()]
    # Step 7:纵向拼接三张子表;行不会互相融合,type 列负责区分预测目标。
    long = pl.concat(parts)

    # Block C · 贴标签:
    # 1) LEFT JOIN 保留左表 long 的全部候选;右表 labels 只补 GT,不会新增未召回 aid。
    # 2) 必须按 (session,type) 对齐,不能把别的 session/type 的正例算进来。
    # 3) contains 得到 bool;无该 type 答案时 null→False;最后 False/True→0/1。
    # 这里只如实贴完整标签,不做 train 负采样。train label 用于拟合,valid label 只用于验证。
    long = (long.join(labels, on=["session", "type"], how="left")
                .with_columns(label=pl.col("ground_truth").list.contains(pl.col("aid"))
                                      .fill_null(False).cast(pl.Int8))
                .drop("ground_truth")
                .with_columns(fold=pl.lit(fold))
                .select("fold", "session", "type", "aid", "rule_rank",
                        "self_flag", "self_score", "self_rank",
                        "click_flag", "click_score", "click_rank",
                        "buy_weighted_flag", "buy_weighted_score", "buy_weighted_rank",
                        "buy2buy_flag", "buy2buy_score", "buy2buy_rank",
                        "pop_flag", "pop_score", "pop_rank",
                        "source_count", "arm_count", "label"))
    out_path = RANK_ART / mode / fold / "candidates.parquet"
    long.write_parquet(out_path)
    return long


def _strict_oracle_at_20(long: pl.DataFrame, labels: pl.DataFrame) -> dict:
    """候选池内正例全部排最前时的严格 Recall@20 上限;不是模型实际成绩。"""
    hits = (long.group_by(["session", "type"])
                .agg(pl.col("label").sum().clip(upper_bound=20).alias("hits")))
    q = (labels.with_columns(
            pl.col("ground_truth").list.len().clip(upper_bound=20).alias("den"))
          .join(hits, on=["session", "type"], how="left")
          .with_columns(pl.col("hits").fill_null(0)))
    by_type = (q.group_by("type")
                 .agg(pl.col("hits").sum().alias("num"), pl.col("den").sum().alias("den")))
    result = {name: 0.0 for name in INV_TYPE.values()}
    for t, num, den in by_type.iter_rows():
        result[INV_TYPE[t]] = num / den if den else 0.0
    result["weighted"] = (0.10 * result["clicks"]
                          + 0.30 * result["carts"]
                          + 0.60 * result["orders"])
    return result


def _report(mode: str, fold: str, long: pl.DataFrame):
    labels = pl.read_parquet(FOLDS_DIR / mode / fold / "labels.parquet")
    # rule Recall@K 描述当前排序;strict oracle@20 才是 Top-100 候选池的最终天花板。
    preds = (long.filter(pl.col("rule_rank") < K_LONG)
                 .sort(["session", "type", "rule_rank"])
                 .group_by(["session", "type"], maintain_order=True)
                 .agg(prediction=pl.col("aid")))
    res = evaluate_at_ks(preds, labels, ks=[20, 50, 100])
    oracle = _strict_oracle_at_20(long, labels)
    # zero-hit:有 GT 但候选零命中的 query 比例
    hit = (long.group_by(["session", "type"]).agg(hits=pl.col("label").sum()))
    q = labels.join(hit, on=["session", "type"], how="left").with_columns(pl.col("hits").fill_null(0))
    zero = q.filter(pl.col("hits") == 0).height / q.height
    dup = long.group_by(["session", "type", "aid"]).len().filter(pl.col("len") > 1).height

    print(f"\n=== {fold} 候选长表报告 ===")
    print(f"  行数 {long.height:,} | 候选唯一键重复 {dup}(应 0)")
    for k in (20, 50, 100):
        s = res[k]
        print(f"  rule recall@{k:<3} clicks {s['clicks']:.4f} | carts {s['carts']:.4f} | "
              f"orders {s['orders']:.4f} | weighted {s['weighted']:.4f}")
    print(f"  strict oracle@20 clicks {oracle['clicks']:.4f} | carts {oracle['carts']:.4f} | "
          f"orders {oracle['orders']:.4f} | weighted {oracle['weighted']:.4f}")
    print(f"  zero-hit query 比例 = {zero:.1%}(候选零命中,精排也救不回)")
    return {"rule_recall": res, "strict_oracle20": oracle}


if __name__ == "__main__":
    if "--sample" not in sys.argv:
        raise SystemExit(
            "当前 candidates_long.py 只允许 --sample;full 约 22 亿行,请先实现分块落盘。"
        )
    mode = "sample"
    t0 = time.time()
    long_valid = None
    for fold in ("rank_train", "rank_valid"):
        lg = build_fold_long(mode, fold)
        res = _report(mode, fold, lg)
        if fold == "rank_valid":
            long_valid = lg

    # Gate A(自洽):rank_valid 长表 rule Top-20 == 直接跑 generate_predictions 的 recall@20
    fold = "rank_valid"
    vi = pl.read_parquet(FOLDS_DIR / mode / fold / "input.parquet")
    vl = pl.read_parquet(FOLDS_DIR / mode / fold / "labels.parquet")
    mats = {m: pl.read_parquet(RANK_ART / mode / fold / "covis" / f"{m}.parquet") for m in KINDS}
    pop = _fold_popularity(RANK_ART / mode / fold / "corpus.parquet", k=100)
    gp = evaluate(generate_predictions(vi, mats, pop, k=20), vl)["weighted"]
    lt = evaluate_at_ks(
        long_valid.filter(pl.col("rule_rank") < 20).sort(["session", "type", "rule_rank"])
                  .group_by(["session", "type"], maintain_order=True).agg(prediction=pl.col("aid")),
        vl, ks=[20])[20]["weighted"]
    print(f"\n[Gate A · 自洽] generate_predictions weighted@20 = {gp:.4f}")
    print(f"[Gate A · 自洽] 长表 rule Top-20  weighted@20 = {lt:.4f}")
    print(f"[Gate A · 自洽] {'✅ 一致' if abs(gp - lt) < 1e-6 else '❌ 不一致,rule_rank 没复现 Phase 2!'}")
    print(f"\n[candidates_long] ✅ mode={mode} 用时 {time.time() - t0:.0f}s")
