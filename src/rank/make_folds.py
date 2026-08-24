"""
Phase 3 · 精排 · Step -1:造两个时间折(rank-train / rank-valid)
================================================================
⚠️ 这是 **Phase 3 精排** 的地基,和 Phase 1 的 `src/validation.py` 是两回事:
   - `validation.py`(Phase 1/2):切**一个**验证集,给召回打分。
   - 本文件(Phase 3):切**两个时间折** —— 精排要"用更早的折训练、在更晚的折
     early-stopping/选型/评估"。直接拿同一份 labels 既训练又报分 = 训练集成绩,
     这是 Phase 3 头号泄漏,靠两折隔离掉。

产出(按 mode 分目录,sample 与 full 互不覆盖):
    data/parquet/folds/{sample|full}/rank_train/{input,labels}.parquet   ← 倒数第二周
    data/parquet/folds/{sample|full}/rank_valid/{input,labels}.parquet   ← 最后一周(复用现有 val)
    data/parquet/folds/{sample|full}/manifest.json

时间窗(半开区间 [window_start, window_end)):
    rank_train: [max_ts-14d, max_ts-7d)
    rank_valid: 复用 data/parquet/val/(最后一周),保证 == Phase 2 用的那份

★ 建 co-vis 语料的历史边界是 `history_end_exclusive`(= window_start),不是 window_end:
    rank_train co-vis 语料 = 原始 train 中 ts < history_end_exclusive 的历史
                            + rank_train/input.parquet(可见前半段)
                            − 全部 hidden tail(那是训练 label,绝不进语料)
  manifest 用 corpus_rule 字段把这条写死,防止 Step 0 误用 window_end 把 tail 数进去。

纪律:同 session 不跨折(anti-join 全部 val session)· rank_valid == 现有 val
     · sort→sample 固定 seed 可复现 · cut 排序带 aid 兜底键。

已知口径限制(为保持 Phase 2 可比,当前接受):
    现有 val 是“按事件时间取最后一周”,不是 OTTO 官方的“按 session 首次事件归属”。
    原始数据中有 1,514,189 个 session 在倒数第二周和最后一周均有事件
    (约占未排除 rank-train session 的 32.6%)。当前代码将这些 session 保留在 rank-valid,
    并从 rank-train query 中整体排除;其更早事件仍可作为 valid 时刻已发生的历史语料。
    这是当前本地 CV 的实用口径,不是“所有序列推荐都必须 user/session 不重叠”的通用定律。
本文件只造折,不碰 co-vis / 特征 / 模型。

跑法:
    python src/rank/make_folds.py            # 全量 → folds/full/
    python src/rank/make_folds.py --sample   # MVP:train 20万/valid 10万 → folds/sample/
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
TRAIN_PARQUET = DATA / "parquet" / "train" / "*.parquet"
VAL_DIR = DATA / "parquet" / "val"           # Phase 1/2 现有验证集 = rank_valid 来源
FOLDS_DIR = DATA / "parquet" / "folds"
DAY_MS = 86_400_000
VAL_DAYS = 7
INV_TYPE = {0: "clicks", 1: "carts", 2: "orders"}


def _query_counts(fl: pl.DataFrame) -> dict:
    """每类有 ground truth 的 session 数(= 该 type 的 query 数),用于估三模型训练规模。"""
    return {INV_TYPE[t]: int(fl.filter(pl.col("type") == t).height) for t in (0, 1, 2)}


def _make_train_fold(window_start: int, window_end: int, exclude_sessions: pl.DataFrame,
                     seed: int = 42, sample_sessions: int | None = None):
    """半开区间 [window_start, window_end) 内切 rank_train 折,排除 exclude_sessions。"""
    ev = (pl.scan_parquet(TRAIN_PARQUET)
            .filter((pl.col("ts") >= window_start) & (pl.col("ts") < window_end))   # 半开区间
            .collect()
            .filter(pl.len().over("session") >= 2)
            .join(exclude_sessions, on="session", how="anti"))                       # 同 session 不跨折

    # ★ 先给全部 session 定随机数 r,再抽样并把 r 带过去 →
    #   同 session + 同 seed 下,sample 的切点 == full 的切点,sample 是 full 的严格子集。
    all_ids = ev.select("session").unique().sort("session")
    rng = np.random.default_rng(seed)
    rand = all_ids.with_columns(pl.Series("r", rng.random(all_ids.height)))

    if sample_sessions is not None and sample_sessions < all_ids.height:
        keep = all_ids.sample(n=sample_sessions, seed=seed)
        ev = ev.join(keep, on="session", how="semi")
        rand = rand.join(keep, on="session", how="semi")     # 保留已分配好的 r,不重新生成

    ev = (ev.join(rand, on="session")
            .sort(["session", "ts", "aid"])                       # aid 兜底键:tied ts 也确定
            .with_columns(idx=pl.int_range(pl.len()).over("session"),
                          n=pl.len().over("session"))
            .with_columns(cut=(1 + (pl.col("r") * (pl.col("n") - 1)).floor()).cast(pl.Int32)))

    fold_input = ev.filter(pl.col("idx") < pl.col("cut")).select("session", "aid", "ts", "type")
    tail = ev.filter(pl.col("idx") >= pl.col("cut"))

    clicks = (tail.filter(pl.col("type") == 0).group_by("session")
                  .agg(pl.col("aid").sort_by("idx").head(1).alias("ground_truth"))
                  .with_columns(type=pl.lit(0, dtype=pl.Int8)))
    carts = (tail.filter(pl.col("type") == 1).group_by("session")
                 .agg(pl.col("aid").unique().sort().alias("ground_truth"))
                 .with_columns(type=pl.lit(1, dtype=pl.Int8)))
    orders = (tail.filter(pl.col("type") == 2).group_by("session")
                  .agg(pl.col("aid").unique().sort().alias("ground_truth"))
                  .with_columns(type=pl.lit(2, dtype=pl.Int8)))
    fold_labels = pl.concat([clicks, carts, orders]).select("session", "type", "ground_truth")
    return fold_input, fold_labels


def _load_and_validate_val():
    """读现有 val 作为 rank_valid 来源,并校验它确实是干净的完整验证集。"""
    vi = pl.read_parquet(VAL_DIR / "input.parquet")
    vl = pl.read_parquet(VAL_DIR / "labels.parquet")
    dup = vl.group_by(["session", "type"]).len().filter(pl.col("len") > 1).height
    assert dup == 0, f"val labels 有重复 (session,type):{dup}"
    orphan = vl.join(vi.select("session").unique(), on="session", how="anti").height
    assert orphan == 0, f"val 有 {orphan} 个 label session 不在 input 里"
    prov = dict(
        source=str(VAL_DIR.relative_to(ROOT)),
        n_sessions=int(vi["session"].n_unique()),
        n_input_events=int(vi.height), n_label_rows=int(vl.height),
        ts_min=int(vi["ts"].min()), ts_max=int(vi["ts"].max()),
        n_query=_query_counts(vl),
    )
    return vi, vl, prov


def _fold_meta(fi: pl.DataFrame, fl: pl.DataFrame,
               window_start: int, window_end_exclusive: int, seed: int) -> dict:
    return {
        "window_start": int(window_start),
        "window_end_exclusive": int(window_end_exclusive),
        "history_end_exclusive": int(window_start),   # ★ co-vis 语料的历史边界(不是 window_end)
        "corpus_rule": ("co-vis 语料 = 原始 train 中 ts < history_end_exclusive 的历史 "
                        "+ 本折 input;绝不含 hidden tail(那是训练 label 来源)"),
        "val_days": VAL_DAYS, "seed": seed,
        "n_sessions": int(fi["session"].n_unique()),
        "n_input_events": int(fi.height), "n_label_rows": int(fl.height),
        "n_query": _query_counts(fl),
    }


if __name__ == "__main__":
    sample = "--sample" in sys.argv
    seed = 42
    n_train = 200_000 if sample else None
    n_valid = 100_000 if sample else None
    out_dir = FOLDS_DIR / ("sample" if sample else "full")     # ← sample/full 分目录,不互相覆盖
    t0 = time.time()

    max_ts = pl.scan_parquet(TRAIN_PARQUET).select(pl.col("ts").max()).collect().item()

    # rank_valid = 现有 val(校验来源);排除跨折用全部 val session,不只 sample 的
    valid_in_full, valid_lb_full, val_prov = _load_and_validate_val()
    all_val_sessions = valid_in_full.select("session").unique()
    if n_valid is not None and n_valid < valid_in_full["session"].n_unique():
        keep = valid_in_full.select("session").unique().sort("session").sample(n=n_valid, seed=seed)
        valid_in = valid_in_full.join(keep, on="session", how="semi")
        valid_lb = valid_lb_full.join(keep, on="session", how="semi")
    else:
        valid_in, valid_lb = valid_in_full, valid_lb_full

    # rank_train = 往前推一周,半开区间 [max_ts-14d, max_ts-7d)
    train_start = max_ts - 2 * VAL_DAYS * DAY_MS
    train_end = max_ts - VAL_DAYS * DAY_MS
    train_in, train_lb = _make_train_fold(train_start, train_end, all_val_sessions,
                                          seed=seed, sample_sessions=n_train)

    # ---- 收尾断言 ----
    overlap = (train_in.select("session").unique()
               .join(valid_in.select("session").unique(), on="session", how="semi").height)
    assert overlap == 0, f"两折 session 交集 = {overlap}(应为 0)"
    assert train_lb.group_by(["session", "type"]).len().filter(pl.col("len") > 1).height == 0, "train 有重复 (session,type)"
    assert train_in["session"].n_unique() == train_lb["session"].n_unique(), "train input/label session 数不一致"
    assert train_in["ts"].min() >= train_start, "train input 时间越下界"
    assert train_in["ts"].max() < train_end, "train input 时间越上界(半开区间)"

    # ---- 落盘 ----
    for name, fi, fl in [("rank_valid", valid_in, valid_lb), ("rank_train", train_in, train_lb)]:
        d = out_dir / name
        d.mkdir(parents=True, exist_ok=True)
        fi.write_parquet(d / "input.parquet")
        fl.write_parquet(d / "labels.parquet")

    manifest = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": "sample" if sample else "full",
        "max_ts": int(max_ts),
        "rank_valid": {**_fold_meta(valid_in, valid_lb, max_ts - VAL_DAYS * DAY_MS, max_ts + 1, seed),
                       "source": "复用现有 val", "val_source_provenance": val_prov},
        "rank_train": _fold_meta(train_in, train_lb, train_start, train_end, seed),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"\n[自查] 两折 session 交集 = {overlap} ✅")
    print(f"[自查] 断言全过(无重复 label / input==label session / 半开区间时间界)✅")
    print(f"[make_folds] ✅ mode={manifest['mode']} 用时 {time.time() - t0:.0f}s -> {out_dir}")
