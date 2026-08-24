"""
Phase 3 · 精排 · Step 0 · Block A:给每折建 cutoff-specific co-vis
================================================================
每折用它自己 manifest 里的 history_end_exclusive 建**防泄漏语料**,再建 3 张 co-vis,
矩阵写到**独立路径** rank_artifacts/,绝不碰 Phase 2 的 data/parquet/covis/。

防泄漏语料 = 原始 train 中 ts < history_end_exclusive 的历史
           + 本折 input(只有可见前半段,make_folds 已剔除 hidden tail)
所以语料天然不含该折的答案;再断言 max_ts < window_end,确保 rank_train 不含 val 周。

★ 历史用**全量**(不采样),只有 fold input 是采样的 —— 否则矩阵稀疏、MVP 结论失真。
矩阵有缓存;建一次,后面候选/特征/模型都读它。

跑法:
    python src/rank/build_fold_covis.py --sample   # 读 folds/sample/,写 rank_artifacts/sample/
    python src/rank/build_fold_covis.py            # 读 folds/full/,写 rank_artifacts/full/
"""
import json
import sys
import time
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import src.recall.covis as covis                    # noqa: E402  复用 build_covis

DATA = ROOT / "data"
FOLDS_DIR = DATA / "parquet" / "folds"
RANK_ART = DATA / "parquet" / "rank_artifacts"
KINDS = ["click", "buy_weighted", "buy2buy"]


def build_fold_covis(mode: str, fold: str, history_end: int, window_end: int) -> None:
    fold_dir = FOLDS_DIR / mode / fold
    out_covis = RANK_ART / mode / fold / "covis"
    corpus = RANK_ART / mode / fold / "corpus.parquet"
    out_covis.mkdir(parents=True, exist_ok=True)

    # 1. 防泄漏语料 = 历史(ts < history_end) + 本折 input(无 tail)
    history = pl.scan_parquet(covis.TRAIN_PARQUET).filter(pl.col("ts") < history_end)
    fin = pl.scan_parquet(fold_dir / "input.parquet").select("session", "aid", "ts", "type")
    pl.concat([history, fin]).sink_parquet(corpus)

    # 2. 断言:语料没越过本折窗口(rank_train 因此不含 val 周)
    cmax = pl.scan_parquet(corpus).select(pl.col("ts").max()).collect().item()
    assert cmax < window_end, f"{fold} 语料越界:max_ts {cmax} >= window_end {window_end}"
    print(f"[fold-covis] {mode}/{fold}: 语料建好,max_ts < window_end 断言过 ✅", flush=True)

    # 3. 建 3 张矩阵到 rank_artifacts(monkeypatch 把 covis 的语料/输出指到本折)
    covis.CORPUS_PARQUET = corpus        # 已 sink,build_corpus 直接复用,不会重建成 Phase2 口径
    covis.COVIS_DIR = out_covis
    for kind in KINDS:
        covis.build_covis(kind)          # 全量历史,不 max_chunks
    print(f"[fold-covis] {mode}/{fold}: 3 张矩阵 -> {out_covis}", flush=True)


if __name__ == "__main__":
    mode = "sample" if "--sample" in sys.argv else "full"
    manifest = json.loads((FOLDS_DIR / mode / "manifest.json").read_text())
    t0 = time.time()
    for fold in ("rank_train", "rank_valid"):
        m = manifest[fold]
        build_fold_covis(mode, fold, m["history_end_exclusive"], m["window_end_exclusive"])
    print(f"[fold-covis] ✅ 全部完成 mode={mode} 用时 {time.time() - t0:.0f}s")
