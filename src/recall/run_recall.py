"""
Phase 2 · 召回 · 编排 + 量分 + 消融
==================================
把召回各路串起来,用 Phase 1 的 evaluate 打分,和 recent baseline 对比。
每加一路跑一次 → 攒消融表。

跑法:
    python src/recall/run_recall.py --sample   # 随机 10 万 val；复用已有矩阵，缺失时建全量(不再建 1/6)
    python src/recall/run_recall.py            # 全量 val + 全量 co-vis(真实分数)
    python src/recall/run_recall.py --build     # 强制重建 co-vis 矩阵
"""
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # 让 src.* 可导入

from src.validation import (evaluate, evaluate_at_ks, popular_share,   # noqa: E402
                            baseline_recent, VAL_DIR)
from src.recall.covis import build_covis, load_covis, COVIS_DIR       # noqa: E402
from src.recall.popularity import build_popularity                    # noqa: E402
from src.recall.candidates import generate_predictions_chunked        # noqa: E402


def _fmt(name: str, s: dict) -> None:
    print(f"  {name:<18} clicks {s['clicks']:.4f} | carts {s['carts']:.4f} | "
          f"orders {s['orders']:.4f} | weighted {s['weighted']:.4f}")


if __name__ == "__main__":
    sample = "--sample" in sys.argv
    rebuild = "--build" in sys.argv
    # 输入
    val_input = pl.read_parquet(VAL_DIR / "input.parquet")
    # 正确答案
    val_labels = pl.read_parquet(VAL_DIR / "labels.parquet")

    if sample:
        # 随机抽 10 万 session(不覆盖 val 文件)。用随机而非按 id 取前 N:
        # 按 id 取前 N 是偏样本(长 session 扎堆,baseline 只有 0.33);随机采样
        # 才和全量同分布(baseline ~0.41),小旋钮实验的相对排序才可信。
        keep = val_input.select("session").unique().sort("session").sample(100_000, seed=42)
        val_input = val_input.join(keep, on="session", how="semi")
        val_labels = val_labels.join(keep, on="session", how="semi")

    KINDS = ["click", "buy_weighted", "buy2buy"]

    for kind in KINDS:
        # 建 / 载当前 kind 的 co-vis 矩阵(用防泄漏语料;--build 时连语料一起重建)
        # 检查是否需要建矩阵:若 parquet 文件不存在或 --build,就建
        if rebuild or not (COVIS_DIR / f"{kind}.parquet").exists():
            # 永远建全量矩阵:--sample 只切 val,不再建 1/6 采样矩阵(防口径混用/覆盖)
            build_covis(kind, rebuild_corpus=rebuild)
    
    # 把三张构建好的co-vis 矩阵载入
    # output:  {'click': DataFrame, 'buy_weighted': DataFrame, 'buy2buy': DataFrame}
    matrices = {kind: load_covis(kind) for kind in KINDS}

    popular = build_popularity(k=100)      # 出 top-100 候选,兜底表也要更长

    gen_k = 100 if sample else 20          # sample 出 top-100 看天花板;全量只需正式 @20
    eval_ks = [20, 50, 100] if sample else [20]
    n_chunks = 1 if sample else 20         # 全量按 session 分 20 批,控内存 + 提速
    preds = generate_predictions_chunked(val_input, matrices, popular,
                                         k=gen_k, n_chunks=n_chunks)

    scope = "sample(随机10万)" if sample else "全量(分块)"
    print(f"\n=== 召回评估 · {scope} ===")
    base = evaluate(baseline_recent(val_input), val_labels)
    print(f"  recent baseline      weighted@20 {base['weighted']:.4f}")
    res = evaluate_at_ks(preds, val_labels, ks=eval_ks)
    for kk in eval_ks:
        s = res[kk]
        print(f"  multi-covis @{kk:<3}    clicks {s['clicks']:.4f} | carts {s['carts']:.4f} | "
              f"orders {s['orders']:.4f} | weighted {s['weighted']:.4f}")
    if sample:
        sh = popular_share(preds, popular, k=20)
        print(f"  热门占比(top20,上界) clicks {sh.get('clicks', 0):.1%} | "
              f"carts {sh.get('carts', 0):.1%} | orders {sh.get('orders', 0):.1%}")
