"""
Phase 2 · 召回 · 编排 + 量分 + 消融
==================================
把召回各路串起来,用 Phase 1 的 evaluate 打分,和 recent baseline 对比。
每加一路跑一次 → 攒消融表。

跑法:
    python src/recall/run_recall.py --sample   # 10 万 val 切片 + 1/6 训练建矩阵,快速验证
    python src/recall/run_recall.py            # 全量 val + 全量 co-vis(真实分数)
    python src/recall/run_recall.py --build     # 强制重建 co-vis 矩阵
"""
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # 让 src.* 可导入

from src.validation import evaluate, baseline_recent, VAL_DIR   # noqa: E402
from src.recall.covis import build_covis, load_covis, COVIS_DIR  # noqa: E402
from src.recall.popularity import build_popularity              # noqa: E402
from src.recall.candidates import generate_predictions          # noqa: E402


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
        # 取前 10 万 session 切片,不覆盖 val 文件
        keep = val_input.select("session").unique().sort("session").head(100_000)
        # 只保留切片(keep)的 session,防止泄漏
        val_input = val_input.join(keep, on="session", how="semi")
        val_labels = val_labels.join(keep, on="session", how="semi")

    KINDS = ["click", "buy_weighted", "buy2buy"]

    for kind in KINDS:
        # 建 / 载 click co-vis 矩阵(用防泄漏语料;--build 时连语料一起重建)
        # 检查是否需要建矩阵:若 parquet 文件不存在或 --build,就建
        if rebuild or not (COVIS_DIR / f"{kind}.parquet").exists():
            build_covis(kind, max_chunks=5 if sample else None, rebuild_corpus=rebuild)
    
    # 把三张构建好的co-vis 矩阵载入
    # output:  {'click': DataFrame, 'buy_weighted': DataFrame, 'buy2buy': DataFrame}
    matrices = {kind: load_covis(kind) for kind in KINDS}

    popular = build_popularity()

    scope = "sample(10万 val · 1/6 训练)" if sample else "全量"
    print(f"\n=== 召回评估 · {scope} ===")
    _fmt("recent(自身臂)", evaluate(baseline_recent(val_input), val_labels))
    _fmt("multi-covis", evaluate(generate_predictions(val_input, matrices, popular), val_labels))
