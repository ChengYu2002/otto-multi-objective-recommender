"""
生成 Kaggle 提交文件 → results/submission.csv
============================================
和本地召回**同一套 pipeline**,仅两处口径不同(因为要预测真 test):
  1. co-vis 语料 = 全量 train + test 输入
     (transductive:test 无"未来"可泄漏,标签在 Kaggle 手里,不需要 build_corpus 那套排除);
  2. 预测对象 = test 全部 session;热门兜底用 train 真实最后一周(不排除 val 周)。
矩阵写到单独目录 covis_submit/,**不碰本地 covis/**(靠 monkeypatch 改 covis 模块的路径全局量)。

输出格式(OTTO 官方):
    session_type,labels
    12899779_clicks,588 923 471 ...     # labels = 空格分隔的 top-20 aid
每个 test session 出 3 行(clicks/carts/orders)。

⚠️ 本脚本只生成文件。**提交动作需你手动**(以你的 Kaggle 账号):
    kaggle competitions submit -c otto-recommender-system \\
        -f results/submission.csv -m "covis multi-recall"
  前提:已在比赛页 Join 并接受规则、~/.kaggle/kaggle.json 配好。

跑法:
    python src/recall/submit.py            # 复用已建的 submit 矩阵
    python src/recall/submit.py --build     # 重建 submit 语料 + 矩阵(改了召回逻辑后)
"""
import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import src.recall.covis as covis                                   # noqa: E402
from src.recall.candidates import generate_predictions_chunked     # noqa: E402

DATA = ROOT / "data"
TRAIN_PARQUET = DATA / "parquet" / "train" / "*.parquet"
TEST_PARQUET = DATA / "parquet" / "test" / "*.parquet"
DAY_MS = 86_400_000
KINDS = ["click", "buy_weighted", "buy2buy"]
TYPE_NAME = {0: "clicks", 1: "carts", 2: "orders"}
OUT = ROOT / "results" / "submission.csv"

# 把 co-vis 的语料/输出目录指到提交专用路径 → build_covis/load_covis 都走这里,
# 本地 covis/ 与 covis_corpus.parquet 一律不受影响。
covis.CORPUS_PARQUET = DATA / "parquet" / "submit_corpus.parquet"
covis.COVIS_DIR = DATA / "parquet" / "covis_submit"


def build_submit_corpus(force: bool = False) -> None:
    """提交语料 = 全量 train + 全量 test(都是已观测事件,无未来)。流式写。"""
    if covis.CORPUS_PARQUET.exists() and not force:
        return
    covis.CORPUS_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    pl.concat([pl.scan_parquet(TRAIN_PARQUET),
               pl.scan_parquet(TEST_PARQUET)]).sink_parquet(covis.CORPUS_PARQUET)
    print("[submit] 语料已建:train + test", flush=True)


def submit_popularity(days: int = 7, k: int = 100) -> list[int]:
    """热门兜底:train 真实最后一周最高频 aid(提交口径,不排除 val 周;aid 兜底键)。"""
    lf = pl.scan_parquet(TRAIN_PARQUET)
    max_ts = lf.select(pl.col("ts").max()).collect().item()
    top = (lf.filter(pl.col("ts") >= max_ts - days * DAY_MS)
             .group_by("aid").agg(pl.len().alias("cnt"))
             .sort(["cnt", "aid"], descending=[True, False]).head(k).collect())
    return top["aid"].to_list()


if __name__ == "__main__":
    build = "--build" in sys.argv

    # 1. 提交语料(train+test)→ 2. 从它建 3 张矩阵到 covis_submit/
    build_submit_corpus(force=build)
    for kind in KINDS:
        if build or not (covis.COVIS_DIR / f"{kind}.parquet").exists():
            covis.build_covis(kind)     # 语料已存在会自动复用,不会重建成本地口径
    matrices = {kind: covis.load_covis(kind) for kind in KINDS}
    popular = submit_popularity(k=100)

    # 3. 对全部 test session 生成 top-20
    test_input = pl.read_parquet(TEST_PARQUET)
    preds = generate_predictions_chunked(test_input, matrices, popular,
                                         k=20, n_chunks=20)

    # 4. 格式化成 session_type,labels
    sub = (preds.with_columns(
                session_type=pl.col("session").cast(pl.Utf8) + "_"
                             + pl.col("type").replace_strict(TYPE_NAME, return_dtype=pl.Utf8),
                labels=pl.col("prediction").cast(pl.List(pl.Utf8)).list.join(" "),
           ).select("session_type", "labels"))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    sub.write_csv(OUT)

    n_sess = test_input["session"].n_unique()
    ok = sub.height == n_sess * 3
    print(f"[submit] ✅ 写出 {OUT}", flush=True)
    print(f"[submit] 行数 {sub.height:,} | test session {n_sess:,} × 3 = {n_sess * 3:,} "
          f"| 覆盖完整: {ok}", flush=True)
    if not ok:
        print("[submit] ⚠️ 行数对不上!有 session/type 缺预测,Kaggle 会判不完整,需排查。", flush=True)
