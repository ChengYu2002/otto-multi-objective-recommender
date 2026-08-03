# Recall Ablation Log

本文件记录 OTTO 多路召回的本地开发实验。指标为与比赛一致的加权
Recall@20：

```text
weighted = 0.10 × clicks + 0.30 × carts + 0.60 × orders
```

## 2026-08-03 · Click co-vis 全量时间隔离实验

运行范围：全量本地验证集、全量 click co-vis 邻居表。当前可引用的主结果为
R002；旧的 `0.4911` 使用了会混入验证未来的建表方式，已作废并移到历史记录。

| ID | 候选方案 | Clicks | Carts | Orders | Weighted | Δ Weighted |
|---|---|---:|---:|---:|---:|---:|
| R001 | Recent baseline（最近去重商品） | 0.3065 | 0.2554 | 0.5046 | 0.4100 | — |
| R002 | 加权 self + click co-vis + popularity（时间隔离） | **0.4144** | **0.3047** | **0.5427** | **0.4585** | **+0.0485** |

三个目标都高于 baseline：clicks `+0.1079`、carts `+0.0493`、orders
`+0.0381`。Weighted Recall@20 提升 `0.0485`（4.85 个百分点），相对
R001 提升约 `11.8%`。加权提升可拆成：

```text
0.10 × 0.1079 + 0.30 × 0.0493 + 0.60 × 0.0381
= 0.01079 + 0.01479 + 0.02286
= 0.04844 ≈ 0.0485
```

虽然 clicks 的原始涨幅最大，但 orders 的指标权重是 0.60，因此它对最终提升的
贡献最大。这说明 click co-vis 不只补到了下一次浏览候选，也给 carts/orders
提供了有效相关商品。

### 时间隔离怎么做

- `src/validation.py` 以原始 train 最后 7 天为验证窗口，再把每个验证 session
  随机切成可见的 `val_input` 和隐藏未来 `val_labels`。
- `src/recall/covis.py` 的建表语料是“验证窗口之前的历史 + `val_input`”，不含
  `val_labels` 对应的未来事件。
- `src/recall/popularity.py` 只统计验证窗口边界之前的近期热门，不使用窗口内部事件。
- 本次矩阵用 `python src/recall/run_recall.py --build` 强制重建，避免读取旧缓存。

产物核对：窗口前历史有 `163,955,180` 行，`val_input` 有 `25,888,554` 行，
两者合计 `189,843,734` 行，与 `covis_corpus.parquet` 实际行数完全一致。

### 结果状态

**Current local development CV（当前本地开发分数，已做时间隔离）**。

这个结果可以公开写进项目 README 和面试材料，用来说明时间隔离后的真实进展；
但仍不能把 R001→R002 的全部提升直接归因于 click co-vis：

1. R002 同时改变了 self 排序、加入 click co-vis 和 popularity，并非严格的单变量消融。
2. R002 暂时将同一份 prediction 复制给 clicks、carts、orders，尚未完成分目标融合。
3. 这是自建的本地切分，不等同于 Kaggle leaderboard 分数。
4. `VAL_DAYS`/`val_days` 目前在多个文件中分别配置；以后改变验证窗口时必须同步并用
   `--build` 重建语料和矩阵。
5. sample 与 full 当前共用 `data/parquet/covis/click.parquet`；执行
   `--sample --build` 会把样本矩阵写到同一路径。之后跑全量前必须再次只用
   `--build` 重建，后续应通过不同文件名或 manifest 消除这个缓存风险。

### 已作废的历史结果（仅留审计记录）

| 版本 | Clicks | Carts | Orders | Weighted | 与当前结果的关系 |
|---|---:|---:|---:|---:|---|
| 旧 R002（存在泄漏风险） | 0.4621 | 0.3409 | 0.5710 | 0.4911 | 作废；比时间隔离结果虚高 0.0326 |

旧矩阵曾直接从完整原始 train 建表，而验证答案也来自 train 的最后 7 天，因而隐藏未来
可能进入共现和热门统计。隔离后分数下降是合理现象，不代表模型退步；`0.4585` 才是当前
应该继续比较的基准。

### 下一轮严格消融

- R003：weighted self only。
- R004：weighted self + popularity。
- R005：weighted self + click co-vis（不加 popularity）。
- 再加入 buy-weighted 与 buy2buy，并分别生成 clicks、carts、orders 候选。
