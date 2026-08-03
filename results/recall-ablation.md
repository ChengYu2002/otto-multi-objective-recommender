# Recall Ablation Log

本文件记录 OTTO 多路召回的本地开发实验。指标为与比赛一致的加权
Recall@20：

```text
weighted = 0.10 × clicks + 0.30 × carts + 0.60 × orders
```

## 2026-08-03 · Click co-vis 首次全量实验

运行范围：全量本地验证集、全量 click co-vis 邻居表。

| ID | 候选方案 | Clicks | Carts | Orders | Weighted | Δ Weighted |
|---|---|---:|---:|---:|---:|---:|
| R001 | Recent baseline（最近去重商品） | 0.3065 | 0.2554 | 0.5046 | 0.4100 | — |
| R002 | 加权 self + click co-vis + popularity 兜底 | 0.4621 | 0.3409 | 0.5710 | **0.4911** | **+0.0811** |

观察：三个目标均有提升，其中 clicks 的绝对提升最大（+0.1556），符合
click co-vis 主要补充共同浏览候选的预期。Weighted Recall@20 提升 8.11 个
百分点，相对 R001 提升约 19.8%。

### 结果状态

**Preliminary / development-only（开发期初步结果）**。

当前结果适合证明召回流水线已经跑通，但不应作为最终 CV 或 Kaggle 成绩引用：

1. co-vis 与 popularity 当前从完整原始 train 构建，而验证标签也从该 train
   的最后 7 天切出，存在隐藏未来进入统计表的风险，分数可能偏高。
2. R002 同时改变了 self 排序、加入 click co-vis 和 popularity，因此它是开发期
   方案对照，还不是单独隔离 click co-vis 贡献的严格消融。
3. R002 暂时将同一份 prediction 复制给 clicks、carts、orders，尚未完成分目标融合。

### 下一轮实验

- 使用验证时间边界之前的历史事件重建 co-vis 与 popularity，消除未来信息。
- 补齐 `weighted self`、`weighted self + popularity`、
  `weighted self + click co-vis` 三个中间版本，形成严格消融。
- 加入 buy-weighted 与 buy2buy，并分别生成 clicks、carts、orders 候选。

