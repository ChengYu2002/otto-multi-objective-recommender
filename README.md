# OTTO — Multi-Objective Recommender System

多路召回 → GBDT 多目标精排的推荐流水线(Kaggle OTTO 比赛)。
项目当前处于开发阶段：数据预处理、本地验证框架与 click co-vis 首个召回版本
已经跑通，并完成了召回建表的时间隔离；严格单变量消融与多目标精排仍在实现中。

## 快速开始

```bash
conda activate otto
python src/data_prep.py           # 原始 JSONL 转 Parquet
python src/validation.py --sample # 构造样本验证集并运行 recent baseline
```

## 结构

```
src/
├── validation.py   # 本地评估框架(与 LB 一致的加权 Recall@20)
├── recall/         # 召回:co-visitation、popularity
├── features/       # 特征工程
└── rank/           # GBDT 精排(LambdaMART)
```

## 指标

加权 Recall@20 = 0.10 × clicks + 0.30 × carts + 0.60 × orders

## 开发实验

时间隔离后的全量召回对照实验将 Weighted Recall@20 从 `0.4100` 提升至
`0.4585`（绝对 `+0.0485`，相对约 `+11.8%`）。旧的 `0.4911` 结果存在
验证未来混入统计表的风险，已作废并只保留在实验日志中用于审计。当前对照仍同时
改变了 self 排序、click co-vis 和 popularity，因此还不是严格的单变量消融。

完整结果、限制说明与下一轮实验见
[`results/recall-ablation.md`](results/recall-ablation.md)。
