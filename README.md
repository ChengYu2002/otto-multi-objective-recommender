# OTTO — Multi-Objective Recommender System

多路召回 → GBDT 多目标精排的推荐流水线(Kaggle OTTO 比赛)。
项目当前处于开发阶段：数据预处理、本地验证框架与 click co-vis 首个召回版本
已经跑通；严格时间隔离的召回验证与多目标精排仍在实现中。

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

首个全量召回对照实验将 Weighted Recall@20 从 `0.4100` 提升至 `0.4911`。
该结果属于开发期初步结果，当前仍存在验证时间隔离待修复、实验变量尚未完全拆分等限制。

完整结果、限制说明与下一轮实验见
[`results/recall-ablation.md`](results/recall-ablation.md)。
