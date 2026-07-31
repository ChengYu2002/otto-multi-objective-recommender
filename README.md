# OTTO — Multi-Objective Recommender System

多路召回 → GBDT 多目标精排的推荐流水线(Kaggle OTTO 比赛)。
项目当前处于开发阶段：数据预处理与本地验证框架已经完成，召回与精排模块尚在实现中。

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
