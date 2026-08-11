# OTTO — Multi-Objective Recommender System

多路召回 → GBDT 多目标精排的推荐流水线(Kaggle OTTO 比赛)。
项目当前已完成功能层面的 Phase 2 纯规则多路召回：三张 co-visitation 矩阵、分目标候选生成、
防泄漏验证、分块内存控制与 Recall@K 诊断均已跑通。确定性 tie-break 已进入代码，待重建矩阵并
重跑一次全量锁定最终分数后，进入候选级特征与 GBDT 多目标精排；Kaggle CV↔LB 对齐仍待完成。

## 快速开始

```bash
conda activate otto
python src/data_prep.py           # 原始 JSONL 转 Parquet
python src/validation.py --sample # 构造样本验证集并运行 recent baseline
python src/recall/run_recall.py --sample # 用现有矩阵复核 multi-covis @20/@50/@100
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

时间隔离后的最终全量召回将 Weighted Recall@20 从 `0.4100` 提升至
`0.4755`（绝对 `+0.0655`，相对约 `+16.0%`）。随机 10 万 session 的当前复核中，
确定性修复后的两次连续复核中，Weighted Recall@20/@50/@100 均为
`0.4777 / 0.5147 / 0.5398`，其中 @100 用于估计
Phase 3 候选天花板。旧的 `0.4911` 结果因验证未来混入统计表而作废。

`run_recall.py --sample` 已不再构建部分矩阵，主入口的 sample/full 混用风险已消除；工程上仍需
补充缓存 manifest/API 防护、自动化回归测试，以及包含 `source/rank/score` 的 Top-100 候选长表。

完整结果、限制说明与下一轮实验见
[`results/experiments.md`](results/experiments.md)。
