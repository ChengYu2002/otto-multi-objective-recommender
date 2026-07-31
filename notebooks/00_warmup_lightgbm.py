"""
Phase 0 热身:环境自检 + LightGBM 手感
=====================================
目的:
  1. 确认 conda 环境 otto 装好了(polars / lightgbm / sklearn 都能 import)
  2. 走一遍 LightGBM 的标准流程:数据 -> train/valid 划分 -> 训练(early stopping)
     -> 评估 -> 看特征重要性。这套流程 Phase 3 精排会原样复用。

跑法:
  conda activate otto
  python notebooks/00_warmup_lightgbm.py
"""

import numpy as np
import polars as pl
import lightgbm as lgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

print("=" * 60)
print("环境自检")
print("=" * 60)
print("polars   ", pl.__version__)
print("lightgbm ", lgb.__version__)
print("numpy    ", np.__version__)

# ---------------------------------------------------------------
# 1. 造一个玩具数据集(乳腺癌二分类,30 个特征)
#    在 OTTO 里,这一步会换成"候选商品 + 特征 + 是否被点击/加购/下单的标签"
# ---------------------------------------------------------------
X, y = load_breast_cancer(return_X_y=True, as_frame=True)
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
print(f"\n训练集 {X_train.shape} | 验证集 {X_valid.shape}")

# ---------------------------------------------------------------
# 2. 训练 LightGBM(binary),带 early stopping
#    OTTO 精排里会把 objective 换成 'lambdarank'(排序),但训练骨架一样
# ---------------------------------------------------------------
train_set = lgb.Dataset(X_train, y_train)
valid_set = lgb.Dataset(X_valid, y_valid, reference=train_set)

params = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "verbose": -1,
}

print("\n开始训练...")
model = lgb.train(
    params,
    train_set,
    num_boost_round=500,
    valid_sets=[train_set, valid_set],
    valid_names=["train", "valid"],
    callbacks=[
        lgb.early_stopping(stopping_rounds=30),
        lgb.log_evaluation(period=50),
    ],
)

# ---------------------------------------------------------------
# 3. 评估
# ---------------------------------------------------------------
pred = model.predict(X_valid, num_iteration=model.best_iteration)
auc = roc_auc_score(y_valid, pred)
print(f"\n验证集 AUC = {auc:.4f}  (best_iteration={model.best_iteration})")

# ---------------------------------------------------------------
# 4. 特征重要性 —— 检查哪些特征对模型贡献最大
# ---------------------------------------------------------------
imp = (
    pl.DataFrame({
        "feature": model.feature_name(),
        "gain": model.feature_importance(importance_type="gain"),
    })
    .sort("gain", descending=True)
    .head(10)
)
print("\nTop 10 重要特征(按 gain):")
print(imp)

print("\n✅ 环境 OK,LightGBM 流程跑通。Phase 0 的自检部分完成。")
