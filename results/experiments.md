# OTTO 实验记录 / 消融表

> 加权 Recall@20 = 0.10·clicks + 0.30·carts + 0.60·orders
> 最后更新:2026-08-06

## ⚠️ 评估口径说明(**数不能跨口径比**)

| 口径 | co-vis 矩阵 | 验证集 | 用途 | baseline |
|---|---|---|---|---|
| **全量** | 全量(30 分块) | 全量 4.36M val | 官方数,最慢(~28min) | 0.4100 |
| **full矩阵/10万** | 全量 | 前 10 万 session 切片 | 快速隔离实验(几分钟) | 0.3288 |
| **1/6采样** | 5/30 分块 | 前 10 万切片 | 开发期秒级自测 | 0.3288 |

> 10 万切片(按 session id 取前 10 万)系统性偏低(baseline 0.3288 vs 全量 0.4100),**只在同口径内比较**。

---

## Phase 1 · baseline(recent 自身臂)

| 口径 | clicks | carts | orders | **weighted** |
|---|---|---|---|---|
| 全量 | 0.3065 | 0.2554 | 0.5046 | **0.4100** |

---

## Phase 2 Step 1 · click co-vis

| 口径 | 方法 | clicks | carts | orders | **weighted** | 备注 |
|---|---|---|---|---|---|---|
| 全量 | + click covis(**泄漏**) | 0.4621 | 0.3409 | 0.5710 | ~~0.4911~~ | ❌ 验证泄漏,虚高 |
| 全量 | + click covis(**修复**) | 0.4144 | 0.3047 | 0.5427 | **0.4585** | ✅ leak-free,**+11.8%** vs baseline |

**关键发现 — 验证泄漏(target leakage)**
`build_covis` 直接读原始 train,把验证 session 的**未来事件**(val_labels)一起数进了共现矩阵 → "用答案预测答案"。
- **修复**:新增 `build_corpus()`,语料 = 历史(ts<边界)⊕ val_input,**排除所有未来**;`popularity` 同样只数边界前。
- **影响**:泄漏在全量把 weighted 虚高 **0.0326(+6.6%)**;长尾稀有对受影响最重。修复后 lift 仍有 +11.8%,主体是真本事。

---

## Phase 2 Step 2 · buy_weighted + buy2buy + 分目标融合

**隔离实验(full矩阵 / 10万切片,同口径)——只换 carts/orders 的 co-vis 矩阵:**

| 方法 | clicks | carts | orders | **weighted** | Δweighted |
|---|---|---|---|---|---|
| all-click(对照) | 0.3869 | 0.2292 | 0.4399 | 0.3714 | — |
| buy for carts/orders | 0.3869 | 0.2301 | 0.4416 | **0.3727** | **+0.0013** |

**关键发现 — 买矩阵只有边际贡献(+0.0013,+0.35%)**
1. **buy_weighted ≈ click**:同一批源事件(`types=None`, 1h 窗口),只换加权 → 新信息有限。
2. **真正的天花板是融合策略,不是矩阵**:硬分层"自身臂永远压 co-vis",而 carts/orders 的答案常是**用户没碰过的新品**,只能从 co-vis 臂来,却被挤到 20 名开外 → 换哪张 co-vis 矩阵都差不多。

> 结论:Step 2 重构正确、买矩阵作为 Phase 3 的 GBDT 特征必须留;但 **carts/orders 的大头收益在 Step 3**(松分层),不在加矩阵。原定"Step 2 → 0.52+"目标过于乐观,已修正。

**补充(1/6采样,同口径对照)**:Step 1(全 click)0.3624 vs Step 2(分目标)0.3623 — 采样下 buy2buy 太稀(源商品仅 13 万),看不出差异;全量下才显(41 万)。

---

## 下一步 · Step 3(真正抬 carts/orders)

- **松分层**:carts/orders 改 score-blend(self + α·covis 一起排),放强 co-vis 新品盖过陈旧自身候选。← 首个要试的杠杆
- 调 `TYPE_W` / `top_n` / `session_cap` / 种子数 / buy2buy 权重。
- 工程:候选生成分块,解决全量 val 的 28 分钟 join。
- 目标:weighted → ~0.55。
