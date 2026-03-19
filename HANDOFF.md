# HANDOFF.md — 项目交接文档

**项目名**: Tail Nodes Benefit Most from Graph Self-Supervision in Link Prediction
**目标会议**: KDD / WWW (ACM sigconf 格式)
**工作目录**: `/home/soso/experiments/cold_start_lp/`
**环境**: `conda activate research` (Python 3.10 + PyTorch 2.5.1+cu121 + PyG 2.7.0 + OGB)
**GPU**: 1× NVIDIA RTX A6000 (49GB)
**日期**: 2026-03-15 开始, 2026-03-19 至此

---

## 一、研究方向与核心发现

### 研究问题
图自监督学习 (SSL/CL) 在链路预测 (LP) 中被广泛使用, 但现有工作只报告整体指标 (overall AUC), 忽略了改进在图结构中的分布。我们提出 **度分层评估 (degree-stratified evaluation)**, 发现 SSL 的增益 **不均匀地集中在低度 (tail) 节点** 上。

### 四大核心发现

1. **SSL增益与度单调递减**: 低度边 (deg 2-4) 获得高达 +13.6% AUC, 高度边 (deg 20-50) 仅 +3.3%。跨 6 个数据集、3 个 backbone (GCN/SAGE/GAT)、3 种指标 (AUC/AP/Hits@20) 一致。

2. **增益分解**: 增益可分解为 (a) 数据增强正则化 和 (b) 对比学习目标。小/稀疏图上对比目标贡献 +2-3%; 大/稠密图上增强本身已足够。

3. **选择性CL无效 (负面结果)**: DegreeGatedCL ≈ GlobalCL ≈ RandomCL。正则化效果是全局的，不需要针对性设计。

4. **非SSL方法增益较小**: Reweight、FocalLoss 增益仅为 SSL 的 1/3 到 1/5。

### 重要转折
- 最初的假设是 "针对低度节点的选择性CL (ColdCL) 会优于全局CL" → **被实验否定**
- 随后尝试 "uncertainty-guided CL" → **也被否定** (比 GlobalCL 更差)
- 最终 **转向分析论文**: 核心贡献从 "新方法" 变为 "实证发现 + 机制分析"

---

## 二、完整的流水线历史

### Stage 1: Idea Discovery (已完成)
- 文献调研: 20+ 篇论文, 4 个子方向
- GPT-5.4 生成 10 个想法 → 过滤到 6 个 → pilot 3 个
- Pilot 结果: Cold-Only CL (POSITIVE), Calibrated LP (POSITIVE), Perturbation Law (MIXED)
- 用户选择: 合并方案 (ColdCL + 校准)

### Stage 2: 实验 v1 (已完成)
- 300 次实验: 5 数据集 × 2 编码器 × 6 方法 × 5 种子
- ColdCL 在 7/10 数据集-编码器组合中最优
- 文件: `run_full.py` → `results/full/results.json`

### Stage 3: GPT-5.4 审稿循环 (3 轮, 已完成)
- **Round 1 (4/10)**: "cold-start" 表述被拒; 建议改为 "tail-node LP"
- **Round 2 (4/10)**: UncertaintyCL 没有超过 GlobalCL; 转向分析论文
- **Round 3 (6.5/10)**: 机制分析强化; 增益与度的单调关系被确认
- 文件: `run_v2.py` → `results/full/v2_results.json`

### Stage 4: 机制分析 (已完成)
- 细粒度度分层增益 (6 个 bin)
- AugOnly vs CL 分解
- 嵌入稳定性测量 (结果: 大图上 CL **增加**方差, 否定了简单的稳定性假说)
- 文件: `run_mechanism.py` → `results/full/mechanism_analysis.json`

### Stage 5: OGB + 稀疏性实验 (已完成)
- ogbl-collab (235K 节点): AugOnly (.848) > CL (.833) > Vanilla (.799)
- 控制稀疏性: 边保留 100%/75%/50%/25%, CL-AugOnly gap 在 Cora 上 +1-3%
- 文件: `run_ogb.py`, `run_sparsity.py` → `results/full/ogb_results.json`, `results/full/sparsity_results.json`

### Stage 6: 最终实验 — 480 次运行 (已完成)
- 4 数据集 × 3 backbone (GCN/SAGE/GAT) × 8 方法 × 5 种子
- 指标: AUC, AP, Hits@20, MRR, ECE, Brier
- **GAT 结果确认**: 注意力机制下同样存在度依赖增益模式
- 文件: `run_final.py` → `results/full/final_results.json`

### Stage 7: 论文撰写 (已完成)
- GPT-5.4 规划大纲 → 5 个自动生成图表 → 7 个 LaTeX 章节
- 2 轮改进: 5/10 → 6/10
- 文件: `paper/main.tex` + `paper/sections/*.tex`

### Stage 8: 审稿回复实验 (已完成)
审稿意见: **Accept with minor revisions**, 需要修改 4 点:
- **R1 偏相关分析**: 度独立预测SSL增益 (CS: partial ρ=-0.070, p<10⁻¹⁸), 聚类系数不独立预测
- **R2 排名指标**: Hits@20 和 AP 已计算, 模式一致
- **R3 归纳冷启动**: 20%节点完全holdout, CL 在 CiteSeer 上 +11.3% AUC
- **R4 保度稀疏化**: degree-preserving vs uniform, 结果差异 <1%
- 文件: `run_reviewer.py` → `results/full/reviewer_response.json`
- 修订版: `paper/main_revised.pdf`

---

## 三、关键实验结果速查

### 主表 (GCN backbone, 5 seeds)
| 数据集 | 方法 | Overall AUC | Cold(2-5) AUC | Cold AP |
|--------|------|-----------|-------------|---------|
| Cora | Vanilla | .900±.012 | .843±.030 | .708 |
| Cora | GlobalCL | .922±.002 | .884±.019 | .791 |
| CiteSeer | Vanilla | .864±.007 | .782±.012 | .672 |
| CiteSeer | GlobalCL | .926±.008 | .883±.010 | .831 |
| PubMed | Vanilla | .926±.002 | .898±.005 | .824 |
| PubMed | GlobalCL | .949±.012 | .945±.005 | .890 |
| CS | Vanilla | .895±.023 | .749±.047 | .328 |
| CS | GlobalCL | .954±.001 | .911±.005 | .672 |

### 度分层增益 (CS, seed 0)
| 度范围 | Vanilla AUC | CL AUC | 增益 |
|--------|-----------|--------|------|
| 2-4 | .762 | .897 | **+13.6%** |
| 4-6 | .825 | .924 | +9.9% |
| 6-10 | .865 | .938 | +7.3% |
| 10-20 | .901 | .952 | +5.1% |
| 20-50 | .914 | .948 | +3.3% |

### 归纳冷启动 (20% holdout)
| 数据集 | Vanilla | CL | Δ AUC |
|--------|---------|-----|-------|
| CiteSeer | .721 | .834 | +11.3% |
| CS | .845 | .910 | +6.5% |

---

## 四、代码结构

```
cold_start_lp/
├── src/                        # 核心库
│   ├── data.py                 # 数据集加载 (Cora/CiteSeer/PubMed/Photo/CS + OGB)
│   ├── models.py               # GCNEncoder, SAGEEncoder, GATEncoder, LinkPredictor,
│   │                           # GlobalCLModel, ColdCLModel, NodeDupPredictor, MCDropoutPredictor
│   ├── metrics.py              # AUC, AP, Hits@K, MRR, ECE, Brier, 度分层评估
│   └── train.py                # train_standard, train_with_cl, train_nodedup, train_reweight
│
├── run_full.py                 # v1 实验 (5 数据集 × 2 编码器 × 6 方法)
├── run_v2.py                   # v2 + UncertaintyCL + FocalLoss + ablations
├── run_final.py                # 最终 480 次实验 (4 数据集 × 3 编码器 × 8 方法)
├── run_mechanism.py            # 机制分析 (稳定性, 分解, 细粒度度分层)
├── run_ogb.py                  # OGB benchmarks (collab, citation2, ddi)
├── run_sparsity.py             # 控制稀疏性实验
├── run_reviewer.py             # 审稿回复 (R1-R4)
├── generate_figures.py         # 5 个论文图表
│
├── results/full/               # 所有结果 JSON
│   ├── final_results.json      # ← 主要结果 (480 runs)
│   ├── mechanism_analysis.json
│   ├── sparsity_results.json
│   ├── ogb_results.json
│   ├── reviewer_response.json  # ← R1-R4 结果
│   ├── v2_results.json
│   └── results.json            # v1 结果
│
├── paper/                      # LaTeX 论文
│   ├── main.tex                # 主文件 (ACM sigconf)
│   ├── main_revised.pdf        # ← 最新修订版 (6 页)
│   ├── sections/               # 7 个章节文件
│   ├── figures/                # 5 个 PDF 图表
│   └── references.bib          # 12 条参考文献
│
├── IDEA_REPORT.md              # 想法排名报告
├── NARRATIVE_REPORT.md         # 叙事报告 (论文输入)
└── PIPELINE_REPORT.md          # 流水线报告
```

---

## 五、审稿状态与待办事项

### 当前审稿评分
| 阶段 | 评分 |
|------|------|
| GPT-5.4 Round 1 (方法论文) | 4/10 |
| GPT-5.4 Round 2 (加强baseline后) | 4/10 |
| GPT-5.4 Round 3 (分析论文转向) | 6.5/10 |
| 论文初稿 | 5/10 |
| 论文修改后 | 6/10 |
| 外部审稿 (模拟 KDD reviewer) | **Accept with minor revisions** |

### 已完成的修改 (审稿回复)
- [x] R1: 偏相关分析 → Table 5
- [x] R2: Hits@20 + AP 指标 → 全表更新
- [x] R3: 归纳冷启动评估 → Table 6
- [x] R4: 保度稀疏化 → 机制章节讨论

### 仍可改进的方向 (按优先级)
1. **扩展到 8 页**: 当前 6 页, 可加入更多 ablation 表 (threshold τ 敏感性, λ 敏感性)
2. **更多非引用图**: ogbl-ppa (蛋白质网络), ogbl-ddi (药物) — ddi 太稠密不适合, ppa 可尝试
3. **Graph Transformer backbone**: 验证注意力+位置编码架构下的度依赖性
4. **MRR 指标**: 已在 metrics.py 实现但未跑入 final results
5. **图表美化**: 当前 seaborn 生成, 可改为 pgfplots 或更精细的 matplotlib
6. **相关工作扩充**: 加入 subgroup fairness evaluation 相关文献 (审稿建议)

---

## 六、关键配置与复现

### 环境设置
```bash
conda activate research
# Python 3.10, PyTorch 2.5.1+cu121, torch-geometric 2.7.0, ogb 1.3.6
```

### 快速复现主要结果
```bash
cd /home/soso/experiments/cold_start_lp

# 运行最终 480 次实验 (~30 min on A6000)
CUDA_VISIBLE_DEVICES=0 python run_final.py \
  --datasets Cora CiteSeer PubMed CS \
  --encoders GCN SAGE GAT \
  --seeds 0 1 2 3 4

# 运行机制分析
python run_mechanism.py

# 运行审稿回复实验
python run_reviewer.py

# 生成论文图表
python generate_figures.py

# 编译论文
cd paper && latexmk -pdf -interaction=nonstopmode -f main.tex
```

### GPT-5.4 审稿 (via Codex MCP)
```
mcp__codex__codex:
  model: gpt-5.4
  config: {"model_reasoning_effort": "xhigh"}
  prompt: [审稿 prompt]
```
历史 thread IDs:
- 想法生成: `019cef4c-ea5c-7df2-a5b1-cde243dede37`
- 论文审稿 Round 1: `019cefd6-8f73-7c53-82aa-95605c12de4a`
- 论文规划: `019cefcd-be83-71f0-9c68-da92ccf39431`

---

## 七、论文核心论点速记

**标题**: Tail Nodes Benefit Most from Graph Self-Supervision in Link Prediction

**一句话摘要**: 图自监督学习的链路预测增益与节点度单调递减——低度边获益最多，且这种效果来自全局表示正则化而非节点级不变性。

**投稿策略**: KDD/WWW 分析论文 (empirical study)，不要声称新方法，而是声称新发现。

**最强论据**:
1. 度-增益单调曲线 (Fig 1, Fig 2)
2. AugOnly vs CL 分解 (Fig 3)
3. 归纳冷启动也有效 (Table 6)
4. 偏相关排除混淆 (Table 5)
5. 选择性CL无效的负面结果 (Fig 4)

**最弱环节** (审稿可能攻击):
1. 仍然只有 message-passing GNN, 没有 Graph Transformer
2. 度与其他拓扑特征的因果关系未完全解开
3. 每边效应量小 (ρ ≈ -0.07), 虽然统计显著
