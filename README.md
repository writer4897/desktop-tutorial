# MathorCup 2026 D题：多场景、多目标货物运输装箱优化

数学建模竞赛团队项目的代码与计算结果展示。针对 **5 类、300 件货物与 2 种车型**，研究三维装箱、车辆组合和成本优化，并开展参数敏感性分析。

**竞赛成果：**第十六届 MathorCup 数学应用挑战赛本科生组区域赛二等奖。

**王子易的贡献：**数学建模、代码实现，以及后续方法与模型调整。本仓库展示团队成果，不表示全部工作由个人独立完成。

## 方法与结果

- 将车厢边界、顶部安全间隙、载重、货物朝向、支撑关系和易碎件要求转化为程序约束。
- 使用约束感知极点法与多种排序策略构造装箱方案，探索混合遗传算法与多车型组合。
- 针对易碎件支撑槽位不足，采用标准件平台构造及局部拆除、重排和回填，完善最终方案。

| 最终归档方案 | 结果 |
| --- | --- |
| 完成装载 | 300 / 300 件 |
| 车辆组合 | 2 辆 V1 |
| 题设运输费用 | 900 元 |
| 总体有效容积利用率 | 75.07% |
| 原项目约束校验器 | 两车均无报错 |

数据来自 `results/experiments/local_g3_repair/`，按题设有效车厢容积计算。这是竞赛模型下的可行方案，不代表实际运输业务收益，也不据此宣称全局最优。

**版本区别：**`results/problem2/` 保留早期组合搜索结果（费用 1150 元）；最终 900 元方案位于 `results/experiments/local_g3_repair/`。其他实验目录包含中间方案，可能仍有未装入货物或约束问题。

## 快速复验

使用 Python 3.10 或更高版本，在仓库根目录运行：

```bash
python scripts/verify_final.py
```

无需第三方库。该入口检查货物标识、数量、尺寸和重量，并调用原项目校验器逐车检查最终坐标。预期输出：300 件、2 辆车、费用 900 元、利用率约 0.750677，两个 `validation_errors` 列表均为空。

校验沿用原模型的支撑和直接承重规则，并不等同于工程安全认证或更全面的力学验证。

## 重运行与可视化

```bash
# 从归档的 298 件候选方案重新运行最终局部修复
python scripts/experiment_local_g3_repair.py
python scripts/verify_final.py

# 可选：绘制最终方案
python -m pip install -r requirements.txt
python scripts/visualize_packing.py --csv results/experiments/local_g3_repair/best_local_g3_repair.csv --prefix final
```

优化实验可能耗时并覆盖对应结果，请在副本或独立分支上重运行。`pack_problem2.py` 是早期组合搜索流程，不会自动生成最终局部修复方案。

## 目录

| 路径 | 用途 |
| --- | --- |
| `scripts/pack_problem1.py` | 题设参数、极点装箱、约束校验 |
| `scripts/pack_problem2.py` | 多车型组合搜索 |
| `scripts/hybrid_ga_problem2.py` | 混合遗传算法实验 |
| `scripts/experiment_*.py` | 平台构造与局部修复等迭代实验 |
| `scripts/problem3_analysis.py` | 基于归档方案的参数敏感性分析 |
| `scripts/verify_final.py` | 为展示版本补充的最终结果复验入口 |
| `results/` | 团队项目归档的 CSV 坐标与 JSON 结果 |

公开版本选取代码与结构化计算结果，不包含承诺书、签名、个人证件、历史优秀论文或赛事原始文档。`results/attachment2/` 是既有计算输出，原始附件数据及其导入脚本未纳入此版本。竞赛相关材料的权利归各自权利人所有，本仓库未另行授予开源许可证。
