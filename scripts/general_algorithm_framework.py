from __future__ import annotations

"""通用装箱算法框架说明性总入口。

本脚本用于把分散在各问题中的程序模块组织成统一流程，便于复核者理解
本文算法体系。它不是替代所有实验脚本的单一求解器，而是给出一套可迁移
到新车型、新货物数据的通用流程：

1. 数据标准化：车辆、货物类型、单件货物、姿态集合；
2. 单车候选生成：约束感知极点法 + 多排序策略；
3. 特殊规则修复：易碎件支撑平台逆推、槽位扫描、局部大邻域重排；
4. 多车组合决策：车辆数优先或成本优先；
5. 统一校验导出：边界、不重叠、姿态、载重、支撑、承重逐项复验。

实际可运行求解分别见 pack_problem1.py、pack_problem2.py、
experiment_local_g3_repair.py、experiment_v2_slot_insert.py 和
problem3_analysis.py。本文件作为附件中的综合算法框架，帮助说明这些脚本
之间的关系。
"""

from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class PipelineStep:
    """算法流程中的一个阶段。"""

    name: str
    purpose: str
    main_scripts: tuple[str, ...]
    outputs: tuple[str, ...]


FRAMEWORK_STEPS: tuple[PipelineStep, ...] = (
    PipelineStep(
        name="1. 数据标准化",
        purpose="将题目车型和货物类型展开为统一对象，并生成每类货物允许姿态。",
        main_scripts=("pack_problem1.py", "validate_attachment2.py"),
        outputs=("products_normalized.csv", "vehicles_normalized.csv"),
    ),
    PipelineStep(
        name="2. 单车候选生成",
        purpose="在不同排序策略下调用约束感知极点法，生成单车坐标候选方案。",
        main_scripts=("pack_problem1.py",),
        outputs=("single_V1.csv", "single_V2_optimized.csv"),
    ),
    PipelineStep(
        name="3. 易碎件平台修复",
        purpose="针对 G3 易碎件完整支撑要求，构造标准件平台并执行局部大邻域重排。",
        main_scripts=("experiment_inverse_fragile_platform.py", "experiment_local_g3_repair.py"),
        outputs=("best_local_g3_repair.csv", "g3_support_audit.json"),
    ),
    PipelineStep(
        name="4. 车型组合决策",
        purpose="在候选单车方案基础上比较车辆数最少和成本最低目标。",
        main_scripts=("pack_problem2.py", "hybrid_ga_problem2.py"),
        outputs=("min_vehicle.csv", "min_cost.csv", "summary.json"),
    ),
    PipelineStep(
        name="5. 敏感性与通用性验证",
        purpose="复用统一校验器，考察成本、载重、安全间隙、承重上限和附件 2 数据。",
        main_scripts=("problem3_analysis.py", "validate_attachment2.py"),
        outputs=("cost_sensitivity.csv", "performance_summary.csv", "best_attachment2_vehicle.csv"),
    ),
    PipelineStep(
        name="6. 结果可视化与归档",
        purpose="根据坐标 CSV 绘制装载图，并将核心 CSV、JSON、图片归档到计算结果文件夹。",
        main_scripts=("render_paper_figures.py",),
        outputs=("p1_single_3d_overview.png", "p2_final_3d_overview.png", "p2_fragile_support.png"),
    ),
)


def print_framework() -> None:
    """在终端打印通用算法框架，便于快速复核。"""

    print("MathorCup D 题通用装箱算法框架")
    print("=" * 36)
    for step in FRAMEWORK_STEPS:
        print(f"\n{step.name}")
        print(f"  目标：{step.purpose}")
        print("  主要脚本：" + ", ".join(step.main_scripts))
        print("  主要输出：" + ", ".join(step.outputs))

    print("\n推荐复现顺序")
    print("  1) py -3 scripts/pack_problem1.py")
    print("  2) py -3 scripts/experiment_local_g3_repair.py")
    print("  3) py -3 scripts/experiment_v2_slot_insert.py")
    print("  4) py -3 scripts/problem3_analysis.py")
    print("  5) py -3 scripts/render_paper_figures.py")


def main() -> None:
    print_framework()


if __name__ == "__main__":
    main()
