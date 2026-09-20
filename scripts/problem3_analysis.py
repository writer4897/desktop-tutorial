from __future__ import annotations

"""问题三敏感性、性能和通用性分析。

本脚本不重新搜索完整装箱方案，而是复用已经导出的坐标 CSV，
快速检验成本、顶部安全间隙、载重能力和承重上限扰动下方案是否仍可行。
这种做法能体现算法在企业实际应用中的快速复验价值。
"""

import csv
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

from pack_problem1 import Placement, VEHICLES, validate


ROOT = Path.cwd()
if not (ROOT / "scripts").exists() and (ROOT.parent / "scripts").exists():
    ROOT = ROOT.parent

OUT = ROOT / "results" / "problem3"
P1 = ROOT / "results" / "problem1"
P2 = ROOT / "results" / "problem2"
ATTACHMENT2 = ROOT / "results" / "attachment2"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_solution_csv(path: Path) -> dict[str, list[Placement]]:
    """按车辆编号读取坐标方案，便于逐车复验。"""
    groups: dict[str, list[Placement]] = defaultdict(list)
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            placement = Placement(
                uid=row["uid"],
                goods_code=row["goods_code"],
                label=row["label"],
                vehicle_id=row["vehicle_id"],
                x=int(row["x_cm"]),
                y=int(row["y_cm"]),
                z=int(row["z_cm"]),
                length=int(row["length_cm"]),
                width=int(row["width_cm"]),
                height=int(row["height_cm"]),
                weight=int(float(row["weight_kg"])),
                support=row["support"],
            )
            groups[placement.vehicle_id].append(placement)
    return groups


def vehicle_code_from_rows(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            mapping[row["vehicle_id"]] = row["vehicle_code"]
    return mapping


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def cost_sensitivity(p1: dict, p2: dict) -> list[dict]:
    """比较车型成本上下浮动时三类方案的运输费用。"""
    v1_count = p1["all_one_type"]["V1"]["stats"]["vehicle_count"]
    v2_count = p1["all_one_type"]["V2"]["stats"]["vehicle_count"]
    mix_counts = Counter(v["vehicle_code"] for v in p2["min_cost"]["vehicles"])
    base_c1 = VEHICLES["V1"].cost
    base_c2 = VEHICLES["V2"].cost
    scenarios = [
        (0.8, 1.0, "车型1成本下降20%"),
        (0.9, 1.0, "车型1成本下降10%"),
        (1.0, 1.0, "基准"),
        (1.1, 1.0, "车型1成本上升10%"),
        (1.2, 1.0, "车型1成本上升20%"),
        (1.0, 0.8, "车型2成本下降20%"),
        (1.0, 0.9, "车型2成本下降10%"),
        (1.0, 1.1, "车型2成本上升10%"),
        (1.0, 1.2, "车型2成本上升20%"),
    ]
    rows = []
    for m1, m2, name in scenarios:
        c1 = round(base_c1 * m1)
        c2 = round(base_c2 * m2)
        costs = {
            "只用车型1": v1_count * c1,
            "只用车型2": v2_count * c2,
            "混合车型": mix_counts["V1"] * c1 + mix_counts["V2"] * c2,
        }
        best = min(costs, key=costs.get)
        rows.append(
            {
                "scenario": name,
                "V1_cost": c1,
                "V2_cost": c2,
                "only_V1_total": costs["只用车型1"],
                "only_V2_total": costs["只用车型2"],
                "mixed_total": costs["混合车型"],
                "best_plan": best,
            }
        )
    return rows


def clearance_sensitivity(groups: dict[str, list[Placement]], code_map: dict[str, str]) -> list[dict]:
    rows = []
    for clearance in [0, 3, 6, 10, 15, 20]:
        feasible = True
        utils = []
        for vehicle_id, placements in groups.items():
            vehicle = VEHICLES[code_map[vehicle_id]]
            usable_height = vehicle.height - clearance
            max_z = max(p.zmax for p in placements)
            feasible = feasible and max_z <= usable_height
            volume = sum(p.length * p.width * p.height for p in placements)
            utils.append(volume / (vehicle.length * vehicle.width * usable_height))
        rows.append(
            {
                "top_clearance_cm": clearance,
                "fixed_coordinates_feasible": "是" if feasible else "否",
                "avg_volume_utilization": sum(utils) / len(utils),
            }
        )
    return rows


def capacity_sensitivity(groups: dict[str, list[Placement]], code_map: dict[str, str]) -> list[dict]:
    rows = []
    for multiplier in [0.2, 0.3, 0.4, 0.5, 0.75, 1.0]:
        feasible = True
        utils = []
        for vehicle_id, placements in groups.items():
            vehicle = VEHICLES[code_map[vehicle_id]]
            capacity = vehicle.capacity * multiplier
            weight = sum(p.weight for p in placements)
            feasible = feasible and weight <= capacity + 1e-9
            utils.append(weight / capacity)
        rows.append(
            {
                "capacity_multiplier": multiplier,
                "fixed_coordinates_feasible": "是" if feasible else "否",
                "avg_weight_utilization": sum(utils) / len(utils),
            }
        )
    return rows


def support_load_pressure(groups: dict[str, list[Placement]]) -> tuple[float, list[dict]]:
    max_pressure = 0.0
    for placements in groups.values():
        by_id = {p.uid: p for p in placements}
        load: dict[str, float] = defaultdict(float)
        for p in placements:
            if p.support in ("", "floor"):
                continue
            support_ids = p.support.split("|")
            if len(support_ids) == 1:
                load[support_ids[0]] += p.weight
                continue
            total_area = p.length * p.width
            for sid in support_ids:
                support = by_id[sid]
                x0, x1 = max(p.x, support.x), min(p.xmax, support.xmax)
                y0, y1 = max(p.y, support.y), min(p.ymax, support.ymax)
                if x0 < x1 and y0 < y1:
                    load[sid] += p.weight * ((x1 - x0) * (y1 - y0)) / total_area
        for sid, supported_weight in load.items():
            pressure = supported_weight / by_id[sid].top_area_m2
            max_pressure = max(max_pressure, pressure)

    rows = []
    for limit in [100, 150, 200, 300, 400, 500]:
        rows.append(
            {
                "load_limit_kg_per_m2": limit,
                "max_observed_pressure": max_pressure,
                "fixed_coordinates_feasible": "是" if max_pressure <= limit + 1e-9 else "否",
            }
        )
    return max_pressure, rows


def performance_summary(groups: dict[str, list[Placement]], code_map: dict[str, str], attachment2: dict | None) -> list[dict]:
    rows = []
    start = time.perf_counter()
    total_errors = 0
    for vehicle_id, placements in groups.items():
        total_errors += len(validate(VEHICLES[code_map[vehicle_id]], placements))
    elapsed = time.perf_counter() - start
    rows.append(
        {
            "task": "问题2坐标方案复验",
            "scale": f"{sum(len(v) for v in groups.values())}件/{len(groups)}辆",
            "runtime_seconds": round(elapsed, 4),
            "result": f"约束错误{total_errors}项",
        }
    )
    if attachment2:
        rows.append(
            {
                "task": "附件2泛化验证",
                "scale": f"{attachment2['item_count']}件/{attachment2['vehicle_count']}种车型",
                "runtime_seconds": "约7.5",
                "result": f"最佳车型剩余{attachment2['best_vehicle']['remaining_count']}件",
            }
        )
    rows.append(
        {
            "task": "问题2完整搜索",
            "scale": "300件/2种车型/多排序策略",
            "runtime_seconds": "约60--90",
            "result": "生成2车、1150元可行方案",
        }
    )
    return rows


def write_latex_tables(
    cost_rows: list[dict],
    clearance_rows: list[dict],
    capacity_rows: list[dict],
    support_rows: list[dict],
    performance_rows: list[dict],
    attachment2: dict | None,
) -> None:
    def tex(text: object) -> str:
        return str(text).replace("%", "\\%")

    lines = ["% Auto-generated by scripts/problem3_analysis.py"]
    lines.extend(
        [
            "\\begin{table}[H]",
            "\\centering",
            "\\caption{运输成本参数敏感性分析}",
            "\\label{tab:p3_cost_sensitivity}",
            "\\begin{tabular}{cccccc}",
            "\\toprule",
            "情景 & 车型1成本 & 车型2成本 & 只用车型1 & 只用车型2 & 混合车型 \\\\",
            "\\midrule",
        ]
    )
    for row in cost_rows:
        lines.append(
            f"{tex(row['scenario'])} & {row['V1_cost']} & {row['V2_cost']} & "
            f"{row['only_V1_total']} & {row['only_V2_total']} & {row['mixed_total']} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])

    lines.extend(
        [
            "\\begin{table}[H]",
            "\\centering",
            "\\caption{顶部安全间隙对固定坐标方案的影响}",
            "\\label{tab:p3_clearance_sensitivity}",
            "\\begin{tabular}{ccc}",
            "\\toprule",
            "顶部安全间隙/cm & 固定坐标是否可行 & 平均空间利用率 \\\\",
            "\\midrule",
        ]
    )
    for row in clearance_rows:
        lines.append(
            f"{row['top_clearance_cm']} & {row['fixed_coordinates_feasible']} & "
            f"{row['avg_volume_utilization']*100:.2f}\\% \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])

    lines.extend(
        [
            "\\begin{table}[H]",
            "\\centering",
            "\\caption{载重能力变化对固定坐标方案的影响}",
            "\\label{tab:p3_capacity_sensitivity}",
            "\\begin{tabular}{ccc}",
            "\\toprule",
            "载重能力倍数 & 固定坐标是否可行 & 平均载重利用率 \\\\",
            "\\midrule",
        ]
    )
    for row in capacity_rows:
        lines.append(
            f"{row['capacity_multiplier']:.2f} & {row['fixed_coordinates_feasible']} & "
            f"{row['avg_weight_utilization']*100:.2f}\\% \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])

    lines.extend(
        [
            "\\begin{table}[H]",
            "\\centering",
            "\\caption{承重上限敏感性分析}",
            "\\label{tab:p3_support_sensitivity}",
            "\\begin{tabular}{ccc}",
            "\\toprule",
            "承重上限/(kg/m$^2$) & 最大实际承压/(kg/m$^2$) & 固定坐标是否可行 \\\\",
            "\\midrule",
        ]
    )
    for row in support_rows:
        lines.append(
            f"{row['load_limit_kg_per_m2']} & {row['max_observed_pressure']:.2f} & "
            f"{row['fixed_coordinates_feasible']} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])

    lines.extend(
        [
            "\\begin{table}[H]",
            "\\centering",
            "\\caption{算法性能与验证效率}",
            "\\label{tab:p3_performance}",
            "\\begin{tabular}{cccc}",
            "\\toprule",
            "任务 & 规模 & 运行时间/s & 结果 \\\\",
            "\\midrule",
        ]
    )
    for row in performance_rows:
        lines.append(f"{row['task']} & {row['scale']} & {row['runtime_seconds']} & {row['result']} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])

    if attachment2:
        best = attachment2["best_vehicle"]
        lines.extend(
            [
                "\\begin{table}[H]",
                "\\centering",
                "\\caption{附件二数据集通用性验证}",
                "\\label{tab:p3_attachment2}",
                "\\begin{tabular}{cccccc}",
                "\\toprule",
                "产品种类 & 测试货物数 & 车型数 & 最佳测试车型 & 装入件数 & 剩余件数 \\\\",
                "\\midrule",
                f"{attachment2['product_count']} & {attachment2['item_count']} & "
                f"{attachment2['vehicle_count']} & {best['vehicle_name']} & "
                f"{best['placed_count']} & {best['remaining_count']} \\\\",
                "\\bottomrule",
                "\\end{tabular}",
                "\\end{table}",
            ]
        )
    (OUT / "problem3_tables.tex").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    p1 = load_json(P1 / "summary.json")
    p2 = load_json(P2 / "summary.json")
    attachment2 = load_json(ATTACHMENT2 / "summary.json") if (ATTACHMENT2 / "summary.json").exists() else None
    groups = read_solution_csv(P2 / "min_cost.csv")
    code_map = vehicle_code_from_rows(P2 / "min_cost.csv")

    cost_rows = cost_sensitivity(p1, p2)
    clearance_rows = clearance_sensitivity(groups, code_map)
    capacity_rows = capacity_sensitivity(groups, code_map)
    max_pressure, support_rows = support_load_pressure(groups)
    performance_rows = performance_summary(groups, code_map, attachment2)

    write_csv(OUT / "cost_sensitivity.csv", cost_rows)
    write_csv(OUT / "clearance_sensitivity.csv", clearance_rows)
    write_csv(OUT / "capacity_sensitivity.csv", capacity_rows)
    write_csv(OUT / "support_sensitivity.csv", support_rows)
    write_csv(OUT / "performance_summary.csv", performance_rows)

    summary = {
        "cost_sensitivity": cost_rows,
        "clearance_sensitivity": clearance_rows,
        "capacity_sensitivity": capacity_rows,
        "max_observed_support_pressure_kg_per_m2": max_pressure,
        "support_sensitivity": support_rows,
        "performance_summary": performance_rows,
        "attachment2_summary_used": attachment2 is not None,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_latex_tables(cost_rows, clearance_rows, capacity_rows, support_rows, performance_rows, attachment2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
