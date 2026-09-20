from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from pack_problem1 import (
    Item,
    Placement,
    VEHICLES,
    add_support_load,
    expand_items,
    goods_counts,
    place_one_item,
    sort_items,
    utilization,
    validate,
)


ROOT = Path.cwd()
if not (ROOT / "scripts").exists() and (ROOT.parent / "scripts").exists():
    ROOT = ROOT.parent
OUT = ROOT / "results" / "problem2"


STRATEGIES = [
    "constraint_volume",
    "volume",
    "base",
    "dir_std_frag",
]


@dataclass
class TruckPlan:
    vehicle_code: str
    vehicle_name: str
    strategy: str
    vehicle_id: str
    placements: list[Placement]
    stats: dict[str, float]

    @property
    def cost(self) -> int:
        return VEHICLES[self.vehicle_code].cost


@dataclass
class SearchState:
    remaining: list[Item]
    plans: list[TruckPlan]
    cost: int

    @property
    def used_count(self) -> int:
        return len(self.plans)

    @property
    def remaining_volume(self) -> int:
        return sum(item.volume for item in self.remaining)

    @property
    def remaining_weight(self) -> int:
        return sum(item.weight for item in self.remaining)


def order_items(items: list[Item], strategy: str) -> list[Item]:
    if strategy in {"constraint_volume", "volume", "base"}:
        return sort_items(items, strategy)

    label_orders = {
        "std_frag_dir": {"standard": 0, "fragile": 1, "directed": 2},
        "std_dir_frag": {"standard": 0, "directed": 1, "fragile": 2},
        "frag_std_dir": {"fragile": 0, "standard": 1, "directed": 2},
        "dir_std_frag": {"directed": 0, "standard": 1, "fragile": 2},
    }
    if strategy not in label_orders:
        raise ValueError(strategy)
    order = label_orders[strategy]
    return sorted(items, key=lambda item: (order[item.label], -item.volume, -item.base_area0, item.uid))


def pack_vehicle(vehicle_code: str, items: list[Item], strategy: str, vehicle_id: str) -> tuple[list[Placement], list[Item]]:
    vehicle = VEHICLES[vehicle_code]
    placements: list[Placement] = []
    current_weight = 0
    load_on_support: dict[str, float] = {}

    for item in order_items(items, strategy):
        placed = place_one_item(item, vehicle, vehicle_id, placements, current_weight, load_on_support)
        if placed is None:
            continue
        placements.append(placed)
        current_weight += placed.weight
        add_support_load(placed, placements, load_on_support)

    placed_ids = {placement.uid for placement in placements}
    remaining = [item for item in items if item.uid not in placed_ids]
    return placements, remaining


PACK_CACHE: dict[tuple[tuple[str, ...], str, str], tuple[TruckPlan, list[Item]] | None] = {}


def clone_plan(plan: TruckPlan, vehicle_id: str) -> TruckPlan:
    cloned_placements = [
        Placement(
            uid=p.uid,
            goods_code=p.goods_code,
            label=p.label,
            vehicle_id=vehicle_id,
            x=p.x,
            y=p.y,
            z=p.z,
            length=p.length,
            width=p.width,
            height=p.height,
            weight=p.weight,
            support=p.support,
        )
        for p in plan.placements
    ]
    return TruckPlan(
        vehicle_code=plan.vehicle_code,
        vehicle_name=plan.vehicle_name,
        strategy=plan.strategy,
        vehicle_id=vehicle_id,
        placements=cloned_placements,
        stats=plan.stats,
    )


def make_candidate(items: list[Item], vehicle_code: str, strategy: str, vehicle_id: str) -> tuple[TruckPlan, list[Item]] | None:
    signature = tuple(sorted(item.uid for item in items))
    cache_key = (signature, vehicle_code, strategy)
    if cache_key in PACK_CACHE:
        cached = PACK_CACHE[cache_key]
        if cached is None:
            return None
        plan, remaining = cached
        return clone_plan(plan, vehicle_id), remaining
    vehicle = VEHICLES[vehicle_code]
    placements, remaining = pack_vehicle(vehicle_code, items, strategy, vehicle_id)
    if not placements:
        PACK_CACHE[cache_key] = None
        return None
    errors = validate(vehicle, placements)
    if errors:
        PACK_CACHE[cache_key] = None
        return None
    stats = utilization(vehicle, placements)
    plan = TruckPlan(
        vehicle_code=vehicle_code,
        vehicle_name=vehicle.name,
        strategy=strategy,
        vehicle_id=vehicle_id,
        placements=placements,
        stats=stats,
    )
    PACK_CACHE[cache_key] = (plan, remaining)
    return plan, remaining


def candidate_plans(state: SearchState, objective: str) -> list[tuple[TruckPlan, list[Item]]]:
    candidates: list[tuple[TruckPlan, list[Item]]] = []
    next_no = state.used_count + 1
    for vehicle_code in ["V1", "V2"]:
        for strategy in STRATEGIES:
            vehicle_id = f"{objective}-{next_no}-{vehicle_code}"
            candidate = make_candidate(state.remaining, vehicle_code, strategy, vehicle_id)
            if candidate is not None:
                candidates.append(candidate)
    return candidates


def beam_key(state: SearchState, objective: str) -> tuple:
    avg_fullness = 0.0
    if state.plans:
        avg_fullness = sum(plan.stats["combined_fullness"] for plan in state.plans) / len(state.plans)
    if objective == "min_vehicle":
        return (state.used_count, state.remaining_volume, state.cost, -avg_fullness)
    if objective == "min_cost":
        return (state.cost, state.used_count, state.remaining_volume, -avg_fullness)
    raise ValueError(objective)


def enumerate_depth(objective: str, depth: int) -> list[SearchState]:
    states = [SearchState(remaining=expand_items(), plans=[], cost=0)]
    complete: list[SearchState] = []
    for level in range(depth):
        next_states: list[SearchState] = []
        for state in states:
            for plan, remaining in candidate_plans(state, objective):
                new_state = SearchState(
                    remaining=remaining,
                    plans=state.plans + [plan],
                    cost=state.cost + plan.cost,
                )
                if not remaining:
                    complete.append(new_state)
                elif level < depth - 1:
                    next_states.append(new_state)

        # Keep only representative states to prevent repeated equivalent branches.
        best_by_remaining: dict[tuple[str, ...], SearchState] = {}
        for state in next_states:
            signature = tuple(sorted(item.uid for item in state.remaining))
            old = best_by_remaining.get(signature)
            if old is None or beam_key(state, objective) < beam_key(old, objective):
                best_by_remaining[signature] = state
        states = sorted(best_by_remaining.values(), key=lambda state: beam_key(state, objective))[:40]
    return complete


def solve(objective: str, max_depth: int = 3) -> SearchState:
    complete: list[SearchState] = []
    for depth in range(1, max_depth + 1):
        found = enumerate_depth(objective, depth)
        complete.extend(found)
        if objective == "min_vehicle" and found:
            return sorted(found, key=lambda state: beam_key(state, objective))[0]
        if objective == "min_cost" and found:
            # After depth 2, the cheaper two-truck combinations have already
            # been tested in the same depth.
            best = sorted(complete, key=lambda state: beam_key(state, objective))[0]
            if depth >= 2:
                return best
    if complete:
        return sorted(complete, key=lambda state: beam_key(state, objective))[0]
    raise RuntimeError(f"No complete solution found for {objective} within depth={max_depth}")


def solution_summary(state: SearchState) -> dict:
    total_volume = sum(plan.stats["volume_cm3"] for plan in state.plans)
    total_weight = sum(plan.stats["weight_kg"] for plan in state.plans)
    avg_volume = sum(plan.stats["volume_utilization"] for plan in state.plans) / len(state.plans)
    avg_weight = sum(plan.stats["weight_utilization"] for plan in state.plans) / len(state.plans)
    avg_fullness = sum(plan.stats["combined_fullness"] for plan in state.plans) / len(state.plans)
    return {
        "vehicle_count": state.used_count,
        "total_cost": state.cost,
        "total_volume_cm3": total_volume,
        "total_weight_kg": total_weight,
        "avg_volume_utilization": avg_volume,
        "avg_weight_utilization": avg_weight,
        "avg_combined_fullness": avg_fullness,
        "vehicles": [
            {
                "vehicle_id": plan.vehicle_id,
                "vehicle_code": plan.vehicle_code,
                "vehicle_name": plan.vehicle_name,
                "strategy": plan.strategy,
                "cost": plan.cost,
                "stats": plan.stats,
                "goods_counts": goods_counts(plan.placements),
            }
            for plan in state.plans
        ],
    }


def write_solution_csv(path: Path, state: SearchState) -> None:
    fields = [
        "vehicle_id",
        "vehicle_code",
        "uid",
        "goods_code",
        "label",
        "x_cm",
        "y_cm",
        "z_cm",
        "length_cm",
        "width_cm",
        "height_cm",
        "weight_kg",
        "support",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for plan in state.plans:
            for placement in plan.placements:
                writer.writerow(
                    {
                        "vehicle_id": placement.vehicle_id,
                        "vehicle_code": plan.vehicle_code,
                        "uid": placement.uid,
                        "goods_code": placement.goods_code,
                        "label": placement.label,
                        "x_cm": placement.x,
                        "y_cm": placement.y,
                        "z_cm": placement.z,
                        "length_cm": placement.length,
                        "width_cm": placement.width,
                        "height_cm": placement.height,
                        "weight_kg": placement.weight,
                        "support": placement.support,
                    }
                )


def strategy_label(strategy: str) -> str:
    return {
        "constraint_volume": "约束-体积",
        "volume": "体积优先",
        "base": "底面积优先",
        "std_frag_dir": "标准-易碎-定向",
        "std_dir_frag": "标准-定向-易碎",
        "frag_std_dir": "易碎-标准-定向",
        "dir_std_frag": "定向-标准-易碎",
    }[strategy]


def write_latex_tables(summary: dict) -> None:
    lines = []
    lines.append("% Auto-generated by scripts/pack_problem2.py")
    lines.append("\\begin{table}[H]")
    lines.append("\\centering")
    lines.append("\\caption{问题二多车型组合配送结果对比}")
    lines.append("\\label{tab:p2_summary}")
    lines.append("\\begin{tabular}{cccccc}")
    lines.append("\\toprule")
    lines.append("优化目标 & 使用车辆数 & 车型组合 & 总成本/元 & 平均空间利用率 & 平均载重利用率 \\\\")
    lines.append("\\midrule")
    for key, title in [("min_vehicle", "车辆数最少"), ("min_cost", "成本最低")]:
        sol = summary[key]
        combo = "+".join(vehicle["vehicle_name"] for vehicle in sol["vehicles"])
        lines.append(
            f"{title} & {sol['vehicle_count']} & {combo} & {sol['total_cost']} & "
            f"{sol['avg_volume_utilization']*100:.2f}\\% & "
            f"{sol['avg_weight_utilization']*100:.2f}\\% \\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")
    lines.append("\\begin{table}[H]")
    lines.append("\\centering")
    lines.append("\\caption{问题二成本最低方案逐车装载情况}")
    lines.append("\\label{tab:p2_min_cost_detail}")
    lines.append("\\begin{tabular}{cccccc}")
    lines.append("\\toprule")
    lines.append("车辆 & 车型 & 排序策略 & 装入件数 & 空间利用率 & 载重利用率 \\\\")
    lines.append("\\midrule")
    for idx, vehicle in enumerate(summary["min_cost"]["vehicles"], 1):
        lines.append(
            f"第{idx}辆 & {vehicle['vehicle_name']} & {strategy_label(vehicle['strategy'])} & "
            f"{vehicle['stats']['item_count']} & "
            f"{vehicle['stats']['volume_utilization']*100:.2f}\\% & "
            f"{vehicle['stats']['weight_utilization']*100:.2f}\\% \\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    (OUT / "problem2_tables.tex").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    min_vehicle = solve("min_vehicle")
    min_cost = solve("min_cost")

    summary = {
        "min_vehicle": solution_summary(min_vehicle),
        "min_cost": solution_summary(min_cost),
    }
    write_solution_csv(OUT / "min_vehicle.csv", min_vehicle)
    write_solution_csv(OUT / "min_cost.csv", min_cost)
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_latex_tables(summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
