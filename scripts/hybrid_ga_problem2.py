from __future__ import annotations

import csv
import json
import random
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
    utilization,
    validate,
)
from pack_problem2 import TruckPlan, SearchState, solution_summary, write_solution_csv


ROOT = Path.cwd()
if not (ROOT / "scripts").exists() and (ROOT.parent / "scripts").exists():
    ROOT = ROOT.parent
OUT = ROOT / "results" / "hybrid_ga"


LABELS = ["standard", "fragile", "directed"]
VEHICLE_CODES = ["V1", "V2"]


@dataclass(frozen=True)
class Chromosome:
    vehicle_sequence: tuple[str, ...]
    label_priority: tuple[str, ...]
    volume_weight: float
    base_weight: float
    height_weight: float
    weight_weight: float
    fragile_late: float


def item_key(item: Item, gene: Chromosome) -> tuple[float, str]:
    label_rank = {label: idx for idx, label in enumerate(gene.label_priority)}
    score = (
        1000000 * label_rank[item.label]
        - gene.volume_weight * item.volume
        - gene.base_weight * item.base_area0
        - gene.height_weight * item.height0
        - gene.weight_weight * item.weight
    )
    if item.label == "fragile":
        score += gene.fragile_late
    return score, item.uid


def ordered_items(items: list[Item], gene: Chromosome) -> list[Item]:
    return sorted(items, key=lambda item: item_key(item, gene))


def pack_vehicle_gene(vehicle_code: str, items: list[Item], gene: Chromosome, vehicle_id: str) -> tuple[TruckPlan, list[Item]] | None:
    vehicle = VEHICLES[vehicle_code]
    placements: list[Placement] = []
    current_weight = 0
    load_on_support: dict[str, float] = {}

    for item in ordered_items(items, gene):
        placed = place_one_item(item, vehicle, vehicle_id, placements, current_weight, load_on_support)
        if placed is None:
            continue
        placements.append(placed)
        current_weight += placed.weight
        add_support_load(placed, placements, load_on_support)

    if not placements:
        return None
    errors = validate(vehicle, placements)
    if errors:
        return None
    placed_ids = {p.uid for p in placements}
    remaining = [item for item in items if item.uid not in placed_ids]
    plan = TruckPlan(
        vehicle_code=vehicle_code,
        vehicle_name=vehicle.name,
        strategy="GA-height-aware-extreme-point",
        vehicle_id=vehicle_id,
        placements=placements,
        stats=utilization(vehicle, placements),
    )
    return plan, remaining


def decode(gene: Chromosome) -> SearchState:
    remaining = expand_items()
    plans: list[TruckPlan] = []
    cost = 0

    for idx, vehicle_code in enumerate(gene.vehicle_sequence, 1):
        packed = pack_vehicle_gene(vehicle_code, remaining, gene, f"ga-{idx}-{vehicle_code}")
        if packed is None:
            continue
        plan, remaining = packed
        plans.append(plan)
        cost += VEHICLES[vehicle_code].cost
        if not remaining:
            break

    return SearchState(remaining=remaining, plans=plans, cost=cost)


def fitness(state: SearchState) -> tuple:
    remaining_volume = sum(item.volume for item in state.remaining)
    remaining_count = len(state.remaining)
    avg_fullness = 0.0
    if state.plans:
        avg_fullness = sum(plan.stats["combined_fullness"] for plan in state.plans) / len(state.plans)
    # Lexicographic objective: feasibility first, then vehicle count, cost,
    # remaining volume/count, and fullness. This avoids hiding infeasibility
    # behind weighted scores.
    return (
        remaining_count,
        remaining_volume,
        state.used_count if remaining_count == 0 else 99,
        state.cost if remaining_count == 0 else 999999,
        -avg_fullness,
    )


def baseline_genes() -> list[Chromosome]:
    return [
        Chromosome(("V1", "V2"), ("directed", "standard", "fragile"), 1.0, 0.2, 0.0, 0.0, 0.0),
        Chromosome(("V2", "V1"), ("directed", "standard", "fragile"), 1.0, 0.2, 0.0, 0.0, 0.0),
    ]


def random_gene(rng: random.Random) -> Chromosome:
    length = rng.choice([2, 2, 3])
    vehicles = tuple(rng.choice(VEHICLE_CODES) for _ in range(length))
    labels = LABELS[:]
    rng.shuffle(labels)
    return Chromosome(
        vehicle_sequence=vehicles,
        label_priority=tuple(labels),
        volume_weight=rng.uniform(0.6, 1.4),
        base_weight=rng.uniform(0.0, 0.6),
        height_weight=rng.uniform(0.0, 500.0),
        weight_weight=rng.uniform(0.0, 500.0),
        fragile_late=rng.uniform(-500000.0, 800000.0),
    )


def mutate(gene: Chromosome, rng: random.Random) -> Chromosome:
    vehicles = list(gene.vehicle_sequence)
    labels = list(gene.label_priority)
    if rng.random() < 0.35:
        pos = rng.randrange(len(vehicles))
        vehicles[pos] = rng.choice(VEHICLE_CODES)
    if rng.random() < 0.20 and len(vehicles) < 3:
        vehicles.append(rng.choice(VEHICLE_CODES))
    if rng.random() < 0.20 and len(vehicles) > 2:
        vehicles.pop(rng.randrange(len(vehicles)))
    if rng.random() < 0.35:
        a, b = rng.sample(range(3), 2)
        labels[a], labels[b] = labels[b], labels[a]

    def jitter(value: float, scale: float, low: float, high: float) -> float:
        return min(high, max(low, value + rng.uniform(-scale, scale)))

    return Chromosome(
        vehicle_sequence=tuple(vehicles),
        label_priority=tuple(labels),
        volume_weight=jitter(gene.volume_weight, 0.25, 0.1, 2.0),
        base_weight=jitter(gene.base_weight, 0.15, 0.0, 1.0),
        height_weight=jitter(gene.height_weight, 120.0, 0.0, 900.0),
        weight_weight=jitter(gene.weight_weight, 120.0, 0.0, 900.0),
        fragile_late=jitter(gene.fragile_late, 250000.0, -800000.0, 1200000.0),
    )


def crossover(a: Chromosome, b: Chromosome, rng: random.Random) -> Chromosome:
    vehicles = a.vehicle_sequence if rng.random() < 0.5 else b.vehicle_sequence
    labels = a.label_priority if rng.random() < 0.5 else b.label_priority
    return Chromosome(
        vehicle_sequence=vehicles,
        label_priority=labels,
        volume_weight=(a.volume_weight + b.volume_weight) / 2,
        base_weight=(a.base_weight + b.base_weight) / 2,
        height_weight=(a.height_weight + b.height_weight) / 2,
        weight_weight=(a.weight_weight + b.weight_weight) / 2,
        fragile_late=(a.fragile_late + b.fragile_late) / 2,
    )


def run_ga(population_size: int = 3, generations: int = 1, seed: int = 2026) -> tuple[Chromosome, SearchState, list[dict]]:
    rng = random.Random(seed)
    population = baseline_genes()
    while len(population) < population_size:
        population.append(random_gene(rng))

    history: list[dict] = []
    best_gene: Chromosome | None = None
    best_state: SearchState | None = None
    best_fit: tuple | None = None

    for generation in range(generations + 1):
        evaluated = []
        for gene in population:
            state = decode(gene)
            fit = fitness(state)
            evaluated.append((fit, gene, state))
            if best_fit is None or fit < best_fit:
                best_fit = fit
                best_gene = gene
                best_state = state
        evaluated.sort(key=lambda row: row[0])
        top_fit, _, top_state = evaluated[0]
        history.append(
            {
                "generation": generation,
                "remaining_count": len(top_state.remaining),
                "remaining_volume": top_state.remaining_volume,
                "vehicle_count": top_state.used_count,
                "cost": top_state.cost,
                "fitness": list(top_fit),
            }
        )

        elites = [gene for _, gene, _ in evaluated[: max(2, population_size // 3)]]
        next_population = elites[:]
        while len(next_population) < population_size:
            if rng.random() < 0.55 and len(elites) >= 2:
                child = crossover(*rng.sample(elites, 2), rng)
            else:
                child = rng.choice(elites)
            next_population.append(mutate(child, rng))
        population = next_population

    assert best_gene is not None and best_state is not None
    return best_gene, best_state, history


def write_history(path: Path, history: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        fields = ["generation", "remaining_count", "remaining_volume", "vehicle_count", "cost", "fitness"]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)


def write_latex_tables(summary: dict) -> None:
    sol = summary["best_solution"]
    lines = [
        "% Auto-generated by scripts/hybrid_ga_problem2.py",
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{混合遗传约束感知极点法得到的问题二可行方案}",
        "\\label{tab:hybrid_ga_result}",
        "\\begin{tabular}{ccccc}",
        "\\toprule",
        "算法 & 车辆数 & 车型组合 & 总成本/元 & 约束错误数 \\\\",
        "\\midrule",
    ]
    combo = "+".join(vehicle["vehicle_name"] for vehicle in sol["vehicles"])
    error_count = sum(len(errors) for errors in summary["validation_errors"].values())
    lines.append(f"HGEP & {sol['vehicle_count']} & {combo} & {sol['total_cost']} & {error_count} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    lines.extend(
        [
            "\\begin{table}[H]",
            "\\centering",
            "\\caption{混合遗传搜索逐代最优个体变化}",
            "\\label{tab:hybrid_ga_history}",
            "\\begin{tabular}{ccccc}",
            "\\toprule",
            "代数 & 剩余件数 & 剩余体积/cm$^3$ & 已用车辆数 & 当前成本/元 \\\\",
            "\\midrule",
        ]
    )
    for row in summary["history"]:
        lines.append(
            f"{row['generation']} & {row['remaining_count']} & {row['remaining_volume']} & "
            f"{row['vehicle_count']} & {row['cost']} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    (OUT / "hybrid_ga_tables.tex").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    best_gene, best_state, history = run_ga()
    summary = {
        "algorithm": "Hybrid Genetic Constraint-aware Extreme Point heuristic",
        "note": "The chromosome optimizes vehicle sequence, cargo type priority and height-aware ordering weights; coordinates are decoded by the constraint-aware extreme-point placer.",
        "best_chromosome": {
            "vehicle_sequence": best_gene.vehicle_sequence,
            "label_priority": best_gene.label_priority,
            "volume_weight": best_gene.volume_weight,
            "base_weight": best_gene.base_weight,
            "height_weight": best_gene.height_weight,
            "weight_weight": best_gene.weight_weight,
            "fragile_late": best_gene.fragile_late,
        },
        "best_solution": solution_summary(best_state) if not best_state.remaining else {
            "remaining_count": len(best_state.remaining),
            "remaining_volume": best_state.remaining_volume,
            "vehicle_count": best_state.used_count,
            "total_cost": best_state.cost,
            "vehicles": [
                {
                    "vehicle_id": plan.vehicle_id,
                    "vehicle_code": plan.vehicle_code,
                    "vehicle_name": plan.vehicle_name,
                    "stats": plan.stats,
                    "goods_counts": goods_counts(plan.placements),
                }
                for plan in best_state.plans
            ],
        },
        "validation_errors": {
            plan.vehicle_id: validate(VEHICLES[plan.vehicle_code], plan.placements)
            for plan in best_state.plans
        },
        "history": history,
    }
    write_solution_csv(OUT / "best_solution.csv", best_state)
    write_history(OUT / "history.csv", history)
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_latex_tables(summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
