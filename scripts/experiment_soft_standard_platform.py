from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from experiment_inverse_fragile_platform import (
    make_floor_g2_platforms,
    max_support_pressure,
    g3_support_stats,
    platform_cells,
    split_remaining_by_code,
)
from experiment_v1_pair_sa import Gene, candidate_points_for_dims
from pack_problem1 import (
    Item,
    Placement,
    VEHICLES,
    add_support_load,
    can_place,
    expand_items,
    goods_counts,
    orientations,
    utilization,
    validate,
    write_placements_csv,
)


ROOT = Path.cwd()
if not (ROOT / "scripts").exists() and (ROOT.parent / "scripts").exists():
    ROOT = ROOT.parent
OUT = ROOT / "results" / "experiments" / "soft_standard_platform"


def build_load(placements: list[Placement]) -> dict[str, float]:
    load: dict[str, float] = {}
    seen: list[Placement] = []
    for placement in placements:
        seen.append(placement)
        add_support_load(placement, seen, load)
    return load


def overlap_cells(x: int, y: int, length: int, width: int, cells: list[tuple[int, int]]) -> int:
    total = 0
    for cx, cy in cells:
        x0, x1 = max(x, cx), min(x + length, cx + 70)
        y0, y1 = max(y, cy), min(y + width, cy + 50)
        if x0 < x1 and y0 < y1:
            total += (x1 - x0) * (y1 - y0)
    return total


def item_key(gene: Gene, item: Item) -> tuple:
    goods_rank = {code: idx for idx, code in enumerate(gene.goods_order)}
    label_rank = {label: idx for idx, label in enumerate(gene.label_order)}
    return (
        goods_rank.get(item.goods_code, 99),
        label_rank.get(item.label, 99),
        -gene.volume_weight * item.volume - gene.base_weight * item.base_area0 - gene.height_weight * item.height0,
        item.uid,
    )


def platform_score(item: Item, candidate: Placement, reserved_cells: list[tuple[int, int]]) -> tuple:
    overlap = overlap_cells(candidate.x, candidate.y, candidate.length, candidate.width, reserved_cells)
    g1_penalty = overlap if item.goods_code == "G1" and candidate.z < 145 else 0
    g2_reward = -overlap if item.goods_code == "G2" and candidate.z in {0, 50, 75, 100, 130, 150} else 0
    # Keep directed goods compact and low; let standards shape the future G3 platforms.
    directed_penalty = 0 if item.goods_code in {"G4", "G5"} else 1
    return (g1_penalty, g2_reward, candidate.z, directed_penalty, candidate.y, candidate.x, candidate.zmax)


def place_one_soft(
    item: Item,
    vehicle_id: str,
    placements: list[Placement],
    current_weight: float,
    load: dict[str, float],
    gene: Gene,
    reserved_cells: list[tuple[int, int]],
) -> Placement | None:
    vehicle = VEHICLES["V1"]
    best: Placement | None = None
    best_support: str | None = None
    best_score: tuple | None = None
    for dims in orientations(item):
        for point in candidate_points_for_dims(dims, placements, vehicle, gene.center_points):
            cand = Placement(item.uid, item.goods_code, item.label, vehicle_id, point[0], point[1], point[2], dims[0], dims[1], dims[2], item.weight, "")
            ok, support = can_place(cand, vehicle, placements, current_weight, load)
            if not ok:
                continue
            score = platform_score(item, cand, reserved_cells)
            if best_score is None or score < best_score:
                best = cand
                best_support = support
                best_score = score
    if best is not None:
        best.support = best_support or ""
    return best


def pack_remaining(
    vehicle_id: str,
    initial: list[Placement],
    remaining: list[Item],
    gene: Gene,
    reserved_cells: list[tuple[int, int]],
) -> tuple[list[Placement], list[Item]]:
    placements = initial[:]
    current_weight = sum(p.weight for p in placements)
    load = build_load(placements)
    for item in sorted(remaining, key=lambda it: item_key(gene, it)):
        placed = place_one_soft(item, vehicle_id, placements, current_weight, load, gene, reserved_cells)
        if placed is None:
            continue
        placements.append(placed)
        current_weight += placed.weight
        add_support_load(placed, placements, load)
    placed_ids = {p.uid for p in placements}
    return placements, [it for it in remaining if it.uid not in placed_ids]


def evaluate(split: tuple[int, int], style: str, gene: Gene, balanced: bool) -> dict:
    items_by_code: dict[str, list[Item]] = defaultdict(list)
    for item in expand_items():
        items_by_code[item.goods_code].append(item)
    cells = platform_cells(style)
    first_initial = make_floor_g2_platforms("soft-V1-1", split[0], items_by_code, style)
    second_initial = make_floor_g2_platforms("soft-V1-2", split[1], items_by_code, style)
    reserved = cells[split[0] :]
    if balanced:
        rem_a, rem_b = split_remaining_by_code(items_by_code, first_initial, second_initial)
        first, left_a = pack_remaining("soft-V1-1", first_initial, rem_a, gene, reserved)
        second, left_b = pack_remaining("soft-V1-2", second_initial, rem_b, gene, reserved)
        second, repair_a = pack_remaining("soft-V1-2", second, left_a, gene, [])
        first, repair_b = pack_remaining("soft-V1-1", first, left_b, gene, [])
        remaining = repair_a + repair_b
    else:
        remaining_items = [item for values in items_by_code.values() for item in values]
        first, rem1 = pack_remaining("soft-V1-1", first_initial, remaining_items, gene, reserved)
        second, remaining = pack_remaining("soft-V1-2", second_initial, rem1, gene, reserved)
    all_placements = first + second
    errors = validate(VEHICLES["V1"], first) + validate(VEHICLES["V1"], second)
    stats1 = utilization(VEHICLES["V1"], first)
    stats2 = utilization(VEHICLES["V1"], second)
    return {
        "split": split,
        "style": style,
        "balanced": balanced,
        "gene": {
            "goods_order": gene.goods_order,
            "label_order": gene.label_order,
            "volume_weight": gene.volume_weight,
            "base_weight": gene.base_weight,
            "height_weight": gene.height_weight,
            "center_points": gene.center_points,
        },
        "placed": len(all_placements),
        "remaining": len(remaining),
        "remaining_counts": goods_counts(
            [Placement(it.uid, it.goods_code, it.label, "", 0, 0, 0, it.length0, it.width0, it.height0, it.weight, "") for it in remaining]
        ),
        "first_count": len(first),
        "second_count": len(second),
        "first_goods": dict(Counter(p.goods_code for p in first)),
        "second_goods": dict(Counter(p.goods_code for p in second)),
        "avg_volume_utilization": (stats1["volume_utilization"] + stats2["volume_utilization"]) / 2,
        "avg_weight_utilization": (stats1["weight_utilization"] + stats2["weight_utilization"]) / 2,
        "max_support_pressure_kg_per_m2": max_support_pressure([first, second]),
        "validation_error_count": len(errors),
        **g3_support_stats(all_placements),
        "placements": all_placements,
    }


def write_trials(path: Path, rows: list[dict]) -> None:
    fields = [
        "split",
        "style",
        "balanced",
        "placed",
        "remaining",
        "remaining_counts",
        "first_count",
        "second_count",
        "first_goods",
        "second_goods",
        "avg_volume_utilization",
        "avg_weight_utilization",
        "max_support_pressure_kg_per_m2",
        "g3_total",
        "g3_floor",
        "g3_platform",
        "validation_error_count",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    genes = [
        Gene(("G4", "G5", "G2", "G1", "G3"), ("directed", "standard", "fragile"), 1.0, 0.45, 150.0, True),
        Gene(("G4", "G5", "G1", "G2", "G3"), ("directed", "standard", "fragile"), 1.0, 0.2, 0.0, True),
        Gene(("G5", "G4", "G2", "G1", "G3"), ("standard", "directed", "fragile"), 0.8, 0.4, 80.0, True),
    ]
    rows: list[dict] = []
    best: dict | None = None
    for split in [(2, 2), (3, 3), (4, 4), (6, 6)]:
        for style in ["checker", "front_rows", "side_rows"]:
            for balanced in [True, False]:
                for gene in genes:
                    result = evaluate(split, style, gene, balanced)
                    row = {k: v for k, v in result.items() if k != "placements"}
                    rows.append(row)
                    write_trials(OUT / "soft_standard_platform_trials.csv", rows)
                    if best is None or (result["remaining"], -result["placed"]) < (best["remaining"], -best["placed"]):
                        best = result
                        write_placements_csv(OUT / "best_soft_standard_platform.csv", best["placements"])
                        print("NEW_BEST", json.dumps(row, ensure_ascii=False))
    assert best is not None
    summary = {k: v for k, v in best.items() if k != "placements"}
    (OUT / "soft_standard_platform_best.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("BEST")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
