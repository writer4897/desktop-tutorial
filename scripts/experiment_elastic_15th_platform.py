from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from experiment_layer_platform_v1_pair import g3_support_stats, max_support_pressure
from experiment_v1_pair_sa import Gene, candidate_points_for_dims, place_one_item_enhanced
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
OUT = ROOT / "results" / "experiments" / "elastic_15th_platform"


def pop_item(items_by_code: dict[str, list[Item]], code: str) -> Item:
    if not items_by_code[code]:
        raise ValueError(f"not enough {code}")
    return items_by_code[code].pop(0)


def platform_cells(style: str) -> list[tuple[int, int]]:
    xs = [0, 70, 140, 210, 280, 350]
    ys = [0, 50, 100, 150]
    cells = [(x, y) for y in ys for x in xs if x + 70 <= 420 and y + 50 <= 210]
    if style == "checker":
        return sorted(cells, key=lambda p: ((p[0] // 70 + p[1] // 50) % 2, p[1], p[0]))
    if style == "left_first":
        return cells
    if style == "right_first":
        return sorted(cells, key=lambda p: (p[1], -p[0]))
    if style == "rear_first":
        return sorted(cells, key=lambda p: (-p[1], p[0]))
    raise ValueError(style)


def make_initial_platforms(
    vehicle_id: str,
    items_by_code: dict[str, list[Item]],
    cells: list[tuple[int, int]],
    fixed_count: int,
) -> list[Placement]:
    placements: list[Placement] = []
    for x, y in cells[:fixed_count]:
        g2a = pop_item(items_by_code, "G2")
        g2b = pop_item(items_by_code, "G2")
        g3 = pop_item(items_by_code, "G3")
        placements.append(Placement(g2a.uid, "G2", "standard", vehicle_id, x, y, 0, 35, 50, 25, g2a.weight, "floor"))
        placements.append(Placement(g2b.uid, "G2", "standard", vehicle_id, x + 35, y, 0, 35, 50, 25, g2b.weight, "floor"))
        placements.append(Placement(g3.uid, "G3", "fragile", vehicle_id, x, y, 25, 70, 50, 40, g3.weight, f"{g2a.uid}|{g2b.uid}"))
    return placements


def build_load(placements: list[Placement]) -> dict[str, float]:
    load: dict[str, float] = {}
    seen: list[Placement] = []
    for placement in placements:
        seen.append(placement)
        add_support_load(placement, seen, load)
    return load


def overlap_reserved(x: int, y: int, length: int, width: int, cells: list[tuple[int, int]]) -> int:
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


def place_with_g1_avoidance(
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
            reserved_penalty = overlap_reserved(cand.x, cand.y, cand.length, cand.width, reserved_cells)
            if item.goods_code == "G1" and cand.z < 145 and reserved_penalty > 0:
                continue
            if item.goods_code in {"G4", "G5"}:
                reserved_penalty = 0
            score = (
                reserved_penalty,
                cand.z,
                cand.y,
                cand.x,
                cand.zmax,
                cand.ymax,
                cand.xmax,
            )
            if best_score is None or score < best_score:
                best = cand
                best_support = support
                best_score = score
    if best is not None:
        best.support = best_support or ""
    return best


def pack_items(
    vehicle_id: str,
    initial: list[Placement],
    items: list[Item],
    gene: Gene,
    reserved_cells: list[tuple[int, int]],
) -> tuple[list[Placement], list[Item]]:
    placements = initial[:]
    current_weight = sum(p.weight for p in placements)
    load = build_load(placements)
    for item in sorted(items, key=lambda it: item_key(gene, it)):
        placed = place_with_g1_avoidance(item, vehicle_id, placements, current_weight, load, gene, reserved_cells)
        if placed is None:
            continue
        placements.append(placed)
        current_weight += placed.weight
        add_support_load(placed, placements, load)
    placed_ids = {p.uid for p in placements}
    return placements, [it for it in items if it.uid not in placed_ids]


def try_add_elastic_g3(
    vehicle_id: str,
    placements: list[Placement],
    items_by_code: dict[str, list[Item]],
    candidate_cells: list[tuple[int, int]],
) -> bool:
    if not items_by_code["G2"] or not items_by_code["G3"]:
        return False
    vehicle = VEHICLES["V1"]
    current_weight = sum(p.weight for p in placements)
    load = build_load(placements)
    for x, y in candidate_cells:
        g2a = items_by_code["G2"][0]
        g2b = items_by_code["G2"][1] if len(items_by_code["G2"]) > 1 else None
        g3 = items_by_code["G3"][0]
        if g2b is None:
            return False
        cand_a = Placement(g2a.uid, "G2", "standard", vehicle_id, x, y, 0, 35, 50, 25, g2a.weight, "")
        ok_a, sup_a = can_place(cand_a, vehicle, placements, current_weight, load)
        if not ok_a:
            continue
        temp = placements + [cand_a]
        temp_load = load.copy()
        cand_a.support = sup_a or ""
        add_support_load(cand_a, temp, temp_load)
        cand_b = Placement(g2b.uid, "G2", "standard", vehicle_id, x + 35, y, 0, 35, 50, 25, g2b.weight, "")
        ok_b, sup_b = can_place(cand_b, vehicle, temp, current_weight + g2a.weight, temp_load)
        if not ok_b:
            continue
        cand_b.support = sup_b or ""
        temp.append(cand_b)
        add_support_load(cand_b, temp, temp_load)
        cand_g3 = Placement(g3.uid, "G3", "fragile", vehicle_id, x, y, 25, 70, 50, 40, g3.weight, "")
        ok_g3, sup_g3 = can_place(cand_g3, vehicle, temp, current_weight + g2a.weight + g2b.weight, temp_load)
        if not ok_g3:
            continue
        cand_g3.support = sup_g3 or ""
        placements.extend([cand_a, cand_b, cand_g3])
        items_by_code["G2"] = items_by_code["G2"][2:]
        items_by_code["G3"] = items_by_code["G3"][1:]
        return True
    return False


def split_items(items_by_code: dict[str, list[Item]]) -> tuple[list[Item], list[Item]]:
    first: list[Item] = []
    second: list[Item] = []
    for code in ["G4", "G5", "G1", "G2", "G3"]:
        items = items_by_code[code]
        half = len(items) // 2
        first.extend(items[:half])
        second.extend(items[half:])
    return first, second


def evaluate(style: str, fixed_count: int, gene: Gene) -> dict:
    items_by_code: dict[str, list[Item]] = defaultdict(list)
    for item in expand_items():
        items_by_code[item.goods_code].append(item)

    cells = platform_cells(style)
    fixed_cells = cells[:fixed_count]
    elastic_cells = cells[fixed_count:]
    first_initial = make_initial_platforms("elastic-V1-1", items_by_code, fixed_cells, fixed_count)
    second_initial = make_initial_platforms("elastic-V1-2", items_by_code, fixed_cells, fixed_count)

    first_items, second_items = split_items(items_by_code)
    first, left_a = pack_items("elastic-V1-1", first_initial, first_items, gene, elastic_cells)
    second, left_b = pack_items("elastic-V1-2", second_initial, second_items, gene, elastic_cells)

    left_by_code: dict[str, list[Item]] = defaultdict(list)
    for item in left_a + left_b:
        left_by_code[item.goods_code].append(item)
    try_add_elastic_g3("elastic-V1-1", first, left_by_code, elastic_cells)
    try_add_elastic_g3("elastic-V1-2", second, left_by_code, elastic_cells)

    remaining_items = [item for values in left_by_code.values() for item in values]
    second, rem_a = pack_items("elastic-V1-2", second, remaining_items, gene, [])
    first, rem_b = pack_items("elastic-V1-1", first, rem_a, gene, [])
    remaining = rem_b

    all_placements = first + second
    errors = validate(VEHICLES["V1"], first) + validate(VEHICLES["V1"], second)
    stats1 = utilization(VEHICLES["V1"], first)
    stats2 = utilization(VEHICLES["V1"], second)
    return {
        "style": style,
        "fixed_count": fixed_count,
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
        "first_volume_utilization": stats1["volume_utilization"],
        "second_volume_utilization": stats2["volume_utilization"],
        "max_support_pressure_kg_per_m2": max_support_pressure([first, second]),
        "validation_error_count": len(errors),
        **g3_support_stats(all_placements),
        "placements": all_placements,
    }


def write_trials(path: Path, rows: list[dict]) -> None:
    fields = [
        "style",
        "fixed_count",
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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    genes = [
        Gene(("G4", "G5", "G1", "G2", "G3"), ("directed", "standard", "fragile"), 1.0, 0.2, 0.0, True),
        Gene(("G4", "G5", "G2", "G1", "G3"), ("directed", "standard", "fragile"), 1.0, 0.45, 150.0, True),
        Gene(("G5", "G4", "G1", "G2", "G3"), ("directed", "standard", "fragile"), 0.8, 0.4, 80.0, True),
    ]
    best: dict | None = None
    rows: list[dict] = []
    for style in ["checker", "left_first", "right_first", "rear_first"]:
        for fixed_count in [12, 13, 14]:
            for gene in genes:
                result = evaluate(style, fixed_count, gene)
                row = {k: v for k, v in result.items() if k != "placements"}
                rows.append(row)
                write_trials(OUT / "elastic_15th_trials.csv", rows)
                if best is None or (result["remaining"], -result["placed"]) < (best["remaining"], -best["placed"]):
                    best = result
                    write_placements_csv(OUT / "best_elastic_15th_platform.csv", best["placements"])
                    print("NEW_BEST", json.dumps(row, ensure_ascii=False))
    assert best is not None
    summary = {k: v for k, v in best.items() if k != "placements"}
    (OUT / "elastic_15th_best.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("BEST")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
