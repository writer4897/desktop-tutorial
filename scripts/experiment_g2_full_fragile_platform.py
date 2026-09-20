from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from experiment_layer_platform_v1_pair import g3_support_stats, max_support_pressure
from experiment_v1_pair_sa import Gene, place_one_item_enhanced
from pack_problem1 import (
    Item,
    Placement,
    VEHICLES,
    add_support_load,
    expand_items,
    goods_counts,
    utilization,
    validate,
    write_placements_csv,
)


ROOT = Path.cwd()
if not (ROOT / "scripts").exists() and (ROOT.parent / "scripts").exists():
    ROOT = ROOT.parent
OUT = ROOT / "results" / "experiments" / "g2_full_fragile_platform"


def pop_item(items_by_code: dict[str, list[Item]], code: str) -> Item:
    if not items_by_code[code]:
        raise ValueError(f"not enough {code}")
    return items_by_code[code].pop(0)


def build_load(placements: list[Placement]) -> dict[str, float]:
    load: dict[str, float] = {}
    seen: list[Placement] = []
    for placement in placements:
        seen.append(placement)
        add_support_load(placement, seen, load)
    return load


def platform_cells(style: str) -> list[tuple[int, int]]:
    """Return 15 non-overlapping 70 x 50 G3 platform cells in V1."""
    if style == "left_block":
        xs = [0, 70, 140, 210, 280]
        ys = [0, 50, 100]
    elif style == "right_block":
        xs = [70, 140, 210, 280, 350]
        ys = [0, 50, 100]
    elif style == "rear_block":
        xs = [0, 70, 140, 210, 280]
        ys = [50, 100, 150]
    elif style == "striped":
        xs = [0, 70, 140, 210, 280]
        ys = [0, 80, 160]
    else:
        raise ValueError(style)
    return [(x, y) for y in ys for x in xs if x + 70 <= 420 and y + 50 <= 210][:15]


def place_g2_g3_platform_layer(
    vehicle_id: str,
    items_by_code: dict[str, list[Item]],
    style: str,
    z_base: int,
    support_floor: bool,
) -> list[Placement]:
    placements: list[Placement] = []
    for x, y in platform_cells(style):
        g2a = pop_item(items_by_code, "G2")
        g2b = pop_item(items_by_code, "G2")
        g3 = pop_item(items_by_code, "G3")
        support = "floor" if support_floor else ""
        placements.append(Placement(g2a.uid, "G2", "standard", vehicle_id, x, y, z_base, 35, 50, 25, g2a.weight, support))
        placements.append(Placement(g2b.uid, "G2", "standard", vehicle_id, x + 35, y, z_base, 35, 50, 25, g2b.weight, support))
        placements.append(Placement(g3.uid, "G3", "fragile", vehicle_id, x, y, z_base + 25, 70, 50, 40, g3.weight, f"{g2a.uid}|{g2b.uid}"))
    return placements


def reserved_overlap_area(item: Item | Placement, point: tuple[int, int, int], dims: tuple[int, int, int], cells: list[tuple[int, int]]) -> int:
    x, y, _ = point
    length, width, _ = dims
    total = 0
    for cx, cy in cells:
        x0, x1 = max(x, cx), min(x + length, cx + 70)
        y0, y1 = max(y, cy), min(y + width, cy + 50)
        if x0 < x1 and y0 < y1:
            total += (x1 - x0) * (y1 - y0)
    return total


def pack_remaining(
    vehicle_id: str,
    initial: list[Placement],
    remaining: list[Item],
    gene: Gene,
    reserved_cells: list[tuple[int, int]],
) -> tuple[list[Placement], list[Item]]:
    vehicle = VEHICLES["V1"]
    placements = initial[:]
    current_weight = sum(p.weight for p in placements)
    load = build_load(placements)

    def key(item: Item) -> tuple:
        rank = {code: idx for idx, code in enumerate(gene.goods_order)}
        label_rank = {label: idx for idx, label in enumerate(gene.label_order)}
        return (
            rank.get(item.goods_code, 99),
            label_rank.get(item.label, 99),
            -gene.volume_weight * item.volume - gene.base_weight * item.base_area0 - gene.height_weight * item.height0,
            item.uid,
        )

    for item in sorted(remaining, key=key):
        placed = place_one_item_enhanced(item, vehicle, vehicle_id, placements, current_weight, load, gene.center_points)
        if placed is None:
            continue
        # G1 is useful as filler, but it should not occupy the horizontal
        # platform block before all G3 support cells are already reserved.
        if item.goods_code == "G1" and placed.z < 120:
            overlap = reserved_overlap_area(item, (placed.x, placed.y, placed.z), (placed.length, placed.width, placed.height), reserved_cells)
            if overlap > 0:
                continue
        placements.append(placed)
        current_weight += placed.weight
        add_support_load(placed, placements, load)
    placed_ids = {p.uid for p in placements}
    return placements, [it for it in remaining if it.uid not in placed_ids]


def split_remaining(items_by_code: dict[str, list[Item]]) -> tuple[list[Item], list[Item]]:
    a: list[Item] = []
    b: list[Item] = []
    for code in ["G1", "G2", "G3", "G4", "G5"]:
        items = items_by_code[code]
        half = len(items) // 2
        a.extend(items[:half])
        b.extend(items[half:])
    return a, b


def evaluate(style: str, gene: Gene) -> dict:
    items_by_code: dict[str, list[Item]] = defaultdict(list)
    for item in expand_items():
        items_by_code[item.goods_code].append(item)

    cells = platform_cells(style)
    first_initial = place_g2_g3_platform_layer("g2full-V1-1", items_by_code, style, 0, True)
    second_initial = place_g2_g3_platform_layer("g2full-V1-2", items_by_code, style, 0, True)
    rem_a, rem_b = split_remaining(items_by_code)
    first, left_a = pack_remaining("g2full-V1-1", first_initial, rem_a, gene, cells)
    second, left_b = pack_remaining("g2full-V1-2", second_initial, rem_b, gene, cells)
    second, repair_a = pack_remaining("g2full-V1-2", second, left_a, gene, cells)
    first, repair_b = pack_remaining("g2full-V1-1", first, left_b, gene, cells)
    remaining = repair_a + repair_b
    all_placements = first + second
    errors = validate(VEHICLES["V1"], first) + validate(VEHICLES["V1"], second)
    stats1 = utilization(VEHICLES["V1"], first)
    stats2 = utilization(VEHICLES["V1"], second)
    pressure = max_support_pressure([first, second])
    return {
        "style": style,
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
        "max_support_pressure_kg_per_m2": pressure,
        "validation_error_count": len(errors),
        **g3_support_stats(all_placements),
        "placements": all_placements,
    }


def write_trials(path: Path, rows: list[dict]) -> None:
    fields = [
        "style",
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
        Gene(("G4", "G5", "G1", "G2"), ("directed", "standard", "fragile"), 1.0, 0.2, 0.0, True),
        Gene(("G4", "G5", "G2", "G1"), ("directed", "standard", "fragile"), 1.0, 0.45, 150.0, True),
        Gene(("G5", "G4", "G1", "G2"), ("directed", "standard", "fragile"), 0.8, 0.4, 80.0, True),
    ]
    best: dict | None = None
    rows: list[dict] = []
    for style in ["left_block", "right_block", "rear_block", "striped"]:
        for gene in genes:
            result = evaluate(style, gene)
            row = {k: v for k, v in result.items() if k != "placements"}
            rows.append(row)
            write_trials(OUT / "g2_full_platform_trials.csv", rows)
            if best is None or (result["remaining"], -result["placed"]) < (best["remaining"], -best["placed"]):
                best = result
                write_placements_csv(OUT / "best_g2_full_platform.csv", best["placements"])
                print("NEW_BEST", json.dumps(row, ensure_ascii=False))
    assert best is not None
    summary = {k: v for k, v in best.items() if k != "placements"}
    (OUT / "g2_full_platform_best.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("BEST")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
