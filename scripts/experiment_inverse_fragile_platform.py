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
    rects_cover,
    utilization,
    validate,
    write_placements_csv,
)


ROOT = Path.cwd()
if not (ROOT / "scripts").exists() and (ROOT.parent / "scripts").exists():
    ROOT = ROOT.parent
OUT = ROOT / "results" / "experiments" / "inverse_fragile_platform"


def pop_item(items_by_code: dict[str, list[Item]], code: str) -> Item:
    if not items_by_code[code]:
        raise ValueError(f"not enough {code}")
    return items_by_code[code].pop(0)


def platform_cells(style: str) -> list[tuple[int, int]]:
    """Candidate 70 x 50 G3 platform lower-left coordinates in V1."""
    if style == "front_rows":
        xs = [0, 70, 140, 210, 280, 350]
        ys = [0, 50, 100, 150]
        return [(x, y) for y in ys for x in xs if x + 70 <= 420 and y + 50 <= 210]
    if style == "side_rows":
        xs = [0, 70, 140, 210, 280, 350]
        ys = [160, 110, 60, 10]
        return [(x, y) for y in ys for x in xs if x + 70 <= 420 and y + 50 <= 210]
    if style == "checker":
        xs = [0, 70, 140, 210, 280, 350]
        ys = [0, 50, 100, 150]
        cells = [(x, y) for y in ys for x in xs if x + 70 <= 420 and y + 50 <= 210]
        return sorted(cells, key=lambda p: ((p[0] // 70 + p[1] // 50) % 2, p[1], p[0]))
    raise ValueError(style)


def make_floor_g2_platforms(
    vehicle_id: str,
    count: int,
    items_by_code: dict[str, list[Item]],
    style: str,
) -> list[Placement]:
    placements: list[Placement] = []
    for x, y in platform_cells(style)[:count]:
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


def rect_union_area(rects: list[tuple[int, int, int, int]]) -> int:
    if not rects:
        return 0
    xs = sorted({x for x0, x1, _, _ in rects for x in (x0, x1)})
    ys = sorted({y for _, _, y0, y1 in rects for y in (y0, y1)})
    area = 0
    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            mx = (xs[i] + xs[i + 1]) / 2
            my = (ys[j] + ys[j + 1]) / 2
            if any(x0 <= mx <= x1 and y0 <= my <= y1 for x0, x1, y0, y1 in rects):
                area += (xs[i + 1] - xs[i]) * (ys[j + 1] - ys[j])
    return area


def standard_layer_areas(placements: list[Placement]) -> dict[int, int]:
    levels = sorted({p.zmax for p in placements if p.label == "standard"})
    areas: dict[int, int] = {}
    for z in levels:
        rects = [(p.x, p.xmax, p.y, p.ymax) for p in placements if p.label == "standard" and p.zmax == z]
        areas[z] = rect_union_area(rects)
    return areas


def layer_metrics(placements: list[Placement]) -> dict[str, float]:
    areas = standard_layer_areas(placements)
    high_areas = {z: area for z, area in areas.items() if z >= 100}
    slot_info = standard_layer_slot_metrics(placements)
    return {
        "max_standard_layer_z": max(areas) if areas else 0,
        "max_standard_layer_area_cm2": max(areas.values()) if areas else 0,
        "max_high_standard_layer_area_cm2": max(high_areas.values()) if high_areas else 0,
        "weighted_high_layer_area_cm3": sum(z * area for z, area in high_areas.items()),
        **slot_info,
    }


def overlap_2d(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return max(a[0], b[0]) < min(a[2], b[2]) and max(a[1], b[1]) < min(a[3], b[3])


def supportable_g3_slots(placements: list[Placement], z: int, step: int = 5) -> list[tuple[int, int, int, int]]:
    slots: list[tuple[int, int, int, int]] = []
    supports = [p for p in placements if p.label == "standard" and p.zmax == z]
    if not supports:
        return slots
    for length, width in [(70, 50), (50, 70)]:
        for x in range(0, VEHICLES["V1"].length - length + 1, step):
            for y in range(0, VEHICLES["V1"].width - width + 1, step):
                candidate = Placement("slot", "G3", "fragile", "", x, y, z, length, width, 40, 15, "")
                if rects_cover(candidate, supports):
                    slots.append((x, y, x + length, y + width))
    return slots


def greedy_nonoverlap_count(slots: list[tuple[int, int, int, int]]) -> int:
    chosen: list[tuple[int, int, int, int]] = []
    for slot in sorted(slots, key=lambda rect: (rect[1], rect[0], rect[3], rect[2])):
        if all(not overlap_2d(slot, old) for old in chosen):
            chosen.append(slot)
    return len(chosen)


def standard_layer_slot_metrics(placements: list[Placement]) -> dict[str, float]:
    total_capacity = 0
    total_used = 0
    total_area_capacity = 0
    total_fragment_loss = 0
    total_surplus_area = 0
    high_capacity = 0
    high_used = 0
    for z, area in standard_layer_areas(placements).items():
        if area < 3500:
            continue
        area_capacity = area // 3500
        slot_capacity = greedy_nonoverlap_count(supportable_g3_slots(placements, z))
        used = sum(1 for p in placements if p.goods_code == "G3" and p.z == z)
        total_capacity += slot_capacity
        total_used += used
        total_area_capacity += area_capacity
        total_fragment_loss += max(0, area_capacity - slot_capacity)
        total_surplus_area += max(0, area - slot_capacity * 3500)
        if z >= 100:
            high_capacity += slot_capacity
            high_used += used
    return {
        "g3_slot_capacity": total_capacity,
        "g3_slot_used": total_used,
        "g3_slot_surplus": total_capacity - total_used,
        "g3_area_capacity": total_area_capacity,
        "g3_fragment_loss_slots": total_fragment_loss,
        "g3_surplus_area_after_slots_cm2": total_surplus_area,
        "high_g3_slot_capacity": high_capacity,
        "high_g3_slot_used": high_used,
        "high_g3_slot_surplus": high_capacity - high_used,
    }


def item_sort_key(gene: Gene, item: Item) -> tuple:
    rank = {code: idx for idx, code in enumerate(gene.goods_order)}
    label_rank = {label: idx for idx, label in enumerate(gene.label_order)}
    return (
        rank.get(item.goods_code, 99),
        label_rank.get(item.label, 99),
        -gene.volume_weight * item.volume - gene.base_weight * item.base_area0 - gene.height_weight * item.height0,
        item.uid,
    )


def pack_remaining(vehicle_id: str, initial: list[Placement], remaining: list[Item], gene: Gene) -> tuple[list[Placement], list[Item]]:
    vehicle = VEHICLES["V1"]
    placements = initial[:]
    current_weight = sum(p.weight for p in placements)
    load = build_load(placements)
    for item in sorted(remaining, key=lambda it: item_sort_key(gene, it)):
        placed = place_one_item_enhanced(item, vehicle, vehicle_id, placements, current_weight, load, gene.center_points)
        if placed is None:
            continue
        placements.append(placed)
        current_weight += placed.weight
        add_support_load(placed, placements, load)
    placed_ids = {p.uid for p in placements}
    return placements, [it for it in remaining if it.uid not in placed_ids]


def split_remaining_by_code(items_by_code: dict[str, list[Item]], initial_a: list[Placement], initial_b: list[Placement]) -> tuple[list[Item], list[Item]]:
    used_a = Counter(p.goods_code for p in initial_a)
    used_b = Counter(p.goods_code for p in initial_b)
    left_a: list[Item] = []
    left_b: list[Item] = []
    for code in ["G1", "G2", "G3", "G4", "G5"]:
        items = items_by_code[code]
        total = len(items) + used_a[code] + used_b[code]
        target_a = total // 2
        take_a = max(0, min(len(items), target_a - used_a[code]))
        left_a.extend(items[:take_a])
        left_b.extend(items[take_a:])
    return left_a, left_b


def evaluate(split: tuple[int, int], style: str, gene: Gene, balanced: bool) -> dict:
    items_by_code: dict[str, list[Item]] = defaultdict(list)
    for item in expand_items():
        items_by_code[item.goods_code].append(item)

    initial_a = make_floor_g2_platforms("inverse-V1-1", split[0], items_by_code, style)
    initial_b = make_floor_g2_platforms("inverse-V1-2", split[1], items_by_code, style)
    if balanced:
        rem_a, rem_b = split_remaining_by_code(items_by_code, initial_a, initial_b)
        first, left_a = pack_remaining("inverse-V1-1", initial_a, rem_a, gene)
        second, left_b = pack_remaining("inverse-V1-2", initial_b, rem_b, gene)
        second, repair_a = pack_remaining("inverse-V1-2", second, left_a, gene)
        first, repair_b = pack_remaining("inverse-V1-1", first, left_b, gene)
        remaining = repair_a + repair_b
    else:
        remaining_items = [item for values in items_by_code.values() for item in values]
        first, rem1 = pack_remaining("inverse-V1-1", initial_a, remaining_items, gene)
        second, remaining = pack_remaining("inverse-V1-2", initial_b, rem1, gene)

    all_placements = first + second
    errors = validate(VEHICLES["V1"], first) + validate(VEHICLES["V1"], second)
    stats_a = utilization(VEHICLES["V1"], first)
    stats_b = utilization(VEHICLES["V1"], second)
    layer_a = layer_metrics(first)
    layer_b = layer_metrics(second)
    count_gap = abs(len(first) - len(second))
    volume_gap = abs(stats_a["volume_utilization"] - stats_b["volume_utilization"])
    pressure = max_support_pressure([first, second])
    layer_bonus = (
        (layer_a["max_high_standard_layer_area_cm2"] + layer_b["max_high_standard_layer_area_cm2"]) / 10000
        + (layer_a["weighted_high_layer_area_cm3"] + layer_b["weighted_high_layer_area_cm3"]) / 10000000
    )
    fragmentation_penalty = (
        layer_a["g3_fragment_loss_slots"]
        + layer_b["g3_fragment_loss_slots"]
        + 0.0001 * (layer_a["g3_surplus_area_after_slots_cm2"] + layer_b["g3_surplus_area_after_slots_cm2"])
    )
    slot_bonus = 2 * (layer_a["g3_slot_capacity"] + layer_b["g3_slot_capacity"])
    score = (
        len(remaining) * 1000
        + count_gap * 2
        + volume_gap * 100
        + abs(stats_a["weight_utilization"] - stats_b["weight_utilization"]) * 100
        + max(0.0, pressure - 500.0) * 10
        + fragmentation_penalty
        - layer_bonus
        - slot_bonus
    )
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
        "score": score,
        "placed": len(all_placements),
        "remaining": len(remaining),
        "remaining_counts": goods_counts(
            [Placement(it.uid, it.goods_code, it.label, "", 0, 0, 0, it.length0, it.width0, it.height0, it.weight, "") for it in remaining]
        ),
        "first_count": len(first),
        "second_count": len(second),
        "first_goods": dict(Counter(p.goods_code for p in first)),
        "second_goods": dict(Counter(p.goods_code for p in second)),
        "avg_volume_utilization": (stats_a["volume_utilization"] + stats_b["volume_utilization"]) / 2,
        "avg_weight_utilization": (stats_a["weight_utilization"] + stats_b["weight_utilization"]) / 2,
        "first_volume_utilization": stats_a["volume_utilization"],
        "second_volume_utilization": stats_b["volume_utilization"],
        "first_weight_utilization": stats_a["weight_utilization"],
        "second_weight_utilization": stats_b["weight_utilization"],
        "max_support_pressure_kg_per_m2": pressure,
        "validation_error_count": len(errors),
        "first_layer_metrics": layer_a,
        "second_layer_metrics": layer_b,
        **g3_support_stats(all_placements),
        "placements": all_placements,
    }


def write_trials(path: Path, rows: list[dict]) -> None:
    fields = [
        "split",
        "style",
        "balanced",
        "score",
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
        "first_layer_metrics",
        "second_layer_metrics",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    genes = [
        Gene(("G4", "G5", "G1", "G2", "G3"), ("directed", "standard", "fragile"), 1.0, 0.2, 0.0, True),
        Gene(("G4", "G5", "G2", "G1", "G3"), ("directed", "standard", "fragile"), 1.0, 0.45, 150.0, True),
        Gene(("G5", "G4", "G2", "G1", "G3"), ("standard", "directed", "fragile"), 0.8, 0.4, 80.0, True),
    ]
    splits = [(0, 0), (2, 2), (4, 4), (6, 6), (8, 8), (10, 10), (12, 12), (14, 14), (6, 8), (8, 6)]
    styles = ["front_rows", "side_rows", "checker"]
    best: dict | None = None
    rows: list[dict] = []
    for split in splits:
        for style in styles:
            for balanced in [True, False]:
                for gene in genes:
                    result = evaluate(split, style, gene, balanced)
                    row = {k: v for k, v in result.items() if k != "placements"}
                    rows.append(row)
                    if best is None or (result["remaining"], result["score"]) < (best["remaining"], best["score"]):
                        best = result
                        write_placements_csv(OUT / "best_inverse_platform.csv", best["placements"])
                        print("NEW_BEST", json.dumps(row, ensure_ascii=False))
                    write_trials(OUT / "inverse_platform_trials.csv", rows)
    assert best is not None
    summary = {k: v for k, v in best.items() if k != "placements"}
    (OUT / "inverse_platform_best.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("BEST")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
