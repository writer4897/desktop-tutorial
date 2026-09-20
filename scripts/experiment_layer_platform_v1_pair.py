from __future__ import annotations

import csv
import json
import time
from collections import defaultdict
from pathlib import Path

from experiment_v1_pair_sa import Gene, item_key, place_one_item_enhanced
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
OUT = ROOT / "results" / "experiments" / "layer_platform_v1_pair"


def pop_item(items_by_code: dict[str, list[Item]], code: str) -> Item:
    if not items_by_code[code]:
        raise ValueError(f"not enough {code}")
    return items_by_code[code].pop(0)


def g4_cells(style: str) -> list[tuple[int, int]]:
    xs = [0, 80, 160, 240, 320]
    ys = [0, 60, 120]
    cells = [(x, y) for y in ys for x in xs]
    if style == "reverse":
        return list(reversed(cells))
    if style == "columns":
        return [(x, y) for x in xs for y in ys]
    return cells


def make_g4_g2_g3_platforms(
    vehicle_id: str,
    count: int,
    items_by_code: dict[str, list[Item]],
    style: str,
) -> list[Placement]:
    placements: list[Placement] = []
    for idx, (x, y) in enumerate(g4_cells(style)[:count]):
        g4 = pop_item(items_by_code, "G4")
        g2a = pop_item(items_by_code, "G2")
        g2b = pop_item(items_by_code, "G2")
        g3 = pop_item(items_by_code, "G3")

        # G4 is a directed bottom module. Two G2 boxes on top create a 70x50
        # standard support platform, and G3 is placed on that standard platform.
        placements.append(Placement(g4.uid, g4.goods_code, g4.label, vehicle_id, x, y, 0, 80, 60, 50, g4.weight, "floor"))
        placements.append(Placement(g2a.uid, g2a.goods_code, g2a.label, vehicle_id, x, y, 50, 35, 50, 25, g2a.weight, g4.uid))
        placements.append(Placement(g2b.uid, g2b.goods_code, g2b.label, vehicle_id, x + 35, y, 50, 35, 50, 25, g2b.weight, g4.uid))
        placements.append(
            Placement(g3.uid, g3.goods_code, g3.label, vehicle_id, x, y, 75, 70, 50, 40, g3.weight, f"{g2a.uid}|{g2b.uid}")
        )
    return placements


def build_load(placements: list[Placement]) -> dict[str, float]:
    load: dict[str, float] = {}
    seen: list[Placement] = []
    for placement in placements:
        seen.append(placement)
        add_support_load(placement, seen, load)
    return load


def pack_remaining(vehicle_id: str, initial: list[Placement], remaining: list[Item], gene: Gene) -> tuple[list[Placement], list[Item]]:
    vehicle = VEHICLES["V1"]
    placements = initial[:]
    current_weight = sum(p.weight for p in placements)
    load = build_load(placements)
    for item in sorted(remaining, key=lambda it: item_key(gene, it)):
        placed = place_one_item_enhanced(item, vehicle, vehicle_id, placements, current_weight, load, gene.center_points)
        if placed is None:
            continue
        placements.append(placed)
        current_weight += placed.weight
        add_support_load(placed, placements, load)
    placed_ids = {p.uid for p in placements}
    return placements, [it for it in remaining if it.uid not in placed_ids]


def max_support_pressure(placements_by_vehicle: list[list[Placement]]) -> float:
    best = 0.0
    for placements in placements_by_vehicle:
        by_id = {p.uid: p for p in placements}
        loads: dict[str, float] = defaultdict(float)
        for p in placements:
            if p.support in ("", "floor"):
                continue
            ids = p.support.split("|")
            if len(ids) == 1:
                loads[ids[0]] += p.weight
                continue
            total_area = p.length * p.width
            for sid in ids:
                support = by_id[sid]
                x0, x1 = max(p.x, support.x), min(p.xmax, support.xmax)
                y0, y1 = max(p.y, support.y), min(p.ymax, support.ymax)
                if x0 < x1 and y0 < y1:
                    loads[sid] += p.weight * ((x1 - x0) * (y1 - y0)) / total_area
        for sid, load in loads.items():
            best = max(best, load / by_id[sid].top_area_m2)
    return best


def g3_support_stats(placements: list[Placement]) -> dict[str, int]:
    g3 = [p for p in placements if p.goods_code == "G3"]
    return {
        "g3_total": len(g3),
        "g3_floor": sum(1 for p in g3 if p.support == "floor" or p.z == 0),
        "g3_platform": sum(1 for p in g3 if p.support not in ("", "floor") and p.z > 0),
    }


def evaluate(split: tuple[int, int], style: str, gene: Gene) -> dict:
    items_by_code: dict[str, list[Item]] = defaultdict(list)
    for item in expand_items():
        items_by_code[item.goods_code].append(item)

    first_initial = make_g4_g2_g3_platforms("layer-V1-1", split[0], items_by_code, style)
    second_initial = make_g4_g2_g3_platforms("layer-V1-2", split[1], items_by_code, style)
    remaining_items = [item for values in items_by_code.values() for item in values]
    first, rem1 = pack_remaining("layer-V1-1", first_initial, remaining_items, gene)
    second, rem2 = pack_remaining("layer-V1-2", second_initial, rem1, gene)
    errors = validate(VEHICLES["V1"], first) + validate(VEHICLES["V1"], second)
    stats1 = utilization(VEHICLES["V1"], first)
    stats2 = utilization(VEHICLES["V1"], second)
    all_placements = first + second
    g3_stats = g3_support_stats(all_placements)
    return {
        "split": split,
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
        "remaining": len(rem2),
        "remaining_counts": goods_counts(
            [Placement(it.uid, it.goods_code, it.label, "", 0, 0, 0, it.length0, it.width0, it.height0, it.weight, "") for it in rem2]
        ),
        "first_count": len(first),
        "second_count": len(second),
        "avg_volume_utilization": (stats1["volume_utilization"] + stats2["volume_utilization"]) / 2,
        "avg_weight_utilization": (stats1["weight_utilization"] + stats2["weight_utilization"]) / 2,
        "first_volume_utilization": stats1["volume_utilization"],
        "second_volume_utilization": stats2["volume_utilization"],
        "max_support_pressure_kg_per_m2": max_support_pressure([first, second]),
        "validation_error_count": len(errors),
        **g3_stats,
        "placements": all_placements,
    }


def write_summary_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "split",
        "style",
        "placed",
        "remaining",
        "remaining_counts",
        "avg_volume_utilization",
        "avg_weight_utilization",
        "max_support_pressure_kg_per_m2",
        "g3_total",
        "g3_floor",
        "g3_platform",
        "validation_error_count",
    ]
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
    ]
    # Representative layer-platform allocations. Each number is how many
    # G4-G2-G3 platform modules are prebuilt in vehicle 1 and vehicle 2.
    splits = [(0, 0), (6, 6), (6, 7), (7, 6), (8, 6), (6, 8)]
    styles = ["rows"]

    best: dict | None = None
    rows = []
    start = time.perf_counter()
    for split in splits:
        for style in styles:
            for gene in genes:
                result = evaluate(split, style, gene)
                row = {k: v for k, v in result.items() if k != "placements"}
                rows.append(row)
                write_summary_csv(OUT / "trials.csv", rows)
                if best is None or (result["remaining"], -result["placed"]) < (best["remaining"], -best["placed"]):
                    best = result
                    print("NEW_BEST", json.dumps(row, ensure_ascii=False))
                    write_placements_csv(OUT / "best_layer_platform_v1_pair.csv", best["placements"])
                if result["remaining"] == 0:
                    break
            if best is not None and best["remaining"] == 0:
                break
        if best is not None and best["remaining"] == 0:
            break

    assert best is not None
    write_placements_csv(OUT / "best_layer_platform_v1_pair.csv", best["placements"])
    write_summary_csv(OUT / "trials.csv", rows)
    summary = {k: v for k, v in best.items() if k != "placements"}
    summary["elapsed_seconds"] = time.perf_counter() - start
    summary["trial_count"] = len(rows)
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("BEST")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
