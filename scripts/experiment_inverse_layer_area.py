from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from experiment_inverse_fragile_platform import (
    make_floor_g2_platforms,
    platform_cells,
    split_remaining_by_code,
)
from experiment_layer_platform_v1_pair import g3_support_stats, max_support_pressure
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
OUT = ROOT / "results" / "experiments" / "inverse_layer_area"


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


def standard_layer_area(placements: list[Placement], z: int, extra: Placement | None = None) -> int:
    rects: list[tuple[int, int, int, int]] = []
    for placement in placements:
        if placement.label == "standard" and placement.zmax == z:
            rects.append((placement.x, placement.xmax, placement.y, placement.ymax))
    if extra is not None and extra.label == "standard" and extra.zmax == z:
        rects.append((extra.x, extra.xmax, extra.y, extra.ymax))
    return rect_union_area(rects)


def build_load(placements: list[Placement]) -> dict[str, float]:
    load: dict[str, float] = {}
    seen: list[Placement] = []
    for placement in placements:
        seen.append(placement)
        add_support_load(placement, seen, load)
    return load


def item_sort_key(gene: Gene, item: Item) -> tuple:
    rank = {code: idx for idx, code in enumerate(gene.goods_order)}
    label_rank = {label: idx for idx, label in enumerate(gene.label_order)}
    return (
        rank.get(item.goods_code, 99),
        label_rank.get(item.label, 99),
        -gene.volume_weight * item.volume - gene.base_weight * item.base_area0 - gene.height_weight * item.height0,
        item.uid,
    )


def layer_score(candidate: Placement, placements: list[Placement], mode: str) -> tuple:
    if candidate.label == "fragile":
        support_area = standard_layer_area(placements, candidate.z)
        if mode == "strong":
            return (-candidate.z, -support_area, candidate.y, candidate.x, candidate.zmax)
        return (-candidate.z, -0.5 * support_area, candidate.y, candidate.x, candidate.zmax)

    if candidate.label == "standard":
        top_area = standard_layer_area(placements, candidate.zmax, candidate)
        if mode == "strong":
            return (-candidate.zmax, -top_area, candidate.z, candidate.y, candidate.x)
        if mode == "moderate":
            return (-0.6 * candidate.zmax, -top_area, candidate.z, candidate.y, candidate.x)
        return (-top_area, -candidate.zmax, candidate.z, candidate.y, candidate.x)

    # Directed goods are the main low-level volume skeleton, so keep them compact.
    return (candidate.z, candidate.y, candidate.x, candidate.zmax, candidate.ymax, candidate.xmax)


def place_one_layer_area(
    item: Item,
    vehicle_id: str,
    placements: list[Placement],
    current_weight: float,
    load: dict[str, float],
    gene: Gene,
    mode: str,
) -> Placement | None:
    vehicle = VEHICLES["V1"]
    best: Placement | None = None
    best_support: str | None = None
    best_score: tuple | None = None
    for dims in orientations(item):
        for point in candidate_points_for_dims(dims, placements, vehicle, gene.center_points):
            candidate = Placement(
                item.uid,
                item.goods_code,
                item.label,
                vehicle_id,
                point[0],
                point[1],
                point[2],
                dims[0],
                dims[1],
                dims[2],
                item.weight,
                "",
            )
            ok, support = can_place(candidate, vehicle, placements, current_weight, load)
            if not ok:
                continue
            score = layer_score(candidate, placements, mode)
            if best_score is None or score < best_score:
                best = candidate
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
    mode: str,
) -> tuple[list[Placement], list[Item]]:
    placements = initial[:]
    current_weight = sum(p.weight for p in placements)
    load = build_load(placements)
    for item in sorted(remaining, key=lambda it: item_sort_key(gene, it)):
        placed = place_one_layer_area(item, vehicle_id, placements, current_weight, load, gene, mode)
        if placed is None:
            continue
        placements.append(placed)
        current_weight += placed.weight
        add_support_load(placed, placements, load)
    placed_ids = {p.uid for p in placements}
    return placements, [it for it in remaining if it.uid not in placed_ids]


def layer_metrics(placements: list[Placement]) -> dict[str, float]:
    areas = {
        z: standard_layer_area(placements, z)
        for z in sorted({p.zmax for p in placements if p.label == "standard"})
    }
    high = {z: area for z, area in areas.items() if z >= 100}
    return {
        "max_standard_layer_z": max(areas) if areas else 0,
        "max_standard_layer_area_cm2": max(areas.values()) if areas else 0,
        "max_high_standard_layer_area_cm2": max(high.values()) if high else 0,
    }


def evaluate(split: tuple[int, int], style: str, gene: Gene, balanced: bool, mode: str) -> dict:
    items_by_code: dict[str, list[Item]] = defaultdict(list)
    for item in expand_items():
        items_by_code[item.goods_code].append(item)

    first_initial = make_floor_g2_platforms("layerarea-V1-1", split[0], items_by_code, style)
    second_initial = make_floor_g2_platforms("layerarea-V1-2", split[1], items_by_code, style)

    if balanced:
        rem_a, rem_b = split_remaining_by_code(items_by_code, first_initial, second_initial)
        first, left_a = pack_remaining("layerarea-V1-1", first_initial, rem_a, gene, mode)
        second, left_b = pack_remaining("layerarea-V1-2", second_initial, rem_b, gene, mode)
        second, repair_a = pack_remaining("layerarea-V1-2", second, left_a, gene, mode)
        first, repair_b = pack_remaining("layerarea-V1-1", first, left_b, gene, mode)
        remaining = repair_a + repair_b
    else:
        remaining_items = [item for values in items_by_code.values() for item in values]
        first, rem1 = pack_remaining("layerarea-V1-1", first_initial, remaining_items, gene, mode)
        second, remaining = pack_remaining("layerarea-V1-2", second_initial, rem1, gene, mode)

    all_placements = first + second
    errors = validate(VEHICLES["V1"], first) + validate(VEHICLES["V1"], second)
    stats_first = utilization(VEHICLES["V1"], first)
    stats_second = utilization(VEHICLES["V1"], second)
    metrics_first = layer_metrics(first)
    metrics_second = layer_metrics(second)
    pressure = max_support_pressure([first, second])
    score = (
        len(remaining) * 1000
        + abs(len(first) - len(second)) * 2
        + abs(stats_first["volume_utilization"] - stats_second["volume_utilization"]) * 100
        - (metrics_first["max_high_standard_layer_area_cm2"] + metrics_second["max_high_standard_layer_area_cm2"]) / 10000
    )
    return {
        "split": split,
        "style": style,
        "balanced": balanced,
        "mode": mode,
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
        "avg_volume_utilization": (stats_first["volume_utilization"] + stats_second["volume_utilization"]) / 2,
        "avg_weight_utilization": (stats_first["weight_utilization"] + stats_second["weight_utilization"]) / 2,
        "max_support_pressure_kg_per_m2": pressure,
        "validation_error_count": len(errors),
        "first_layer_metrics": metrics_first,
        "second_layer_metrics": metrics_second,
        **g3_support_stats(all_placements),
        "placements": all_placements,
    }


def write_trials(path: Path, rows: list[dict]) -> None:
    fields = [
        "split",
        "style",
        "balanced",
        "mode",
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
        Gene(("G5", "G4", "G2", "G1", "G3"), ("standard", "directed", "fragile"), 0.8, 0.4, 80.0, True),
    ]
    rows: list[dict] = []
    best: dict | None = None
    # Center on the combinations that produced the previous 298/300 result.
    cases = [
        ((2, 2), "checker", True),
        ((2, 2), "front_rows", True),
        ((2, 2), "side_rows", True),
        ((4, 4), "side_rows", False),
        ((6, 8), "side_rows", False),
    ]
    for split, style, balanced in cases:
        for mode in ["moderate", "strong", "area_first"]:
            for gene in genes:
                result = evaluate(split, style, gene, balanced, mode)
                row = {k: v for k, v in result.items() if k != "placements"}
                rows.append(row)
                write_trials(OUT / "inverse_layer_area_trials.csv", rows)
                if best is None or (result["remaining"], result["score"]) < (best["remaining"], best["score"]):
                    best = result
                    write_placements_csv(OUT / "best_inverse_layer_area.csv", best["placements"])
                    print("NEW_BEST", json.dumps(row, ensure_ascii=False))
    assert best is not None
    summary = {k: v for k, v in best.items() if k != "placements"}
    (OUT / "inverse_layer_area_best.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("BEST")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
