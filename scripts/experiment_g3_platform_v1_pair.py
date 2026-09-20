from __future__ import annotations

import json
import random
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
OUT = ROOT / "results" / "experiments" / "g3_platform_v1_pair"


def pop_items(items_by_code: dict[str, list[Item]], code: str, count: int) -> list[Item]:
    if len(items_by_code[code]) < count:
        raise ValueError(f"not enough {code}: need {count}, have {len(items_by_code[code])}")
    selected = items_by_code[code][:count]
    del items_by_code[code][:count]
    return selected


def platform_cells(style: str) -> list[tuple[int, int]]:
    xs = list(range(0, 420, 70))
    ys = list(range(0, 200, 50))
    if style == "rows":
        return [(x, y) for y in ys for x in xs]
    if style == "cols":
        return [(x, y) for x in xs for y in ys]
    if style == "checker":
        cells = [(x, y) for y in ys for x in xs]
        return sorted(cells, key=lambda p: ((p[0] // 70 + p[1] // 50) % 2, p[1], p[0]))
    raise ValueError(style)


def make_g3_platforms(
    vehicle_id: str,
    count: int,
    items_by_code: dict[str, list[Item]],
    style: str,
) -> list[Placement]:
    g3_items = pop_items(items_by_code, "G3", count)
    g2_items = pop_items(items_by_code, "G2", 2 * count)
    placements: list[Placement] = []
    cells = platform_cells(style)
    for idx in range(count):
        x, y = cells[idx]
        left = g2_items[2 * idx]
        right = g2_items[2 * idx + 1]
        placements.append(
            Placement(left.uid, left.goods_code, left.label, vehicle_id, x, y, 0, 35, 50, 25, left.weight, "floor")
        )
        placements.append(
            Placement(right.uid, right.goods_code, right.label, vehicle_id, x + 35, y, 0, 35, 50, 25, right.weight, "floor")
        )
        fragile = g3_items[idx]
        placements.append(
            Placement(
                fragile.uid,
                fragile.goods_code,
                fragile.label,
                vehicle_id,
                x,
                y,
                25,
                70,
                50,
                40,
                fragile.weight,
                f"{left.uid}|{right.uid}",
            )
        )
    return placements


def build_load(placements: list[Placement]) -> dict[str, float]:
    load: dict[str, float] = {}
    seen: list[Placement] = []
    for placement in placements:
        seen.append(placement)
        add_support_load(placement, seen, load)
    return load


def pack_remaining(
    vehicle_id: str,
    initial: list[Placement],
    remaining_items: list[Item],
    gene: Gene,
) -> tuple[list[Placement], list[Item]]:
    vehicle = VEHICLES["V1"]
    placements = initial[:]
    current_weight = sum(p.weight for p in placements)
    load_on_support = build_load(placements)
    for item in sorted(remaining_items, key=lambda it: item_key(gene, it)):
        placed = place_one_item_enhanced(
            item,
            vehicle,
            vehicle_id,
            placements,
            current_weight,
            load_on_support,
            gene.center_points,
        )
        if placed is None:
            continue
        placements.append(placed)
        current_weight += placed.weight
        add_support_load(placed, placements, load_on_support)
    placed_ids = {p.uid for p in placements}
    return placements, [it for it in remaining_items if it.uid not in placed_ids]


def evaluate(split: tuple[int, int], style: str, gene: Gene) -> dict:
    items_by_code: dict[str, list[Item]] = defaultdict(list)
    for item in expand_items():
        items_by_code[item.goods_code].append(item)

    first_initial = make_g3_platforms("platform-V1-1", split[0], items_by_code, style)
    second_initial = make_g3_platforms("platform-V1-2", split[1], items_by_code, style)
    remaining_items = [item for values in items_by_code.values() for item in values]

    first, rem1 = pack_remaining("platform-V1-1", first_initial, remaining_items, gene)
    second, rem2 = pack_remaining("platform-V1-2", second_initial, rem1, gene)
    errors = validate(VEHICLES["V1"], first) + validate(VEHICLES["V1"], second)
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
        "placed": len(first) + len(second),
        "remaining": len(rem2),
        "remaining_counts": goods_counts(
            [Placement(it.uid, it.goods_code, it.label, "", 0, 0, 0, it.length0, it.width0, it.height0, it.weight, "") for it in rem2]
        ),
        "first_count": len(first),
        "second_count": len(second),
        "first_volume_utilization": utilization(VEHICLES["V1"], first)["volume_utilization"],
        "second_volume_utilization": utilization(VEHICLES["V1"], second)["volume_utilization"],
        "validation_error_count": len(errors),
        "placements": first + second,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(2607662)
    genes = [
        Gene(("G4", "G5", "G1", "G2", "G3"), ("directed", "standard", "fragile"), 1.0, 0.2, 0.0, True),
        Gene(("G5", "G4", "G2", "G1", "G3"), ("standard", "fragile", "directed"), 0.8, 0.4, 100.0, True),
        Gene(("G4", "G5", "G2", "G1", "G3"), ("directed", "standard", "fragile"), 1.0, 0.5, 200.0, True),
    ]
    for _ in range(2):
        goods = ["G1", "G2", "G4", "G5"]
        rng.shuffle(goods)
        goods.append("G3")
        labels = ["standard", "directed", "fragile"]
        rng.shuffle(labels)
        genes.append(
            Gene(tuple(goods), tuple(labels), rng.uniform(0.5, 1.5), rng.uniform(0.0, 0.8), rng.uniform(0.0, 700.0), True)
        )

    trials = []
    best: dict | None = None
    start = time.perf_counter()
    splits = [(15, 15), (12, 18), (18, 12)]
    styles = ["rows"]
    for split in splits:
        for style in styles:
            for gene in genes:
                result = evaluate(split, style, gene)
                row = {k: v for k, v in result.items() if k != "placements"}
                trials.append(row)
                if best is None or (result["remaining"], -result["placed"]) < (best["remaining"], -best["placed"]):
                    best = result
                    print("NEW_BEST", json.dumps(row, ensure_ascii=False))
                if result["remaining"] == 0:
                    break
            if best is not None and best["remaining"] == 0:
                break
        if best is not None and best["remaining"] == 0:
            break

    assert best is not None
    write_placements_csv(OUT / "best_platform_v1_pair.csv", best["placements"])
    summary = {k: v for k, v in best.items() if k != "placements"}
    summary["elapsed_seconds"] = time.perf_counter() - start
    summary["trial_count"] = len(trials)
    summary["trials"] = trials
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("BEST")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
