from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from pack_problem1 import (
    GOODS_BY_CODE,
    Item,
    Placement,
    VEHICLES,
    add_support_load,
    can_place,
    expand_items,
    goods_counts,
    place_one_item,
    utilization,
    validate,
    write_placements_csv,
)


ROOT = Path.cwd()
if not (ROOT / "scripts").exists() and (ROOT.parent / "scripts").exists():
    ROOT = ROOT.parent
OUT = ROOT / "results" / "experiments" / "v2_platform_check"


def pop_item(items_by_code: dict[str, list[Item]], code: str) -> Item:
    return items_by_code[code].pop()


def build_g3_platform(vehicle_id: str, items_by_code: dict[str, list[Item]], cols: int, rows: int) -> list[Placement]:
    """Use two G2 items as a 70x50 support cell, then place one G3 on it."""
    placements: list[Placement] = []
    g3_count = GOODS_BY_CODE["G3"].quantity
    made = 0
    for row in range(rows):
        for col in range(cols):
            if made >= g3_count:
                return placements
            x = col * 70
            y = row * 50
            if x + 70 > VEHICLES["V2"].length or y + 50 > VEHICLES["V2"].width:
                continue
            left = pop_item(items_by_code, "G2")
            right = pop_item(items_by_code, "G2")
            g3 = pop_item(items_by_code, "G3")
            p1 = Placement(left.uid, "G2", "standard", vehicle_id, x, y, 0, 35, 50, 25, left.weight, "floor")
            p2 = Placement(right.uid, "G2", "standard", vehicle_id, x + 35, y, 0, 35, 50, 25, right.weight, "floor")
            p3 = Placement(g3.uid, "G3", "fragile", vehicle_id, x, y, 25, 70, 50, 40, g3.weight, f"{p1.uid}|{p2.uid}")
            placements.extend([p1, p2, p3])
            made += 1
    return placements


def sort_remaining(items: list[Item], strategy: str) -> list[Item]:
    def volume(item: Item) -> int:
        return item.length0 * item.width0 * item.height0

    if strategy == "directed_first":
        order = {"directed": 0, "standard": 1, "fragile": 2}
        return sorted(items, key=lambda it: (order[it.label], -volume(it), -it.weight, it.uid))
    if strategy == "volume":
        return sorted(items, key=lambda it: (-volume(it), -it.weight, it.uid))
    if strategy == "base":
        return sorted(items, key=lambda it: (-(it.length0 * it.width0), -volume(it), it.uid))
    if strategy == "small_first":
        return sorted(items, key=lambda it: (volume(it), it.uid))
    raise ValueError(strategy)


def repack_remaining(vehicle_id: str, initial: list[Placement], remaining: list[Item], strategy: str):
    vehicle = VEHICLES["V2"]
    placements = initial[:]
    load: dict[str, float] = {}
    for p in placements:
        add_support_load(p, placements, load)
    current_weight = sum(p.weight for p in placements)
    leftover: list[Item] = []
    for item in sort_remaining(remaining, strategy):
        placed = place_one_item(item, vehicle, vehicle_id, placements, current_weight, load)
        if placed is None:
            leftover.append(item)
            continue
        placements.append(placed)
        current_weight += placed.weight
        add_support_load(placed, placements, load)
    return placements, leftover


def run_case(cols: int, rows: int, strategy: str):
    items_by_code: dict[str, list[Item]] = defaultdict(list)
    for item in expand_items():
        items_by_code[item.goods_code].append(item)
    initial = build_g3_platform("v2-platform-1", items_by_code, cols, rows)
    remaining = [item for values in items_by_code.values() for item in values]
    placements, leftover = repack_remaining("v2-platform-1", initial, remaining, strategy)
    errors = validate(VEHICLES["V2"], placements)
    stats = utilization(VEHICLES["V2"], placements)
    return {
        "cols": cols,
        "rows": rows,
        "strategy": strategy,
        "placed": len(placements),
        "leftover": len(leftover),
        "leftover_counts": {k: v for k, v in sorted(goods_counts([Placement(i.uid, i.goods_code, i.label, "", 0, 0, 0, i.length0, i.width0, i.height0, i.weight, "") for i in leftover]).items())},
        "validation_error_count": len(errors),
        "errors": errors[:8],
        "stats": stats,
        "goods_counts": goods_counts(placements),
        "placements": placements,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    best = None
    rows = []
    for cols, rows_count in [(6, 5), (7, 5), (8, 4), (9, 4), (10, 3)]:
        for strategy in ["directed_first", "volume", "base", "small_first"]:
            result = run_case(cols, rows_count, strategy)
            row = {k: v for k, v in result.items() if k != "placements"}
            rows.append(row)
            if best is None or (result["leftover"], -result["placed"], -result["stats"]["volume_utilization"]) < (
                best["leftover"],
                -best["placed"],
                -best["stats"]["volume_utilization"],
            ):
                best = result
                write_placements_csv(OUT / "best_v2_platform.csv", result["placements"])
                print("NEW_BEST", json.dumps(row, ensure_ascii=False))
    assert best is not None
    summary = {k: v for k, v in best.items() if k != "placements"}
    (OUT / "best_v2_platform_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
