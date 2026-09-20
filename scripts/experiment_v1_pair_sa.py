from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from pack_problem1 import (
    Item,
    Placement,
    VEHICLES,
    add_support_load,
    can_place,
    expand_items,
    generate_points,
    goods_counts,
    orientations,
    utilization,
    validate,
    write_placements_csv,
)


ROOT = Path.cwd()
if not (ROOT / "scripts").exists() and (ROOT.parent / "scripts").exists():
    ROOT = ROOT.parent
OUT = ROOT / "results" / "experiments" / "v1_pair_sa"


@dataclass(frozen=True)
class Gene:
    goods_order: tuple[str, ...]
    label_order: tuple[str, ...]
    volume_weight: float
    base_weight: float
    height_weight: float
    center_points: bool


def candidate_points_for_dims(
    dims: tuple[int, int, int],
    placements: list[Placement],
    vehicle,
    center_points: bool,
) -> list[tuple[int, int, int]]:
    points = set(generate_points(placements, vehicle, limit=500))
    if center_points:
        length, width, _ = dims
        for support in placements:
            x = round(support.cx - length / 2)
            y = round(support.cy - width / 2)
            z = support.zmax
            if 0 <= x <= vehicle.length and 0 <= y <= vehicle.width and 0 <= z <= vehicle.usable_height:
                points.add((x, y, z))
    return sorted(points, key=lambda p: (p[2], p[1], p[0]))


def candidate_score(candidate: Placement, vehicle) -> tuple:
    touches = int(candidate.x == 0) + int(candidate.y == 0) + int(candidate.z == 0)
    touches += int(candidate.xmax == vehicle.length) + int(candidate.ymax == vehicle.width)
    return (candidate.z, candidate.y, candidate.x, -touches, candidate.zmax, candidate.ymax, candidate.xmax)


def place_one_item_enhanced(
    item: Item,
    vehicle,
    vehicle_id: str,
    placements: list[Placement],
    current_weight: int,
    load_on_support: dict[str, float],
    center_points: bool,
) -> Placement | None:
    best: Placement | None = None
    best_support: str | None = None
    best_score: tuple | None = None
    for dims in orientations(item):
        for point in candidate_points_for_dims(dims, placements, vehicle, center_points):
            cand = Placement(
                uid=item.uid,
                goods_code=item.goods_code,
                label=item.label,
                vehicle_id=vehicle_id,
                x=point[0],
                y=point[1],
                z=point[2],
                length=dims[0],
                width=dims[1],
                height=dims[2],
                weight=item.weight,
                support="",
            )
            ok, support = can_place(cand, vehicle, placements, current_weight, load_on_support)
            if not ok:
                continue
            score = candidate_score(cand, vehicle)
            if best_score is None or score < best_score:
                best = cand
                best_support = support
                best_score = score
    if best is not None:
        best.support = best_support or ""
    return best


def item_key(gene: Gene, item: Item) -> tuple:
    goods_rank = {code: idx for idx, code in enumerate(gene.goods_order)}
    label_rank = {label: idx for idx, label in enumerate(gene.label_order)}
    score = (
        goods_rank.get(item.goods_code, 99),
        label_rank.get(item.label, 99),
        -gene.volume_weight * item.volume - gene.base_weight * item.base_area0 - gene.height_weight * item.height0,
        item.uid,
    )
    return score


def pack_vehicle_gene(items: list[Item], gene: Gene, vehicle_id: str) -> tuple[list[Placement], list[Item]]:
    vehicle = VEHICLES["V1"]
    placements: list[Placement] = []
    current_weight = 0
    load_on_support: dict[str, float] = {}
    for item in sorted(items, key=lambda it: item_key(gene, it)):
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
    remaining = [item for item in items if item.uid not in placed_ids]
    return placements, remaining


def evaluate(gene: Gene) -> dict:
    items = expand_items()
    first, rem1 = pack_vehicle_gene(items, gene, "sa-V1-1")
    second, rem2 = pack_vehicle_gene(rem1, gene, "sa-V1-2")
    errors = validate(VEHICLES["V1"], first) + validate(VEHICLES["V1"], second)
    return {
        "gene": asdict(gene),
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


def random_gene(rng: random.Random) -> Gene:
    goods = ["G1", "G2", "G3", "G4", "G5"]
    labels = ["standard", "fragile", "directed"]
    rng.shuffle(goods)
    rng.shuffle(labels)
    return Gene(
        goods_order=tuple(goods),
        label_order=tuple(labels),
        volume_weight=rng.uniform(0.5, 1.5),
        base_weight=rng.uniform(0.0, 0.8),
        height_weight=rng.uniform(0.0, 900.0),
        center_points=rng.random() < 0.7,
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(2607662)
    seeds = [
        Gene(("G4", "G5", "G1", "G2", "G3"), ("directed", "standard", "fragile"), 1.0, 0.2, 0.0, False),
        Gene(("G2", "G3", "G4", "G5", "G1"), ("standard", "fragile", "directed"), 1.0, 0.4, 0.0, True),
        Gene(("G3", "G2", "G4", "G5", "G1"), ("fragile", "standard", "directed"), 1.0, 0.4, 0.0, True),
        Gene(("G4", "G5", "G2", "G3", "G1"), ("directed", "standard", "fragile"), 1.0, 0.4, 0.0, True),
    ]
    genes = seeds + [random_gene(rng) for _ in range(16)]
    best: dict | None = None
    history = []
    start = time.perf_counter()
    for idx, gene in enumerate(genes):
        result = evaluate(gene)
        row = {k: v for k, v in result.items() if k != "placements"}
        row["trial"] = idx
        history.append(row)
        if best is None or (result["remaining"], -result["placed"]) < (best["remaining"], -best["placed"]):
            best = result
        print(json.dumps(row, ensure_ascii=False))
        if result["remaining"] == 0:
            break
    assert best is not None
    write_placements_csv(OUT / "best_v1_pair.csv", best["placements"])
    summary = {
        k: v for k, v in best.items() if k != "placements"
    }
    summary["elapsed_seconds"] = time.perf_counter() - start
    summary["history"] = history
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("BEST")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
