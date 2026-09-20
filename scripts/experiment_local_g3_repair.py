from __future__ import annotations

"""问题二最终局部修复实验。

输入为逆推标准件平台得到的 298/300 候选方案。该方案只剩少量 G3 易碎件
无法装入，说明总体体积足够但局部支撑槽位碎片化。脚本在指定局部区域内
拆除相关货物及其依赖货物，再按“标准件平台优先、易碎件随后、其余回填”
的顺序重排，最终得到 300/300 可行坐标。
"""

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
OUT = ROOT / "results" / "experiments" / "local_g3_repair"
SOURCE = ROOT / "results" / "experiments" / "inverse_fragile_platform" / "best_inverse_platform.csv"


def read_placements(path: Path) -> list[Placement]:
    """读取已有坐标方案，转换为 Placement 对象。"""
    placements: list[Placement] = []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            placements.append(
                Placement(
                    uid=row["uid"],
                    goods_code=row["goods_code"],
                    label=row["label"],
                    vehicle_id=row["vehicle_id"],
                    x=int(float(row["x_cm"])),
                    y=int(float(row["y_cm"])),
                    z=int(float(row["z_cm"])),
                    length=int(float(row["length_cm"])),
                    width=int(float(row["width_cm"])),
                    height=int(float(row["height_cm"])),
                    weight=float(row["weight_kg"]),
                    support=row["support"],
                )
            )
    return placements


def build_load(placements: list[Placement]) -> dict[str, float]:
    """按当前坐标方案重建每个下层货物的承重记录。"""
    load: dict[str, float] = {}
    seen: list[Placement] = []
    for placement in placements:
        seen.append(placement)
        add_support_load(placement, seen, load)
    return load


def to_item(placement: Placement) -> Item:
    """将已放置货物转回 Item，便于参与局部回填。"""
    return Item(placement.uid, placement.goods_code, placement.label, placement.length, placement.width, placement.height, int(placement.weight))


def missing_items(placements: list[Placement]) -> list[Item]:
    """找出当前方案中尚未装入的货物。"""
    placed = {p.uid for p in placements}
    return [item for item in expand_items() if item.uid not in placed]


def overlaps_xy(p: Placement, x0: int, y0: int, x1: int, y1: int) -> bool:
    """判断货物底面投影是否与给定矩形相交。"""
    return max(p.x, x0) < min(p.xmax, x1) and max(p.y, y0) < min(p.ymax, y1)


def dependent_closure(placements: list[Placement], seed_ids: set[str]) -> set[str] | None:
    """求局部拆除集合的依赖闭包。

    如果拆掉某个下层货物，则所有由它直接或间接支撑的上层货物也必须一起拆掉，
    否则回填前会出现悬空结构。
    """
    selected = set(seed_ids)
    changed = True
    while changed:
        changed = False
        for p in placements:
            if p.uid in selected:
                continue
            if p.support and p.support not in ("", "floor"):
                supporters = set(p.support.split("|"))
                if supporters & selected:
                    selected.add(p.uid)
                    changed = True
    return selected


def region_seed(placements: list[Placement], rect: tuple[int, int, int, int], z_min: int, z_max: int) -> set[str]:
    """选择需要重排的局部空间区域。"""
    x0, y0, x1, y1 = rect
    return {
        p.uid
        for p in placements
        if z_min <= p.z < z_max and overlaps_xy(p, x0, y0, x1, y1) and p.support != "floor"
    }


def item_key(gene: Gene, item: Item) -> tuple:
    """根据遗传搜索得到的权重对回填货物排序。"""
    rank = {code: idx for idx, code in enumerate(gene.goods_order)}
    label_rank = {label: idx for idx, label in enumerate(gene.label_order)}
    return (
        rank.get(item.goods_code, 99),
        label_rank.get(item.label, 99),
        -gene.volume_weight * item.volume - gene.base_weight * item.base_area0 - gene.height_weight * item.height0,
        item.uid,
    )


def pack_into_existing(vehicle_id: str, base: list[Placement], items: list[Item], gene: Gene) -> tuple[list[Placement], list[Item]]:
    """在保留 base 货物的基础上，将拆出的货物重新装回车厢。"""
    vehicle = VEHICLES["V1"]
    placements = base[:]
    current_weight = sum(p.weight for p in placements)
    load = build_load(placements)
    for item in sorted(items, key=lambda it: item_key(gene, it)):
        placed = place_one_item_enhanced(item, vehicle, vehicle_id, placements, current_weight, load, gene.center_points)
        if placed is None:
            continue
        placements.append(placed)
        current_weight += placed.weight
        add_support_load(placed, placements, load)
    placed_ids = {p.uid for p in placements}
    return placements, [item for item in items if item.uid not in placed_ids]


def repair_vehicle(vehicle_id: str, placements: list[Placement], missing_g3: Item) -> dict:
    """对单辆车执行一次局部大邻域修复。"""
    genes = [
        Gene(("G2", "G1", "G3", "G5", "G4"), ("standard", "fragile", "directed"), 1.0, 0.5, 80.0, True),
        Gene(("G1", "G2", "G3", "G5", "G4"), ("standard", "fragile", "directed"), 1.0, 0.4, 60.0, True),
        Gene(("G2", "G3", "G1", "G5", "G4"), ("standard", "fragile", "directed"), 1.0, 0.6, 120.0, True),
        Gene(("G5", "G1", "G2", "G3", "G4"), ("standard", "fragile", "directed"), 0.8, 0.4, 80.0, True),
    ]
    rects = [
        (0, 40, 150, 160),
        (0, 80, 150, 200),
        (140, 0, 230, 120),
        (150, 20, 230, 140),
        (200, 0, 360, 160),
        (260, 0, 420, 160),
        (340, 0, 420, 150),
        (0, 0, 420, 80),
        (0, 80, 420, 160),
    ]
    z_windows = [(130, 218), (100, 218), (120, 218), (150, 218)]
    best: dict | None = None
    by_id = {p.uid: p for p in placements}
    for rect in rects:
        for z_min, z_max in z_windows:
            seed = region_seed(placements, rect, z_min, z_max)
            if not seed:
                continue
            selected = dependent_closure(placements, seed)
            if not selected:
                continue
            removed = [by_id[uid] for uid in selected]
            if len(removed) > 32:
                continue
            base = [p for p in placements if p.uid not in selected]
            if validate(VEHICLES["V1"], base):
                continue
            items = [to_item(p) for p in removed] + [missing_g3]
            for gene in genes:
                candidate, leftover = pack_into_existing(vehicle_id, base, items, gene)
                errors = validate(VEHICLES["V1"], candidate)
                g3_added = any(p.uid == missing_g3.uid for p in candidate)
                row = {
                    "vehicle_id": vehicle_id,
                    "rect": rect,
                    "z_window": (z_min, z_max),
                    "removed_count": len(removed),
                    "placed_count": len(candidate),
                    "leftover_count": len(leftover),
                    "leftover_counts": goods_counts(
                        [
                            Placement(it.uid, it.goods_code, it.label, "", 0, 0, 0, it.length0, it.width0, it.height0, it.weight, "")
                            for it in leftover
                        ]
                    ),
                    "g3_added": g3_added,
                    "validation_error_count": len(errors),
                    "placements": candidate,
                    "leftover": leftover,
                }
                key = (
                    0 if g3_added else 1,
                    len(leftover),
                    len(errors),
                    -len(candidate),
                    len(removed),
                )
                if best is None or key < best["key"]:
                    row["key"] = key
                    best = row
                if g3_added and not leftover and not errors:
                    return row
    assert best is not None
    return best


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    all_placements = read_placements(SOURCE)
    missing = missing_items(all_placements)
    missing_g3 = [item for item in missing if item.goods_code == "G3"]
    by_vehicle: dict[str, list[Placement]] = defaultdict(list)
    for placement in all_placements:
        by_vehicle[placement.vehicle_id].append(placement)

    repaired_by_vehicle: dict[str, list[Placement]] = {}
    reports = []
    for idx, (vehicle_id, placements) in enumerate(sorted(by_vehicle.items())):
        result = repair_vehicle(vehicle_id, placements, missing_g3[idx])
        reports.append({k: v for k, v in result.items() if k not in ("placements", "leftover", "key")})
        repaired_by_vehicle[vehicle_id] = result["placements"]

    combined = [p for group in repaired_by_vehicle.values() for p in group]
    errors = []
    for placements in repaired_by_vehicle.values():
        errors.extend(validate(VEHICLES["V1"], placements))
    stats = [utilization(VEHICLES["V1"], placements) for placements in repaired_by_vehicle.values()]
    summary = {
        "reports": reports,
        "placed": len(combined),
        "remaining": 300 - len(combined),
        "goods_counts": dict(Counter(p.goods_code for p in combined)),
        "avg_volume_utilization": sum(s["volume_utilization"] for s in stats) / len(stats),
        "avg_weight_utilization": sum(s["weight_utilization"] for s in stats) / len(stats),
        "max_support_pressure_kg_per_m2": max_support_pressure(list(repaired_by_vehicle.values())),
        "validation_error_count": len(errors),
        **g3_support_stats(combined),
    }
    write_placements_csv(OUT / "best_local_g3_repair.csv", combined)
    (OUT / "local_g3_repair_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
