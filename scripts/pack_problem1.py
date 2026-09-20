from __future__ import annotations

"""问题一基础装箱模块。

本文件负责把题目数据转化为单件货物，采用约束感知极点法生成坐标方案，
并提供统一的可行性校验函数。后续问题二、问题三和局部修复实验均复用
这里的车辆、货物、坐标、支撑和承重定义。
"""

import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "problem1"


@dataclass(frozen=True)
class VehicleType:
    """车辆参数。尺寸单位为 cm，载重单位为 kg，费用单位为元/次。"""
    code: str
    name: str
    length: int
    width: int
    height: int
    capacity: int
    cost: int

    @property
    def usable_height(self) -> int:
        # 题目要求顶部预留 3 cm 安全间隙，因此有效装载高度比车厢高度低 3 cm。
        return self.height - 3

    @property
    def volume(self) -> int:
        return self.length * self.width * self.usable_height


@dataclass(frozen=True)
class GoodsType:
    """题目给出的货物类别参数，quantity 表示该类货物件数。"""
    code: str
    label: str
    length: int
    width: int
    height: int
    weight: int
    quantity: int

    @property
    def volume(self) -> int:
        return self.length * self.width * self.height


@dataclass(frozen=True)
class Item:
    """展开后的单件货物，用 uid 区分同一类别下的不同个体。"""
    uid: str
    goods_code: str
    label: str
    length0: int
    width0: int
    height0: int
    weight: int

    @property
    def volume(self) -> int:
        return self.length0 * self.width0 * self.height0

    @property
    def base_area0(self) -> int:
        return self.length0 * self.width0


@dataclass
class Placement:
    """一件货物的实际装载记录，包括坐标、姿态尺寸和支撑对象。"""
    uid: str
    goods_code: str
    label: str
    vehicle_id: str
    x: int
    y: int
    z: int
    length: int
    width: int
    height: int
    weight: int
    support: str

    @property
    def xmax(self) -> int:
        return self.x + self.length

    @property
    def ymax(self) -> int:
        return self.y + self.width

    @property
    def zmax(self) -> int:
        return self.z + self.height

    @property
    def cx(self) -> float:
        return self.x + self.length / 2

    @property
    def cy(self) -> float:
        return self.y + self.width / 2

    @property
    def top_area_m2(self) -> float:
        return (self.length * self.width) / 10000


VEHICLES = {
    "V1": VehicleType("V1", "车型1", 420, 210, 220, 6000, 450),
    "V2": VehicleType("V2", "车型2", 680, 245, 250, 10000, 700),
}

GOODS = [
    GoodsType("G1", "standard", 60, 40, 30, 12, 80),
    GoodsType("G2", "standard", 50, 35, 25, 8, 100),
    GoodsType("G3", "fragile", 70, 50, 40, 15, 30),
    GoodsType("G4", "directed", 80, 60, 50, 25, 40),
    GoodsType("G5", "directed", 40, 40, 60, 18, 50),
]


def expand_items() -> list[Item]:
    """将 G1--G5 按数量展开为 300 件可独立分配的货物。"""
    items: list[Item] = []
    for g in GOODS:
        for idx in range(1, g.quantity + 1):
            items.append(
                Item(
                    uid=f"{g.code}-{idx:03d}",
                    goods_code=g.code,
                    label=g.label,
                    length0=g.length,
                    width0=g.width,
                    height0=g.height,
                    weight=g.weight,
                )
            )
    return items


def unique_permutations(values: tuple[int, int, int]) -> list[tuple[int, int, int]]:
    perms = {
        (values[0], values[1], values[2]),
        (values[0], values[2], values[1]),
        (values[1], values[0], values[2]),
        (values[1], values[2], values[0]),
        (values[2], values[0], values[1]),
        (values[2], values[1], values[0]),
    }
    return sorted(perms)


def orientations(item: Item) -> list[tuple[int, int, int]]:
    """按货物类型返回允许姿态。

    标准件允许六种正交旋转；易碎件只允许底面水平旋转；定向件保持题目给定姿态。
    """
    if item.label == "standard":
        return unique_permutations((item.length0, item.width0, item.height0))
    if item.label == "fragile":
        # 保持高度方向不变，可交换长宽朝向以适配底面支撑。
        return [(item.length0, item.width0, item.height0), (item.width0, item.length0, item.height0)]
    # 定向件仅允许题目给定姿态。
    return [(item.length0, item.width0, item.height0)]


def overlap_1d(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(a0, b0) < min(a1, b1)


def intersects(candidate: Placement, other: Placement) -> bool:
    return (
        overlap_1d(candidate.x, candidate.xmax, other.x, other.xmax)
        and overlap_1d(candidate.y, candidate.ymax, other.y, other.ymax)
        and overlap_1d(candidate.z, candidate.zmax, other.z, other.zmax)
    )


def within_projection_center(candidate: Placement, support: Placement) -> bool:
    return support.x <= candidate.cx <= support.xmax and support.y <= candidate.cy <= support.ymax


def within_projection_full(candidate: Placement, support: Placement) -> bool:
    return (
        support.x <= candidate.x
        and support.y <= candidate.y
        and candidate.xmax <= support.xmax
        and candidate.ymax <= support.ymax
    )


def rects_cover(candidate: Placement, supports: list[Placement]) -> bool:
    """Return True if support top rectangles fully cover candidate's bottom rectangle."""
    xs = {candidate.x, candidate.xmax}
    ys = {candidate.y, candidate.ymax}
    rects: list[tuple[int, int, int, int]] = []
    for p in supports:
        x0, x1 = max(candidate.x, p.x), min(candidate.xmax, p.xmax)
        y0, y1 = max(candidate.y, p.y), min(candidate.ymax, p.ymax)
        if x0 < x1 and y0 < y1:
            rects.append((x0, x1, y0, y1))
            xs.update([x0, x1])
            ys.update([y0, y1])
    xs_sorted = sorted(xs)
    ys_sorted = sorted(ys)
    for xi in range(len(xs_sorted) - 1):
        for yi in range(len(ys_sorted) - 1):
            cx0, cx1 = xs_sorted[xi], xs_sorted[xi + 1]
            cy0, cy1 = ys_sorted[yi], ys_sorted[yi + 1]
            if cx0 == cx1 or cy0 == cy1:
                continue
            mx = (cx0 + cx1) / 2
            my = (cy0 + cy1) / 2
            covered = any(x0 <= mx <= x1 and y0 <= my <= y1 for x0, x1, y0, y1 in rects)
            if not covered:
                return False
    return True


def fragile_support_at(
    candidate: Placement,
    placements: list[Placement],
    load_on_support: dict[str, float],
) -> str | None:
    """检查易碎件是否由车厢底面或若干标准件顶面完整支撑。

    易碎件的底面可能大于单个标准件顶面，因此这里允许多个同高度标准件
    拼接成支撑平台，并按重叠面积分摊承重。
    """
    if candidate.z == 0:
        return "floor"
    supports = [p for p in placements if p.label == "standard" and p.zmax == candidate.z]
    if not supports or not rects_cover(candidate, supports):
        return None
    used: list[str] = []
    for p in supports:
        x0, x1 = max(candidate.x, p.x), min(candidate.xmax, p.xmax)
        y0, y1 = max(candidate.y, p.y), min(candidate.ymax, p.ymax)
        if x0 < x1 and y0 < y1:
            overlap_area = (x1 - x0) * (y1 - y0)
            share = candidate.weight * overlap_area / (candidate.length * candidate.width)
            max_load = 500 * p.top_area_m2
            if load_on_support.get(p.uid, 0.0) + share > max_load + 1e-9:
                return None
            used.append(p.uid)
    return "|".join(sorted(used)) if used else None


def supporter_at(
    candidate: Placement,
    placements: list[Placement],
    load_on_support: dict[str, float],
) -> str | None:
    """返回候选货物的支撑对象；不可支撑时返回 None。"""
    if candidate.z == 0:
        return "floor"
    if candidate.label == "fragile":
        return fragile_support_at(candidate, placements, load_on_support)

    for p in placements:
        if p.zmax != candidate.z:
            continue
        if p.label == "fragile":
            continue
        if not within_projection_center(candidate, p):
            continue

        max_load = 500 * p.top_area_m2
        if load_on_support.get(p.uid, 0.0) + candidate.weight <= max_load + 1e-9:
            return p.uid
    return None


def point_inside_any(point: tuple[int, int, int], placements: list[Placement]) -> bool:
    x, y, z = point
    for p in placements:
        if p.x <= x < p.xmax and p.y <= y < p.ymax and p.z <= z < p.zmax:
            return True
    return False


def generate_points(placements: list[Placement], vehicle: VehicleType, limit: int = 360) -> list[tuple[int, int, int]]:
    """根据已放置货物生成极点候选集。

    每次放置后，在货物右侧、前侧、上侧及组合角点处产生新候选点，
    再删除位于已有货物内部或超出车厢范围的点。
    """
    points = {(0, 0, 0)}
    for p in placements:
        candidates = [
            (p.xmax, p.y, p.z),
            (p.x, p.ymax, p.z),
            (p.x, p.y, p.zmax),
            (p.xmax, p.ymax, p.z),
            (p.xmax, p.y, p.zmax),
            (p.x, p.ymax, p.zmax),
        ]
        for point in candidates:
            x, y, z = point
            if 0 <= x <= vehicle.length and 0 <= y <= vehicle.width and 0 <= z <= vehicle.usable_height:
                points.add(point)
    filtered = [p for p in points if not point_inside_any(p, placements)]
    return sorted(filtered, key=lambda p: (p[2], p[1], p[0]))[:limit]


def candidate_score(candidate: Placement, vehicle: VehicleType) -> tuple:
    # 下、右、后优先，并轻微偏好贴边，减少碎片。
    touches = int(candidate.x == 0) + int(candidate.y == 0) + int(candidate.z == 0)
    touches += int(candidate.xmax == vehicle.length) + int(candidate.ymax == vehicle.width)
    return (candidate.z, candidate.y, candidate.x, -touches, candidate.zmax, candidate.ymax, candidate.xmax)


def can_place(
    candidate: Placement,
    vehicle: VehicleType,
    placements: list[Placement],
    current_weight: int,
    load_on_support: dict[str, float],
) -> tuple[bool, str | None]:
    """完整检查一个候选放置位置是否可行。"""
    if candidate.xmax > vehicle.length or candidate.ymax > vehicle.width or candidate.zmax > vehicle.usable_height:
        return False, None
    if current_weight + candidate.weight > vehicle.capacity:
        return False, None
    for p in placements:
        if intersects(candidate, p):
            return False, None
    support = supporter_at(candidate, placements, load_on_support)
    if support is None:
        return False, None
    return True, support


def place_one_item(
    item: Item,
    vehicle: VehicleType,
    vehicle_id: str,
    placements: list[Placement],
    current_weight: int,
    load_on_support: dict[str, float],
) -> Placement | None:
    """在当前车厢状态中为单件货物选择评分最优的可行极点。"""
    best: Placement | None = None
    best_support: str | None = None
    best_score: tuple | None = None
    for point in generate_points(placements, vehicle):
        for dims in orientations(item):
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


def sort_items(items: Iterable[Item], strategy: str) -> list[Item]:
    """按照不同启发式策略对待装货物排序。"""
    if strategy == "constraint_volume":
        rank = {"fragile": 0, "directed": 1, "standard": 2}
        return sorted(items, key=lambda it: (rank[it.label], -it.volume, -it.base_area0, it.uid))
    if strategy == "volume":
        return sorted(items, key=lambda it: (-it.volume, -it.base_area0, it.uid))
    if strategy == "base":
        return sorted(items, key=lambda it: (-it.base_area0, -it.volume, it.uid))
    if strategy == "height":
        return sorted(items, key=lambda it: (-it.height0, -it.volume, it.uid))
    if strategy == "weight":
        return sorted(items, key=lambda it: (-it.weight, -it.volume, it.uid))
    raise ValueError(strategy)


def pack_single_vehicle(
    vehicle: VehicleType,
    items: list[Item],
    strategy: str,
    vehicle_id: str,
) -> tuple[list[Placement], list[Item]]:
    """用指定排序策略生成单车装载方案。"""
    placements: list[Placement] = []
    remaining: list[Item] = []
    current_weight = 0
    load_on_support: dict[str, float] = {}
    for item in sort_items(items, strategy):
        placed = place_one_item(item, vehicle, vehicle_id, placements, current_weight, load_on_support)
        if placed is None:
            remaining.append(item)
            continue
        placements.append(placed)
        current_weight += placed.weight
        add_support_load(placed, placements, load_on_support)
    return placements, remaining


def add_support_load(placed: Placement, placements: list[Placement], load_on_support: dict[str, float]) -> None:
    """更新下层货物承受的重量。

    单一支撑时整件重量计入支撑件；多标准件共同支撑时按底面重叠面积比例分摊。
    """
    if placed.support in ("", "floor"):
        return
    by_id = {p.uid: p for p in placements}
    support_ids = placed.support.split("|")
    if len(support_ids) == 1:
        sid = support_ids[0]
        load_on_support[sid] = load_on_support.get(sid, 0.0) + placed.weight
        return
    total_area = placed.length * placed.width
    for sid in support_ids:
        support = by_id[sid]
        x0, x1 = max(placed.x, support.x), min(placed.xmax, support.xmax)
        y0, y1 = max(placed.y, support.y), min(placed.ymax, support.ymax)
        if x0 < x1 and y0 < y1:
            share = placed.weight * ((x1 - x0) * (y1 - y0)) / total_area
            load_on_support[sid] = load_on_support.get(sid, 0.0) + share


def utilization(vehicle: VehicleType, placements: list[Placement]) -> dict[str, float]:
    """计算空间利用率、载重利用率和综合满载率。"""
    volume = sum(p.length * p.width * p.height for p in placements)
    weight = sum(p.weight for p in placements)
    uv = volume / vehicle.volume
    uq = weight / vehicle.capacity
    combined = 0.5 * uv + 0.5 * uq - 0.1 * abs(uv - uq)
    return {
        "item_count": len(placements),
        "volume_cm3": volume,
        "weight_kg": weight,
        "volume_utilization": uv,
        "weight_utilization": uq,
        "combined_fullness": combined,
    }


def choose_best_single(vehicle: VehicleType, items: list[Item]) -> tuple[str, list[Placement], list[Item], dict[str, float]]:
    best = None
    for strategy in ["constraint_volume", "volume", "base"]:
        placements, remaining = pack_single_vehicle(vehicle, items, strategy, f"{vehicle.code}-single")
        stats = utilization(vehicle, placements)
        key = (stats["combined_fullness"], stats["volume_utilization"], stats["item_count"])
        if best is None or key > best[0]:
            best = (key, strategy, placements, remaining, stats)
    assert best is not None
    return best[1], best[2], best[3], best[4]


def pack_all_one_type(vehicle: VehicleType, items: list[Item]) -> tuple[str, list[list[Placement]], dict[str, float]]:
    best_result = None
    for strategy in ["constraint_volume", "volume", "base"]:
        unpacked = sort_items(items, strategy)
        bins: list[list[Placement]] = []
        vehicle_no = 1
        while unpacked:
            placements, remaining = pack_single_vehicle(vehicle, unpacked, strategy, f"{vehicle.code}-{vehicle_no}")
            if not placements:
                raise RuntimeError(f"{vehicle.code} cannot place remaining items under {strategy}")
            bins.append(placements)
            placed_ids = {p.uid for p in placements}
            unpacked = [it for it in unpacked if it.uid not in placed_ids]
            vehicle_no += 1
        total_stats = summarize_bins(vehicle, bins)
        key = (-len(bins), total_stats["avg_volume_utilization"], total_stats["avg_combined_fullness"])
        if best_result is None or key > best_result[0]:
            best_result = (key, strategy, bins, total_stats)
    assert best_result is not None
    return best_result[1], best_result[2], best_result[3]


def summarize_bins(vehicle: VehicleType, bins: list[list[Placement]]) -> dict[str, float]:
    stats = [utilization(vehicle, b) for b in bins]
    return {
        "vehicle_count": len(bins),
        "total_items": sum(s["item_count"] for s in stats),
        "total_volume_cm3": sum(s["volume_cm3"] for s in stats),
        "total_weight_kg": sum(s["weight_kg"] for s in stats),
        "avg_volume_utilization": sum(s["volume_utilization"] for s in stats) / len(stats),
        "avg_weight_utilization": sum(s["weight_utilization"] for s in stats) / len(stats),
        "avg_combined_fullness": sum(s["combined_fullness"] for s in stats) / len(stats),
        "total_cost": len(bins) * vehicle.cost,
    }


def validate(vehicle: VehicleType, placements: list[Placement]) -> list[str]:
    """对一辆车的坐标方案进行独立复验。"""
    errors: list[str] = []
    by_id = {p.uid: p for p in placements}
    total_weight = sum(p.weight for p in placements)
    if total_weight > vehicle.capacity:
        errors.append(f"overweight: {total_weight}>{vehicle.capacity}")
    for p in placements:
        if p.x < 0 or p.y < 0 or p.z < 0 or p.xmax > vehicle.length or p.ymax > vehicle.width or p.zmax > vehicle.usable_height:
            errors.append(f"{p.uid}: outside vehicle")
        if p.label == "directed" and (p.length, p.width, p.height) != (GOODS_BY_CODE[p.goods_code].length, GOODS_BY_CODE[p.goods_code].width, GOODS_BY_CODE[p.goods_code].height):
            errors.append(f"{p.uid}: directed orientation violated")
    for i, p in enumerate(placements):
        for q in placements[i + 1 :]:
            if intersects(p, q):
                errors.append(f"{p.uid}/{q.uid}: overlap")
    for p in placements:
        if p.support == "floor":
            if p.z != 0:
                errors.append(f"{p.uid}: floor support but z != 0")
            continue
        if p.label == "fragile":
            support_ids = p.support.split("|") if p.support else []
            supports = [by_id[sid] for sid in support_ids if sid in by_id]
            if len(supports) != len(support_ids):
                errors.append(f"{p.uid}: missing fragile support {p.support}")
                continue
            if any(s.label != "standard" or s.zmax != p.z for s in supports):
                errors.append(f"{p.uid}: fragile support type/height invalid")
            if not rects_cover(p, supports):
                errors.append(f"{p.uid}: fragile bottom not fully covered")
        else:
            support = by_id.get(p.support)
            if support is None:
                errors.append(f"{p.uid}: missing support {p.support}")
                continue
            if support.label == "fragile":
                errors.append(f"{p.uid}: supported by fragile {support.uid}")
            if p.z != support.zmax:
                errors.append(f"{p.uid}: support height mismatch")
            if not within_projection_center(p, support):
                errors.append(f"{p.uid}: center outside support")
    load: dict[str, float] = {}
    for p in placements:
        if p.support not in ("", "floor"):
            if "|" not in p.support:
                load[p.support] = load.get(p.support, 0.0) + p.weight
            else:
                for sid in p.support.split("|"):
                    support = by_id[sid]
                    x0, x1 = max(p.x, support.x), min(p.xmax, support.xmax)
                    y0, y1 = max(p.y, support.y), min(p.ymax, support.ymax)
                    if x0 < x1 and y0 < y1:
                        share = p.weight * ((x1 - x0) * (y1 - y0)) / (p.length * p.width)
                        load[sid] = load.get(sid, 0.0) + share
    for sid, supported_weight in load.items():
        support = by_id[sid]
        max_load = 500 * support.top_area_m2
        if supported_weight > max_load + 1e-9:
            errors.append(f"{sid}: load {supported_weight}>{max_load}")
    return errors


def write_placements_csv(path: Path, placements: list[Placement]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "vehicle_id",
        "uid",
        "goods_code",
        "label",
        "x_cm",
        "y_cm",
        "z_cm",
        "length_cm",
        "width_cm",
        "height_cm",
        "weight_kg",
        "support",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for p in placements:
            writer.writerow(
                {
                    "vehicle_id": p.vehicle_id,
                    "uid": p.uid,
                    "goods_code": p.goods_code,
                    "label": p.label,
                    "x_cm": p.x,
                    "y_cm": p.y,
                    "z_cm": p.z,
                    "length_cm": p.length,
                    "width_cm": p.width,
                    "height_cm": p.height,
                    "weight_kg": p.weight,
                    "support": p.support,
                }
            )


def goods_counts(placements: list[Placement]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for p in placements:
        counts[p.goods_code] = counts.get(p.goods_code, 0) + 1
    return counts


def write_latex_tables(summary: dict) -> None:
    strategy_name = {
        "constraint_volume": "约束-体积",
        "volume": "体积优先",
        "base": "底面积优先",
        "height": "高度优先",
        "weight": "重量优先",
    }
    lines = []
    lines.append("% Auto-generated by scripts/pack_problem1.py")
    lines.append("\\begin{table}[H]")
    lines.append("\\centering")
    lines.append("\\caption{问题一单车装载结果汇总}")
    lines.append("\\label{tab:p1_single_summary}")
    lines.append("\\begin{tabular}{cccccc}")
    lines.append("\\toprule")
    lines.append("车型 & 算法排序 & 装入件数 & 空间利用率 & 载重利用率 & 综合满载率 \\\\")
    lines.append("\\midrule")
    for code in ["V1", "V2"]:
        s = summary["single"][code]
        lines.append(
            f"{VEHICLES[code].name} & {strategy_name.get(s['strategy'], s['strategy'])} & {s['stats']['item_count']} & "
            f"{s['stats']['volume_utilization']*100:.2f}\\% & "
            f"{s['stats']['weight_utilization']*100:.2f}\\% & "
            f"{s['stats']['combined_fullness']*100:.2f}\\% \\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")
    lines.append("\\begin{table}[H]")
    lines.append("\\centering")
    lines.append("\\caption{问题一单车型完成全部货物的车辆数结果}")
    lines.append("\\label{tab:p1_all_summary}")
    lines.append("\\begin{tabular}{cccccc}")
    lines.append("\\toprule")
    lines.append("车型 & 算法排序 & 使用车辆数 & 平均空间利用率 & 平均载重利用率 & 总成本/元 \\\\")
    lines.append("\\midrule")
    for code in ["V1", "V2"]:
        s = summary["all_one_type"][code]
        lines.append(
            f"{VEHICLES[code].name} & {strategy_name.get(s['strategy'], s['strategy'])} & {s['stats']['vehicle_count']} & "
            f"{s['stats']['avg_volume_utilization']*100:.2f}\\% & "
            f"{s['stats']['avg_weight_utilization']*100:.2f}\\% & "
            f"{s['stats']['total_cost']:.0f} \\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    (OUT / "problem1_tables.tex").write_text("\n".join(lines), encoding="utf-8")


GOODS_BY_CODE = {g.code: g for g in GOODS}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    items = expand_items()
    summary: dict = {"coordinate_system": "origin=right-rear-lower; x=rear_to_front; y=right_to_left; z=up", "single": {}, "all_one_type": {}}

    for code, vehicle in VEHICLES.items():
        strategy, placements, remaining, stats = choose_best_single(vehicle, items)
        errors = validate(vehicle, placements)
        summary["single"][code] = {
            "strategy": strategy,
            "stats": stats,
            "goods_counts": goods_counts(placements),
            "remaining_count": len(remaining),
            "validation_errors": errors,
        }
        write_placements_csv(OUT / f"single_{code}.csv", placements)

        strategy_all, bins, stats_all = pack_all_one_type(vehicle, items)
        all_errors: dict[str, list[str]] = {}
        all_placements: list[Placement] = []
        for idx, bin_placements in enumerate(bins, 1):
            errs = validate(vehicle, bin_placements)
            if errs:
                all_errors[f"{code}-{idx}"] = errs
            all_placements.extend(bin_placements)
        summary["all_one_type"][code] = {
            "strategy": strategy_all,
            "stats": stats_all,
            "vehicle_goods_counts": [goods_counts(b) for b in bins],
            "validation_errors": all_errors,
        }
        write_placements_csv(OUT / f"all_{code}.csv", all_placements)

    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_latex_tables(summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
