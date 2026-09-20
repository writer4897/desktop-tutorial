from __future__ import annotations

"""车型 2 单车方案的 G3 槽位插入复验。

基础极点法对车型 2 可先装入 270 件非易碎件，但没有识别若干由标准件
拼接形成的 G3 支撑面。本脚本扫描这些标准件顶面槽位，在不移动原有货物
的前提下插入可完全支撑的 G3，用于检查车型 2 的优化潜力。
"""

import csv
import json
from pathlib import Path

from audit_v2_g3_slots import greedy_nonoverlap, read, slots_at_z
from pack_problem1 import GOODS_BY_CODE, Placement, VEHICLES, utilization, validate, write_placements_csv


ROOT = Path.cwd()
if not (ROOT / "scripts").exists() and (ROOT.parent / "scripts").exists():
    ROOT = ROOT.parent
OUT = ROOT / "results" / "experiments" / "v2_slot_insert"


def used_g3_ids(path: Path) -> set[str]:
    """读取基础方案中已经装入的 G3 编号。"""
    used = set()
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            if row["goods_code"] == "G3":
                used.add(row["uid"])
    return used


def support_ids(candidate: Placement, placements: list[Placement]) -> str:
    """记录一个 G3 候选位置下方实际参与支撑的标准件编号。"""
    ids = []
    for p in placements:
        if p.label != "standard" or p.zmax != candidate.z:
            continue
        x0, x1 = max(candidate.x, p.x), min(candidate.xmax, p.xmax)
        y0, y1 = max(candidate.y, p.y), min(candidate.ymax, p.ymax)
        if x0 < x1 and y0 < y1:
            ids.append(p.uid)
    return "|".join(sorted(ids))


def main() -> None:
    """执行槽位扫描、G3 插入、可行性复验和结果导出。"""
    OUT.mkdir(parents=True, exist_ok=True)
    source = ROOT / "results" / "problem1" / "single_V2.csv"
    placements = read(source)
    used = used_g3_ids(source)
    missing = [idx for idx in range(1, GOODS_BY_CODE["G3"].quantity + 1) if f"G3-{idx:03d}" not in used]
    layer_slots = []
    for z in sorted({p.zmax for p in placements if p.label == "standard"}):
        for rect in greedy_nonoverlap(slots_at_z(placements, z)):
            layer_slots.append((z, rect))
    inserted = []
    for idx, (z, rect) in zip(missing, layer_slots):
        x0, y0, x1, y1 = rect
        p = Placement(
            uid=f"G3-{idx:03d}",
            goods_code="G3",
            label="fragile",
            vehicle_id="V2-single-slot-insert",
            x=x0,
            y=y0,
            z=z,
            length=x1 - x0,
            width=y1 - y0,
            height=40,
            weight=15,
            support="",
        )
        p.support = support_ids(p, placements)
        placements.append(p)
        inserted.append(p.uid)
    errors = validate(VEHICLES["V2"], placements)
    write_placements_csv(OUT / "single_V2_slot_insert.csv", placements)
    summary = {
        "source": str(source),
        "inserted_g3": len(inserted),
        "placed": len(placements),
        "remaining_g3": len(missing) - len(inserted),
        "validation_error_count": len(errors),
        "errors": errors[:10],
        "stats": utilization(VEHICLES["V2"], placements),
    }
    (OUT / "single_V2_slot_insert_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
