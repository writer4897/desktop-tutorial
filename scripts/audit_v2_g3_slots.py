from __future__ import annotations

import csv
import json
from pathlib import Path

from pack_problem1 import Placement, VEHICLES, rects_cover


ROOT = Path.cwd()
if not (ROOT / "scripts").exists() and (ROOT.parent / "scripts").exists():
    ROOT = ROOT.parent


def read(path: Path) -> list[Placement]:
    rows: list[Placement] = []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            rows.append(
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
                    weight=int(float(row["weight_kg"])),
                    support=row["support"],
                )
            )
    return rows


def slots_at_z(placements: list[Placement], z: int, step: int = 5) -> list[tuple[int, int, int, int]]:
    supports = [p for p in placements if p.label == "standard" and p.zmax == z]
    occupied = [p for p in placements if p.z < z + 40 and p.zmax > z]
    slots = []
    for length, width in [(70, 50), (50, 70)]:
        for x in range(0, VEHICLES["V2"].length - length + 1, step):
            for y in range(0, VEHICLES["V2"].width - width + 1, step):
                candidate = Placement("slot", "G3", "fragile", "", x, y, z, length, width, 40, 15, "")
                if not rects_cover(candidate, supports):
                    continue
                if any(
                    max(candidate.x, p.x) < min(candidate.xmax, p.xmax)
                    and max(candidate.y, p.y) < min(candidate.ymax, p.ymax)
                    and max(candidate.z, p.z) < min(candidate.zmax, p.zmax)
                    for p in occupied
                ):
                    continue
                slots.append((x, y, x + length, y + width))
    return slots


def greedy_nonoverlap(rects: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    chosen = []
    for rect in sorted(rects, key=lambda r: (r[1], r[0], r[3], r[2])):
        if all(not (max(rect[0], c[0]) < min(rect[2], c[2]) and max(rect[1], c[1]) < min(rect[3], c[3])) for c in chosen):
            chosen.append(rect)
    return chosen


def main() -> None:
    placements = read(ROOT / "results" / "problem1" / "single_V2.csv")
    layer_rows = []
    total_slots = 0
    for z in sorted({p.zmax for p in placements if p.label == "standard"}):
        raw = slots_at_z(placements, z)
        chosen = greedy_nonoverlap(raw)
        layer_rows.append({"z": z, "raw_candidates": len(raw), "nonoverlap_slots": len(chosen)})
        total_slots += len(chosen)
    print(json.dumps({"layers": layer_rows, "total_nonoverlap_slots": total_slots}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
