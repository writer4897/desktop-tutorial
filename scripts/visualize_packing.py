from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from pack_problem1 import VEHICLES


ROOT = Path.cwd()
if not (ROOT / "scripts").exists() and (ROOT.parent / "scripts").exists():
    ROOT = ROOT.parent

COLORS = {
    "G1": "#4E79A7",
    "G2": "#59A14F",
    "G3": "#E15759",
    "G4": "#F28E2B",
    "G5": "#B07AA1",
}


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def vehicle_code(row: dict) -> str:
    if row.get("vehicle_code"):
        return row["vehicle_code"]
    vid = row["vehicle_id"]
    return "V2" if "V2" in vid else "V1"


def cuboid_faces(x: float, y: float, z: float, dx: float, dy: float, dz: float):
    p = [
        (x, y, z),
        (x + dx, y, z),
        (x + dx, y + dy, z),
        (x, y + dy, z),
        (x, y, z + dz),
        (x + dx, y, z + dz),
        (x + dx, y + dy, z + dz),
        (x, y + dy, z + dz),
    ]
    return [
        [p[0], p[1], p[2], p[3]],
        [p[4], p[5], p[6], p[7]],
        [p[0], p[1], p[5], p[4]],
        [p[2], p[3], p[7], p[6]],
        [p[1], p[2], p[6], p[5]],
        [p[0], p[3], p[7], p[4]],
    ]


def plot_vehicle_3d(rows: list[dict], out: Path, title: str) -> None:
    code = vehicle_code(rows[0])
    vehicle = VEHICLES[code]
    fig = plt.figure(figsize=(10, 7), dpi=180)
    ax = fig.add_subplot(111, projection="3d")

    for row in rows:
        x = float(row["x_cm"])
        y = float(row["y_cm"])
        z = float(row["z_cm"])
        dx = float(row["length_cm"])
        dy = float(row["width_cm"])
        dz = float(row["height_cm"])
        goods = row["goods_code"]
        color = COLORS.get(goods, "#9C755F")
        faces = cuboid_faces(x, y, z, dx, dy, dz)
        poly = Poly3DCollection(faces, facecolors=color, edgecolors="#222222", linewidths=0.25, alpha=0.58)
        ax.add_collection3d(poly)

    # Vehicle frame.
    ax.set_xlim(0, vehicle.length)
    ax.set_ylim(0, vehicle.width)
    ax.set_zlim(0, vehicle.usable_height)
    ax.set_box_aspect((vehicle.length, vehicle.width, vehicle.usable_height))
    ax.set_xlabel("x / cm")
    ax.set_ylabel("y / cm")
    ax.set_zlabel("z / cm")
    ax.view_init(elev=24, azim=-58)
    ax.set_title(title)
    handles = [
        plt.Line2D([0], [0], color=COLORS[k], lw=7, label=k)
        for k in ["G1", "G2", "G3", "G4", "G5"]
        if any(row["goods_code"] == k for row in rows)
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_vehicle_top(rows: list[dict], out: Path, title: str) -> None:
    code = vehicle_code(rows[0])
    vehicle = VEHICLES[code]
    fig, ax = plt.subplots(figsize=(10, 4), dpi=180)
    for row in rows:
        x = float(row["x_cm"])
        y = float(row["y_cm"])
        dx = float(row["length_cm"])
        dy = float(row["width_cm"])
        goods = row["goods_code"]
        rect = plt.Rectangle((x, y), dx, dy, facecolor=COLORS.get(goods, "#9C755F"), edgecolor="#222222", alpha=0.55, linewidth=0.3)
        ax.add_patch(rect)
    ax.set_xlim(0, vehicle.length)
    ax.set_ylim(0, vehicle.width)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x / cm")
    ax.set_ylabel("y / cm")
    ax.set_title(title + " top view")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_vehicle_xz(rows: list[dict], out: Path, title: str) -> None:
    code = vehicle_code(rows[0])
    vehicle = VEHICLES[code]
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=180)
    for row in rows:
        x = float(row["x_cm"])
        z = float(row["z_cm"])
        dx = float(row["length_cm"])
        dz = float(row["height_cm"])
        goods = row["goods_code"]
        rect = plt.Rectangle((x, z), dx, dz, facecolor=COLORS.get(goods, "#9C755F"), edgecolor="#222222", alpha=0.5, linewidth=0.25)
        ax.add_patch(rect)
    ax.set_xlim(0, vehicle.length)
    ax.set_ylim(0, vehicle.usable_height)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x / cm")
    ax.set_ylabel("z / cm")
    ax.set_title(title + " side view (x-z)")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_vehicle_yz(rows: list[dict], out: Path, title: str) -> None:
    code = vehicle_code(rows[0])
    vehicle = VEHICLES[code]
    fig, ax = plt.subplots(figsize=(7, 5), dpi=180)
    for row in rows:
        y = float(row["y_cm"])
        z = float(row["z_cm"])
        dy = float(row["width_cm"])
        dz = float(row["height_cm"])
        goods = row["goods_code"]
        rect = plt.Rectangle((y, z), dy, dz, facecolor=COLORS.get(goods, "#9C755F"), edgecolor="#222222", alpha=0.5, linewidth=0.25)
        ax.add_patch(rect)
    ax.set_xlim(0, vehicle.width)
    ax.set_ylim(0, vehicle.usable_height)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("y / cm")
    ax.set_ylabel("z / cm")
    ax.set_title(title + " rear view (y-z)")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def visualize(csv_path: Path, out_dir: Path, prefix: str) -> list[Path]:
    rows = read_rows(csv_path)
    by_vehicle: dict[str, list[dict]] = {}
    for row in rows:
        by_vehicle.setdefault(row["vehicle_id"], []).append(row)
    outputs: list[Path] = []
    for idx, (vehicle_id, group) in enumerate(sorted(by_vehicle.items()), 1):
        title = f"{prefix} {vehicle_id} ({len(group)} items)"
        out3d = out_dir / f"{prefix}_{idx}_{vehicle_id}_3d.png"
        outtop = out_dir / f"{prefix}_{idx}_{vehicle_id}_top.png"
        outxz = out_dir / f"{prefix}_{idx}_{vehicle_id}_side_xz.png"
        outyz = out_dir / f"{prefix}_{idx}_{vehicle_id}_rear_yz.png"
        plot_vehicle_3d(group, out3d, title)
        plot_vehicle_top(group, outtop, title)
        plot_vehicle_xz(group, outxz, title)
        plot_vehicle_yz(group, outyz, title)
        outputs.extend([out3d, outtop, outxz, outyz])
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=ROOT / "results" / "problem2" / "min_cost.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "visuals")
    parser.add_argument("--prefix", default="min_cost")
    args = parser.parse_args()
    outputs = visualize(args.csv, args.out, args.prefix)
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
