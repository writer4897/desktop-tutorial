"""Verify archived final coordinates against the contest model; no optimizer run."""
from collections import Counter, defaultdict
from pathlib import Path
import json
from experiment_local_g3_repair import read_placements
from pack_problem1 import VEHICLES, expand_items, validate

ROOT = Path(__file__).resolve().parents[1]

def main():
    rows = read_placements(ROOT / 'results/experiments/local_g3_repair/best_local_g3_repair.csv')
    expected = {x.uid: x for x in expand_items()}
    if Counter(p.uid for p in rows) != Counter(expected.keys()):
        raise ValueError('Missing, duplicate, or unexpected item identifiers')
    groups = defaultdict(list)
    for p in rows:
        item = expected[p.uid]
        if (p.goods_code, p.label, p.weight) != (item.goods_code, item.label, item.weight):
            raise ValueError(f'{p.uid}: item metadata mismatch')
        if sorted((p.length, p.width, p.height)) != sorted((item.length0, item.width0, item.height0)):
            raise ValueError(f'{p.uid}: dimensions mismatch')
        groups[p.vehicle_id].append(p)
    if len(groups) != 2 or any('V1' not in k for k in groups):
        raise ValueError('Expected two V1 vehicles')
    errors = {k: validate(VEHICLES['V1'], ps) for k, ps in groups.items()}
    if any(errors.values()):
        raise ValueError(errors)
    print(json.dumps({'items': len(rows), 'vehicles': len(groups),
        'cost_yuan': len(groups) * VEHICLES['V1'].cost,
        'volume_utilization': sum(p.length*p.width*p.height for p in rows)/(len(groups)*VEHICLES['V1'].volume),
        'validation_errors': errors}, indent=2))

if __name__ == '__main__':
    main()
