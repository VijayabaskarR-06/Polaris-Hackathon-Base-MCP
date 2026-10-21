"""Refit the Points Engine calibration from judged eval results.

    python scripts/fit_calibration.py RUN_DIR/grades.json [--cases eval/cases.jsonl] [--write]

Re-runs every judged case in-process on the rule path, pairs each returned
product's rule estimate (1-3) with the judge's grade, prints the confusion
table, and with --write saves assets/calibration.json with the refitted
'estimate' rows (the 'twin' rows are kept; refit those from LLM twin logs).
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import search  # noqa: E402
from engine import points  # noqa: E402
from engine.engine import state_key  # noqa: E402
from engine.llm import NullLLM  # noqa: E402


def estimates_for(catalog, args):
    """{product id: rule estimate} for every pooled product of this request."""
    prepared = catalog.prepare(args)
    catalog.semantic(prepared)
    state = catalog.engine.cache.get(state_key(prepared))
    out = {}
    for pool in state.pools if state else []:
        for pos, est in zip(pool.positions.tolist(), pool.estimates.tolist()):
            pid = catalog.index.ids[pos]
            out[pid] = max(out.get(pid, 0), int(est))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('grades')
    parser.add_argument('--cases', default=str(ROOT / 'eval/cases.jsonl'))
    parser.add_argument('--write', action='store_true')
    opts = parser.parse_args()
    grades = json.load(open(opts.grades))
    cases = {json.loads(line)['id']: json.loads(line) for line in open(opts.cases) if line.strip()}
    catalog = search.Catalog.load(ROOT / 'data/products.json', llm=NullLLM())
    pairs = []
    for cid, judged in grades.items():
        case = cases.get(cid)
        if case is None:
            continue
        est = estimates_for(catalog, case['args'])
        for pid, g in judged.get('primary', {}).items():
            if g is not None and pid in est:
                pairs.append((est[pid], int(g)))
    table = Counter(pairs)
    print(f'{len(pairs)} pairs (rule estimate, judge grade)')
    for e in (1, 2, 3):
        row = [table[(e, g)] for g in range(4)]
        total = sum(row) or 1
        print(f'  estimate {e}: n={sum(row):4d}  judge 0/1/2/3 = ' +
              ' '.join(f'{c / total:5.2f}' for c in row))
    calibration = points.fit_calibration(pairs, source='estimate')
    print('fitted estimate rows:', {k: tuple(round(p, 3) for p in v) for k, v in calibration['estimate'].items()})
    if opts.write:
        path = ROOT / 'assets/calibration.json'
        path.parent.mkdir(exist_ok=True)
        current = points.DEFAULT_CALIBRATION
        if path.exists():
            current = points.calibration_from_json(json.load(open(path)))
        merged = dict(current, estimate=calibration['estimate'])
        json.dump(points.calibration_to_json(merged), open(path, 'w'), indent=1)
        print('wrote', path)


if __name__ == '__main__':
    main()
