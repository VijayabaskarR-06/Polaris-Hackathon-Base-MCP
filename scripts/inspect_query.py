"""Show how the engine understands and ranks queries: python scripts/inspect_query.py "black jeans" ...

Prints the plan (choices, hard constraints, soft hints), the abstention decision
and the top results with their estimated grade. Uses the rule path unless
LLM_PROVIDER is set.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import search  # noqa: E402
from engine import points  # noqa: E402
from engine.rules import signals_of  # noqa: E402


def describe_choice(choice):
    fields = {k: getattr(choice, k) for k in ('label', 'product_types', 'keywords', 'colors', 'exclude_colors',
                                              'patterns', 'min_price', 'max_price', 'soft_sections', 'soft_words',
                                              'price_preference', 'rating_preference')}
    for extra in ('descriptors', 'unknown', 'exclude_words', 'primary_types', 'needs_keyword', 'occasion_types'):
        if getattr(choice, extra, None):
            fields[extra] = getattr(choice, extra)
    return {k: v for k, v in fields.items() if v not in (None, [], False)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('queries', nargs='+')
    parser.add_argument('--limit', type=int, default=10)
    parser.add_argument('--filters', default='{}')
    parser.add_argument('--recs', action='store_true')
    opts = parser.parse_args()
    catalog = search.Catalog.load(ROOT / 'data/products.json')
    engine = catalog.engine
    products = catalog.products
    for query in opts.queries:
        args = catalog.prepare({'query': query, 'limit': opts.limit, 'filters': json.loads(opts.filters),
                                'include_recommendations': opts.recs})
        out = catalog.search({k: v for k, v in args.items()})
        state = engine.cache.get(__import__('engine.engine', fromlist=['state_key']).state_key(args))
        print(f'\n=== {query!r}')
        plan = state.plan
        signals = signals_of(plan)
        print(f'kind={plan.kind} source={state.source} audience={plan.audience_group} abstain={state.abstain} '
              f'{state.reason} corrected={signals.corrected} unknown={signals.unknown} brands={signals.brands}')
        for choice in plan.choices:
            print('  choice:', describe_choice(choice))
        if plan.complement is not None:
            print('  complement:', describe_choice(plan.complement))
        for rank, item in enumerate(out['results'], 1):
            pos = catalog.index.pos_of[item['id']]
            p = products[pos]
            cand = state.candidate(pos)
            eg = points.expected_gain(cand.probs) if cand else 0
            print(f"  {rank:2d}. eg={eg:4.2f} {p['name'][:30]:30s} | {p['product_type']:14s} | {p['color']:14s} | "
                  f"INR {p['price']['amount']:5d} | {p['audience'][:22]:22s} | {p['section'][:24]}")
        for rec in out['recommendations']:
            p = products[catalog.index.pos_of[rec['id']]]
            print(f"   rec {p['name'][:30]:30s} | {p['product_type']:14s} | {p['color']:14s} | "
                  f"INR {p['price']['amount']:5d} | {rec['reason']}")


if __name__ == '__main__':
    main()
