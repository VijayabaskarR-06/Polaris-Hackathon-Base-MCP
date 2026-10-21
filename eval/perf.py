"""Performance workload as the brief describes it: python eval/perf.py URL.

20 unscored warm-up calls, then 50 calls at concurrency 1 and 50 at concurrency
5, repeated three times. Every call is an empty-query filter/sort page with
limit=10 and offset=0 whose result must equal the oracle (eval/reference.py).
Failed, incorrect or timed-out calls count as 10,000 ms. p95 is nearest-rank
over all 300 timed calls.
"""
import argparse
import asyncio
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mcp import ClientSession  # noqa: E402
from mcp.client.streamable_http import streamable_http_client  # noqa: E402

from eval import reference  # noqa: E402

PENALTY_MS = 10_000.0
SORTS = ['relevance', 'id', 'price', 'rating', 'review_count']


def build_pool(products, size, seed):
    """A fixed, ordered pool of realistic empty-query filter and field-sort calls."""
    rng = random.Random(seed)
    pool = []
    while len(pool) < size:
        seed_product = rng.choice(products)
        filters = {}
        for field in ('category', 'product_type', 'color', 'audience', 'section', 'pattern'):
            if rng.random() < 0.3:
                filters[field] = seed_product[field]
        if rng.random() < 0.35:
            filters['max_price'] = rng.choice([999, 1499, 2000, 2999, 4999])
        if rng.random() < 0.15:
            filters['min_price'] = rng.choice([0, 499, 999])
        if rng.random() < 0.15:
            filters['min_rating'] = rng.choice([3, 3.5, 4, 4.5])
        if rng.random() < 0.1:
            filters['min_review_count'] = rng.choice([1, 3, 5])
        if filters.get('min_price', 0) > filters.get('max_price', math.inf):
            del filters['min_price']
        args = {'filters': filters, 'sort_by': rng.choice(SORTS),
                'sort_order': rng.choice(['asc', 'desc']), 'offset': 0, 'limit': 10}
        if rng.random() < 0.2:
            args['query'] = ''
        pool.append(args)
    return pool


def nearest_rank(values, pct):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(pct / 100 * len(ordered)) - 1)]


async def timed_call(session, args, expected, timeout):
    start = time.perf_counter()
    try:
        result = await asyncio.wait_for(session.call_tool('search_products', args), timeout)
        elapsed = (time.perf_counter() - start) * 1000
        ok = (not result.isError and result.structuredContent == expected
              and json.loads(result.content[0].text) == expected)
    except Exception:
        elapsed, ok = PENALTY_MS, False
    return (elapsed if ok else PENALTY_MS), ok


async def workload(session, pool, expected, concurrency, timeout):
    semaphore = asyncio.Semaphore(concurrency)
    samples = [None] * len(pool)

    async def one(i):
        async with semaphore:
            samples[i] = await timed_call(session, pool[i], expected[i], timeout)

    await asyncio.gather(*(one(i) for i in range(len(pool))))
    return samples


async def run(url, pool, expected, repeats, timeout):
    async with streamable_http_client(url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            for i in range(20):
                await session.call_tool('search_products', pool[i % len(pool)])
            results = {1: [], 5: []}
            wall = 0.0
            for _ in range(repeats):
                for concurrency in (1, 5):
                    start = time.perf_counter()
                    results[concurrency] += await workload(session, pool, expected, concurrency, timeout)
                    wall += time.perf_counter() - start
    return results, wall


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('url', nargs='?', default='http://127.0.0.1:8000/mcp')
    parser.add_argument('--pool', type=int, default=50)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--timeout', type=float, default=10.0)
    opts = parser.parse_args()

    oracle = reference.Catalog.load(ROOT / 'data/products.json')
    pool = build_pool(oracle.products, opts.pool, opts.seed)
    expected = [oracle.search(args) for args in pool]
    results, wall = asyncio.run(run(opts.url, pool, expected, opts.repeats, opts.timeout))

    every = results[1] + results[5]
    latencies = [ms for ms, _ in every]
    correct = sum(ok for _, ok in every) / len(every)
    p95 = nearest_rank(latencies, 95)
    factor = max(0.0, min(1.0, (2000 - p95) / 1800))
    for concurrency in (1, 5):
        ms = [m for m, _ in results[concurrency]]
        print(f'concurrency {concurrency}: p50 {statistics.median(ms):7.1f} ms  '
              f'p95 {nearest_rank(ms, 95):7.1f} ms  max {max(ms):7.1f} ms  '
              f'correct {sum(ok for _, ok in results[concurrency])}/{len(ms)}')
    print(f'pooled {len(every)} calls: p50 {statistics.median(latencies):.1f} ms  p95 {p95:.1f} ms  '
          f'correct rate {correct:.3f}  correct rps {sum(ok for _, ok in every) / wall:.1f}')
    print(f'latency factor {factor:.3f} -> performance points if eligible: {5 * correct * factor:.2f} / 5')
    return 0 if correct >= 0.95 else 1


if __name__ == '__main__':
    sys.exit(main())
