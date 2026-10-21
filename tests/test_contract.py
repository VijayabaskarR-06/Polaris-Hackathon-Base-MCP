"""Exact contract checks against the untouched starter implementation (eval/reference.py).

Empty-query pages are what the validity category compares exactly, so every
random page here must equal the oracle's page. CONTRACT_CASES overrides the
number of random pages (default 2000).
"""
import os
import random
import unittest
from pathlib import Path

import search
from eval import reference

ROOT = Path(__file__).resolve().parent.parent
CATALOG_PATH = os.getenv('CATALOG_PATH', str(ROOT / 'data/products.json'))
CASES = int(os.getenv('CONTRACT_CASES', '2000'))
SORT_BY = ['relevance', 'id', 'price', 'rating', 'review_count']

INVALID = [
    {'limit': 0}, {'limit': 101}, {'limit': True}, {'limit': 3.5}, {'limit': '10'}, {'limit': None},
    {'offset': -1}, {'offset': 10001}, {'offset': 1.5}, {'offset': False},
    {'sort_by': 'name'}, {'sort_by': 'Price'}, {'sort_order': 'ASC'}, {'sort_order': None},
    {'query': None}, {'query': 5}, {'query': 'x' * 1001}, {'query': ['jeans']},
    {'filters': None}, {'filters': []}, {'filters': 'Black'}, {'filters': {'colour': 'Black'}},
    {'filters': {'color': ''}}, {'filters': {'color': '  '}}, {'filters': {'section': '\t'}},
    {'filters': {'color': 5}}, {'filters': {'color': None}}, {'filters': {'color': ['Black']}},
    {'filters': {'min_price': -1}}, {'filters': {'max_price': -0.5}},
    {'filters': {'min_price': 5, 'max_price': 1}}, {'filters': {'min_price': '100'}},
    {'filters': {'min_rating': 0}}, {'filters': {'min_rating': 5.5}}, {'filters': {'min_rating': 0.99}},
    {'filters': {'min_review_count': -1}}, {'filters': {'min_review_count': 1.5}},
    {'filters': {'min_review_count': True}},
    {'filters': {'min_price': float('nan')}}, {'filters': {'max_price': float('inf')}},
    {'include_recommendations': 'yes'}, {'include_recommendations': 1}, {'include_recommendations': None},
    {'unknown': 1}, {'query': '', 'extra': None},
]


def random_case_variant(rng, value):
    variant = rng.choice([value, value.lower(), value.upper(), value.swapcase(), value.title()])
    return rng.choice(['', ' ', '  ', '\t']) + variant + rng.choice(['', ' ', '\n'])


def random_arguments(rng, products, values):
    """Random valid empty-query arguments, biased so that filters often intersect."""
    args = {}
    roll = rng.random()
    if roll < 0.2:
        args['query'] = ''
    elif roll < 0.3:
        args['query'] = rng.choice([' ', '   ', '\t', '\n  ', '　', ' \r\n'])
    filters = {}
    seed = rng.choice(products)
    for field in search.STRING_FIELDS:
        roll = rng.random()
        if roll < 0.18:
            filters[field] = random_case_variant(rng, seed[field])
        elif roll < 0.22:
            filters[field] = random_case_variant(rng, rng.choice(values[field]))
        elif roll < 0.235:
            filters[field] = rng.choice(['Navy', 'unknown value', 'Dog wear', 'DOG WEAR', 'Brown'])
    prices = [199, 499, 999, 1999, 2499, 4499, 8999, seed['price']['amount']]
    if rng.random() < 0.3:
        filters['min_price'] = rng.choice(prices + [0, 0.0, 998.5, 1500.25])
    if rng.random() < 0.3:
        low = filters.get('min_price', 0)
        filters['max_price'] = max(low, rng.choice(prices + [100000, 2000.0, 1e9]))
    if rng.random() < 0.2:
        filters['min_rating'] = rng.choice([1, 1.0, 2, 2.5, 3, 3.33, 3.34, 4, 4.5, 4.67, 5, 5.0])
    if rng.random() < 0.2:
        filters['min_review_count'] = rng.choice([0, 1, 2, 3, 5, 8, 9, 0.0, 4.0])
    if filters or rng.random() < 0.5:
        args['filters'] = filters
    if rng.random() < 0.85:
        args['sort_by'] = rng.choice(SORT_BY)
    if rng.random() < 0.85:
        args['sort_order'] = rng.choice(['asc', 'desc'])
    roll = rng.random()
    if roll < 0.6:
        args['offset'] = rng.choice([0, 0, 0, 1, 5, 10, 37, 90])
    elif roll < 0.75:
        args['offset'] = rng.choice([500, 2000, 9999, 10000, 105000 % 10001])
    elif roll < 0.8:
        args['offset'] = rng.choice([0.0, 10.0, 3.0])
    roll = rng.random()
    if roll < 0.75:
        args['limit'] = rng.choice([1, 2, 5, 10, 10, 10, 20, 50, 99, 100])
    elif roll < 0.82:
        args['limit'] = rng.choice([10.0, 3.0, 100.0, 1.0])
    if rng.random() < 0.3:
        args['include_recommendations'] = rng.random() < 0.5
    return args


def oracle_arguments(args):
    """The starter crashes on integral floats it accepts, so feed it ints."""
    fixed = dict(args)
    for key in ('offset', 'limit'):
        if key in fixed:
            fixed[key] = int(fixed[key])
    if 'min_review_count' in fixed.get('filters', {}):
        fixed['filters'] = dict(fixed['filters'], min_review_count=int(fixed['filters']['min_review_count']))
    return fixed


class ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.oracle = reference.Catalog.load(CATALOG_PATH)
        cls.catalog = search.Catalog.load(CATALOG_PATH, semantic=False)
        cls.values = {f: sorted({p[f] for p in cls.oracle.products}) for f in search.STRING_FIELDS}

    def test_random_empty_query_pages_equal_oracle(self):
        rng = random.Random(20261021)
        nonempty = 0
        for case in range(CASES):
            args = random_arguments(rng, self.oracle.products, self.values)
            expected = self.oracle.search(oracle_arguments(args))
            got = self.catalog.search(args)
            nonempty += bool(expected['results'])
            if got != expected:
                self.fail(f'case {case}: {args}\n got {got["results"][:5]}\n want {expected["results"][:5]}')
        self.assertGreater(nonempty, CASES // 3, 'random cases should mostly produce non-empty pages')

    def test_sort_edges(self):
        for sort_by in SORT_BY:
            for order in ('asc', 'desc'):
                for offset in (0, 11559, 93560, 93566, 105120, 105126, 10000):
                    args = {'sort_by': sort_by, 'sort_order': order, 'offset': min(offset, 10000), 'limit': 100}
                    with self.subTest(args=args):
                        self.assertEqual(self.catalog.search(args), self.oracle.search(args))

    def test_invalid_arguments_rejected_by_both(self):
        for args in INVALID:
            with self.subTest(args=args):
                with self.assertRaises(ValueError):
                    self.oracle.search(args)
                with self.assertRaises(ValueError):
                    self.catalog.search(args)

    def test_integral_floats_are_accepted(self):
        got = self.catalog.search({'limit': 3.0, 'offset': 2.0, 'filters': {'min_review_count': 2.0}})
        want = self.oracle.search({'limit': 3, 'offset': 2, 'filters': {'min_review_count': 2}})
        self.assertEqual(got, want)
        self.assertEqual(len(got['results']), 3)

    def test_whitespace_query_is_empty_and_has_no_recommendations(self):
        for query in ('', ' ', '\t\n', '　'):
            got = self.catalog.search({'query': query, 'include_recommendations': True, 'limit': 7})
            self.assertEqual(got, self.oracle.search({'limit': 7}))
            self.assertEqual(got['recommendations'], [])

    def test_casefold_collision_matches_both_spellings(self):
        got = self.catalog.search({'filters': {'product_type': 'dog wear'}, 'limit': 100})
        self.assertEqual(got, self.oracle.search({'filters': {'product_type': 'dog wear'}, 'limit': 100}))
        self.assertEqual(len(got['results']), 27)


if __name__ == '__main__':
    unittest.main()
