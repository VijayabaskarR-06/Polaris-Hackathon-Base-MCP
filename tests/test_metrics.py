"""Benchmark formulas (eval/metrics.py) and the exam simulator's deterministic checks.

Expected values are worked out by hand from the brief: gain 2^g - 1, discount
log2(rank + 1) with ranks from 1, and a fixed ideal of k grade-3 items.
No catalog or server is needed: the simulator checks run on a tiny fixture.
"""
import contextlib
import io
import json
import math
import sys
import tempfile
import threading
import types
import unittest
from collections import Counter
from pathlib import Path

from eval import local_judge, metrics, run_eval

# 7 x (1/log2 2 + 1/log2 3 + ... + 1/log2 11) and the same over ranks 1..5.
IDEAL10 = 31.804915366618417
IDEAL5 = 20.639213832155747
LOG3 = math.log2(3)


def product(pid, name='Slim jeans', product_type='Trousers', color='Black', price=999, average=4.0,
            count=2, audience='Ladieswear', section='Womens Everyday Collection', pattern='Denim',
            category='Garment Lower body', description='Five-pocket jeans in stretch denim.'):
    return {'id': pid, 'name': name, 'description': description, 'product_type': product_type,
            'category': category, 'audience': audience, 'section': section, 'color': color,
            'pattern': pattern, 'price': {'amount': price, 'currency': 'INR', 'synthetic': True},
            'rating': {'average': average, 'count': count, 'synthetic': True}}


PRODUCTS = [
    product('0001'),
    product('0002', 'Basic tee', 'T-shirt', 'White', 499, None, 0, pattern='Solid', category='Garment Upper body',
            description='Cotton jersey T-shirt.'),
    product('0003', 'Shoulder bag', 'Bag', 'Black', 1299, 5.0, 3, 'Ladies Accessories', 'Womens Big accessories',
            'Solid', 'Accessories', 'Small bag with a shoulder strap.'),
    product('0004', 'Regular jeans', 'Trousers', 'Dark Blue', 999, 4.0, 1, 'Menswear', 'Denim Men'),
    product('0005', 'Party dress', 'Dress', 'Pink', 799, 3.0, 1, 'Children Sizes 92-140', 'Kids Girl', 'Solid',
            'Garment Full body', 'Tulle dress with a bow.'),
    product('0006', 'Body', 'Bodysuit', 'White', 299, None, 0, 'Baby Sizes 50-98', 'Baby Essentials & Complements',
            'Solid', 'Garment Full body', 'Soft cotton body.'),
    product('0007', 'Sports top', 'Top', 'Black', 999, 2.5, 4, 'Sport', 'Men H&M Sport', 'Solid',
            'Garment Upper body', 'Fast-drying sports top.'),
    product('0008', 'Dog jumper', 'Dog wear', 'Grey', 699, 4.5, 2, 'Divided', 'Divided Accessories', 'Solid',
            'Accessories', 'Knitted jumper for dogs.'),
]
CATALOG = run_eval.Catalog([dict(p) for p in PRODUCTS])
BY_ID = CATALOG.by_id


def out(*ids, recs=()):
    return {'results': [{'id': i} for i in ids],
            'recommendations': [{'id': r, 'for_product_id': f, 'reason': 'goes with it'} for r, f in recs]}


def ok_call(output, args):
    return {'kind': 'result', 'ms': 1.0, 'args': args, 'output': output, 'text_ok': True, 'error': None}


def case(category='ranking', args=None, **expect):
    return {'id': 'c1', 'category': category, 'args': args if args is not None else {'query': 'jeans'},
            'expect': expect}


def run(test_case, output, extra_calls=None, shift=None, case_ms=5.0):
    calls = {'main': ok_call(output, test_case['args'])}
    calls.update(extra_calls or {})
    return run_eval.evaluate(test_case, calls, shift, case_ms, CATALOG)


class NdcgTests(unittest.TestCase):
    def test_ideal_is_ten_grade_3_slots(self):
        self.assertAlmostEqual(metrics.ideal_dcg(10), IDEAL10, places=12)
        self.assertAlmostEqual(metrics.ideal_dcg(5), IDEAL5, places=12)

    def test_perfect_list_is_one(self):
        self.assertEqual(metrics.ndcg([3] * 10), 1.0)
        self.assertEqual(metrics.ndcg([3] * 5, k=5), 1.0)

    def test_ten_grade_2_is_three_sevenths(self):
        self.assertAlmostEqual(metrics.ndcg([2] * 10), 3 / 7, places=12)
        self.assertAlmostEqual(metrics.ndcg([1] * 10), 1 / 7, places=12)

    def test_missing_slots_earn_zero(self):
        self.assertAlmostEqual(metrics.ndcg([3]), 7 / IDEAL10, places=12)
        self.assertAlmostEqual(metrics.ndcg([3]), 0.2200917662980802, places=12)
        self.assertAlmostEqual(metrics.ndcg([3], k=5), 7 / IDEAL5, places=12)
        self.assertEqual(metrics.ndcg([]), 0.0)

    def test_rank_discount(self):
        self.assertAlmostEqual(metrics.ndcg([3, 2]), (7 + 3 / LOG3) / IDEAL10, places=12)
        self.assertAlmostEqual(metrics.ndcg([0, 3]), (7 / LOG3) / IDEAL10, places=12)
        self.assertGreater(metrics.ndcg([3, 0]), metrics.ndcg([0, 3]))

    def test_grades_past_k_are_ignored(self):
        self.assertEqual(metrics.ndcg([3] * 12), 1.0)
        self.assertEqual(metrics.ndcg([0] * 10 + [3, 3]), 0.0)
        self.assertEqual(metrics.ndcg([3] * 5 + [0] * 5, k=5), 1.0)

    def test_violating_products_earn_no_gain(self):
        self.assertEqual(metrics.effective([3, 3, 2], [True, False, True]), [3, 0, 2])
        self.assertAlmostEqual(metrics.ndcg(metrics.effective([3, 3], [False, True])), (7 / LOG3) / IDEAL10,
                               places=12)
        with self.assertRaises(ValueError):
            metrics.effective([3, 3], [True])

    def test_invalid_grades_are_rejected(self):
        for bad in (4, -1, 2.5, True, None, '3'):
            with self.subTest(grade=bad), self.assertRaises(ValueError):
                metrics.ndcg([bad])
        with self.assertRaises(ValueError):
            metrics.ndcg([3], k=0)


class MixedTests(unittest.TestCase):
    def test_both_choices_covered(self):
        score = metrics.mixed_score([3, 2], [[3, 0], [0, 2]], 2, [True, True])
        self.assertAlmostEqual(score, 0.5 + 0.5 * (7 + 3 / LOG3) / IDEAL10, places=12)
        self.assertEqual(metrics.choice_coverage([3, 2], [[3, 0], [0, 2]], 2, [True, True]), 1.0)

    def test_one_of_two_choices_covered(self):
        self.assertEqual(metrics.choice_coverage([3, 3], [[3, 1], [2, 0]], 2, [True, True]), 0.5)
        self.assertAlmostEqual(metrics.mixed_score([3] * 10, [[3, 0]] * 10, 2, [True] * 10), 0.75, places=12)

    def test_coverage_needs_primary_grade_2(self):
        self.assertEqual(metrics.choice_coverage([1], [[3]], 1, [True]), 0.0)
        self.assertEqual(metrics.choice_coverage([2], [[3]], 1, [True]), 1.0)

    def test_coverage_needs_choice_grade_2(self):
        self.assertEqual(metrics.choice_coverage([3], [[1]], 1, [True]), 0.0)
        self.assertEqual(metrics.choice_coverage([3], [[2]], 1, [True]), 1.0)

    def test_coverage_needs_hard_constraints(self):
        self.assertEqual(metrics.choice_coverage([3], [[3, 3]], 2, [False]), 0.0)
        self.assertEqual(metrics.mixed_score([3], [[3, 3]], 2, [False]), 0.0)

    def test_three_choices(self):
        rows = [[3, 0, 0], [0, 3, 0], [0, 0, 1]]
        self.assertAlmostEqual(metrics.choice_coverage([3, 3, 3], rows, 3, [True] * 3), 2 / 3, places=12)

    def test_empty_results_score_zero(self):
        self.assertEqual(metrics.mixed_score([], [], 2, []), 0.0)

    def test_coverage_counts_every_returned_result(self):
        # The brief puts no cutoff on coverage: a covering result at rank 11 counts
        # for coverage, while nDCG only sees the first 10.
        rows = [[3, 0]] * 10 + [[0, 3]]
        self.assertEqual(metrics.mixed_score([3] * 11, rows, 2, [True] * 11), 1.0)

    def test_bad_shapes_raise(self):
        with self.assertRaises(ValueError):
            metrics.choice_coverage([3], [[3]], 0, [True])
        with self.assertRaises(ValueError):
            metrics.choice_coverage([3, 3], [[3]], 1, [True, True])
        with self.assertRaises(ValueError):
            metrics.choice_coverage([3], [[3]], 2, [True])


class RecommendationTests(unittest.TestCase):
    def test_perfect_case(self):
        self.assertEqual(metrics.rec_case_score([3] * 10, [3] * 5, [3] * 5, [True] * 5), 1.0)

    def test_anchor_needs_primary_grade_2(self):
        self.assertEqual(metrics.rec_quality([3], [1], [True]), 0.0)
        self.assertAlmostEqual(metrics.rec_quality([3], [2], [True]), 7 / IDEAL5, places=12)
        self.assertAlmostEqual(metrics.rec_quality([3, 3], [0, 3], [True, True]), (7 / LOG3) / IDEAL5, places=12)

    def test_rec_must_obey_its_constraints(self):
        self.assertEqual(metrics.rec_quality([3], [3], [False]), 0.0)

    def test_empty_recs(self):
        self.assertEqual(metrics.rec_quality([], [], [], no_complement=True), 1.0)
        self.assertEqual(metrics.rec_quality([], [], []), 0.0)
        self.assertEqual(metrics.rec_case_score([3] * 10, [], [], [], no_complement=True), 1.0)
        self.assertEqual(metrics.rec_case_score([3] * 10, [], [], []), 0.5)

    def test_forced_recs_in_a_no_complement_case_are_judged(self):
        self.assertEqual(metrics.rec_quality([0], [3], [True], no_complement=True), 0.0)

    def test_case_score_is_the_mean(self):
        score = metrics.rec_case_score([3], [3, 2], [3, 3], [True, True])
        self.assertAlmostEqual(score, (7 / IDEAL10 + (7 + 3 / LOG3) / IDEAL5) / 2, places=12)

    def test_only_five_recommendation_slots(self):
        self.assertEqual(metrics.rec_quality([3] * 5 + [0], [3] * 6, [True] * 6), 1.0)

    def test_misaligned_inputs_raise(self):
        with self.assertRaises(ValueError):
            metrics.rec_quality([3], [], [True])


class ValidityTests(unittest.TestCase):
    def test_mean_of_four_binary_checks(self):
        self.assertEqual(metrics.validity_case_score(True, True, True, True), 1.0)
        self.assertEqual(metrics.validity_case_score(True, False, True, True), 0.75)
        self.assertEqual(metrics.validity_case_score(1, 0, 0, 1), 0.5)
        self.assertEqual(metrics.validity_case_score(False, False, False, False), 0.0)

    def test_invalid_call_scores_one_only_for_an_error(self):
        self.assertEqual(metrics.invalid_case_score(True), 1.0)
        self.assertEqual(metrics.invalid_case_score(False), 0.0)


class TotalTests(unittest.TestCase):
    ALL = {c: 1.0 for c in metrics.CATEGORIES}

    def test_weights(self):
        self.assertEqual(metrics.WEIGHTS, {'validity': 25, 'ranking': 30, 'unusual': 7.5, 'irrelevant': 7.5,
                                           'mixed': 15, 'recs': 10})
        self.assertEqual(metrics.FUNCTIONAL_MAX, 95)

    def test_perfect_run(self):
        result = metrics.total(self.ALL, perf=(1.0, 200))
        self.assertEqual((result['functional'], result['performance'], result['total']), (95, 5, 100))
        self.assertEqual(metrics.total(self.ALL)['total'], 95)

    def test_claude_md_target_table(self):
        means = {'validity': 1, 'irrelevant': 1, 'ranking': 0.88, 'unusual': 0.8, 'mixed': 0.9, 'recs': 0.85}
        result = metrics.total(means, perf=(1.0, 150))
        self.assertAlmostEqual(result['points']['ranking'], 26.4, places=9)
        self.assertAlmostEqual(result['functional'], 86.9, places=9)
        self.assertAlmostEqual(result['total'], 91.9, places=9)

    def test_perf_eligibility_threshold(self):
        at = dict(self.ALL, irrelevant=0.0, recs=0.8)       # 25 + 30 + 7.5 + 0 + 15 + 8 = 85.5
        below = dict(self.ALL, irrelevant=0.0, recs=0.799)  # 85.49
        result = metrics.total(at, perf=(1.0, 380))
        self.assertEqual(result['functional'], 85.5)
        self.assertTrue(result['perf_eligible'])
        self.assertAlmostEqual(result['performance'], 4.5, places=12)
        self.assertAlmostEqual(result['total'], 90.0, places=12)
        result = metrics.total(below, perf=(1.0, 100))
        self.assertAlmostEqual(result['functional'], 85.49, places=9)
        self.assertFalse(result['perf_eligible'])
        self.assertEqual(result['performance'], 0.0)
        self.assertAlmostEqual(result['total'], 85.49, places=9)

    def test_perf_needs_95_percent_correct_calls(self):
        self.assertEqual(metrics.total(self.ALL, perf=(0.94, 100))['performance'], 0.0)
        self.assertAlmostEqual(metrics.total(self.ALL, perf=(0.95, 100))['performance'], 4.75, places=12)
        self.assertAlmostEqual(metrics.performance_points(0.98, 1100, 90.0), 5 * 0.98 * 0.5, places=12)

    def test_latency_factor(self):
        for p95, factor in ((100, 1.0), (200, 1.0), (380, 0.9), (1100, 0.5), (2000, 0.0), (2500, 0.0)):
            with self.subTest(p95=p95):
                self.assertAlmostEqual(metrics.latency_factor(p95), factor, places=12)

    def test_missing_categories_earn_nothing(self):
        result = metrics.total({'validity': 1.0, 'ranking': None})
        self.assertEqual(result['functional'], 25)
        self.assertEqual(result['measured_max'], 25)
        self.assertEqual(result['points']['mixed'], 0.0)
        with self.assertRaises(ValueError):
            metrics.total({'speed': 1.0})


class CaseFormatTests(unittest.TestCase):
    def test_good_case(self):
        good = {'id': 'mix-01', 'category': 'mixed', 'args': {'query': 'black jeans and white tees'},
                'expect': {'choices': ['black jeans', 'white tees'],
                           'choice_constraints': [{'colors': ['Black']}, {'colors': ['White']}],
                           'constraints': {'max_price': 2000, 'audience_group': 'women'}}}
        self.assertEqual(run_eval.case_problems(good), [])

    def test_bad_cases(self):
        bad = [
            {'id': 'a', 'category': 'speed', 'args': {}},
            {'id': 'b', 'category': 'mixed', 'args': {}, 'expect': {}},
            {'id': 'c', 'category': 'ranking', 'args': {}, 'expect': {'constraints': {'colour': ['Black']}}},
            {'id': 'd', 'category': 'mixed', 'args': {}, 'expect': {'choices': ['x', 'y'],
                                                                     'choice_constraints': [{}]}},
            {'id': 'e', 'category': 'irrelevant', 'args': {}, 'expect': {'empty': False}},
            {'id': 'f', 'category': 'ranking', 'args': {},
             'expect': {'constraints': {'audience_group': 'teens', 'max_price': '2000'}}},
            {'category': 'ranking', 'args': []},
        ]
        for item in bad:
            with self.subTest(case=item):
                self.assertTrue(run_eval.case_problems(item))

    def test_select_by_category_and_pattern(self):
        cases = [{'id': 'rank-01', 'category': 'ranking'}, {'id': 'rank-02', 'category': 'ranking'},
                 {'id': 'val-01', 'category': 'validity'}]
        self.assertEqual([c['id'] for c in run_eval.select_cases(cases, 'validity')], ['val-01'])
        self.assertEqual([c['id'] for c in run_eval.select_cases(cases, 'rank-0*, val-01')],
                         ['rank-01', 'rank-02', 'val-01'])
        with self.assertRaises(SystemExit):
            run_eval.select_cases(cases, 'nothing')


class ConstraintTests(unittest.TestCase):
    def test_audience_groups(self):
        groups = {pid: {g for g in run_eval.AUDIENCE_GROUPS if run_eval.in_audience_group(BY_ID[pid], g)}
                  for pid in BY_ID}
        self.assertEqual(groups['0001'], {'women'})
        self.assertEqual(groups['0003'], {'women'})
        self.assertEqual(groups['0004'], {'men'})
        self.assertEqual(groups['0005'], {'kids', 'girls'})
        self.assertEqual(groups['0006'], {'kids', 'girls', 'boys', 'baby'})
        self.assertEqual(groups['0007'], {'men'})
        self.assertEqual(groups['0008'], {'women'})

    def test_violations(self):
        jeans = BY_ID['0001']
        self.assertEqual(run_eval.violations(jeans, {'colors': ['black'], 'max_price': 999, 'min_price': 999,
                                                     'product_types': ['trousers'], 'audience_group': 'women',
                                                     'exclude_colors': []}), [])
        self.assertEqual(len(run_eval.violations(jeans, {'colors': ['White', 'Off White']})), 1)
        self.assertEqual(len(run_eval.violations(jeans, {'exclude_colors': ['Black']})), 1)
        self.assertEqual(len(run_eval.violations(jeans, {'max_price': 998.5})), 1)
        self.assertEqual(len(run_eval.violations(jeans, {'min_price': 1000})), 1)
        self.assertEqual(len(run_eval.violations(jeans, {'audience_group': 'men'})), 1)
        self.assertEqual(run_eval.violations(BY_ID['0008'], {'product_types': ['Dog Wear']}), [])

    def test_explicit_filters_use_the_oracle_rules(self):
        args = {'filters': {'color': ' black ', 'max_price': 999, 'min_rating': 4}}
        self.assertEqual(run_eval.primary_violations(BY_ID['0001'], args, {}), [])
        self.assertEqual(len(run_eval.primary_violations(BY_ID['0007'], args, {})), 1)  # rating 2.5
        self.assertEqual(len(run_eval.primary_violations(BY_ID['0002'], {'filters': {'min_rating': 1}}, {})), 1)


class StructureTests(unittest.TestCase):
    KNOWN = set(BY_ID)

    def problems(self, output, **args):
        return run_eval.structure_problems(output, args, self.KNOWN)

    def test_valid_pages(self):
        self.assertEqual(self.problems(out('0001', '0002')), [])
        self.assertEqual(self.problems(out('0001', recs=[('0003', '0001')]), include_recommendations=True), [])
        self.assertEqual(self.problems(out()), [])

    def test_each_rule(self):
        cases = [
            (out('0001', '0001'), {}, ['duplicate result']),
            (out('9999'), {}, ['not in the catalog']),
            (out('0001', '0002', '0003'), {'limit': 2}, ['exceed limit']),
            (out('0001', recs=[('0003', '0001')]), {}, ['include_recommendations is false']),
            (out(recs=[('0003', '0001')]), {'include_recommendations': True}, ['without primary', 'not among']),
            (out('0001', recs=[('0001', '0001')]), {'include_recommendations': True}, ['repeat primary']),
            (out('0001', recs=[('0003', '0002')]), {'include_recommendations': True}, ['not among']),
            (out('0001', recs=[('0003', '0001'), ('0003', '0001')]), {'include_recommendations': True},
             ['duplicate recommendation']),
        ]
        for output, args, expected in cases:
            with self.subTest(expected=expected):
                problems = self.problems(output, **args)
                self.assertEqual(len(problems), len(expected), problems)
                for words, problem in zip(expected, problems):
                    self.assertIn(words, problem)

    def test_schema(self):
        self.assertEqual(run_eval.schema_problems(out('0001')), [])
        self.assertTrue(run_eval.schema_problems({'results': [{'id': '0001', 'score': 1}], 'recommendations': []}))
        self.assertTrue(run_eval.schema_problems({'results': []}))


class OrderTests(unittest.TestCase):
    def check(self, ids, sort_by, order):
        return run_eval.order_problems(ids, sort_by, order, BY_ID)

    def test_price_ties_break_by_id_ascending_both_ways(self):
        self.assertEqual(self.check(['0006', '0002', '0001', '0004', '0007', '0003'], 'price', 'asc'), [])
        self.assertEqual(self.check(['0003', '0001', '0004', '0007', '0002'], 'price', 'desc'), [])
        self.assertTrue(self.check(['0003', '0004', '0001'], 'price', 'desc'))
        self.assertTrue(self.check(['0002', '0001'], 'price', 'desc'))

    def test_null_ratings_last_in_both_directions(self):
        self.assertEqual(self.check(['0003', '0008', '0001', '0004', '0005', '0007', '0002', '0006'],
                                    'rating', 'desc'), [])
        self.assertEqual(self.check(['0007', '0005', '0001', '0004', '0008', '0003', '0002', '0006'],
                                    'rating', 'asc'), [])
        self.assertTrue(self.check(['0002', '0003'], 'rating', 'desc'))
        self.assertTrue(self.check(['0006', '0002'], 'rating', 'asc'))

    def test_id_and_review_count(self):
        self.assertEqual(self.check(['0008', '0003', '0001'], 'id', 'desc'), [])
        self.assertTrue(self.check(['0001', '0003'], 'id', 'desc'))
        self.assertEqual(self.check(['0007', '0003', '0001', '0008', '0004'], 'review_count', 'desc'), [])
        self.assertIn('ranks 1-2', self.check(['0004', '0007'], 'review_count', 'desc')[0])


class ProbeTests(unittest.TestCase):
    def test_prefix_requests(self):
        cases = [((0, 10), (0, 20, 0)), ((10, 10), (0, 30, 10)), ((50, 50), (0, 100, 50)),
                 ((95, 10), (95, 20, 0)), ((5, 100), None), ((0, 100), None), ((0, 60), (0, 100, 0))]
        for (offset, limit), want in cases:
            with self.subTest(offset=offset, limit=limit):
                args, shift = run_eval.prefix_request({'query': 'x', 'offset': offset, 'limit': limit})
                got = None if args is None else (args['offset'], args['limit'], shift)
                self.assertEqual(got, want)
                if args is not None:
                    self.assertEqual(args['query'], 'x')

    def probe(self, main, repeat, prefix, shift, limit):
        args = {'query': 'x', 'limit': limit}
        calls = {'main': ok_call(main, args), 'repeat': ok_call(repeat, args),
                 'prefix': ok_call(prefix, {'offset': 0, 'limit': 2 * limit})}
        return run_eval.probe_problems(calls, shift, limit)

    def test_consistent_pages_pass(self):
        self.assertEqual(self.probe(out('0002', '0003'), out('0002', '0003'), out('0001', '0002', '0003', '0004'),
                                    1, 2), [])
        self.assertEqual(self.probe(out('0002'), out('0002'), out('0001', '0002'), 1, 2), [])

    def test_inconsistent_pages_fail(self):
        self.assertTrue(self.probe(out('0002', '0003'), out('0003', '0002'), out('0002', '0003'), 0, 2))
        self.assertTrue(self.probe(out('0002', '0003'), out('0002', '0003'), out('0002', '0004'), 0, 2))
        # A short page means the list ended there; a larger request may not show more.
        self.assertTrue(self.probe(out('0002'), out('0002'), out('0002', '0003'), 0, 2))


class EvaluateTests(unittest.TestCase):
    EMPTY = {'filters': {'color': 'Black'}, 'sort_by': 'price', 'sort_order': 'asc', 'limit': 2}

    def test_exact_empty_query_page(self):
        record = run(case('validity', dict(self.EMPTY)), out('0001', '0007'))
        self.assertEqual(record['validity_score'], 1.0, record['problems'])
        record = run(case('validity', dict(self.EMPTY)), out('0007', '0001'))
        self.assertEqual(record['checks']['sorting'], False)
        self.assertEqual(record['validity_score'], 0.75)

    def test_integral_float_limit_and_whitespace_query(self):
        record = run(case('validity', dict(self.EMPTY, limit=2.0, query='  ')), out('0001', '0007'))
        self.assertEqual(record['validity_score'], 1.0, record['problems'])

    def test_probe_case(self):
        args = dict(self.EMPTY)
        test_case = case('validity', args, probe=True)
        prefix_args, shift = run_eval.prefix_request(args)
        extra = {'repeat': ok_call(out('0001', '0007'), args),
                 'prefix': ok_call(out('0001', '0007', '0003'), prefix_args)}
        record = run(test_case, out('0001', '0007'), extra, shift)
        self.assertEqual(record['validity_score'], 1.0, record['problems'])
        record = run(test_case, out('0001', '0007'), extra, shift, case_ms=31_000)
        self.assertEqual(record['validity_score'], 0.0)

    def test_field_sort_with_a_query_checks_order_only(self):
        args = {'query': 'black', 'sort_by': 'price', 'sort_order': 'desc'}
        self.assertEqual(run(case('validity', args), out('0003', '0001'))['validity_score'], 1.0)
        self.assertEqual(run(case('validity', args), out('0001', '0003'))['checks']['sorting'], False)

    def test_invalid_call_cases(self):
        bad = case('validity', {'limit': 0}, error=True)
        for kind, score in (('tool_error', 1.0), ('rpc_error', 1.0), ('result', 0.0), ('timeout', 0.0),
                            ('transport', 0.0)):
            with self.subTest(kind=kind):
                call = dict(ok_call(out(), bad['args']), kind=kind)
                record = run_eval.evaluate(bad, {'main': call}, None, 1.0, CATALOG)
                self.assertEqual(record['validity_score'], score)

    def test_failed_calls_zero_the_case(self):
        for kind in ('timeout', 'transport', 'malformed', 'tool_error'):
            with self.subTest(kind=kind):
                call = {'kind': kind, 'ms': 1.0, 'args': {}, 'output': None, 'text_ok': False, 'error': 'x'}
                record = run_eval.evaluate(case('ranking'), {'main': call}, None, 1.0, CATALOG)
                self.assertEqual((record['validity_score'], record['structural_ok']), (0.0, False))

    def test_text_block_must_match(self):
        test_case = case('ranking')
        calls = {'main': dict(ok_call(out('0001'), test_case['args']), text_ok=False)}
        record = run_eval.evaluate(test_case, calls, None, 1.0, CATALOG)
        self.assertFalse(record['structural_ok'])

    def test_irrelevant_cases(self):
        self.assertTrue(run(case('irrelevant', {'query': 'spaceship engine'}), out())['correct_empty'])
        record = run(case('irrelevant', {'query': 'spaceship engine'}), out('0001'))
        self.assertFalse(record['correct_empty'])
        self.assertFalse(record['checks']['presence'])

    def test_constraints_flag_products(self):
        record = run(case('ranking', {'query': 'black jeans', 'filters': {'max_price': 1500}},
                          constraints={'colors': ['Black'], 'audience_group': 'women'}), out('0001', '0004', '0003'))
        self.assertEqual([r['ok'] for r in record['results']], [True, False, True])
        self.assertTrue(record['structural_ok'])
        self.assertFalse(record['checks']['constraints'])

    def test_mixed_choice_constraints(self):
        test_case = case('mixed', {'query': 'black jeans and white tees'}, choices=['black jeans', 'white tees'],
                         choice_constraints=[{'colors': ['Black']}, {'colors': ['White']}])
        record = run(test_case, out('0001', '0002', '0004'))
        self.assertEqual([r['ok'] for r in record['results']], [True, True, False])
        self.assertEqual([r['choice_ok'] for r in record['results']], [[True, False], [False, True], [False, False]])

    def test_rec_constraints(self):
        test_case = case('recs', {'query': 'jeans with a matching bag', 'include_recommendations': True},
                         rec_constraints={'max_price': 1000})
        record = run(test_case, out('0001', recs=[('0003', '0001')]))
        self.assertEqual(record['recommendations'][0]['ok'], False)
        self.assertTrue(record['structural_ok'])


class ScoreRunTests(unittest.TestCase):
    def test_scores_every_category(self):
        records = [
            run(dict(case('validity', {'filters': {'color': 'Black'}, 'limit': 2}), id='val'), out('0001', '0003')),
            run(dict(case('irrelevant', {'query': 'spaceship'}), id='irr'), out()),
            run(dict(case('ranking', {'query': 'jeans'}, constraints={'colors': ['Black']}), id='rank'),
                out('0001', '0004')),
            run(dict(case('mixed', {'query': 'jeans and tees'}, choices=['jeans', 'tees']), id='mix'),
                out('0001', '0002')),
            run(dict(case('recs', {'query': 'jeans with a bag', 'include_recommendations': True},
                          complement='bag'), id='rec'), out('0001', '0004', recs=[('0003', '0001'), ('0002', '0004')])),
            run(dict(case('unusual', {'query': 'comfy'}), id='unu'), out('0001', '0001')),
        ]
        grades = {
            'rank': {'primary': {'0001': 3, '0004': 3}},
            'mix': {'primary': {'0001': 3, '0002': 2}, 'choices': {'0001': {'jeans': 3, 'tees': 0},
                                                                   '0002': {'jeans': 0, 'tees': 3}}},
            'rec': {'primary': {'0001': 3, '0004': 1}, 'recs': {'0003': 3, '0002': 3}},
        }
        summary = run_eval.score_run(records, grades)
        scores = {c['id']: c['score'] for c in summary['cases']}
        self.assertEqual(scores['val'], 1.0)
        self.assertEqual(scores['irr'], 1.0)
        self.assertAlmostEqual(scores['rank'], 7 / IDEAL10, places=12)           # 0004 is not black
        self.assertAlmostEqual(scores['mix'], 0.5 + 0.5 * (7 + 3 / LOG3) / IDEAL10, places=12)
        rec_quality = 7 / IDEAL5                                                 # 0002's anchor graded 1
        self.assertAlmostEqual(scores['rec'], ((7 + 1 / LOG3) / IDEAL10 + rec_quality) / 2, places=12)
        self.assertEqual(scores['unu'], 0.0)                                     # duplicate ids
        self.assertEqual(summary['missing_grades'], {})
        self.assertEqual(summary['categories']['ranking']['cases'], 1)
        self.assertAlmostEqual(summary['functional'], 25 + 7.5 + 30 * scores['rank'] + 15 * scores['mix']
                               + 10 * scores['rec'], places=9)

    def test_choice_constraints_gate_coverage(self):
        # The black sports top fits the "black jeans" choice's colour, so it may stay in the
        # list, but it breaks "white tees", so its grade 3 for that choice cannot cover it.
        test_case = case('mixed', {'query': 'black jeans and white tees'}, choices=['black jeans', 'white tees'],
                         choice_constraints=[{'colors': ['Black']}, {'colors': ['White']}])
        grades = {'c1': {'primary': {'0001': 3, '0007': 3},
                         'choices': {'0001': {'black jeans': 3, 'white tees': 0},
                                     '0007': {'black jeans': 0, 'white tees': 3}}}}
        summary = run_eval.score_run([run(test_case, out('0001', '0007'))], grades)
        self.assertEqual(summary['cases'][0]['parts']['coverage'], 0.5)
        self.assertAlmostEqual(summary['cases'][0]['score'], 0.25 + 0.5 * (7 + 7 / LOG3) / IDEAL10, places=12)
        self.assertEqual(summary['missing_grades'], {})

    def test_missing_grades_count_as_zero(self):
        record = run(case('ranking'), out('0001', '0004'))
        summary = run_eval.score_run([record], {'c1': {'primary': {'0001': 3}}})
        self.assertAlmostEqual(summary['cases'][0]['score'], 7 / IDEAL10, places=12)
        self.assertEqual(summary['missing_grades'], {'c1': 1})

    def test_no_complement_case_with_empty_recs(self):
        test_case = case('recs', {'query': 'swimsuit', 'include_recommendations': True}, complement='none')
        summary = run_eval.score_run([run(test_case, out('0001'))], {'c1': {'primary': {'0001': 3}}})
        self.assertAlmostEqual(summary['cases'][0]['score'], (7 / IDEAL10 + 1.0) / 2, places=12)

    def test_twin_pairs_join_single_intent_results(self):
        records = [run(case('ranking', {'query': 'black  jeans '}), out('0001', '0004')),
                   run(dict(case('mixed', {'query': 'black jeans'}, choices=['a', 'b']), id='c2'), out('0001'))]
        grades = {'c1': {'primary': {'0001': 3, '0004': 1}}, 'c2': {'primary': {'0001': 3}}}
        lines = ['{"query": "black jeans", "pid": "0001", "twin_grades": [3, 2]}',
                 'not json',
                 '{"query": "black jeans", "pid": "0004", "twin_grades": [0]}',
                 '{"query": "black jeans", "pid": "0004", "twin_grades": [1]}',
                 '{"query": "other", "pid": "0001", "twin_grades": [0]}']
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'grades.jsonl'
            path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
            pairs = run_eval.twin_pairs(records, grades, path)
        self.assertEqual([(p['case'], p['pid'], p['twin'], p['judge']) for p in pairs],
                         [('c1', '0001', 3, 3), ('c1', '0001', 2, 3), ('c1', '0004', 1, 1)])

    def test_invalid_grade_values_stop_scoring(self):
        with self.assertRaises(SystemExit):
            run_eval.score_run([run(case('ranking'), out('0001'))], {'c1': {'primary': {'0001': 7}}})


class PacketAndReportTests(unittest.TestCase):
    def records(self):
        return [
            run(dict(case('mixed', {'query': 'jeans and tees'}, choices=['jeans', 'tees']), id='mix'),
                out('0001', '0002')),
            run(dict(case('recs', {'query': 'jeans with a bag', 'include_recommendations': True},
                          rec_constraints={'max_price': 1000}), id='rec'), out('0001', recs=[('0003', '0001')])),
            run(dict(case('ranking'), id='empty'), out()),
            run(dict(case('validity', {'filters': {'color': 'Black'}, 'limit': 2}), id='val'), out('0003', '0001')),
        ]

    def test_build_packet(self):
        packet = run_eval.build_packet(self.records(), CATALOG, {'url': 'x'})
        self.assertEqual([c['id'] for c in packet['cases']], ['mix', 'rec'])
        self.assertEqual(packet['skipped'], [{'id': 'empty', 'reason': 'no results, nothing to grade'}])
        self.assertEqual(packet['meta']['grades_needed'], 2 * 3 + 1 + 1)
        rec = packet['cases'][1]['recommendations'][0]
        self.assertEqual((rec['id'], rec['for_product_id'], rec['constraint_ok']), ('0003', '0001', False))
        self.assertTrue(rec['anchor_card'].startswith('Slim jeans | type: Trousers'))
        self.assertEqual(packet['grades_template']['mix']['choices']['0002'], {'jeans': None, 'tees': None})
        self.assertEqual(packet['grades_template']['rec'],
                         {'primary': {'0001': None}, 'choices': {}, 'recs': {'0003': None}})
        json.dumps(packet)

    def test_reports_render(self):
        records = self.records()
        grades = {'mix': {'primary': {'0001': 3, '0002': 3}, 'choices': {'0001': {'jeans': 3}}},
                  'rec': {'primary': {'0001': 3}, 'recs': {'0003': 3}}}
        summary = run_eval.score_run(records, grades, perf=(1.0, 300))
        summary['perf'] = {'correct_rate': 1.0, 'p95_ms': 300}
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            run_eval.print_checks({'discovery': {'ok': True, 'protocol': '2025-11-25', 'server': 's', 'ms': 1.0},
                                   'cases': records})
            run_eval.print_score(summary, records, worst=10)
        text = buffer.getvalue()
        self.assertIn('R1. g3 X Shoulder bag', text)
        self.assertIn('choices [3, 0]', text)
        self.assertIn('3 grades missing', text)
        self.assertIn('val              0.75  sorting:', text)
        json.dumps(summary)


class StubLLM:
    """Answers judge prompts with grade 3 for jeans and 0 otherwise; can fail the first calls."""

    def __init__(self, failures=0):
        self.failures, self.calls, self.lock = failures, [], threading.Lock()

    def complete(self, system, user, max_tokens, timeout, model=None):
        with self.lock:
            self.calls.append((system, user, model))
            if len(self.calls) <= self.failures:
                return 'not json'
        cards = [line for line in user.splitlines() if line.startswith('[')]
        grades = [{'n': n, 'g': 3 if 'jeans' in line.lower() else 0} for n, line in enumerate(cards, 1)]
        return json.dumps({'grades': grades})


def extract_json(text):
    start, end = text.find('{'), text.rfind('}')
    try:
        return json.loads(text[start:end + 1]) if start >= 0 < end else None
    except ValueError:
        return None


class LocalJudgeTests(unittest.TestCase):
    def setUp(self):
        self.stubbed = False
        try:
            import engine.llm  # noqa: F401
        except ImportError:  # engine/llm.py not written yet: provide its extract_json
            sys.modules['engine.llm'] = types.SimpleNamespace(extract_json=extract_json)
            self.stubbed = True

    def tearDown(self):
        if self.stubbed:
            del sys.modules['engine.llm']

    def packet(self):
        cards = {pid: local_judge.fact_card(p) for pid, p in BY_ID.items()}
        return {'cases': [
            {'id': 'mix', 'category': 'mixed', 'query': 'jeans and tees', 'args': {'query': 'jeans and tees'},
             'choices': ['jeans', 'tees'],
             'results': [{'id': '0001', 'card': cards['0001']}, {'id': '0002', 'card': cards['0002']}]},
            {'id': 'rec', 'category': 'recs', 'query': 'jeans with a bag', 'args': {'query': 'jeans with a bag'},
             'choices': [], 'results': [{'id': '0001', 'card': cards['0001']}],
             'recommendations': [{'id': '0003', 'for_product_id': '0001', 'card': cards['0003'],
                                  'anchor_card': cards['0001']}]},
        ]}

    def test_fact_card(self):
        self.assertEqual(local_judge.fact_card(BY_ID['0001']),
                         'Slim jeans | type: Trousers | category: Garment Lower body | audience: Ladieswear | '
                         'section: Womens Everyday Collection | colour: Black | pattern: Denim | price: INR 999 | '
                         'rating: 4 (2) | Five-pocket jeans in stretch denim.')
        self.assertIn('rating: no reviews (0)', local_judge.fact_card(BY_ID['0002']))
        self.assertEqual(local_judge.rating_text(3.25), '3.25')
        self.assertEqual(local_judge.rating_text(4.5), '4.5')
        with_extras = dict(BY_ID['0001'], reviews=[{'title': 'Great', 'text': 'great'}], image_url='http://x')
        self.assertEqual(local_judge.fact_card(with_extras), local_judge.fact_card(BY_ID['0001']))

    def test_parse_grades(self):
        reply = 'Sure: {"grades": [{"n": 1, "g": 3}, {"n": 2, "g": "2"}, {"n": 3, "g": 5}, {"n": 9, "g": 1}, ' \
                '{"n": 1, "g": 0}, {"n": 4.0, "g": 1.0}]}'
        self.assertEqual(local_judge.parse_grades(reply, 4, extract_json), {1: 3, 2: 2, 4: 1})
        self.assertEqual(local_judge.parse_grades(None, 4, extract_json), {})
        self.assertEqual(local_judge.parse_grades('{"grades": "x"}', 4, extract_json), {})

    def test_tasks_and_batches(self):
        tasks = local_judge.grading_tasks(self.packet(), batch_size=1)
        self.assertEqual(Counter(t['kind'] for t in tasks), {'primary': 3, 'choice': 4, 'rec': 1})
        rec = next(t for t in tasks if t['kind'] == 'rec')
        prompt = local_judge.user_prompt(rec, rec['items'])
        self.assertIn('Main product the shopper is buying: Slim jeans', prompt)
        self.assertIn('[1] Shoulder bag', prompt)
        choice = next(t for t in tasks if t['kind'] == 'choice')
        self.assertIn('requested choice only: "jeans"', local_judge.user_prompt(choice, choice['items']))

    def test_grade_packet(self):
        llm = StubLLM()
        grades = local_judge.grade_packet(self.packet(), llm, 'judge-model', workers=3)
        self.assertEqual(grades, {
            'mix': {'primary': {'0001': 3, '0002': 0}, 'choices': {'0001': {'jeans': 3, 'tees': 3},
                                                                   '0002': {'jeans': 0, 'tees': 0}}, 'recs': {}},
            'rec': {'primary': {'0001': 3}, 'choices': {}, 'recs': {'0003': 0}},
        })
        self.assertEqual(len(llm.calls), 5)
        self.assertTrue(all(model == 'judge-model' and system == local_judge.SYSTEM_PROMPT
                            for system, _, model in llm.calls))

    def test_make_judge_refuses_a_disabled_provider(self):
        if self.stubbed:
            self.skipTest('engine/llm.py is not available')
        with self.assertRaises(SystemExit):
            local_judge.make_judge('none')

    def test_grade_packet_retries_and_gives_up(self):
        grades = local_judge.grade_packet(self.packet(), StubLLM(failures=1), workers=1)
        self.assertEqual(grades['mix']['primary'], {'0001': 3, '0002': 0})
        with self.assertLogs('polaris.local_judge', level='WARNING') as logs:
            grades = local_judge.grade_packet(self.packet(), StubLLM(failures=100), workers=2)
        self.assertEqual(grades['mix'], {'primary': {}, 'choices': {}, 'recs': {}})
        self.assertEqual(len(logs.records), 5)


if __name__ == '__main__':
    unittest.main()
