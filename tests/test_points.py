"""Points Engine checks: brute force proves each decision maximizes the expected score.

No catalog is needed; the whole file runs in about a second.
"""
import itertools
import json
import math
import random
import time
import unittest

from engine.points import (
    DEFAULT_CALIBRATION, GAINS, Candidate, build_mixed_list, calibration_from_json,
    calibration_to_json, choose_anchor, choose_field_sorted_set, coverage_prob, expected_coverage,
    expected_dcg, expected_gain, expected_ndcg, expected_rec_score, fit_calibration, grade_probs,
    ideal_dcg, mixed_objective, order_by_expected_gain, p_answerable, p_at_least,
    select_recommendations, should_abstain)

TWIN = DEFAULT_CALIBRATION['twin']
ESTIMATE = DEFAULT_CALIBRATION['estimate']
CERTAIN = {g: tuple(1.0 if i == g else 0.0 for i in range(4)) for g in range(4)}


def random_probs(rng):
    """A random grade distribution; often an exact calibration row, so gains tie."""
    if rng.random() < 0.3:
        return DEFAULT_CALIBRATION[rng.choice(['twin', 'estimate'])][rng.randint(0, 3)]
    weights = [rng.random() ** rng.choice([1, 2, 4]) for _ in range(4)]
    total = sum(weights)
    return tuple(w / total for w in weights)


def random_candidates(rng, n, n_choices=1):
    return [Candidate(pid=pid, probs=random_probs(rng), choice=rng.randrange(n_choices),
                      prior=rng.choice([0.0, 0.5, round(rng.random(), 3)]),
                      choice_probs=random_probs(rng) if rng.random() < 0.5 else None)
            for pid in rng.sample(range(10_000), n)]


def shuffled(rng, items):
    items = list(items)
    rng.shuffle(items)
    return items


def key_by_pid(keys):
    """A total sort_key for choose_field_sorted_set from a pid -> sort value map."""
    return lambda cand: keys[cand.pid]


def pids(cands):
    return [c.pid for c in cands]


class ProbsAssertions(unittest.TestCase):
    def assertProbs(self, got, want, places=12):
        self.assertEqual(len(got), 4)
        for g, w in zip(got, want):
            self.assertAlmostEqual(g, w, places=places)


class ScoreFormulaTests(ProbsAssertions):
    def test_gains_and_fixed_ideal(self):
        self.assertEqual(GAINS, (0.0, 1.0, 3.0, 7.0))
        self.assertEqual(ideal_dcg(1), 7.0)
        self.assertAlmostEqual(ideal_dcg(10), 7 * sum(1 / math.log2(r + 1) for r in range(1, 11)))
        self.assertEqual(ideal_dcg(0), 0.0)

    def test_expected_gain_and_tail_probabilities(self):
        probs = (0.1, 0.2, 0.3, 0.4)
        self.assertAlmostEqual(expected_gain(probs), 0.2 + 0.9 + 2.8)
        self.assertAlmostEqual(p_at_least(probs, 2), 0.7)
        self.assertAlmostEqual(p_at_least(probs, 0), 1.0)
        self.assertEqual(p_at_least(probs, 4), 0.0)
        for g in range(4):
            self.assertEqual(expected_gain(CERTAIN[g]), GAINS[g])

    def test_fixed_ideal_and_missing_slots(self):
        threes = [Candidate(pid, CERTAIN[3]) for pid in range(12)]
        self.assertAlmostEqual(expected_ndcg(threes, 10), 1.0)
        self.assertAlmostEqual(expected_ndcg(threes[:5], 5), 1.0)
        # Missing slots earn nothing: five perfect results are not a perfect top ten.
        self.assertAlmostEqual(expected_ndcg(threes[:5], 10), ideal_dcg(5) / ideal_dcg(10))
        self.assertEqual(expected_ndcg([], 10), 0.0)
        self.assertAlmostEqual(expected_ndcg([Candidate(0, CERTAIN[2])], 10), 3 / ideal_dcg(10))
        # Only the first k count.
        self.assertEqual(expected_dcg([Candidate(0, CERTAIN[0])] * 10 + threes, 10), 0.0)


class OrderTests(ProbsAssertions):
    def test_order_maximizes_expected_ndcg_over_all_permutations(self):
        # Every permutation of the whole set also covers every choice of which
        # products fill the first k slots, so this proves the selection too.
        rng = random.Random(20261021)
        for trial in range(120):
            cands = random_candidates(rng, rng.randint(1, 7))
            k = rng.randint(1, 10)
            ours = expected_ndcg(order_by_expected_gain(cands, min_gain=0.0), k)
            best = max(expected_ndcg(perm, k) for perm in itertools.permutations(cands))
            self.assertGreaterEqual(ours, best - 1e-12, f'trial {trial}')

    def test_min_gain_drops_only_weak_candidates(self):
        rng = random.Random(5)
        for _ in range(50):
            cands = random_candidates(rng, rng.randint(1, 7))
            got = order_by_expected_gain(cands, min_gain=0.5)
            kept = [c for c in cands if expected_gain(c.probs) >= 0.5]
            self.assertEqual(sorted(pids(got)), sorted(pids(kept)))
            if kept:
                best = max(expected_ndcg(perm) for perm in itertools.permutations(kept))
                self.assertGreaterEqual(expected_ndcg(got), best - 1e-12)
        # Products we predict irrelevant leave an empty list, as an irrelevant case needs.
        self.assertEqual(order_by_expected_gain([Candidate(p, grade_probs([0])) for p in range(20)]), [])

    def test_ties_break_by_prior_then_pid(self):
        row = TWIN[2]
        cands = [Candidate(5, row, prior=0.1), Candidate(3, row, prior=0.9),
                 Candidate(1, row, prior=0.1), Candidate(9, TWIN[3])]
        self.assertEqual(pids(order_by_expected_gain(cands)), [9, 3, 1, 5])

    def test_split_graders_beat_an_agreed_two(self):
        # Grades 3 and 1 average to a higher expected gain than two 2s.
        split, agreed = Candidate(1, grade_probs([3, 1])), Candidate(0, grade_probs([2, 2]))
        self.assertGreater(expected_gain(split.probs), expected_gain(agreed.probs))
        self.assertEqual(pids(order_by_expected_gain([agreed, split])), [1, 0])

    def test_deterministic_and_duplicate_free(self):
        rng = random.Random(9)
        cands = random_candidates(rng, 30)
        cands += [Candidate(cands[0].pid, TWIN[0]), Candidate(cands[1].pid, TWIN[3])]
        first = order_by_expected_gain(cands)
        for _ in range(5):
            self.assertEqual(order_by_expected_gain(shuffled(rng, cands)), first)
        self.assertEqual(len(pids(first)), len(set(pids(first))))
        kept = {c.pid: c for c in first}
        for pid in (cands[0].pid, cands[1].pid):
            best = max(expected_gain(c.probs) for c in cands if c.pid == pid)
            if best >= 0.5:   # the best-placed copy survives
                self.assertEqual(expected_gain(kept[pid].probs), best)


class FieldSortTests(ProbsAssertions):
    @staticmethod
    def brute_force(cands, sort_key, k):
        """Best E[DCG@k] over every subset, shown in sort order."""
        ordered = sorted(cands, key=sort_key)
        return max(expected_dcg([c for i, c in enumerate(ordered) if mask >> i & 1], k)
                   for mask in range(1 << len(ordered)))

    def test_first_k_are_optimal_and_later_pages_follow_the_last_chosen(self):
        rng = random.Random(77)
        for trial in range(150):
            n = rng.randint(1, 10)
            cands = random_candidates(rng, n)
            sort_key = key_by_pid(dict(zip(pids(cands), rng.sample(range(1000), n))))
            k = rng.choice([1, 2, 3, 4, 5, 10])
            min_gain = rng.choice([0.0, 0.5])
            got = choose_field_sorted_set(cands, sort_key, k=k, min_gain=min_gain)
            eligible = [c for c in cands if expected_gain(c.probs) >= min_gain]
            msg = f'trial {trial}'
            # No subset of the eligible candidates shown in sort order beats the first k.
            self.assertGreaterEqual(expected_dcg(got, k), self.brute_force(eligible, sort_key, k) - 1e-9, msg)
            # Sort order, no duplicates, nothing below min_gain.
            keys = [sort_key(c) for c in got]
            self.assertEqual(keys, sorted(set(keys)), msg)
            self.assertTrue(all(expected_gain(c.probs) >= min_gain for c in got), msg)
            if not got:
                self.assertEqual(eligible, [], msg)
                continue
            # Later-page items come after the last chosen item, and every eligible
            # candidate after it is kept; skipped ones all sort before it.
            last = sort_key(got[min(k, len(got)) - 1])
            self.assertTrue(all(sort_key(c) > last for c in got[k:]), msg)
            shown = set(pids(got))
            self.assertTrue(all(c.pid in shown for c in eligible if sort_key(c) > last), msg)

    def test_weak_early_products_do_not_push_a_strong_one_down(self):
        weak1, weak2, strong = Candidate(1, TWIN[1]), Candidate(2, TWIN[1]), Candidate(3, CERTAIN[3])
        key = key_by_pid({1: 10, 2: 20, 3: 30})
        for k in (1, 3, 10):
            self.assertEqual(pids(choose_field_sorted_set([weak1, weak2, strong], key, k=k)), [3])
        # Two strong products: the weak one between them is dropped too.
        s1, w, s2 = Candidate(1, CERTAIN[3]), Candidate(2, TWIN[1]), Candidate(3, CERTAIN[3])
        self.assertEqual(pids(choose_field_sorted_set([s2, w, s1], key, k=3)), [1, 3])
        # Once k are chosen, everything after the last chosen follows for later pages.
        key = key_by_pid({1: 10, 3: 20, 2: 30})
        self.assertEqual(pids(choose_field_sorted_set([w, s2, s1], key, k=1)), [1, 3, 2])

    def test_junk_duplicates_and_input_order(self):
        key = key_by_pid({p: -p for p in range(40)})
        self.assertEqual(choose_field_sorted_set([Candidate(p, TWIN[0]) for p in range(10)], key), [])
        self.assertEqual(choose_field_sorted_set([], key), [])
        rng = random.Random(12)
        cands = random_candidates(rng, 25)
        key = key_by_pid({c.pid: rng.random() for c in cands})
        cands.append(Candidate(cands[0].pid, CERTAIN[3]))
        first = choose_field_sorted_set(cands, key)
        self.assertEqual(len(pids(first)), len(set(pids(first))))
        for _ in range(5):
            self.assertEqual(choose_field_sorted_set(shuffled(rng, cands), key), first)


class MixedTests(ProbsAssertions):
    @staticmethod
    def brute_force(cands, n_choices, k):
        """Best mixed_objective over every ordered list of at most k candidates."""
        return max(mixed_objective(perm, n_choices, k)
                   for size in range(min(k, len(cands)) + 1)
                   for perm in itertools.permutations(cands, size))

    def test_reaches_the_brute_force_optimum(self):
        rng = random.Random(4242)
        for trial in range(60):
            n_choices = rng.randint(2, 3)
            cands = random_candidates(rng, rng.randint(1, 9), n_choices)
            k = rng.randint(1, 4)
            got = mixed_objective(build_mixed_list(cands, n_choices, k=k), n_choices, k)
            self.assertGreaterEqual(got, 0.99 * self.brute_force(cands, n_choices, k), f'trial {trial}')

    def test_local_search_fixes_a_greedy_first_pick(self):
        # Greedy first takes a sure 3 that only flips a coin on suiting its choice;
        # the best pair swaps it for a sure 2 that certainly covers choice 0.
        flip = (0.5, 0.0, 0.5, 0.0)
        cands = [Candidate(0, CERTAIN[2], choice=0), Candidate(1, CERTAIN[3], choice=0, choice_probs=flip),
                 Candidate(2, CERTAIN[3], choice=1, choice_probs=flip)]
        got = build_mixed_list(cands, 2, k=2)
        self.assertEqual(pids(got), [2, 0, 1])
        self.assertAlmostEqual(mixed_objective(got, 2, 2), self.brute_force(cands, 2, 2))
        self.assertGreater(mixed_objective(got, 2, 2), mixed_objective(cands[1:], 2, 2))

    def test_one_choice_is_order_by_expected_gain(self):
        rng = random.Random(8)
        for _ in range(20):
            cands = random_candidates(rng, rng.randint(0, 40))
            want = order_by_expected_gain(cands, min_gain=0.5)
            for n_choices in (0, 1):
                self.assertEqual(build_mixed_list(cands, n_choices), want)
            self.assertEqual(build_mixed_list(cands, 1, depth=7), want[:7])

    def test_covers_a_weaker_choice(self):
        # Covering choice 1 is worth more than a tenth sure hit for choice 0.
        strong = [Candidate(pid, TWIN[3], choice=0) for pid in range(20)]
        other = Candidate(100, TWIN[2], choice=1)
        got = build_mixed_list(strong + [other], 2)
        self.assertEqual(got[9].pid, 100)   # in the top ten, placed by expected gain
        self.assertEqual(len(got), 21)
        self.assertGreater(mixed_objective(got, 2), mixed_objective(strong, 2))

    def test_deterministic_duplicate_free_depth_and_tail(self):
        rng = random.Random(31)
        cands = random_candidates(rng, 60, 3)
        # The same product pooled for two choices: at most one copy may appear.
        cands += [Candidate(c.pid, c.probs, choice=(c.choice + 1) % 3, prior=c.prior) for c in cands[:10]]
        first = build_mixed_list(cands, 3)
        for _ in range(3):
            self.assertEqual(build_mixed_list(shuffled(rng, cands), 3), first)
        self.assertEqual(len(pids(first)), len(set(pids(first))))
        self.assertEqual(build_mixed_list(cands, 3, depth=12), first[:12])
        # After the first ten, the rest follow by expected gain without weak products.
        head = set(pids(first[:10]))
        self.assertEqual(first[10:], [c for c in order_by_expected_gain(cands) if c.pid not in head])
        self.assertEqual(build_mixed_list([], 3), [])

    def test_fast_at_production_size(self):
        rng = random.Random(1)
        cands = random_candidates(rng, 200, 4)
        start = time.perf_counter()
        build_mixed_list(cands, 4)
        self.assertLess(time.perf_counter() - start, 0.5)   # about 10 ms in practice

    def test_coverage_probability_and_objective(self):
        self.assertAlmostEqual(coverage_prob(Candidate(0, TWIN[3])), 0.90)
        self.assertAlmostEqual(coverage_prob(Candidate(0, TWIN[3], choice_probs=TWIN[1])), 0.25)
        self.assertAlmostEqual(coverage_prob(Candidate(0, TWIN[1], choice_probs=CERTAIN[3])), 0.25)
        cands = [Candidate(0, TWIN[3], choice=0), Candidate(1, TWIN[2], choice=0),
                 Candidate(2, TWIN[2], choice=1)]
        both = 1 - 0.1 * 0.28
        self.assertAlmostEqual(expected_coverage(cands, 2), (both + 0.72) / 2)
        self.assertAlmostEqual(expected_coverage(cands, 3), (both + 0.72) / 3)
        self.assertAlmostEqual(expected_coverage(cands, 2, k=1), 0.90 / 2)
        self.assertAlmostEqual(mixed_objective(cands, 2),
                               0.5 * expected_coverage(cands, 2) + 0.5 * expected_ndcg(cands))


class CalibrationTests(ProbsAssertions):
    def test_grade_probs_rows_averaging_and_normalization(self):
        for source, rows in DEFAULT_CALIBRATION.items():
            for g, row in rows.items():
                self.assertProbs(grade_probs([g], source=source), row)
        self.assertProbs(grade_probs([3, 1]), [(a + b) / 2 for a, b in zip(TWIN[3], TWIN[1])])
        self.assertEqual(grade_probs([1, 3, 2]), grade_probs([2, 3, 1]))   # bit for bit
        self.assertAlmostEqual(sum(grade_probs([0, 1, 2, 3, 3])), 1.0, places=12)
        # Unnormalized rows are normalized before averaging.
        cal = {'twin': {1: (0, 0, 1, 1), 3: (0, 0, 0, 5)}}
        self.assertProbs(grade_probs([1], cal), (0, 0, 0.5, 0.5))
        self.assertProbs(grade_probs([1, 3], cal), (0, 0, 0.25, 0.75))
        # Rows a calibration lacks come from the defaults.
        self.assertProbs(grade_probs([2], cal), TWIN[2])

    def test_no_usable_grade_gives_the_uninformative_row(self):
        for grades in ([], [None], ['x', float('nan')]):
            self.assertProbs(grade_probs(grades), ESTIMATE[1])
            self.assertProbs(grade_probs(grades, source='estimate'), ESTIMATE[1])
        self.assertProbs(grade_probs([], {'estimate': {1: (1, 1, 1, 1)}}), (0.25,) * 4)
        # Out-of-range grades clamp; numeric strings and str keys (raw JSON) work.
        self.assertProbs(grade_probs([5]), TWIN[3])
        self.assertProbs(grade_probs([-1]), TWIN[0])
        self.assertProbs(grade_probs(['2']), TWIN[2])
        self.assertProbs(grade_probs([2], json.loads('{"twin": {"2": [0, 0, 1, 0]}}')), CERTAIN[2])

    def test_default_rows_are_monotone(self):
        for rows in DEFAULT_CALIBRATION.values():
            self.assertEqual(sorted(rows), [0, 1, 2, 3])
            for t in range(4):
                self.assertAlmostEqual(sum(rows[t]), 1.0, places=12)
            for t in range(3):
                self.assertLess(expected_gain(rows[t]), expected_gain(rows[t + 1]))
                for g in (1, 2, 3):
                    self.assertLessEqual(p_at_least(rows[t], g), p_at_least(rows[t + 1], g))
            self.assertLess(expected_gain(rows[0]), 0.5)   # dropped by the default min_gain
        # A twin 3 is more trustworthy than a rule estimate of 3.
        self.assertGreater(TWIN[3][3], ESTIMATE[3][3])

    def test_fit_calibration_recovers_a_known_confusion_matrix(self):
        truth = {0: (0.70, 0.20, 0.08, 0.02), 1: (0.20, 0.50, 0.20, 0.10),
                 2: (0.05, 0.15, 0.60, 0.20), 3: (0.02, 0.03, 0.15, 0.80)}
        rng = random.Random(3)
        pairs = []
        for _ in range(40_000):
            t = rng.randrange(4)
            pairs.append((t, rng.choices(range(4), weights=truth[t])[0]))
        fitted = fit_calibration(pairs)
        for t in range(4):
            for got, want in zip(fitted['twin'][t], truth[t]):
                self.assertAlmostEqual(got, want, delta=0.02)
        self.assertEqual(fitted['estimate'], ESTIMATE)
        self.assertProbs(grade_probs([3], fitted), fitted['twin'][3])

    def test_fit_calibration_smoothing_and_base(self):
        before = json.dumps(calibration_to_json(DEFAULT_CALIBRATION))
        fitted = fit_calibration([])
        for t in range(4):
            self.assertProbs(fitted['twin'][t], TWIN[t])
        # Heavy smoothing keeps a row near its base row.
        fitted = fit_calibration([(3, 0)] * 2, smoothing=100)
        self.assertLess(abs(fitted['twin'][3][0] - TWIN[3][0]), 0.03)
        # Unusable pairs are ignored; a row with no data and no smoothing keeps its base.
        fitted = fit_calibration([(2, 3), (None, 1), ('x', 2)] * 50, source='estimate', smoothing=0)
        self.assertProbs(fitted['estimate'][2], CERTAIN[3])
        self.assertProbs(fitted['estimate'][1], ESTIMATE[1])
        self.assertEqual(fitted['twin'], TWIN)
        # A custom base is used and never mutated.
        base = fit_calibration([(1, 3)] * 30)
        refit = fit_calibration([], source='estimate', base=base)
        self.assertEqual(refit['twin'], base['twin'])
        self.assertEqual(json.dumps(calibration_to_json(DEFAULT_CALIBRATION)), before)

    def test_json_round_trip(self):
        fitted = fit_calibration([(t, (t + 1) % 4) for t in range(4)] * 7)
        text = json.dumps(calibration_to_json(fitted))
        self.assertEqual(calibration_from_json(json.loads(text)), fitted)
        self.assertEqual(calibration_from_json(calibration_to_json(DEFAULT_CALIBRATION)), DEFAULT_CALIBRATION)
        for bad in ({'twin': {'0': [0.5, 0.5]}}, {'twin': {'4': [1, 0, 0, 0]}},
                    {'twin': {'1': [-1, 1, 0.5, 0.5]}}, {'twin': {'2': [0, 0, 0, 0]}},
                    {'twin': {'2': [float('nan'), 1, 0, 0]}}, {'twin': {'x': [1, 0, 0, 0]}}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                calibration_from_json(bad)


class DecisionTests(ProbsAssertions):
    def test_p_answerable(self):
        self.assertEqual(p_answerable([]), 0.0)
        self.assertAlmostEqual(p_answerable([Candidate(0, TWIN[2])]), 0.72)
        self.assertAlmostEqual(p_answerable([Candidate(0, TWIN[1]), Candidate(1, TWIN[1])]), 1 - 0.75 ** 2)
        self.assertEqual(p_answerable([Candidate(0, CERTAIN[0])] * 10 + [Candidate(1, CERTAIN[3])]), 0.0)

    def test_should_abstain_at_the_extremes(self):
        # Every product predicted irrelevant: the list empties; even the raw list says abstain.
        junk = [Candidate(pid, grade_probs([0, 0])) for pid in range(30)]
        self.assertEqual(order_by_expected_gain(junk), [])
        self.assertTrue(should_abstain(p_answerable([]), expected_ndcg([])))
        self.assertTrue(should_abstain(p_answerable(junk), expected_ndcg(junk)))
        self.assertTrue(should_abstain(p_answerable(junk), mixed_objective(junk, 2)))
        # A confident grade-3 list answers.
        hits = order_by_expected_gain([Candidate(pid, grade_probs([3, 3])) for pid in range(30)])
        self.assertFalse(should_abstain(p_answerable(hits), expected_ndcg(hits)))
        self.assertFalse(should_abstain(p_answerable(hits), mixed_objective(hits, 2)))
        # The formula and its weights.
        self.assertTrue(should_abstain(0.0, 1.0))
        self.assertFalse(should_abstain(1.0, 0.0))
        self.assertTrue(should_abstain(0.5, 0.5))                  # 0.375 > 0.25
        self.assertFalse(should_abstain(0.5, 0.5, w_answer=1.6))   # 0.375 < 0.40
        self.assertFalse(should_abstain(0.5, 0.5, w_abstain=0.4))  # 0.20 < 0.25

    def test_choose_anchor(self):
        self.assertIsNone(choose_anchor([]))
        a, b, c = Candidate(1, TWIN[2]), Candidate(2, TWIN[3]), Candidate(3, TWIN[3])
        self.assertIs(choose_anchor([a, b, c]), b)   # most likely >= 2; earliest of a tie
        self.assertIs(choose_anchor([a, c, b]), c)
        self.assertIs(choose_anchor([Candidate(9, TWIN[1]), a]), a)
        # P(>= 2) decides, not expected gain: a sure 2 beats a coin flip between 0 and 3.
        sure_two, flip = Candidate(4, CERTAIN[2]), Candidate(5, (0.5, 0.0, 0.0, 0.5))
        self.assertIs(choose_anchor([flip, sure_two]), sure_two)

    def test_select_recommendations(self):
        self.assertEqual(select_recommendations([Candidate(p, TWIN[0]) for p in range(8)]), [])
        self.assertEqual(select_recommendations([Candidate(p, (0.9, 0.1, 0.0, 0.0)) for p in range(8)]), [])
        self.assertEqual(select_recommendations([]), [])
        good = [Candidate(p, TWIN[p % 4]) for p in range(12)]
        self.assertEqual(select_recommendations(good), order_by_expected_gain(good, 0.3)[:5])
        self.assertEqual(len(select_recommendations(good, k=3)), 3)
        self.assertEqual(pids(select_recommendations(good[:3])), [2, 1])   # fewer than k is fine
        # An anchor unlikely to count makes even good complements worthless.
        self.assertEqual(select_recommendations(good, anchor_p2=0.01), [])

    def test_expected_rec_score(self):
        recs = [Candidate(p, CERTAIN[3]) for p in range(5)]
        self.assertAlmostEqual(expected_rec_score(recs, 1.0), 1.0)
        self.assertAlmostEqual(expected_rec_score(recs, 0.5), 0.5)
        self.assertAlmostEqual(expected_rec_score(recs[:1], 1.0), 7 / ideal_dcg(5))


if __name__ == '__main__':
    unittest.main()
