"""Benchmark scoring formulas, exactly as hackathon-brief.md states them.

Pure functions over judge grades (integers 0-3) and binary checks; the local
exam simulator (eval/run_eval.py) feeds them.

* nDCG@k: gain 2^g - 1, discount log2(rank + 1) with ranks from 1, normalized
  by a FIXED ideal of k grade-3 items. Missing slots earn zero, and callers pass
  grade 0 for hard-constraint-violating products (see effective()).
* Mixed case: 0.5 x choice coverage + 0.5 x primary nDCG@10.
* Recommendation case: mean of primary nDCG@10 and recommendation nDCG@5.
* Validity case: mean of four binary checks; an invalid call scores 1 for an
  MCP error and 0 for success.
* Total: weighted category means (F, at most 95) plus performance points,
  which count only when F >= 85.5 and at least 95% of timed calls are correct.
"""
import math

MAX_GRADE = 3
PRIMARY_K = 10
REC_K = 5
COVERED_GRADE = 2          # coverage and the recommendation anchor rule need a grade of at least 2

# Official categories in run order, with their points.
WEIGHTS = {'validity': 25.0, 'ranking': 30.0, 'unusual': 7.5, 'irrelevant': 7.5,
           'mixed': 15.0, 'recs': 10.0}
CATEGORIES = tuple(WEIGHTS)
FUNCTIONAL_MAX = sum(WEIGHTS.values())
PERF_POINTS = 5.0
PERF_MIN_FUNCTIONAL = 85.5
PERF_MIN_CORRECT = 0.95
FULL_CREDIT_MS = 200.0
ZERO_CREDIT_MS = 2000.0


def check_grade(grade):
    """A judge grade must be an integer 0-3 (bools are rejected)."""
    if isinstance(grade, bool) or not isinstance(grade, int) or not 0 <= grade <= MAX_GRADE:
        raise ValueError(f'grade must be an integer 0-{MAX_GRADE}, got {grade!r}')
    return grade


def gain(grade):
    return 2 ** check_grade(grade) - 1


def dcg(grades, k):
    """Sum of (2^g - 1) / log2(rank + 1) over the first k grades, ranks starting at 1."""
    return sum(gain(g) / math.log2(rank + 1) for rank, g in enumerate(list(grades)[:k], start=1))


def ideal_dcg(k):
    """The fixed ceiling: k slots of grade 3."""
    return dcg([MAX_GRADE] * k, k)


def ndcg(grades, k=PRIMARY_K):
    """nDCG@k of a ranked grade list against the fixed ideal of k grade-3 items.

    Fewer than k grades means missing slots, which earn zero. Grades past k are ignored.
    """
    if k < 1:
        raise ValueError('k must be at least 1')
    return dcg(grades, k) / ideal_dcg(k)


def effective(grades, ok_flags):
    """Grades with hard-constraint-violating products set to 0 (they earn no gain)."""
    grades, ok_flags = list(grades), list(ok_flags)
    if len(grades) != len(ok_flags):
        raise ValueError('grades and ok_flags must have the same length')
    return [check_grade(g) if ok else 0 for g, ok in zip(grades, ok_flags)]


def choice_coverage(primary_grades, choice_grades_per_result, n_choices, ok_flags):
    """Fraction of requested choices with at least one covering result.

    A result covers choice c when its grade for that choice is at least 2, its
    primary grade is at least 2 and it obeys the hard constraints.
    choice_grades_per_result[i][c] is result i's grade for choice c; to apply a
    choice's own constraints, pass 0 for products that violate them.
    """
    if n_choices < 1:
        raise ValueError('a mixed case needs at least one declared choice')
    primary_grades, choice_rows, ok_flags = (list(primary_grades), list(choice_grades_per_result),
                                             list(ok_flags))
    if not len(primary_grades) == len(choice_rows) == len(ok_flags):
        raise ValueError('primary_grades, choice_grades_per_result and ok_flags must align')
    if any(len(row) < n_choices for row in choice_rows):
        raise ValueError(f'every result needs a grade for each of the {n_choices} choices')
    covered = 0
    for c in range(n_choices):
        if any(ok and check_grade(primary) >= COVERED_GRADE and check_grade(row[c]) >= COVERED_GRADE
               for primary, row, ok in zip(primary_grades, choice_rows, ok_flags)):
            covered += 1
    return covered / n_choices


def mixed_score(primary_grades, choice_grades_per_result, n_choices, ok_flags):
    """0.5 x choice coverage + 0.5 x nDCG@10 of the pooled list (violators earn no gain)."""
    primary_grades, ok_flags = list(primary_grades), list(ok_flags)
    coverage = choice_coverage(primary_grades, choice_grades_per_result, n_choices, ok_flags)
    return 0.5 * coverage + 0.5 * ndcg(effective(primary_grades, ok_flags), PRIMARY_K)


def rec_quality(rec_grades, rec_anchor_primary_grade, rec_ok, no_complement=False):
    """Recommendation nDCG@5.

    A recommendation earns gain only if it obeys its own constraints and its
    anchor (for_product_id) has a primary grade of at least 2. An empty list
    earns full credit only when the case declares that no complement exists.
    """
    rec_grades, anchors, rec_ok = list(rec_grades), list(rec_anchor_primary_grade), list(rec_ok)
    if not len(rec_grades) == len(anchors) == len(rec_ok):
        raise ValueError('rec_grades, rec_anchor_primary_grade and rec_ok must align')
    if not rec_grades:
        return 1.0 if no_complement else 0.0
    gains = [check_grade(g) if ok and check_grade(anchor) >= COVERED_GRADE else 0
             for g, anchor, ok in zip(rec_grades, anchors, rec_ok)]
    return ndcg(gains, REC_K)


def rec_case_score(primary_grades, rec_grades, rec_anchor_primary_grade, rec_ok, no_complement=False):
    """Mean of primary nDCG@10 and recommendation quality at 5.

    primary_grades must already carry 0 for constraint violators (effective()),
    and rec_anchor_primary_grade should be the anchor's effective grade too.
    """
    primary = ndcg(primary_grades, PRIMARY_K)
    return (primary + rec_quality(rec_grades, rec_anchor_primary_grade, rec_ok, no_complement)) / 2


def validity_case_score(schema_ok, constraints_ok, sorting_ok, presence_ok):
    """Mean of the four binary checks of a valid call (each passed as truthy or falsy)."""
    return sum(bool(c) for c in (schema_ok, constraints_ok, sorting_ok, presence_ok)) / 4


def invalid_case_score(got_error):
    """An invalid call earns 1 for an MCP error (isError or JSON-RPC error) and 0 for success."""
    return 1.0 if got_error else 0.0


def latency_factor(p95_ms):
    """Full credit at 200 ms or below, zero at 2,000 ms or above, linear between."""
    return max(0.0, min(1.0, (ZERO_CREDIT_MS - p95_ms) / (ZERO_CREDIT_MS - FULL_CREDIT_MS)))


def perf_eligible(functional, correct_rate):
    return functional >= PERF_MIN_FUNCTIONAL and correct_rate >= PERF_MIN_CORRECT


def performance_points(correct_rate, p95_ms, functional):
    """5 x correct rate x latency factor, or 0 unless F >= 85.5 and >= 95% of timed calls are correct."""
    if not perf_eligible(functional, correct_rate):
        return 0.0
    return PERF_POINTS * correct_rate * latency_factor(p95_ms)


def total(category_means, perf=None):
    """Weighted score of per-category mean case scores.

    category_means maps a category to its mean case score in 0..1 (None or a
    missing category earns 0 and is left out of measured_max). perf is None
    (not measured: 0 points) or a (correct_rate, p95_ms) pair.
    """
    unknown = set(category_means) - set(WEIGHTS)
    if unknown:
        raise ValueError(f'unknown categories: {sorted(unknown)}')
    points, measured_max = {}, 0.0
    for category in CATEGORIES:
        mean = category_means.get(category)
        points[category] = 0.0 if mean is None else WEIGHTS[category] * mean
        measured_max += 0.0 if mean is None else WEIGHTS[category]
    functional = sum(points.values())
    performance, eligible = 0.0, False
    if perf is not None:
        correct_rate, p95_ms = perf
        eligible = perf_eligible(functional, correct_rate)
        performance = performance_points(correct_rate, p95_ms, functional)
    return {'points': points, 'functional': functional, 'measured_max': measured_max,
            'performance': performance, 'perf_eligible': eligible, 'total': functional + performance}
