"""Points Engine: turn uncertain grade predictions into the decisions that score most.

The official judge grades each returned product 0-3; grade g gains 2^g - 1
(0, 1, 3, 7) at rank r, discounted by log2(r + 1). nDCG@k divides by a FIXED
ideal of k grade-3 items (10 primary slots, 5 recommendation slots), so a
missing slot earns 0 and a weak product costs nothing but the slot it takes.

We never see the official grade, only a distribution over it (Candidate.probs,
built from calibration rows P(official grade | our predicted grade)).
Expectation is linear, E[DCG] = sum_r E[gain_r] / log2(r + 1), so each decision
here maximizes an expected score:

* relevance order: expected gain descending (rearrangement inequality);
* mixed choices: 0.5 x coverage + 0.5 x nDCG@10 is a set-selection problem for
  the first 10 slots (coverage ignores order; nDCG wants expected-gain order);
* field sorts: the order is forced, so an exact DP picks which products to show;
* abstain or answer, recommend or not: compare the expected points of each.

Every function is deterministic: ties break by prior (desc), then pid (asc).
"""
import bisect
import math
from dataclasses import dataclass
from functools import lru_cache
from itertools import islice

GAINS = (0.0, 1.0, 3.0, 7.0)
_EPS = 1e-12             # objective differences this small count as ties
_MAX_SWAP_ROUNDS = 100   # safety cap; the local search converges in a few rounds

# Row t = P(official grade 0..3 | we predicted t). Within a source the rows are
# stochastically ordered: a higher prediction never lowers P(official >= g).
# 'twin' rows are for an LLM grader prompted like the judge (fairly sharp);
# 'estimate' rows are for the rule-based fallback grade (flatter). Row 0 of both
# keeps the expected gain under the default min_gain of 0.5, so products we
# believe irrelevant are dropped. Estimate row 1 doubles as the "no information"
# row. Comments give E[gain] and P(official >= 2). Refit with fit_calibration().
DEFAULT_CALIBRATION = {
    'twin': {
        0: (0.85, 0.10, 0.04, 0.01),   # 0.29, 0.05
        1: (0.35, 0.40, 0.18, 0.07),   # 1.43, 0.25
        2: (0.08, 0.20, 0.50, 0.22),   # 3.24, 0.72
        3: (0.03, 0.07, 0.25, 0.65),   # 5.37, 0.90
    },
    'estimate': {
        0: (0.80, 0.13, 0.05, 0.02),   # 0.42, 0.07
        1: (0.45, 0.25, 0.18, 0.12),   # 1.63, 0.30
        2: (0.20, 0.22, 0.33, 0.25),   # 2.96, 0.58
        3: (0.10, 0.15, 0.30, 0.45),   # 4.20, 0.75
    },
}


@dataclass(frozen=True)
class Candidate:
    """A product we could return, with our belief about its official grade."""
    pid: int                    # catalog position; unique; final tie-break is pid ascending
    probs: tuple                # (p0, p1, p2, p3): P(official judge grade = g); sums to 1
    choice: int = 0             # index of the requested choice this candidate serves (mixed queries)
    prior: float = 0.0          # tie-break signal in [0, 1]; higher wins before pid
    choice_probs: tuple = None  # optional (p0..p3) for the grade w.r.t. its own choice; None: use probs


# ---------------------------------------------------------------------------
# Scoring formula

@lru_cache(maxsize=None)
def _discounts(k):
    """1 / log2(r + 1) for ranks r = 1..k."""
    return tuple(1.0 / math.log2(r + 1) for r in range(1, k + 1))


@lru_cache(maxsize=None)
def ideal_dcg(k):
    """DCG of k grade-3 items: the benchmark's fixed nDCG normalizer."""
    return GAINS[3] * sum(_discounts(k))


def expected_gain(probs):
    """E[2^g - 1] under a grade distribution (p0, p1, p2, p3); GAINS[0] is 0."""
    return GAINS[1] * probs[1] + GAINS[2] * probs[2] + GAINS[3] * probs[3]


def p_at_least(probs, g):
    """P(grade >= g)."""
    return float(sum(probs[max(0, g):]))


def expected_dcg(cands, k=10):
    """E[DCG@k] of the list in the order given."""
    return sum(expected_gain(c.probs) * d for c, d in zip(cands, _discounts(k)))


def expected_ndcg(cands, k=10):
    """E[nDCG@k] of the list in the order given (fixed ideal of k grade-3 items)."""
    ideal = ideal_dcg(k)
    return expected_dcg(cands, k) / ideal if ideal > 0 else 0.0


def _order_key(cand):
    """Expected gain desc, prior desc, pid asc (choice only separates duplicate pids)."""
    return (-expected_gain(cand.probs), -cand.prior, cand.pid, cand.choice)


def _unique(cands):
    """First candidate of every pid, in order: a duplicate id zeroes a whole case."""
    seen = set()
    out = []
    for cand in cands:
        if cand.pid not in seen:
            seen.add(cand.pid)
            out.append(cand)
    return out


# ---------------------------------------------------------------------------
# Calibration: predicted grades -> P(official grade)

def _grade(value):
    """A grade as an int 0-3, or None when unusable (None, NaN, junk)."""
    try:
        grade = int(round(float(value)))
    except (TypeError, ValueError, OverflowError):
        return None
    return min(3, max(0, grade))


def _row(calibration, source, grade):
    """Calibration row for (source, grade), int or str keys, else the default row, else None."""
    for cal in (calibration, DEFAULT_CALIBRATION):
        rows = cal.get(source) or {}
        row = rows.get(grade, rows.get(str(grade)))
        if row is not None:
            return row
    return None


def _normalize(values):
    """Non-negative finite 4-tuple scaled to sum to 1 (uniform when nothing is left)."""
    values = [max(0.0, float(v)) if math.isfinite(v) else 0.0 for v in values]
    total = sum(values)
    if total <= 0.0:
        return (0.25, 0.25, 0.25, 0.25)
    return tuple(v / total for v in values)


def grade_probs(grades, calibration=None, source='twin'):
    """P(official grade = 0..3) for one product from our predicted grades.

    Each predicted grade selects its row P(official | predicted). Several passes
    average their rows: passes share the prompt and facts, so their errors are
    correlated, and multiplying rows as if independent would be overconfident.
    No usable grade gives the uninformative row (the calibration's estimate row 1).
    Grades are sorted first so the same multiset always gives identical floats.
    """
    calibration = calibration or DEFAULT_CALIBRATION
    rows = []
    for grade in sorted(g for g in map(_grade, grades) if g is not None):
        row = _row(calibration, source, grade)
        if row is not None:
            rows.append(_normalize(row))
    if not rows:
        rows = [_normalize(_row(calibration, 'estimate', 1))]
    return _normalize([sum(column) / len(rows) for column in zip(*rows)])


def fit_calibration(pairs, source='twin', smoothing=1.0, base=None):
    """Re-estimate one source's rows from (predicted grade, official grade) pairs.

    Row t = counts of official grades when we predicted t, plus smoothing x the
    base row t (pseudo-counts), normalized: with few pairs a row stays near the
    base, with many it follows the data. Other sources are copied from base.
    """
    base = base or DEFAULT_CALIBRATION
    smoothing = max(0.0, float(smoothing))
    counts = [[0.0] * 4 for _ in range(4)]
    for predicted, official in pairs:
        t, g = _grade(predicted), _grade(official)
        if t is not None and g is not None:
            counts[t][g] += 1.0
    fitted = {name: {int(t): tuple(row) for t, row in rows.items()} for name, rows in base.items()}
    rows = {}
    for t in range(4):
        prior = _row(base, source, t) or _row(base, 'estimate', t)
        mixed = [c + smoothing * p for c, p in zip(counts[t], prior)]
        rows[t] = _normalize(mixed) if sum(mixed) > 0 else _normalize(prior)
    fitted[source] = rows
    return fitted


def calibration_to_json(cal):
    """JSON-ready copy of a calibration: string row keys, list rows."""
    return {source: {str(t): [float(p) for p in row] for t, row in rows.items()}
            for source, rows in cal.items()}


def calibration_from_json(obj):
    """Inverse of calibration_to_json. Malformed rows raise ValueError, so a bad file fails at startup."""
    cal = {}
    for source, rows in obj.items():
        cal[source] = {}
        for key, row in rows.items():
            t = int(key)
            row = tuple(float(p) for p in row)
            if (t not in range(4) or len(row) != 4 or not all(math.isfinite(p) and p >= 0 for p in row)
                    or sum(row) <= 0):
                raise ValueError(f'bad calibration row {source}[{key}]: {row}')
            cal[source][t] = row
    return cal


# ---------------------------------------------------------------------------
# Single-intent ranking

def order_by_expected_gain(cands, min_gain=0.5):
    """The expected-nDCG-optimal order: expected gain desc, then prior desc, then pid asc.

    E[DCG@k] = sum_r E[gain_r] * d_r with discounts d_r decreasing in r, so by the
    rearrangement inequality the largest expected gains belong in the top slots,
    and the top k by expected gain are the best set for any k. Candidates below
    min_gain are dropped: they could only fill trailing slots with near-certain
    zeros, and a list of nothing but such products becomes empty, which is what
    an irrelevant case needs. A duplicate pid keeps its best-placed copy.
    """
    return _unique(sorted((c for c in cands if expected_gain(c.probs) >= min_gain), key=_order_key))


# ---------------------------------------------------------------------------
# Mixed choices

def coverage_prob(cand):
    """P(this result covers its choice): choice grade >= 2 and primary grade >= 2.

    The two grades are strongly positively correlated, so the joint probability
    is taken as the smaller marginal (its upper bound, exact if comonotone).
    """
    choice_probs = cand.probs if cand.choice_probs is None else cand.choice_probs
    return min(p_at_least(choice_probs, 2), p_at_least(cand.probs, 2))


def expected_coverage(cands, n_choices, k=10):
    """Expected fraction of the n_choices choices covered within the first k results.

    A choice is missed only if all of its results miss; results count as
    independent, so P(covered) = 1 - prod(1 - q). Out-of-range choices never cover.
    """
    if n_choices <= 0:
        return 0.0
    miss = [1.0] * n_choices
    for cand in islice(cands, k):
        if 0 <= cand.choice < n_choices:
            miss[cand.choice] *= 1.0 - coverage_prob(cand)
    return sum(1.0 - m for m in miss) / n_choices


def mixed_objective(cands, n_choices, k=10):
    """Expected mixed-choice case score of the list in the order given."""
    head = list(islice(cands, k))
    return 0.5 * expected_coverage(head, n_choices, k) + 0.5 * expected_ndcg(head, k)


def _inserted(members, index):
    """Copy of the sorted index list with index added."""
    out = list(members)
    bisect.insort(out, index)
    return out


def build_mixed_list(cands, n_choices, depth=100, k=10):
    """Ranked list for a mixed query: the first k maximize mixed_objective, the rest follow by gain.

    Coverage ignores order and nDCG prefers expected-gain order, so the best
    arrangement of any fixed set is expected gain descending; what is left is
    choosing the set. Both halves are monotone and submodular in the set
    (probabilistic coverage; DCG of a sorted set is a positive mix of top-r
    sums), so greedy marginal-gain selection is near-optimal; a best-improvement
    1-swap local search then polishes it. Equal values prefer higher prior,
    then lower pid. Copies of one pid serving different choices are alternatives:
    at most one is used. After the first k, the remaining candidates follow in
    order_by_expected_gain(min_gain=0.5) order, up to depth items in total.
    The list never depends on offset or limit. One choice reduces to
    order_by_expected_gain, since single-intent cases are scored by nDCG alone.
    """
    if n_choices <= 1:
        return order_by_expected_gain(cands, min_gain=0.5)[:depth]
    # Pool indices follow _order_key, so a sorted index list is the best arrangement.
    pool = sorted(cands, key=_order_key)
    gains = [expected_gain(c.probs) for c in pool]
    covers = [coverage_prob(c) for c in pool]
    discounts = _discounts(k)
    ideal = ideal_dcg(k) or 1.0

    def value(members):
        """mixed_objective of pool[members], members sorted."""
        dcg = 0.0
        miss = [1.0] * n_choices
        for d, i in zip(discounts, members):
            dcg += gains[i] * d
            choice = pool[i].choice
            if 0 <= choice < n_choices:
                miss[choice] *= 1.0 - covers[i]
        return 0.5 * sum(1.0 - m for m in miss) / n_choices + 0.5 * dcg / ideal

    def tie_key(i):
        return (-pool[i].prior, pool[i].pid)

    # Greedy: add the candidate whose addition gives the best re-sorted set.
    members, used = [], set()
    while len(members) < min(k, depth):
        best, best_value = None, 0.0
        for i, cand in enumerate(pool):
            if cand.pid in used:
                continue
            v = value(_inserted(members, i))
            if (best is None or v > best_value + _EPS
                    or (v >= best_value - _EPS and tie_key(i) < tie_key(best))):
                best, best_value = i, v
        if best is None:
            break
        bisect.insort(members, best)
        used.add(pool[best].pid)

    # Local search: apply the best improving swap until none is left.
    current = value(members)
    for _ in range(_MAX_SWAP_ROUNDS):
        move, move_value = None, current + _EPS
        for out in members:
            rest = [m for m in members if m != out]
            rest_pids = {pool[m].pid for m in rest}
            for i, cand in enumerate(pool):
                if i == out or cand.pid in rest_pids:
                    continue
                v = value(_inserted(rest, i))
                if v > move_value:
                    move, move_value = (out, i), v
        if move is None:
            break
        members.remove(move[0])
        bisect.insort(members, move[1])
        current = move_value

    head = [pool[i] for i in members]
    taken = {c.pid for c in head}
    tail = [c for c in order_by_expected_gain(pool, min_gain=0.5) if c.pid not in taken]
    return (head + tail)[:depth]


# ---------------------------------------------------------------------------
# Abstention

def p_answerable(cands, k=10):
    """P(at least one of the first k results grades >= 2), results treated as independent."""
    miss = 1.0
    for cand in islice(cands, k):
        miss *= 1.0 - p_at_least(cand.probs, 2)
    return 1.0 - miss


def should_abstain(p_answerable, score, w_answer=1.0, w_abstain=0.75):
    """True when an empty response is worth more points in expectation than answering.

    If the case is irrelevant or impossible (probability about 1 - p_answerable),
    only an empty response scores: 7.5 points / 10 cases = 0.75 per case
    (w_abstain). If it is answerable, answering earns weight x score, where score
    is the expected nDCG or mixed objective of the list we would return. The
    weights per category: ranking 30/25 = 1.2 x nDCG, unusual 7.5/10 = 0.75 x
    nDCG, mixed 15/15 = 1.0, recommendations 10/10 = 1.0; over all 60 answerable
    semantic cases that is 62.5 / 60, about 1.0 (w_answer). Since score already
    falls when items are likely irrelevant, multiplying it by p_answerable makes
    the rule lean towards abstaining; with independent items a list with a high
    expected score has p_answerable near 1, so this only matters at the margin.
    Abstain iff (1 - p_answerable) * w_abstain > p_answerable * w_answer * score.
    """
    return (1.0 - p_answerable) * w_abstain > p_answerable * w_answer * score


# ---------------------------------------------------------------------------
# Field sorts with a query

def choose_field_sorted_set(cands, sort_key, k=10, min_gain=0.5):
    """Which candidates to show when a field sort (price, rating, ...) fixes their order.

    The list is shown in sort_key order (ascending), so a weak product that sorts
    early takes a top slot from a strong one that sorts later. Candidates below
    min_gain are left out, as in order_by_expected_gain. Exact DP over the rest in
    sort order: best[i][j] is the most expected DCG@k obtainable from candidates
    i.. when j are already chosen (the next one lands at rank j + 1); ties
    prefer taking the earlier product. Once k are chosen, every later candidate
    is appended too: it can only appear after rank k, on later pages. An
    unchosen candidate that sorts before the last chosen one is never shown,
    since it would push chosen ones down. (If fewer than k are chosen, nothing
    eligible follows the last one: taking it would only add gain.) sort_key
    should be total; pid (= id rank) breaks any tie it leaves.
    """
    eligible = [c for c in cands if expected_gain(c.probs) >= min_gain]
    ordered = _unique(sorted(eligible, key=lambda c: (sort_key(c), c.pid, -expected_gain(c.probs))))
    n, k = len(ordered), max(0, k)
    gains = [expected_gain(c.probs) for c in ordered]
    discounts = _discounts(k)
    best = [[0.0] * (k + 1) for _ in range(n + 1)]   # best[i][k] stays 0: no slot left
    for i in range(n - 1, -1, -1):
        row, below = best[i], best[i + 1]
        for j in range(k):
            row[j] = max(below[j], gains[i] * discounts[j] + below[j + 1])
    chosen = []
    for i, cand in enumerate(ordered):
        j = len(chosen)
        if j == k:
            chosen.extend(ordered[i:])
            break
        if gains[i] * discounts[j] + best[i + 1][j + 1] >= best[i + 1][j] - _EPS:
            chosen.append(cand)
    return chosen


def choose_mixed_field_sorted_set(cands, n_choices, sort_key, k=10, min_gain=0.5):
    """choose_field_sorted_set for a mixed query: every choice keeps a share of the first k.

    A plain field sort can fill the first k with one choice (every cheap T-shirt before any
    jeans) while coverage counts each requested choice. Each choice picks its own best set for
    its share of the k slots (split evenly; a choice with too few candidates passes its unused
    slots on); the union is shown in sort order, then every other eligible candidate that sorts
    after the last one shown, so the whole list stays sorted.
    """
    eligible = _unique([c for c in cands if expected_gain(c.probs) >= min_gain])
    groups = {}
    for c in eligible:
        groups.setdefault(c.choice, []).append(c)
    present = sorted(groups)
    if len(present) <= 1:
        return choose_field_sorted_set(cands, sort_key, k=k, min_gain=min_gain)
    share = {g: 0 for g in present}
    left = k
    while left > 0:
        open_groups = [g for g in present if share[g] < len(groups[g])]
        if not open_groups:
            break
        for g in open_groups:
            if left == 0:
                break
            share[g] += 1
            left -= 1
    top = []
    for g in present:
        if share[g]:
            top += choose_field_sorted_set(groups[g], sort_key, k=share[g], min_gain=min_gain)[:share[g]]
    top = _unique(sorted(top, key=lambda c: (sort_key(c), c.pid)))
    if not top:
        return []
    last = (sort_key(top[-1]), top[-1].pid)
    shown = {c.pid for c in top}
    rest = [c for c in sorted(eligible, key=lambda c: (sort_key(c), c.pid))
            if c.pid not in shown and (sort_key(c), c.pid) > last]
    return top + rest


# ---------------------------------------------------------------------------
# Recommendations

def choose_anchor(page_cands):
    """The page product most likely to grade >= 2 (a recommendation only counts for such an anchor).

    The earliest on the page wins ties; None for an empty page.
    """
    best, best_p = None, -1.0
    for cand in page_cands:
        p = p_at_least(cand.probs, 2)
        if p > best_p + _EPS:
            best, best_p = cand, p
    return best


def expected_rec_score(cands, anchor_p2, k=5):
    """Expected recommendation nDCG@k of the list as given; it counts only if the anchor grades >= 2."""
    return anchor_p2 * expected_ndcg(cands, k)


def select_recommendations(cands, anchor_p2=1.0, k=5, min_gain=0.3, min_value=0.05):
    """Up to k complements, best first, or [] when they are not worth showing.

    cands carry P(complement grade). Expected-gain order is optimal for nDCG@k
    exactly as for primary results. Nothing is forced: when the expected score
    anchor_p2 x nDCG@k is below min_value the list is empty.
    """
    top = order_by_expected_gain(cands, min_gain)[:k]
    if expected_rec_score(top, anchor_p2, k) < min_value:
        return []
    return top
