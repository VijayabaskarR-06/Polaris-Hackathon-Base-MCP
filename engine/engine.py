"""Semantic search: plan -> pools -> grades -> Points Engine -> pages.

One SearchState per (query, filters) is computed once and cached. Every page
and every sort order of a request is a slice of that state, so repeated and
larger-prefix requests always agree.

With an LLM configured, the LLM planner races the instant rule planner, and
the Judge Twin grades the candidate pools; anything late or failing falls back
to the rule plan and rule estimates, so a valid call never errors or times out.
"""
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from engine import points
from engine.cache import LRU, KeyLocks
from engine.constraints import hard_mask, type_mask
from engine.recs import Recommender
from engine.retrieve import BM25
from engine.rules import RulePlanner, signals_of
from engine.text import tokenize
from engine.vocab import Vocab

log = logging.getLogger('polaris.engine')

POOL_SINGLE, POOL_MIXED = 100, 50
EVIDENCE_COVERAGE = 0.6
# Deadlines, seconds after the request starts (the benchmark allows 10 s per call).
PLANNER_DEADLINE = float(os.getenv('PLANNER_DEADLINE', '2.5'))
JUDGE_DEADLINE = float(os.getenv('JUDGE_DEADLINE', '6.0'))
RECS_DEADLINE = float(os.getenv('RECS_DEADLINE', '8.0'))
TWIN_POOL = int(os.getenv('TWIN_POOL', '100'))   # products per choice the Judge Twin grades
CALIBRATION_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'assets', 'calibration.json')


def state_key(args):
    filters = {}
    for name, value in sorted(args['filters'].items()):
        filters[name] = value.strip().casefold() if isinstance(value, str) else float(value)
    return json.dumps({'q': ' '.join(args['query'].split()).casefold(), 'f': filters}, sort_keys=True)


def terms_of(words):
    out = []
    for word in words:
        for term in tokenize(word):
            if term not in out:
                out.append(term)
    return out


def p_answerable(ranked, k=10):
    """P(the request is answerable at all) = the best single product's P(grade >= 2).

    Products in one list are strongly correlated: for an irrelevant request they
    are all weak together. Treating them as independent (1 - prod(1 - p)) would
    make ten weak products look almost surely answerable and never abstain.
    """
    return max((points.p_at_least(c.probs, 2) for c in ranked[:k]), default=0.0)


def load_calibration():
    try:
        with open(CALIBRATION_PATH) as fh:
            return points.calibration_from_json(json.load(fh))
    except (OSError, ValueError, KeyError, TypeError):
        return points.DEFAULT_CALIBRATION


@dataclass
class Pool:
    choice_index: int
    choice: object
    positions: np.ndarray
    scores: np.ndarray
    estimates: np.ndarray
    priors: np.ndarray
    feasible: bool
    hard: object
    support_count: int = 0
    best_coverage: float = 0.0


@dataclass
class SearchState:
    key: str
    plan: object
    abstain: bool
    reason: str
    cands: list = field(default_factory=list)
    ranked: list = field(default_factory=list)
    n_choices: int = 1
    pools: list = field(default_factory=list)
    orderings: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)
    source: str = 'rules'
    graded: int = 0
    by_pid: dict = field(default_factory=dict)
    lock: object = field(default_factory=threading.Lock)

    def candidate(self, pid):
        if not self.by_pid and self.cands:
            self.by_pid = {c.pid: c for c in self.cands}
        return self.by_pid.get(pid)


class SemanticEngine:
    def __init__(self, products, index, llm=None, calibration=None):
        start = time.monotonic()
        self.products = products
        self.index = index
        self.bm25 = BM25(products)
        self.vocab = Vocab(products, index, self.bm25)
        self.rules = RulePlanner(self.vocab)
        self.calibration = calibration or load_calibration()
        self.cache = LRU(5000)
        self.rec_cache = LRU(5000)
        self.locks = KeyLocks()
        self.adult = self.vocab.group_masks['women'] | self.vocab.group_masks['men']
        self.section_codes = {s: i for i, s in enumerate(index.values['section'])}
        self.audience_codes = {a: i for i, a in enumerate(index.values['audience'])}
        self.pattern_codes = {p: i for i, p in enumerate(index.values['pattern'])}
        self.recommender = Recommender(self)
        self.llm = llm
        self.planner = self.twin = None
        if llm is not None and getattr(llm, 'name', 'none') != 'none':
            from engine.judge import JudgeTwin
            from engine.llm import model_for
            from engine.planner import LLMPlanner
            self.planner = LLMPlanner(self.vocab, llm, model_for('planner'))
            self.twin = JudgeTwin(products, llm, model_for('judge'), model_for('judge2'))
        self.grade_log = os.getenv('LOG_GRADES') == '1'
        log.info('semantic engine ready in %.1fs (llm=%s)', time.monotonic() - start,
                 getattr(llm, 'name', 'none') if llm else 'none')

    # ----- public ----------------------------------------------------------
    def search(self, args):
        t0 = time.monotonic()
        key = state_key(args)
        state = self.cache.get(key)
        if state is None:
            with self.locks.get(key):
                state = self.cache.get(key)
                if state is None:
                    state = self.cache.put(key, self.compute(args, key, t0))
        order = self.ordering(state, args['sort_by'], args['sort_order'])
        page = order[args['offset']:args['offset'] + args['limit']]
        ids = self.index.ids
        recs = []
        if args['include_recommendations'] and page:
            try:
                recs = self.recommend(state, page, args, deadline=t0 + RECS_DEADLINE)
            except Exception:  # a recommendation failure must never fail the call
                log.exception('recommendations failed for %r', args['query'][:80])
                recs = []
        return {'results': [{'id': ids[p]} for p in page], 'recommendations': recs}

    def warm_up(self):
        for query in ('black jeans', 'white t-shirt and blue jeans', 'dress with matching bag'):
            state = self.build_state(None, self.rules.plan(query), {})
            page = self.ordering(state, 'relevance', 'desc')[:10]
            if page:
                self.recommender.candidates(state.plan, page[0], page)
        if self.llm is not None and self.planner is not None:
            from engine.llm import warm_up
            warm_up(self.llm, self.planner.model)
            if self.twin.model != self.planner.model:
                warm_up(self.llm, self.twin.model)

    # ----- state -----------------------------------------------------------
    def compute(self, args, key, t0):
        query, filters = args['query'], args['filters']
        rule_plan = self.rules.plan(query)
        state = None
        if self.planner is not None:
            try:
                state = self.compute_llm(query, filters, key, rule_plan, t0)
            except Exception:
                log.exception('LLM path failed for %r; using rules', query[:80])
                state = None
        if state is None:
            state = self.build_state(key, rule_plan, filters)
        state.timings['total'] = time.monotonic() - t0
        self.log_state(args, state)
        return state

    def compute_llm(self, query, filters, key, rule_plan, t0):
        from engine.llm import EXECUTOR
        from engine.planner import merge_safety_net
        future = EXECUTOR.submit(self.planner.plan, query, max(0.5, t0 + PLANNER_DEADLINE - time.monotonic()))
        try:
            llm_plan = future.result(timeout=max(0.05, t0 + PLANNER_DEADLINE + 0.2 - time.monotonic()))
        except Exception:
            llm_plan = None
        plan = merge_safety_net(llm_plan, rule_plan)
        planned = time.monotonic()

        def grader(pools):
            groups = [(p.choice.label if len(pools) > 1 else None, p.positions.tolist()[:TWIN_POOL])
                      for p in pools]
            return self.twin.grade(query, groups, deadline=t0 + JUDGE_DEADLINE)

        state = self.build_state(key, plan, filters, grader=grader, query=query)
        state.timings['plan'] = planned - t0
        state.source = plan.source if llm_plan is not None else 'rules+twin'
        return state

    def abstained(self, key, plan, reason):
        return SearchState(key=key, plan=plan, abstain=True, reason=reason, source=plan.source)

    def build_state(self, key, plan, filters, grader=None, query=None):
        signals = signals_of(plan)
        if plan.kind == 'nonsense' or not plan.choices:
            return self.abstained(key, plan, 'nonsense')
        if signals.brands:
            return self.abstained(key, plan, 'brand not in catalog: ' + ','.join(signals.brands))
        explicit_types = any(c.product_types and not getattr(c, 'occasion_types', False) for c in plan.choices)
        if not explicit_types and signals.service:
            return self.abstained(key, plan, 'service request, no product named')
        if not explicit_types and signals.question and not any(getattr(c, 'occasion', False) for c in plan.choices):
            return self.abstained(key, plan, 'question without a product or occasion')
        heads = [getattr(c, 'head', None) for c in plan.choices]
        if heads and all(h == 'non_catalog' for h in heads):
            return self.abstained(key, plan, 'not sold: ' + ','.join(signals.non_catalog))
        mixed = len(plan.choices) > 1
        pools = []
        for ci, choice in enumerate(plan.choices):
            pool = self.build_pool(plan, ci, choice, filters, POOL_MIXED if mixed else POOL_SINGLE)
            if pool.feasible:
                pools.append(pool)
        if not pools:
            return self.abstained(key, plan, 'infeasible')
        if grader is None:
            pools = [p for p in pools if self.has_evidence(p)]
            if not pools:
                return self.abstained(key, plan, 'weak evidence')
        grades = {}
        if grader is not None:
            started = time.monotonic()
            grades = grader(pools) or {}
            if self.grade_log and query is not None:
                self.write_grades(query, grades)
            log.info('twin graded %d products in %.0fms', len(grades), 1000 * (time.monotonic() - started))
        cands = self.candidates(pools, grades)
        n_choices = len(pools)
        if n_choices > 1:
            ranked = points.build_mixed_list(cands, n_choices, depth=100)
            score = points.mixed_objective(ranked[:10], n_choices)
        else:
            ranked = points.order_by_expected_gain(cands, min_gain=0.5)
            score = points.expected_ndcg(ranked[:10])
        state = SearchState(key=key, plan=plan, abstain=False, reason='', cands=cands, ranked=ranked,
                            n_choices=n_choices, pools=pools, source=plan.source, graded=len(grades))
        if not ranked:
            state.abstain, state.reason = True, 'no candidate above min gain'
        elif grades:
            # Answering earns about weight x score, where score already reflects how likely the
            # products are relevant; passing score / p keeps should_abstain from counting the
            # chance that the request is unanswerable twice.
            p = p_answerable(ranked)
            if points.should_abstain(p, score / max(p, 1e-9)):
                state.abstain, state.reason = True, f'twin: p_answerable {p:.2f}, score {score:.3f}'
        elif max(int(pool.estimates.max()) for pool in pools if len(pool.estimates)) <= 1:
            # Rule path: explicit rules decide (nonsense, infeasible, weak evidence); a list with
            # no product that even matches the requested type or words is not worth showing.
            state.abstain, state.reason = True, 'only weak rule matches'
        return state

    def has_evidence(self, pool):
        """Product types, colours or occasions are evidence; free words must mostly match one product.

        A request whose head noun is unknown to the catalog ("spaceship for
        underwater travel", "travel pillow") is insufficient evidence even if a
        modifier names an occasion: the requested object itself is not sold.
        """
        choice = pool.choice
        explicit = choice.product_types and not getattr(choice, 'occasion_types', False)
        head = getattr(choice, 'head', None)
        if not explicit and head in ('unknown', 'brand', 'non_catalog'):
            return False
        if not explicit and head == 'known':
            # The object asked for is a catalog word but not a product type, occasion, material
            # or style ("pizza" exists only as a print): enough only when it is common in the
            # catalog and nothing else in the request is unknown.
            word = getattr(choice, 'head_word', None) or ''
            if getattr(choice, 'unknown', None) or self.bm25.df(word) < 20:
                return False
        if choice.product_types or choice.colors or getattr(choice, 'occasion', False):
            return True
        return pool.best_coverage >= EVIDENCE_COVERAGE

    # ----- retrieval ---------------------------------------------------------
    def build_pool(self, plan, ci, choice, filters, k):
        idx, bm = self.index, self.bm25
        n = idx.n
        types = list(choice.product_types)
        occasion_types = getattr(choice, 'occasion_types', False)
        tm = type_mask(idx, types) if types else None
        core = terms_of(choice.keywords)
        desc = [t for t in terms_of(getattr(choice, 'descriptors', [])) if t not in core]
        unknown = [t for t in terms_of(getattr(choice, 'unknown', [])) if t not in core and t not in desc
                   and len(t) > 2]
        soft = [t for t in terms_of(choice.soft_words) if t not in core and t not in desc]
        soft_sections = [self.section_codes[s] for s in choice.soft_sections if s in self.section_codes]
        sec_mask = np.isin(idx.codes['section'], soft_sections) if soft_sections else None
        soft_any = bm.any_mask(soft) if soft else None

        if tm is not None and not occasion_types:
            support = (tm | bm.any_mask(core, names_only=True)) if core else tm
        elif tm is not None:
            support = tm.copy()
            if soft_any is not None or sec_mask is not None:
                hint = np.zeros(n, dtype=bool)
                if soft_any is not None:
                    hint |= soft_any
                if sec_mask is not None:
                    hint |= sec_mask
                if (support & hint).any():
                    support &= hint
        elif core or desc or unknown:
            support = bm.any_mask(core + desc + unknown)
        elif soft_any is not None or sec_mask is not None:
            support = np.zeros(n, dtype=bool)
            if soft_any is not None:
                support |= soft_any
            if sec_mask is not None:
                support |= sec_mask
        else:
            support = None

        hard = hard_mask(idx, self.vocab, filters, plan, choice, support=support)
        base = hard.mask if support is None else hard.mask & support
        excluded = terms_of(getattr(choice, 'exclude_words', []))
        if excluded:
            base = base & ~bm.any_mask(excluded)
        positions = np.flatnonzero(base)
        empty = np.empty(0)
        # A named product type must itself be satisfiable: "jacket under 2000" is impossible even
        # though a fleece "jacket" typed Hoodie costs less (the organizer audits by catalog type).
        explicit_type = tm is not None and not occasion_types
        if len(positions) == 0 or (explicit_type and not (base & tm).any()):
            return Pool(ci, choice, np.empty(0, dtype=np.int64), empty, empty, empty, False, hard)

        weights = {t: 1.0 for t in core}
        weights.update({t: 1.3 for t in desc})
        weights.update({t: 0.4 for t in soft})
        weights.update({t: 0.8 for t in unknown})
        terms = core + desc + soft + unknown
        raw = bm.scores(terms, weights)[positions] if terms else np.zeros(len(positions), dtype=np.float32)
        top = float(raw.max()) if len(raw) else 0.0
        bm_norm = 4.0 * raw / top if top > 0 else np.zeros(len(positions))
        zeros = np.zeros(len(positions), dtype=bool)
        type_hit = tm[positions] if tm is not None else zeros
        name_hit = (bm.name_hits(core)[positions] > 0) if core else zeros
        match = getattr(choice, 'match_terms', None) or core
        kw_hit = (bm.coverage(match)[positions] > 0) if match else zeros
        free = desc + unknown
        free_cov = (bm.coverage(free)[positions] / len(free)) if free else np.ones(len(positions))
        free_name = (bm.name_hits(free)[positions] / len(free)) if free else np.zeros(len(positions))
        sec_hit = sec_mask[positions] if sec_mask is not None else zeros
        occ_sections = [self.section_codes[x] for x in getattr(choice, 'occasion_sections', []) if x in self.section_codes]
        occ_audiences = [self.audience_codes[x] for x in getattr(choice, 'occasion_audiences', [])
                         if x in self.audience_codes]
        # An occasion's audience ("yoga" -> Sport) is a requirement for a top grade; its
        # sections only hint (a warm scarf is not in an outerwear section).
        occasion_signal = bool(occ_audiences)
        occ_hit = zeros.copy()
        if occ_audiences:
            occ_hit |= np.isin(idx.codes['audience'][positions], occ_audiences)
        pat_codes = [self.pattern_codes[p] for p in choice.patterns if p in self.pattern_codes]
        pat_hit = np.isin(idx.codes['pattern'][positions], pat_codes) if pat_codes else zeros
        soft_hit = soft_any[positions] if soft_any is not None else zeros

        primary = getattr(choice, 'primary_types', None) or types
        needs_kw = getattr(choice, 'needs_keyword', False)
        primary_hit = type_mask(idx, primary)[positions] if primary and tm is not None else zeros
        type_ok = (primary_hit & (kw_hit | (not needs_kw))) | (type_hit & kw_hit)

        score = (bm_norm + 4.0 * type_ok + 1.5 * type_hit + 2.0 * name_hit + 2.0 * free_cov
                 + 1.5 * sec_hit + 1.5 * occ_hit + 1.0 * pat_hit + 0.5 * soft_hit)
        del occ_sections
        adult_first = (getattr(choice, 'audience_group', None) or plan.audience_group) is None
        if adult_first:
            # No audience stated: adult products first, before the pool is cut to size.
            score = score + 1.0 * self.adult[positions]
        est = np.ones(len(positions), dtype=np.int64)
        if tm is not None and not occasion_types:
            est[type_hit | name_hit] = 2
            top = type_ok & (free_cov >= 0.999)
            if occasion_signal:
                top &= occ_hit       # "yoga leggings": sports leggings, not fashion ones
            est[top] = 3
        elif tm is not None:
            est[type_hit] = 2
            est[type_hit & (occ_hit if occasion_signal else (sec_hit | soft_hit))] = 3
        elif free:
            est[free_cov >= 0.5] = 2
            est[(free_cov >= 0.999) & ((free_name > 0) | (len(free) == 1))] = 3
        else:
            est[:] = 2

        order = np.lexsort((positions, -score))[:k]
        best_cov = float(free_cov.max()) if free else 1.0
        positions, score, est = positions[order], score[order], est[order]
        priors = self.priors(choice, positions, score, adult_first=adult_first)
        return Pool(ci, choice, positions, score, est, priors, True, hard,
                    support_count=int(base.sum()), best_coverage=best_cov)

    def priors(self, choice, positions, score, adult_first=False):
        """Tie-break signal in [0, 1]: retrieval score, plus stated price/rating preferences.

        With no audience stated, adult products win ties: the typical shopper is
        an adult, and a judge may see a kids' item as a looser match.
        """
        top = float(score.max()) if len(score) else 0.0
        prior = score / top if top > 0 else np.zeros(len(positions))
        if adult_first:
            prior = 0.5 * prior + 0.5 * self.adult[positions]
        pref = None
        if choice.price_preference in ('low', 'high'):
            price = self.index.price[positions]
            span = float(price.max() - price.min()) or 1.0
            rel = (price - price.min()) / span
            pref = 1.0 - rel if choice.price_preference == 'low' else rel
        if choice.rating_preference:
            rating = np.nan_to_num(self.index.rating[positions], nan=0.0) / 5.0
            pref = rating if pref is None else 0.5 * (pref + rating)
        if pref is not None:
            prior = 0.4 * prior + 0.6 * pref
        return prior

    def candidates(self, pools, grades=None):
        """One Candidate per product (the better-graded copy wins when pools overlap)."""
        best = {}
        for pool_index, pool in enumerate(pools):
            for i, pid in enumerate(pool.positions.tolist()):
                twin = (grades or {}).get(pid)
                probs = (points.grade_probs(twin, self.calibration, source='twin') if twin
                         else points.grade_probs([int(pool.estimates[i])], self.calibration, source='estimate'))
                cand = points.Candidate(pid=pid, probs=probs, choice=pool_index, prior=float(pool.priors[i]))
                old = best.get(pid)
                if old is None or points.expected_gain(cand.probs) > points.expected_gain(old.probs):
                    best[pid] = cand
        return sorted(best.values(), key=lambda c: c.pid)

    # ----- ordering ------------------------------------------------------------
    def ordering(self, state, sort_by, sort_order):
        key = (sort_by, sort_order)
        with state.lock:
            cached = state.orderings.get(key)
            if cached is not None:
                return cached
            if state.abstain:
                order = []
            elif sort_by == 'relevance':
                order = [c.pid for c in state.ranked]
                if sort_order == 'asc':
                    order.reverse()
            else:
                rank = self.index.order_rank[key]
                chosen = points.choose_field_sorted_set(state.cands, sort_key=lambda c: int(rank[c.pid]), k=10)
                order = [c.pid for c in chosen]
            state.orderings[key] = order
            return order

    # ----- recommendations ---------------------------------------------------
    def recommend(self, state, page, args, deadline):
        grader = None
        if self.twin is not None:
            query = args['query']

            def grader(anchor_pid, pids):
                return self.twin.grade(query, [(None, pids)], deadline=deadline, anchor_pid=anchor_pid)
        return self.recommender.recommend(state, page, grader=grader)

    def write_grades(self, query, grades):
        try:
            os.makedirs('logs', exist_ok=True)
            ids = self.index.ids
            with open(os.path.join('logs', 'grades.jsonl'), 'a') as fh:
                for pid, g in grades.items():
                    fh.write(json.dumps({'query': query, 'pid': ids[pid], 'twin_grades': g}) + '\n')
        except OSError:
            log.warning('could not write logs/grades.jsonl')

    def log_state(self, args, state):
        log.info('q=%r src=%s abstain=%s %s pools=%s ranked=%d graded=%d t=%.0fms', args['query'][:80],
                 state.source, state.abstain, state.reason, [int(p.support_count) for p in state.pools],
                 len(state.ranked), state.graded, 1000 * state.timings.get('total', 0))
