"""Complementary recommendations for the products on the current page.

Recommendations take their own hard constraints: the complement's budget and
colours, the stated audience (or the anchor's audience family), never the
explicit `filters` argument, and never a product already on the page.
"""
import numpy as np

from engine import points
from engine.constraints import type_mask
from engine.text import tokenize

NEUTRALS = {'Black', 'White', 'Off White', 'Beige', 'Light Beige', 'Dark Beige', 'Greyish Beige', 'Gold',
            'Silver', 'Grey', 'Light Grey', 'Dark Grey', 'Transparent'}
FALLBACK_COMPLEMENTS = ['Bag', 'Belt', 'Sunglasses', 'Scarf', 'Necklace', 'Earring']
CANDIDATES = 30
MAX_RECS = 5


def color_family(color):
    if not color:
        return ''
    word = color.split()[-1].lower()
    return {'khaki': 'green', 'copper': 'bronze', 'bronze/copper': 'bronze'}.get(word, word)


def harmonious(anchor_color, color):
    return color in NEUTRALS or color_family(color) == color_family(anchor_color)


def reason(product, anchor):
    text = (f"{product['color']} {product['product_type'].lower()} to pair with the "
            f"{anchor['color'].lower()} {anchor['product_type'].lower()}; INR {product['price']['amount']}.")
    return text[:300]


class Recommender:
    def __init__(self, engine):
        self.engine = engine
        self.index = engine.index
        self.vocab = engine.vocab
        self.products = engine.products

    def targets(self, plan, anchor):
        """(complement product types in priority order, keywords, explicit?)."""
        if plan.complement is not None and plan.complement.product_types:
            return list(plan.complement.product_types), list(plan.complement.keywords), True
        if plan.complement is not None and plan.complement.keywords:
            return [], list(plan.complement.keywords), True
        types = self.vocab.complement_map.get(anchor['product_type']) or FALLBACK_COMPLEMENTS
        return [t for t in types if t != anchor['product_type']], [], False

    def hard_mask(self, plan, anchor_pid, page):
        idx = self.index
        mask = np.ones(idx.n, dtype=bool)
        group = plan.audience_group or self.vocab.group_of(anchor_pid)
        if group:
            mask &= self.vocab.group_masks[group]
        comp = plan.complement
        budget = comp if comp is not None and (comp.min_price is not None or comp.max_price is not None) else None
        if budget is None and len(plan.choices) == 1:
            budget = plan.choices[0]          # budgets are per item unless the complement has its own
        if budget is not None:
            if budget.min_price is not None:
                mask &= idx.price >= budget.min_price
            if budget.max_price is not None:
                mask &= idx.price <= budget.max_price
        if comp is not None and comp.colors:
            colors = [i for i, v in enumerate(idx.values['color']) if v in set(comp.colors)]
            mask &= np.isin(idx.codes['color'], colors)
        mask[np.asarray(page, dtype=np.int64)] = False
        return mask

    def candidates(self, plan, anchor_pid, page):
        anchor = self.products[anchor_pid]
        types, keywords, explicit = self.targets(plan, anchor)
        mask = self.hard_mask(plan, anchor_pid, page)
        bm = self.engine.bm25
        kw_terms = []
        for word in keywords:
            kw_terms += [t for t in tokenize(word) if t not in kw_terms]
        extra = self.vocab.complement_keywords
        if types:
            base = mask & type_mask(self.index, types)
        elif kw_terms:
            base = mask & bm.any_mask(kw_terms)
        else:
            return [], explicit
        positions = np.flatnonzero(base)
        if len(positions) == 0:
            return [], explicit
        anchor_terms = [t for t in tokenize(anchor['name']) if len(t) > 2]
        terms = kw_terms + [t for w in (extra.get(t, []) for t in types[:3]) for x in w for t in tokenize(x)]
        bm_score = bm.scores(terms + anchor_terms, {t: 0.5 for t in anchor_terms})[positions] if terms or \
            anchor_terms else np.zeros(len(positions))
        top = float(bm_score.max()) if len(bm_score) else 0.0
        bm_norm = bm_score / top if top > 0 else np.zeros(len(positions))
        type_codes = self.index.codes['product_type'][positions]
        type_values = self.index.values['product_type']
        type_rank = np.array([types.index(type_values[c]) if type_values[c] in types else len(types)
                              for c in type_codes], dtype=np.float64)
        colors = self.index.values['color']
        color_codes = self.index.codes['color'][positions]
        harmony = np.array([harmonious(anchor['color'], colors[c]) for c in color_codes])
        same_family = np.array([color_family(colors[c]) == color_family(anchor['color']) for c in color_codes])
        kw_hit = (bm.coverage(kw_terms)[positions] > 0) if kw_terms else np.ones(len(positions), dtype=bool)
        rating = np.nan_to_num(self.index.rating[positions], nan=0.0) / 5.0
        # Judged data: neutral accessories (black, white, beige, gold, silver) grade at least as
        # well as same-colour ones, so harmony matters and an exact colour match only a little.
        score = (2.0 * harmony + 0.3 * same_family + 1.5 * bm_norm + 1.0 * kw_hit + 0.3 * rating
                 - 0.6 * np.minimum(type_rank, 5))
        est = np.where(harmony & kw_hit, 3, 2)
        if explicit and not kw_terms:
            est = np.where(harmony, 3, 2)
        order = np.lexsort((positions, -score))
        chosen, per_type = [], {}
        # Implicit complements: spread across complement types so one weak type can't sink all five.
        cap = None if explicit else 2
        for i in order:
            t = type_codes[i]
            if cap is not None and per_type.get(t, 0) >= cap:
                continue
            per_type[t] = per_type.get(t, 0) + 1
            chosen.append(i)
            if len(chosen) >= CANDIDATES:
                break
        if len(chosen) < MAX_RECS:
            chosen += [i for i in order if i not in chosen][:CANDIDATES - len(chosen)]
        top_score = float(score[chosen].max()) if chosen else 1.0
        span = max(1e-9, top_score - float(score[chosen].min())) if chosen else 1.0
        out = []
        for rank, i in enumerate(chosen):
            prior = (float(score[i]) - float(score[chosen].min())) / span
            out.append((int(positions[i]), int(est[i]), prior))
        return out, explicit

    def recommend(self, state, page, grader=None):
        plan = state.plan
        page_cands = [c for c in (state.candidate(pid) for pid in page) if c is not None]
        anchor = points.choose_anchor(page_cands)
        if anchor is None:
            return []
        cache_key = (state.key, anchor.pid, tuple(page))
        cached = self.engine.rec_cache.get(cache_key)
        if cached is not None:
            return cached
        raw, _ = self.candidates(plan, anchor.pid, page)
        grades = grader(anchor.pid, [pid for pid, _, _ in raw]) if grader and raw else {}
        cands = []
        for pid, est, prior in raw:
            g = grades.get(pid)
            probs = (points.grade_probs(g, self.engine.calibration, source='twin') if g
                     else points.grade_probs([est], self.engine.calibration, source='estimate'))
            cands.append(points.Candidate(pid=pid, probs=probs, prior=prior))
        chosen = points.select_recommendations(cands, anchor_p2=points.p_at_least(anchor.probs, 2), k=MAX_RECS)
        anchor_product = self.products[anchor.pid]
        ids = self.index.ids
        recs = [{'id': ids[c.pid], 'for_product_id': ids[anchor.pid],
                 'reason': reason(self.products[c.pid], anchor_product)} for c in chosen]
        return self.engine.rec_cache.put(cache_key, recs)
