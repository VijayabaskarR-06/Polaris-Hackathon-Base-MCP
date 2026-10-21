"""Complementary recommendations for the products on the current page.

Recommendations take their own hard constraints: the complement's budget and
colours, the stated audience (or the anchor's audience family), never the
explicit `filters` argument, and never a product already on the page.
"""
import re

import numpy as np

from engine import points
from engine.constraints import type_mask
from engine.text import tokenize

NEUTRALS = {'Black', 'White', 'Off White', 'Beige', 'Light Beige', 'Dark Beige', 'Greyish Beige', 'Gold',
            'Silver', 'Grey', 'Light Grey', 'Dark Grey', 'Transparent'}
FALLBACK_COMPLEMENTS = ['Bag', 'Belt', 'Sunglasses', 'Scarf', 'Necklace', 'Earring']
CANDIDATES = 30
MAX_RECS = 5
# "a bikini top with matching bottoms": the other half of the same kind of set.
SET_PARTS = {
    'bottom': {'Bikini top': ['Swimwear bottom'], 'Swimwear top': ['Swimwear bottom'],
               'Swimsuit': ['Swimwear bottom'], 'Bra': ['Underwear bottom'],
               'Pyjama set': ['Pyjama bottom'], 'Kids Underwear top': ['Underwear bottom']},
    'top': {'Swimwear bottom': ['Bikini top', 'Swimwear top'], 'Underwear bottom': ['Bra'],
            'Pyjama bottom': ['Top', 'T-shirt', 'Vest top']},
}
# Bags for chores, not for an outfit ("cosmetic bag", "pencil case").
UTILITY_WORDS = ['cosmetic', 'case', 'pencil', 'gymbag', 'packable', 'washbag', 'toiletry', 'laptop', 'lunch',
                 'pouch', 'waistbag', 'bumbag', 'backpack']
# Name words too common to mark a matching line ("basic", "slim").
NAME_NOISE = {'basic', 'slim', 'regular', 'loose', 'relaxed', 'soft', 'long', 'short', 'pack', 'the', 'and', 'with'}


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


def line_word(name):
    """The product line of a name ("Zoey bikini top" -> "zoey"): its first distinctive word."""
    for word in re.findall(r'[a-z]+', (name or '').lower()):
        if len(word) >= 3 and word not in NAME_NOISE:
            return word
    return None


class Recommender:
    def __init__(self, engine):
        self.engine = engine
        self.index = engine.index
        self.vocab = engine.vocab
        self.products = engine.products

    def targets(self, plan, anchor, ref_type=None):
        """(complement product types in priority order, keywords, explicit?).

        ref_type is the kind of item the shopper asked for (a suit's jacket even when the anchor
        is its trousers); it decides which shoes or accessories suit it.
        """
        ref_type = ref_type or anchor['product_type']
        comp = plan.complement
        if comp is not None and comp.product_types:
            types = list(comp.product_types)
            words = ' '.join([comp.label] + list(getattr(comp, 'type_words', [])))
            for part, by_anchor in SET_PARTS.items():
                if re.search(rf'\b{part}s?\b', words) and anchor['product_type'] in by_anchor:
                    types = [t for t in by_anchor[anchor['product_type']] if t in self.vocab.type_set] or types
                    break
            else:
                if getattr(comp, 'from_group', False):
                    # "shoes to match" a party dress: heels and sandals before trainers.
                    pairs = [t for t in self.vocab.complement_map.get(ref_type, []) if t in types]
                    if pairs:
                        types = pairs
            return types, list(comp.keywords), True
        if comp is not None and comp.keywords:
            return [], list(comp.keywords), True
        types = self.vocab.complement_map.get(ref_type) or FALLBACK_COMPLEMENTS
        owned = set(getattr(plan, 'context_types', []) or [])
        # Implicit complements never repeat the anchor's kind or what the shopper already has.
        return [t for t in types if t != anchor['product_type'] and t not in owned], [], False

    def hard_mask(self, plan, anchor_pid, page, anchor_choice=None):
        idx = self.index
        mask = np.ones(idx.n, dtype=bool)
        group = (getattr(anchor_choice, 'audience_group', None) or plan.audience_group
                 or self.vocab.group_of(anchor_pid))
        if group:
            mask &= self.vocab.group_masks[group]
            if group in ('kids', 'girls', 'boys'):
                # A child's outfit stays in the child's size range: no baby socks for a schoolboy.
                baby = self.vocab.group_masks['baby']
                mask &= baby if baby[anchor_pid] else ~baby
        comp = plan.complement
        budget = comp if comp is not None and (comp.min_price is not None or comp.max_price is not None) else None
        if budget is None:
            # Budgets are per item unless the complement has its own.
            budget = anchor_choice if anchor_choice is not None else (
                plan.choices[0] if len(plan.choices) == 1 else None)
        if budget is not None:
            if budget.min_price is not None:
                mask &= idx.price >= budget.min_price
            if budget.max_price is not None:
                mask &= idx.price <= budget.max_price
        if comp is not None and comp.colors:
            colors = [i for i, v in enumerate(idx.values['color']) if v in set(comp.colors)]
            mask &= np.isin(idx.codes['color'], colors)
        if comp is not None and getattr(comp, 'needs_keyword', False) and getattr(comp, 'match_terms', None):
            # "a clutch", "cufflinks": the product must say so; the bare type is too broad.
            mask &= self.engine.bm25.coverage(comp.match_terms) > 0
        mask[np.asarray(page, dtype=np.int64)] = False
        return mask

    def candidates(self, plan, anchor_pid, page, anchor_choice=None):
        anchor = self.products[anchor_pid]
        if getattr(plan, 'complement_missing', False):
            return [], True          # the accompanying item asked for is not sold
        primary = list(getattr(anchor_choice, 'primary_types', None) or [])
        ref_type = anchor['product_type'] if not primary or anchor['product_type'] in primary else primary[0]
        types, keywords, explicit = self.targets(plan, anchor, ref_type)
        mask = self.hard_mask(plan, anchor_pid, page, anchor_choice)
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
        # The occasion of the main item ("a dress for dinner") favours evening bags and heels.
        terms += [t for w in (getattr(anchor_choice, 'soft_words', None) or [])[:6] for t in tokenize(w)
                  if t not in terms]
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
        line = line_word(anchor['name'])
        same_line = np.array([line is not None and line_word(self.products[p]['name']) == line
                              for p in positions.tolist()])
        # Judged data: neutral accessories (black, white, beige, gold, silver) grade at least as
        # well as same-colour ones, so harmony matters and an exact colour match only a little,
        # unless the shopper asked for a matching item.
        matching = 1.0 if getattr(plan, 'matching', False) else 0.3
        utility = bm.any_mask(UTILITY_WORDS, names_only=True)[positions]
        score = (2.0 * harmony + matching * same_family + 1.5 * bm_norm + 1.0 * kw_hit + 0.3 * rating
                 + (1.5 if explicit else 0.5) * same_line - 0.6 * np.minimum(type_rank, 5) - 1.5 * utility)
        est = np.where(harmony & kw_hit, 3, 2)
        if explicit and not kw_terms:
            est = np.where(harmony, 3, 2)
        order = np.lexsort((positions, -score))
        chosen, per_type = [], {}
        # Implicit complements: spread across complement types so one weak type can't sink all five;
        # several named ones ("sunglasses and a straw hat") share the five as well.
        named = len(getattr(plan.complement, 'type_words', None) or []) if plan.complement is not None else 0
        cap = (3 if named > 1 else None) if explicit else 2
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
        anchor_choice = None
        if state.pools and 0 <= anchor.choice < len(state.pools):
            anchor_choice = state.pools[anchor.choice].choice
        raw, _ = self.candidates(plan, anchor.pid, page, anchor_choice)
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
