"""Complementary recommendations for the products on the current page.

Recommendations take their own hard constraints: the complement's budget and
colours, the stated audience (or the anchor's audience family), never the
explicit `filters` argument, and never a product already on the page.
"""
import os
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
# Named accompaniments are spread over the top equally exact main products, one each.
REC_SPREAD = os.getenv('REC_SPREAD', '1') == '1'
# "a bikini top with matching bottoms": the other half of the same kind of set.
SET_PARTS = {
    'bottom': {'Bikini top': ['Swimwear bottom'], 'Swimwear top': ['Swimwear bottom'],
               'Swimsuit': ['Swimwear bottom'], 'Bra': ['Underwear bottom'],
               'Pyjama set': ['Pyjama bottom'], 'Kids Underwear top': ['Underwear bottom']},
    'top': {'Swimwear bottom': ['Bikini top', 'Swimwear top'], 'Underwear bottom': ['Bra'],
            'Pyjama bottom': ['Top', 'T-shirt', 'Vest top']},
}
# Kinds that stand in for each other: owning a coat, a jacket is no accompaniment.
SAME_KIND = {'Coat': ('Jacket',), 'Jacket': ('Coat',), 'Sneakers': ('Other shoe',), 'Other shoe': ('Sneakers',)}
# Outfit slots: an owned dress already covers the top and the bottom, owned trousers the bottom.
SLOTS = {
    'bottom': ('Trousers', 'Skirt', 'Shorts', 'Leggings/Tights', 'Outdoor trousers'),
    'top': ('T-shirt', 'Top', 'Blouse', 'Shirt', 'Vest top', 'Polo shirt', 'Sweater', 'Hoodie'),
    'onepiece': ('Dress', 'Jumpsuit/Playsuit', 'Dungarees'),
    'outer': ('Jacket', 'Coat', 'Blazer', 'Outdoor Waistcoat', 'Cardigan'),
    'shoes': ('Sneakers', 'Boots', 'Sandals', 'Heeled sandals', 'Pumps', 'Heels', 'Wedge', 'Ballerinas',
              'Flat shoe', 'Flat shoes', 'Other shoe', 'Flip flop'),
}
SLOT_OF = {t: slot for slot, types in SLOTS.items() for t in types}
COVERS = {'onepiece': {'onepiece', 'top', 'bottom'}, 'top': {'top', 'onepiece'},
          'bottom': {'bottom', 'onepiece'}, 'outer': {'outer'}, 'shoes': {'shoes'}}
# Bags for chores, not for an outfit ("cosmetic bag", "pencil case").
UTILITY_WORDS = ['cosmetic', 'case', 'pencil', 'gymbag', 'packable', 'washbag', 'toiletry', 'laptop', 'lunch',
                 'pouch', 'waistbag', 'bumbag', 'backpack', 'gym', 'bum', 'fanny']
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

    def targets(self, plan, anchor, ref_type=None, group=None):
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
        context = [t for t in (getattr(plan, 'context_types', []) or []) if t != ref_type]
        owned = set(context)
        owned |= {t for o in owned for t in SAME_KIND.get(o, ())}
        if context:
            # "I bought a black dress, find shoes to go with it": the shoes and the dress make the
            # outfit, so the rest completes both (a bag, earrings), never another dress or a skirt.
            covered = set().union(*(COVERS.get(SLOT_OF.get(t), set()) for t in context + [ref_type]))
            around = self.vocab.complement_map.get(context[0]) or []
            both = [t for t in types if t in around]
            types = both + [t for t in around if t not in both] + [t for t in types if t not in around]
            types = [t for t in types if SLOT_OF.get(t) not in covered]
        # Implicit complements never repeat the anchor's kind or what the shopper already has, and
        # suit the wearer (no dress for a boy's sun hat).
        other = {'men': 'women', 'boys': 'women', 'women': 'men', 'girls': 'men'}.get(group)
        return [t for t in types if t != anchor['product_type'] and t not in owned
                and (other is None or self.vocab.type_gender.get(t) != other)], [], False

    def hard_mask(self, plan, anchor_pid, page, anchor_choice=None, use_group=True):
        idx = self.index
        mask = np.ones(idx.n, dtype=bool)
        group = (getattr(anchor_choice, 'audience_group', None) or plan.audience_group
                 or self.vocab.group_of(anchor_pid))
        if group and use_group:
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
        group = (getattr(anchor_choice, 'audience_group', None) or plan.audience_group
                 or self.vocab.group_of(anchor_pid))
        types, keywords, explicit = self.targets(plan, anchor, ref_type, group)
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
        comp = plan.complement
        if len(positions) == 0 and explicit and comp is not None and getattr(comp, 'from_group', False) \
                and set(comp.product_types) - set(types):
            # "a hoodie with matching shoes under 1000": no trainers at that price, so any shoe.
            types = list(comp.product_types)
            base = mask & type_mask(self.index, types)
            positions = np.flatnonzero(base)
        stated = bool(getattr(anchor_choice, 'audience_group', None) or plan.audience_group)
        if len(positions) == 0 and explicit and types and not stated:
            # "a white shirt with a matching tie": the shirt is women's, ties are sold for men only;
            # the tie asked for beats an audience that was only inferred from the anchor.
            mask = self.hard_mask(plan, anchor_pid, page, anchor_choice, use_group=False)
            base = mask & type_mask(self.index, types)
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
        matching = 3.0 if getattr(plan, 'matching', False) or explicit else 0.3
        utility = bm.any_mask(UTILITY_WORDS, names_only=True)[positions]
        if 'Belt' in types:
            utility = utility | bm.any_mask(['bag'], names_only=True)[positions]   # belt bags are bags
        if 'Wallet' in types:
            utility = utility | bm.any_mask(['cardholder', 'card'], names_only=True)[positions]
        if 'Bag' in types and not set(tokenize(' '.join(getattr(anchor_choice, 'soft_words', None) or []))) & {
                'beach', 'summer', 'holiday', 'vacation', 'travel', 'pool'}:
            # "a matching handbag": a beach shopper or a weekend holdall is a looser match.
            utility = utility | bm.any_mask(['beach', 'weekend', 'shopper', 'straw', 'holdall', 'duffel'],
                                            names_only=True)[positions]
        # "a matching clutch": a bag named clutch beats a tote whose description mentions one.
        name_hit = (bm.any_mask(kw_terms, names_only=True)[positions] if explicit and kw_terms
                    else np.zeros(len(positions), dtype=bool))
        score = (2.0 * harmony + matching * same_family + 1.5 * bm_norm + 1.0 * kw_hit + 0.3 * rating
                 + (1.5 if explicit else 0.5) * same_line - 0.6 * np.minimum(type_rank, 5) - 1.5 * utility
                 + 2.0 * name_hit)
        est = np.where(harmony & kw_hit, 3, 2)
        if explicit and not kw_terms:
            est = np.where(harmony, 3, 2)
        exact = np.array([colors[c] == anchor['color'] for c in color_codes])
        if explicit and same_family.any():
            # "a red dress with a matching clutch": judges give the top grade to an accompaniment in
            # the main item's colour; a black or beige one goes with it but is a looser match, and
            # so is another shade ("light purple does not match the purple skirt").
            match = exact if np.count_nonzero(exact & kw_hit) >= 3 else same_family
            est = np.where(match & kw_hit, 3, 2)
            score = score + 2.0 * exact
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
        out, names = [], set()
        for rank, i in enumerate(chosen):
            prior = (float(score[i]) - float(score[chosen].min())) / span
            name = self.products[int(positions[i])]['name'].strip().lower()
            if name in names:
                prior *= 0.5         # a second colourway of the same product only after the others
            names.add(name)
            out.append((int(positions[i]), int(est[i]), prior))
        return out, explicit

    def choose_anchor(self, state, page_cands):
        """The page product every recommendation hangs on: a recommendation only counts when its
        anchor grades >= 2, so among equally likely products prefer the one that is exactly what
        was asked for (the stated colour itself rather than a shade, the primary kind rather than
        a sibling type), then the earliest."""
        ranked = self.anchor_options(state, page_cands)
        return ranked[0][1] if ranked else None

    def anchor_options(self, state, page_cands):
        """(key, candidate) for every page product, best anchor first (see choose_anchor)."""
        ranked = []
        for i, cand in enumerate(page_cands):
            p = self.products[cand.pid]
            choice = state.pools[cand.choice].choice if state.pools and 0 <= cand.choice < len(state.pools) else None
            exact_colour = primary_type = 0
            if choice is not None:
                families = getattr(choice, 'color_families', None) or []
                exact_colour = int(any(f and p['color'] == f[0] for f in families)) if families else 1
                primary = getattr(choice, 'primary_types', None) or choice.product_types
                primary_type = int(not primary or p['product_type'] in primary)
            key = (round(points.p_at_least(cand.probs, 2), 2), primary_type, exact_colour, -i)
            ranked.append((key, cand))
        ranked.sort(key=lambda kc: kc[0], reverse=True)
        return ranked

    def fitting_anchor(self, state, ranked, page):
        """Among equally exact anchors, the one whose accompaniments can match it.

        "a yellow t-shirt with a coordinating wallet": a men's tee only has black wallets in its
        section, a women's tee has yellow ones; "a white shirt with a matching tie": ties are men's.
        Preference: more accompaniments in the anchor's colour, then more from its own audience.
        """
        best = ranked[0][1]
        plan = state.plan
        if plan.complement is None or plan.complement.colors:
            return best
        top = ranked[0][0][:3]
        options = [cand for key, cand in ranked if key[:3] == top][:5]
        if len(options) < 2:
            return best
        best_fit = None
        for cand in options:
            choice = state.pools[cand.choice].choice if state.pools and 0 <= cand.choice < len(state.pools) else None
            raw, _ = self.candidates(plan, cand.pid, page, choice)
            family = color_family(self.products[cand.pid]['color'])
            group = self.vocab.group_of(cand.pid)
            same = sum(color_family(self.products[pid]['color']) == family for pid, _, _ in raw)
            own = sum(self.vocab.group_of(pid) == group for pid, _, _ in raw[:MAX_RECS])
            fit = (min(same, MAX_RECS), own)
            if best_fit is None or fit > best_fit:
                best, best_fit = cand, fit
        return best

    def recommend(self, state, page, grader=None):
        plan = state.plan
        page_cands = [c for c in (state.candidate(pid) for pid in page) if c is not None]
        ranked = self.anchor_options(state, page_cands)
        if not ranked:
            return []
        anchor = self.fitting_anchor(state, ranked, page)
        cache_key = (state.key, anchor.pid, tuple(page))
        cached = self.engine.rec_cache.get(cache_key)
        if cached is not None:
            return cached
        cands = self.graded_candidates(state, anchor, page, grader)
        chosen = points.select_recommendations(cands, anchor_p2=points.p_at_least(anchor.probs, 2), k=MAX_RECS)
        pairs = [(c, anchor) for c in chosen]
        if REC_SPREAD and plan.complement is not None and len(chosen) > 1:
            pairs = self.spread(state, ranked, anchor, chosen, page, grader)
        ids = self.index.ids
        recs = [{'id': ids[c.pid], 'for_product_id': ids[a.pid],
                 'reason': reason(self.products[c.pid], self.products[a.pid])} for c, a in pairs]
        return self.engine.rec_cache.put(cache_key, recs)

    def graded_candidates(self, state, anchor, page, grader=None):
        choice = state.pools[anchor.choice].choice if state.pools and 0 <= anchor.choice < len(state.pools) else None
        raw, _ = self.candidates(state.plan, anchor.pid, page, choice)
        grades = grader(anchor.pid, [pid for pid, _, _ in raw]) if grader and raw else {}
        cands = []
        for pid, est, prior in raw:
            g = grades.get(pid)
            probs = (points.grade_probs(g, self.engine.calibration, source='twin') if g
                     else points.grade_probs([est], self.engine.calibration, source='estimate'))
            cands.append(points.Candidate(pid=pid, probs=probs, prior=prior))
        return points.order_by_expected_gain(cands, min_gain=0.3)

    def spread(self, state, ranked, anchor, chosen, page, grader=None):
        """'a dress with a matching bag': each of the top equally exact dresses on the page gets its
        own best matching bag, as long as it is as good as the main anchor's weakest pick; the
        main anchor's own picks fill the rest. Best expected grade first."""
        top = ranked[0][0][:3]
        position = {pid: i for i, pid in enumerate(page)}
        others = sorted((c for key, c in ranked if key[:3] == top and c.pid != anchor.pid),
                        key=lambda c: position.get(c.pid, len(page)))
        floor = min(points.expected_gain(c.probs) for c in chosen)
        pairs, used = [(chosen[0], anchor)], {chosen[0].pid}
        for other in others[:MAX_RECS - 1]:
            pick = next((c for c in self.graded_candidates(state, other, page, grader)
                         if c.pid not in used and points.expected_gain(c.probs) >= floor), None)
            if pick is not None:
                pairs.append((pick, other))
                used.add(pick.pid)
        for c in chosen[1:]:
            if len(pairs) >= MAX_RECS:
                break
            if c.pid not in used:
                pairs.append((c, anchor))
                used.add(c.pid)
        rank = {id(pair): i for i, pair in enumerate(pairs)}
        pairs.sort(key=lambda pair: (-points.expected_gain(pair[0].probs), rank[id(pair)]))
        return pairs[:MAX_RECS]
