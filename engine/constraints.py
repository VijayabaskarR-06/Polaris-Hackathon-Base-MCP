"""Hard constraints for a plan choice, and feasibility.

Explicit filters are applied exactly like the empty-query path. Query
requirements (colours, exclusions, budget, stated audience) are added on top,
except where they contradict an explicit filter on the same field: then the
filter wins and the query requirement is dropped (the brief's precedence rule).
"""
from dataclasses import dataclass, field

import numpy as np

STRICT_COLOR_MIN = 20
REQUIRED_SECTION_MIN = 5


@dataclass
class HardInfo:
    mask: np.ndarray
    colors: list = field(default_factory=list)      # colours actually enforced
    dropped: list = field(default_factory=list)     # query requirements dropped in favour of filters
    feasible: bool = True


def _codes_mask(index, field_name, values):
    if not values:
        return None
    codes = [i for i, v in enumerate(index.values[field_name]) if v in set(values)]
    return np.isin(index.codes[field_name], codes)


def type_mask(index, types):
    return _codes_mask(index, 'product_type', types)


def base_mask(index, filters):
    mask = index.mask(filters)
    return np.ones(index.n, dtype=bool) if mask is None else mask.copy()


def _apply(mask, extra, info, label, filter_fields, filters):
    """AND a query requirement into mask, unless an explicit filter on the same
    field contradicts it (empty intersection): then the filter wins."""
    if extra is None:
        return mask
    combined = mask & extra
    if any(f in filters for f in filter_fields) and mask.any() and not combined.any():
        info.dropped.append(label)
        return mask
    return combined


def strict_colors(choice):
    """The exact colour values the shopper named (first of each family), before shades."""
    families = getattr(choice, 'color_families', None)
    if families:
        return [f[0] for f in families if f]
    return choice.colors[:1]


def hard_mask(index, vocab, filters, plan, choice, support=None):
    """Hard-constraint mask for one choice.

    support: products that plausibly are the requested kind of item (type or
    keyword match); used only to decide strict-first colours, so the decision
    is the same for every page of the same request.
    """
    mask = base_mask(index, filters)
    info = HardInfo(mask=mask)
    group = getattr(choice, 'audience_group', None) or plan.audience_group
    if group:
        mask = _apply(mask, vocab.group_masks[group], info, 'audience_group', ('audience', 'section'), filters)
    required = getattr(choice, 'required_sections', None)
    if required and 'section' not in filters:
        # "plus size", "maternity": a stated requirement, enforced when the catalog can satisfy it.
        sections = _codes_mask(index, 'section', required)
        pool = mask if support is None else mask & support
        if np.count_nonzero(pool & sections) >= REQUIRED_SECTION_MIN:
            mask = mask & sections
        else:
            info.dropped.append('required_sections')
    if choice.min_price is not None or choice.max_price is not None:
        low = choice.min_price if choice.min_price is not None else -np.inf
        high = choice.max_price if choice.max_price is not None else np.inf
        price = (index.price >= low) & (index.price <= high)
        mask = _apply(mask, price, info, 'budget', ('min_price', 'max_price'), filters)
    if choice.exclude_colors:
        mask = _apply(mask, ~_codes_mask(index, 'color', choice.exclude_colors), info, 'exclude_colors',
                      ('color',), filters)
    if choice.exclude_patterns:
        mask = _apply(mask, ~_codes_mask(index, 'pattern', choice.exclude_patterns), info, 'exclude_patterns',
                      ('pattern',), filters)
    if choice.colors and 'color' not in filters:
        strict = strict_colors(choice)
        strict_mask = _codes_mask(index, 'color', strict)
        full_mask = _codes_mask(index, 'color', choice.colors)
        pool = mask if support is None else mask & support
        if strict_mask is not None and np.count_nonzero(pool & strict_mask) >= STRICT_COLOR_MIN:
            mask, info.colors = mask & strict_mask, strict
        else:
            mask, info.colors = mask & full_mask, list(choice.colors)
    elif choice.colors:
        info.dropped.append('colors')
    info.mask = mask
    return info
