"""Exact catalog filtering, field sorting and pagination over numpy arrays.

For an empty query every page must equal the starter implementation
(eval/reference.py) exactly. The contract's ordering rules are baked into
precomputed permutations, built once at startup:

* equal sort keys break by id ascending in either direction;
* null ratings sort last for both rating directions;
* relevance with an empty query is id ascending regardless of sort_order.

Products are held sorted by id, so a product's position is also its id rank.
"""
import numpy as np

STRING_FIELDS = ('category', 'product_type', 'color', 'audience', 'section', 'pattern')
SORT_KEYS = [(s, o) for s in ('relevance', 'id', 'price', 'rating', 'review_count')
             for o in ('asc', 'desc')]


def fold(value):
    """The contract's string comparison: strip outer whitespace, compare case-insensitively."""
    return str(value or '').strip().casefold()


class Index:
    def __init__(self, products):
        self.products = products
        self.n = n = len(products)
        self.ids = [p['id'] for p in products]
        if any(self.ids[i] >= self.ids[i + 1] for i in range(n - 1)):
            raise ValueError('products must be sorted by unique id')
        self.pos_of = {pid: i for i, pid in enumerate(self.ids)}
        self.price = np.array([p['price']['amount'] for p in products], dtype=np.float64)
        averages = [p['rating']['average'] for p in products]
        self.rated = np.array([a is not None for a in averages], dtype=bool)
        self.rating = np.array([np.nan if a is None else a for a in averages], dtype=np.float64)
        # Null ratings become -inf so that they fail every min_rating bound.
        self.rating_or_low = np.where(self.rated, np.nan_to_num(self.rating), -np.inf)
        self.count = np.array([p['rating']['count'] for p in products], dtype=np.int64)

        # codes[f][pos] is a small int naming the product's raw value of field f.
        # lookup[f] maps a folded value to every code with that spelling
        # (product_type has both "Dog Wear" and "Dog wear").
        self.codes, self.values, self.lookup = {}, {}, {}
        for field in STRING_FIELDS:
            raw = [str(p.get(field) or '') for p in products]
            values = sorted(set(raw))
            code_of = {v: i for i, v in enumerate(values)}
            self.codes[field] = np.array([code_of[v] for v in raw], dtype=np.int32)
            self.values[field] = values
            lookup = {}
            for value, code in code_of.items():
                lookup.setdefault(fold(value), []).append(code)
            self.lookup[field] = lookup
        self._value_masks = {}
        self._none = np.zeros(n, dtype=bool)
        self._none.flags.writeable = False

        self.positions = np.arange(n, dtype=np.int64)
        self.order = {}
        self.order_rank = {}
        self._build_orders()

    def _build_orders(self):
        pos = self.positions
        by_id = pos
        orders = {
            ('relevance', 'asc'): by_id,
            ('relevance', 'desc'): by_id,
            ('id', 'asc'): by_id,
            ('id', 'desc'): by_id[::-1].copy(),
            # np.lexsort sorts by the last key first; position (= id rank) breaks ties.
            ('price', 'asc'): np.lexsort((pos, self.price)),
            ('price', 'desc'): np.lexsort((pos, -self.price)),
            ('review_count', 'asc'): np.lexsort((pos, self.count)),
            ('review_count', 'desc'): np.lexsort((pos, -self.count)),
        }
        rated, unrated = pos[self.rated], pos[~self.rated]
        rating = self.rating[self.rated]
        orders[('rating', 'asc')] = np.concatenate([rated[np.lexsort((rated, rating))], unrated])
        orders[('rating', 'desc')] = np.concatenate([rated[np.lexsort((rated, -rating))], unrated])
        for key, order in orders.items():
            order = np.ascontiguousarray(order, dtype=np.int64)
            order.flags.writeable = False
            rank = np.empty(self.n, dtype=np.int64)
            rank[order] = pos
            rank.flags.writeable = False
            self.order[key] = order
            self.order_rank[key] = rank

    def value_mask(self, field, value):
        """Read-only boolean mask of products whose field equals value under the contract's folding."""
        key = (field, fold(value))
        mask = self._value_masks.get(key)
        if mask is None:
            codes = self.lookup[field].get(key[1], [])
            if not codes:
                return self._none          # unknown value: shared, never cached
            if len(codes) == 1:
                mask = self.codes[field] == codes[0]
            else:
                mask = np.isin(self.codes[field], codes)
            mask.flags.writeable = False
            if len(self._value_masks) < 4096:
                self._value_masks[key] = mask
        return mask

    def mask(self, filters):
        """AND of every explicit filter; None means no filter (all products pass)."""
        result = None

        def both(mask):
            return mask if result is None else result & mask

        for field in STRING_FIELDS:
            if field in filters:
                result = both(self.value_mask(field, filters[field]))
        if 'min_price' in filters:
            result = both(self.price >= filters['min_price'])
        if 'max_price' in filters:
            result = both(self.price <= filters['max_price'])
        if 'min_rating' in filters:
            result = both(self.rating_or_low >= filters['min_rating'])
        if 'min_review_count' in filters:
            result = both(self.count >= filters['min_review_count'])
        return result

    def sort_positions(self, positions, sort_by, sort_order):
        """Order arbitrary positions by a contract field sort (ties by id, nulls last)."""
        positions = np.asarray(positions, dtype=np.int64)
        rank = self.order_rank[(sort_by, sort_order)]
        return positions[np.argsort(rank[positions], kind='stable')]

    def filtered_page(self, filters, sort_by, sort_order, offset, limit):
        """Positions of the exact empty-query page: filter, order, then offset and limit."""
        order = self.order[(sort_by, sort_order)]
        mask = self.mask(filters)
        if mask is not None:
            order = order[mask[order]]
        return order[offset:offset + limit]

    def page_ids(self, filters, sort_by, sort_order, offset, limit):
        ids = self.ids
        return [ids[i] for i in self.filtered_page(filters, sort_by, sort_order, offset, limit)]
