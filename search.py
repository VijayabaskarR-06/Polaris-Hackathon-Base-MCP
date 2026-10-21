"""Catalog search: the exact contract engine for empty queries, the semantic engine otherwise."""
import json
import math
import re
from pathlib import Path

from jsonschema import Draft202012Validator, ValidationError

from engine.index import Index, STRING_FIELDS

ROOT = Path(__file__).resolve().parent
INPUT_SCHEMA = json.loads((ROOT / 'schemas/input.schema.json').read_text())
OUTPUT_SCHEMA = json.loads((ROOT / 'schemas/output.schema.json').read_text())
VALIDATOR = Draft202012Validator(INPUT_SCHEMA)
# Fields the engine never reads; dropping them keeps memory and GC work down.
UNUSED_FIELDS = ('reviews', 'image_url')


def validate_arguments(arguments):
    # Reject non-JSON floating point values even when called directly from Python.
    json.dumps(arguments, allow_nan=False)
    try:
        VALIDATOR.validate(arguments)
    except ValidationError as exc:
        raise ValueError(exc.message) from exc
    filters = arguments.get('filters', {})
    if any(not filters[k].strip() for k in STRING_FIELDS if k in filters):
        raise ValueError('String filters cannot be blank')
    if filters.get('min_price', 0) > filters.get('max_price', math.inf):
        raise ValueError('min_price cannot exceed max_price')


def normalize_arguments(arguments):
    """Validated arguments with every default applied and integral floats made ints.

    JSON Schema 2020-12 accepts 3.0 as an integer, so limit/offset/min_review_count
    may arrive as floats; slicing needs real ints.
    """
    filters = dict(arguments.get('filters', {}))
    if 'min_review_count' in filters:
        filters['min_review_count'] = int(filters['min_review_count'])
    return {
        'query': arguments.get('query', ''),
        'filters': filters,
        'sort_by': arguments.get('sort_by', 'relevance'),
        'sort_order': arguments.get('sort_order', 'desc'),
        'offset': int(arguments.get('offset', 0)),
        'limit': int(arguments.get('limit', 10)),
        'include_recommendations': arguments.get('include_recommendations', False),
    }


def is_empty_query(args):
    """Only a whitespace-only query is empty; punctuation such as "!!!" is a real (nonsense) query."""
    return not args['query'].strip()


class LexicalEngine:
    """Starter-style AND token matching, used until the semantic engine is available."""

    def __init__(self, products, index):
        self.index = index
        self.words = [set(re.findall(r'\w+', ' '.join(str(p.get(k) or '') for k in
                      ('name', 'description', *STRING_FIELDS)).casefold())) for p in products]
        self.names = [set(re.findall(r'\w+', p['name'].casefold())) for p in products]

    def search(self, args):
        query = re.findall(r'\w+', args['query'].casefold())
        if not query:
            return {'results': [], 'recommendations': []}
        mask = self.index.mask(args['filters'])
        scored = []
        for pos in range(self.index.n):
            if (mask is None or mask[pos]) and all(w in self.words[pos] for w in query):
                scored.append((-(1 + sum(w in self.names[pos] for w in query)), pos))
        if args['sort_by'] == 'relevance':
            scored.sort()
            positions = [pos for _, pos in scored]
            if args['sort_order'] == 'asc':
                positions.reverse()
        else:
            positions = self.index.sort_positions([pos for _, pos in scored],
                                                  args['sort_by'], args['sort_order'])
        page = positions[args['offset']:args['offset'] + args['limit']]
        return {'results': [{'id': self.index.ids[p]} for p in page], 'recommendations': []}


class Catalog:
    def __init__(self, products, semantic=True):
        self.products = sorted(products, key=lambda p: p['id'])
        for product in self.products:
            for field in UNUSED_FIELDS:
                product.pop(field, None)
        self.index = Index(self.products)
        self.engine = None
        if semantic:
            self.engine = LexicalEngine(self.products, self.index)

    @classmethod
    def load(cls, path, semantic=True):
        return cls(json.loads(Path(path).read_text(encoding='utf-8')), semantic=semantic)

    def prepare(self, arguments):
        """Validate (ValueError on invalid input) and normalize tool arguments."""
        validate_arguments(arguments)
        return normalize_arguments(arguments)

    def exact(self, args):
        """Exact empty-query page: catalog filtering, contract ordering, offset, limit."""
        return {'results': [{'id': pid} for pid in self.index.page_ids(
                    args['filters'], args['sort_by'], args['sort_order'],
                    args['offset'], args['limit'])],
                'recommendations': []}

    def semantic(self, args):
        return self.engine.search(args)

    def warm_up(self):
        """Touch every sort order and common filter mask once before serving."""
        for sort_by in ('relevance', 'id', 'price', 'rating', 'review_count'):
            for order in ('asc', 'desc'):
                self.search({'sort_by': sort_by, 'sort_order': order,
                             'filters': {'color': 'Black', 'max_price': 2000}})
        if self.engine is not None and hasattr(self.engine, 'warm_up'):
            self.engine.warm_up()

    def search(self, arguments):
        args = self.prepare(arguments)
        return self.exact(args) if is_empty_query(args) else self.semantic(args)
