"""Catalog search: the exact contract engine for empty queries, the semantic engine otherwise."""
import json
import logging
import math
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
    for bound in ('min_price', 'max_price'):
        if bound in filters:
            # A valid bound can be a JSON integer too large for a float: it means +inf.
            try:
                filters[bound] = float(filters[bound])
            except OverflowError:
                filters[bound] = math.inf
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


class Catalog:
    def __init__(self, products, semantic=True, llm=None):
        self.products = sorted(products, key=lambda p: p['id'])
        for product in self.products:
            for field in UNUSED_FIELDS:
                product.pop(field, None)
        self.index = Index(self.products)
        self.engine = None
        self.semantic_status = 'off'
        if semantic:
            # The exact path (validity + performance) must survive any semantic-engine failure.
            try:
                from engine.engine import SemanticEngine
                if llm is None:
                    from engine.llm import make_llm
                    llm = make_llm()
                self.engine = SemanticEngine(self.products, self.index, llm=llm)
                self.semantic_status = 'ok'
            except Exception:
                logging.getLogger('polaris.search').exception('semantic engine failed to start; degraded mode')
                self.semantic_status = 'degraded'

    @classmethod
    def load(cls, path, semantic=True, llm=None):
        return cls(json.loads(Path(path).read_text(encoding='utf-8')), semantic=semantic, llm=llm)

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
        if self.engine is None:
            return {'results': [], 'recommendations': []}
        return self.engine.search(args)

    def warm_up(self):
        """Touch every sort order and common filter mask once before serving."""
        for sort_by in ('relevance', 'id', 'price', 'rating', 'review_count'):
            for order in ('asc', 'desc'):
                self.search({'sort_by': sort_by, 'sort_order': order,
                             'filters': {'color': 'Black', 'max_price': 2000}})
        if self.engine is not None and hasattr(self.engine, 'warm_up'):
            try:
                self.engine.warm_up()
            except Exception:
                logging.getLogger('polaris.search').exception('semantic warm-up failed')

    def search(self, arguments):
        args = self.prepare(arguments)
        return self.exact(args) if is_empty_query(args) else self.semantic(args)
