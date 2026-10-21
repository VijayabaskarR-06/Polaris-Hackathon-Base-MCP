"""Untouched starter search (schema path fixed): the oracle for exact empty-query pages."""
import json
import math
import re
from functools import lru_cache
from pathlib import Path

from jsonschema import Draft202012Validator, ValidationError

ROOT = Path(__file__).resolve().parent.parent
INPUT_SCHEMA = json.loads((ROOT / 'schemas/input.schema.json').read_text())
OUTPUT_SCHEMA = json.loads((ROOT / 'schemas/output.schema.json').read_text())
VALIDATOR = Draft202012Validator(INPUT_SCHEMA)
STRING_FILTERS = ('category', 'product_type', 'color', 'audience', 'section', 'pattern')


def validate_arguments(arguments):
    # Reject non-JSON floating point values even when called directly from Python.
    json.dumps(arguments, allow_nan=False)
    try:
        VALIDATOR.validate(arguments)
    except ValidationError as exc:
        raise ValueError(exc.message) from exc
    filters = arguments.get('filters', {})
    if any(not filters[k].strip() for k in STRING_FILTERS if k in filters):
        raise ValueError('String filters cannot be blank')
    if filters.get('min_price', 0) > filters.get('max_price', math.inf):
        raise ValueError('min_price cannot exceed max_price')


@lru_cache(maxsize=2048)
def tokens(text):
    return tuple(re.findall(r'\w+', text.casefold()))


class Catalog:
    def __init__(self, products):
        self.products = sorted(products, key=lambda p: p['id'])
        self.words = {
            p['id']: set(tokens(' '.join(str(p.get(k) or '') for k in
                ('name', 'description', *STRING_FILTERS)))) for p in self.products
        }

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text(encoding='utf-8')))

    def relevance(self, product, query_tokens):
        """Student extension: replace this strict lexical baseline with better retrieval."""
        words = self.words[product['id']]
        if not all(word in words for word in query_tokens):
            return 0
        name_words = set(tokens(product['name']))
        return 1 + sum(word in name_words for word in query_tokens)

    def recommend(self, results, arguments):
        """Student extension: return up to five complementary product references."""
        return []

    @staticmethod
    def matches(product, filters):
        for key in STRING_FILTERS:
            if key in filters and str(product.get(key) or '').strip().casefold() != filters[key].strip().casefold():
                return False
        price = product['price']['amount']
        average = product['rating']['average']
        return (price >= filters.get('min_price', 0)
                and price <= filters.get('max_price', math.inf)
                and ('min_rating' not in filters or
                     (average is not None and average >= filters['min_rating']))
                and product['rating']['count'] >= filters.get('min_review_count', 0))

    def search(self, arguments):
        validate_arguments(arguments)
        query = tokens(arguments.get('query', '').strip())
        filters = arguments.get('filters', {})
        candidates = [p for p in self.products if self.matches(p, filters)]
        scores = {}
        if query:
            scores = {p['id']: self.relevance(p, query) for p in candidates}
            candidates = [p for p in candidates if scores[p['id']] > 0]
        sort_by = arguments.get('sort_by', 'relevance')
        reverse = arguments.get('sort_order', 'desc') == 'desc'
        # Stable sort preserves ascending ID for equal keys, in either direction.
        if sort_by == 'relevance':
            if query:
                candidates.sort(key=lambda p: scores[p['id']], reverse=reverse)
        elif sort_by == 'rating':
            rated = [p for p in candidates if p['rating']['average'] is not None]
            unrated = [p for p in candidates if p['rating']['average'] is None]
            rated.sort(key=lambda p: p['rating']['average'], reverse=reverse)
            candidates = rated + unrated
        else:
            key = {'id': lambda p: p['id'], 'price': lambda p: p['price']['amount'],
                   'review_count': lambda p: p['rating']['count']}[sort_by]
            candidates.sort(key=key, reverse=reverse)
        offset, limit = arguments.get('offset', 0), arguments.get('limit', 10)
        results = [{'id': p['id']} for p in candidates[offset:offset + limit]]
        return {'results': results, 'recommendations':
                self.recommend(results, arguments) if results and arguments.get('include_recommendations', False) else []}
