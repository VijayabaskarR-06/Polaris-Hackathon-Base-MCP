"""Catalog grounding checks for engine/lexicon.py.

Every catalog string in the lexicon must exist exactly in its field, every key
must already be in the matcher's canonical token form (engine.text.raw_tokens
joined by spaces), and the word lists must agree with what the catalog sells.
The catalog is loaded once; CATALOG_PATH overrides its location.
"""
import ast
import bisect
import json
import os
import unittest
from pathlib import Path

from engine import lexicon as L
from engine.text import raw_tokens, stem, tokenize

ROOT = Path(__file__).resolve().parent.parent
CATALOG_PATH = os.getenv('CATALOG_PATH', str(ROOT / 'data/products.json'))
LEXICON_PATH = ROOT / 'engine' / 'lexicon.py'
GROUPS = {'women', 'men', 'kids', 'girls', 'boys', 'baby'}
# catch-all colour codes no shopper word names
UNREACHABLE_COLORS = {'Other', 'Unknown'}

KEY_TABLES = ('TYPE_WORDS', 'TYPE_GROUP_WORDS', 'COLOR_WORDS', 'AUDIENCE_WORDS', 'SECTION_WORDS',
              'PATTERN_WORDS', 'OCCASIONS', 'MATERIAL_WORDS', 'STYLE_WORDS', 'PRICE_LOW_WORDS',
              'PRICE_HIGH_WORDS', 'RATING_WORDS', 'NEGATION_WORDS', 'STOPWORDS', 'GENERIC_WORDS',
              'FOREIGN_WORDS', 'NON_CATALOG_WORDS', 'BODY_PART_WORDS')


def canonical(text):
    """The matcher's form of a phrase: lowercase, compounds joined, [a-z0-9] words."""
    return ' '.join(raw_tokens(text))


def padded(text):
    """Stemmed, space-padded token form, for whole-word phrase search."""
    return ' ' + ' '.join(tokenize(text)) + ' '


def stemmed(phrase):
    return ' '.join(stem(t) for t in phrase.split())


class Corpus:
    """Tokenized texts joined into one string, with document counts by phrase."""

    def __init__(self, docs):
        self.items = list(docs)
        self.starts, offset = [], 0
        for doc in self.items:
            self.starts.append(offset)
            offset += len(doc) + 1
        self.text = '\n'.join(self.items)

    def count(self, phrase, limit=10**9):
        """Number of documents containing the padded phrase, stopping at limit."""
        found, pos, last = 0, 0, -1
        while found < limit:
            i = self.text.find(phrase, pos)
            if i < 0:
                break
            doc = bisect.bisect_right(self.starts, i) - 1
            if doc != last:
                found, last = found + 1, doc
            pos = i + 1
        return found

    def docs(self, phrase):
        """Indices of documents containing the padded phrase."""
        out, pos = [], 0
        while True:
            i = self.text.find(phrase, pos)
            if i < 0:
                return out
            doc = bisect.bisect_right(self.starts, i) - 1
            if not out or out[-1] != doc:
                out.append(doc)
            pos = i + 1


class LexiconTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        with open(CATALOG_PATH, encoding='utf-8') as f:
            products = json.load(f)
        fields = ('product_type', 'category', 'audience', 'section', 'color', 'pattern')
        cls.values = {f: {p[f] for p in products} for f in fields}
        cls.type_counts = {}
        for p in products:
            cls.type_counts[p['product_type']] = cls.type_counts.get(p['product_type'], 0) + 1
        texts = {p['name'] or '' for p in products} | {p['description'] or '' for p in products}
        tok = {t: padded(t) for t in texts}
        cls.names = Corpus([tok[p['name'] or ''] for p in products])
        cls.texts = Corpus([tok[p['name'] or ''] + tok[p['description'] or ''] for p in products])
        cls.known = set()
        for name in KEY_TABLES:
            if name not in ('FOREIGN_WORDS', 'NON_CATALOG_WORDS'):
                cls.known.update(getattr(L, name))
        cls.known_stemmed = {stemmed(k) for k in cls.known}

    # --- helpers ------------------------------------------------------------------

    def assert_values(self, where, values, field):
        self.assertIsInstance(values, list, where)
        self.assertTrue(values, f'{where}: empty list')
        self.assertEqual(len(values), len(set(values)), f'{where}: duplicate values')
        missing = [v for v in values if v not in self.values[field]]
        self.assertFalse(missing, f'{where}: not a catalog {field}: {missing}')

    def assert_occurs(self, where, words, minimum):
        counts = {w: self.texts.count(padded(w), minimum) for w in words}
        rare = sorted((w, n) for w, n in counts.items() if n < minimum)
        self.assertFalse(rare, f'{where}: keywords found in fewer than {minimum} products: {rare}')

    def is_known(self, phrase):
        return phrase in self.known or stemmed(phrase) in self.known_stemmed

    # --- keys -----------------------------------------------------------------------

    def test_keys_are_canonical(self):
        for name in KEY_TABLES:
            table = getattr(L, name)
            keys = table.keys() if isinstance(table, dict) else table
            for key in keys:
                self.assertIsInstance(key, str, name)
                self.assertEqual(key, key.strip().lower(), f'{name}: {key!r} not lowercase/stripped')
                self.assertTrue(key, f'{name}: empty key')
                self.assertEqual(canonical(key), key, f'{name}: {key!r} is not in canonical token form')

    def test_no_duplicate_literal_keys(self):
        """Python silently keeps the last of two equal dict keys; catch that in the source."""
        tree = ast.parse(LEXICON_PATH.read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if isinstance(node, ast.Dict):
                items = [k.value for k in node.keys if isinstance(k, ast.Constant)]
            elif isinstance(node, (ast.Set, ast.List)):
                items = [e.value for e in node.elts if isinstance(e, ast.Constant)]
            else:
                continue
            dupes = sorted({i for i in items if items.count(i) > 1})
            self.assertFalse(dupes, f'line {node.lineno}: duplicate literal items {dupes}')

    def test_word_lists_are_unique(self):
        for name in ('PRICE_LOW_WORDS', 'PRICE_HIGH_WORDS', 'RATING_WORDS', 'NEGATION_WORDS'):
            words = getattr(L, name)
            self.assertIsInstance(words, list, name)
            self.assertEqual(len(words), len(set(words)), name)
        for name in ('STOPWORDS', 'GENERIC_WORDS', 'NON_CATALOG_WORDS'):
            self.assertIsInstance(getattr(L, name), set, name)

    # --- product types ----------------------------------------------------------------

    def test_type_words_values(self):
        for key, value in L.TYPE_WORDS.items():
            self.assertIsInstance(value, tuple, key)
            types, keywords = value
            self.assert_values(f'TYPE_WORDS[{key!r}]', types, 'product_type')
            self.assertLessEqual(len(types), 6, f'TYPE_WORDS[{key!r}]: too many types')
            self.assertIsInstance(keywords, list, key)
            self.assertTrue(all(k == k.lower() for k in keywords), key)
        self.assert_occurs('TYPE_WORDS', {w for _, kws in L.TYPE_WORDS.values() for w in kws}, 1)

    def test_type_words_cover_every_product_type(self):
        for product_type in self.values['product_type']:
            key = canonical(product_type)
            self.assertIn(key, L.TYPE_WORDS, f'{product_type!r} has no TYPE_WORDS key {key!r}')
            self.assertIn(product_type, L.TYPE_WORDS[key][0], key)

    def test_type_group_words(self):
        for key, types in L.TYPE_GROUP_WORDS.items():
            self.assert_values(f'TYPE_GROUP_WORDS[{key!r}]', types, 'product_type')
        self.assertFalse(set(L.TYPE_GROUP_WORDS) & set(L.TYPE_WORDS), 'a key is both a type and a group')

    # --- colours, audience, sections, patterns -------------------------------------

    def test_color_words(self):
        for key, colors in L.COLOR_WORDS.items():
            self.assert_values(f'COLOR_WORDS[{key!r}]', colors, 'color')
        covered = {c for colors in L.COLOR_WORDS.values() for c in colors}
        self.assertEqual(self.values['color'] - covered, UNREACHABLE_COLORS)
        for color in self.values['color'] - UNREACHABLE_COLORS:
            if color.startswith('Other '):
                continue   # "other pink" reads as English; neon words reach these
            key = canonical(color)
            self.assertIn(key, L.COLOR_WORDS, color)
            self.assertEqual(L.COLOR_WORDS[key][0], color, f'{key!r} must list {color!r} first')

    def test_audience_words(self):
        for key, group in L.AUDIENCE_WORDS.items():
            self.assertIn(group, GROUPS, key)

    def test_section_words(self):
        for key, value in L.SECTION_WORDS.items():
            self.assertEqual(set(value), {'sections', 'group'}, key)
            self.assert_values(f'SECTION_WORDS[{key!r}]', value['sections'], 'section')
            self.assertIn(value['group'], GROUPS | {None}, key)
        self.assertEqual(L.SECTION_WORDS['maternity']['sections'], ['Mama'])
        self.assertEqual(L.SECTION_WORDS['plussize']['sections'], ['H&M+'])

    def test_pattern_words(self):
        for key, patterns in L.PATTERN_WORDS.items():
            self.assert_values(f'PATTERN_WORDS[{key!r}]', patterns, 'pattern')

    # --- occasions, materials, styles -------------------------------------------------

    def test_occasions(self):
        for key, value in L.OCCASIONS.items():
            where = f'OCCASIONS[{key!r}]'
            self.assertLessEqual({'types', 'keywords', 'sections', 'audience'}, set(value), where)
            self.assertLessEqual(set(value), {'types', 'keywords', 'sections', 'audience', 'colors', 'footwear'}, where)
            if 'footwear' in value:
                self.assert_values(where, value['footwear'], 'product_type')
            if 'colors' in value:
                self.assert_values(where, value['colors'], 'color')
            self.assert_values(where, value['types'], 'product_type')
            self.assertTrue(value['keywords'], f'{where}: no keywords')
            self.assertIsInstance(value['sections'], list, where)
            missing = [s for s in value['sections'] if s not in self.values['section']]
            self.assertFalse(missing, f'{where}: unknown sections {missing}')
            self.assertIn(value['audience'], self.values['audience'] | {None}, where)
        self.assert_occurs('OCCASIONS', {w for v in L.OCCASIONS.values() for w in v['keywords']}, 3)

    def test_material_and_style_words(self):
        for name in ('MATERIAL_WORDS', 'STYLE_WORDS'):
            table = getattr(L, name)
            for key, words in table.items():
                self.assertIsInstance(words, list, key)
                self.assertTrue(words, f'{name}[{key!r}]: empty list')
            self.assert_occurs(name, {w for words in table.values() for w in words}, 3)

    # --- filler, foreign words, non-catalog words ------------------------------------

    def test_stopwords_do_not_hide_meaningful_words(self):
        single = set()
        for name in KEY_TABLES:
            if name != 'STOPWORDS':
                single.update(k for k in getattr(L, name) if ' ' not in k)
        clash = sorted(L.STOPWORDS & single)
        self.assertFalse(clash, f'stopwords that other tables use: {clash}')

    def test_generic_words_are_not_types(self):
        clash = L.GENERIC_WORDS & (set(L.TYPE_WORDS) | set(L.TYPE_GROUP_WORDS))
        self.assertFalse(clash, clash)

    def test_foreign_words(self):
        english = self.known - L.GENERIC_WORDS
        clash = sorted(set(L.FOREIGN_WORDS) & english)
        self.assertFalse(clash, f'foreign keys that are already English table keys: {clash}')
        for key, value in L.FOREIGN_WORDS.items():
            self.assertEqual(canonical(value), value, key)
            ok = self.is_known(value) or all(self.is_known(t) for t in value.split())
            self.assertTrue(ok, f'FOREIGN_WORDS[{key!r}] = {value!r} is not understood by the other tables')

    def test_non_catalog_words(self):
        others = set()
        for name in KEY_TABLES:
            if name != 'NON_CATALOG_WORDS':
                others.update(getattr(L, name))
        clash = sorted(L.NON_CATALOG_WORDS & others)
        self.assertFalse(clash, f'non-catalog words that other tables use: {clash}')
        type_phrases = [k for k in L.TYPE_WORDS if ' ' in k]
        for word in L.NON_CATALOG_WORDS:
            # A name such as "laptop case" sells a case, not a laptop: products
            # whose name contains a multi-word type key with this word are covered.
            covers = [padded(k) for k in type_phrases if padded(word) in padded(k)]
            names = [self.names.items[d] for d in self.names.docs(padded(word))]
            hits = [n for n in names if not any(c in n for c in covers)]
            self.assertLess(len(hits), 5, f'{word!r} names {len(hits)} catalog products: {hits[:5]}')

    # --- complements -------------------------------------------------------------------

    def test_complement_map(self):
        for anchor, complements in L.COMPLEMENT_MAP.items():
            self.assertIn(anchor, self.values['product_type'], anchor)
            self.assert_values(f'COMPLEMENT_MAP[{anchor!r}]', complements, 'product_type')
            self.assertNotIn(anchor, complements, f'{anchor} complements itself')
        common = {t for t, n in self.type_counts.items() if n >= 50}
        self.assertFalse(common - set(L.COMPLEMENT_MAP), 'types with 50+ products need complements')

    def test_complement_keywords(self):
        targets = {t for complements in L.COMPLEMENT_MAP.values() for t in complements}
        self.assertFalse(targets - set(L.COMPLEMENT_KEYWORDS), 'complement types without keywords')
        for product_type, words in L.COMPLEMENT_KEYWORDS.items():
            self.assertIn(product_type, self.values['product_type'], product_type)
            self.assertTrue(words, product_type)
        self.assert_occurs('COMPLEMENT_KEYWORDS', {w for ws in L.COMPLEMENT_KEYWORDS.values() for w in ws}, 3)


if __name__ == '__main__':
    unittest.main()
