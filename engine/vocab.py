"""Catalog vocabulary and shopper-language matching, built once at startup.

Combines the catalog's own values (exact spellings, price bands per type,
audience-group masks) with the static shopper lexicon (engine/lexicon.py)
into a phrase matcher the rule planner scans queries with.
"""
import re
from collections import defaultdict

import numpy as np

from engine import lexicon
from engine.index import STRING_FIELDS, fold
from engine.text import fold_accents, raw_tokens, stem, tokenize

WORD_RE = re.compile(r'[a-z]+')

GROUPS = ('women', 'men', 'kids', 'girls', 'boys', 'baby')
# Function words of common shopping languages, mapped to the English words the planner
# understands ("y" splits choices, "sin" negates, "para" is just "for").
FOREIGN_FUNCTION_WORDS = {
    'para': 'for', 'por': 'for', 'pour': 'for', 'fur': 'for',
    'con': 'with', 'avec': 'with', 'mit': 'with', 'com': 'with',
    'sin': 'without', 'sans': 'without', 'ohne': 'without', 'senza': 'without', 'sem': 'without',
    'y': 'and', 'et': 'and', 'und': 'and',
    'ou': 'or', 'oder': 'or',
    'de': 'of', 'del': 'of', 'des': 'of', 'du': 'of', 'di': 'of', 'da': 'of',
    'en': 'in', 'em': 'in', 'im': 'in', 'nel': 'in', 'al': 'to the', 'au': 'to the', 'aux': 'to the', 'zu': 'to',
    'el': 'the', 'la': 'the', 'los': 'the', 'las': 'the', 'le': 'the', 'les': 'the', 'der': 'the',
    'die': 'the', 'das': 'the', 'il': 'the',
    'un': 'a', 'una': 'a', 'une': 'a', 'ein': 'a', 'eine': 'a', 'uno': 'a', 'um': 'a', 'uma': 'a',
    'a juego': 'matching', 'assorti': 'matching', 'assortie': 'matching', 'passend': 'matching',
    'da abbinare': 'matching', 'abbinato': 'matching', 'abbinata': 'matching', 'combinando': 'matching',
    'menos de': 'under', 'moins de': 'under', 'unter': 'under', 'sotto': 'under', 'abaixo de': 'under',
    'mas de': 'over', 'plus de': 'over', 'uber': 'over', 'sopra': 'over',
    # context words of common shopping languages that the lexicon lacks
    'fiesta': 'party', 'fete': 'party', 'soiree': 'evening party', 'feier': 'party', 'festa': 'party',
    'noche': 'evening', 'nuit': 'evening', 'abend': 'evening', 'sera': 'evening', 'noite': 'evening',
    'boda': 'wedding', 'mariage': 'wedding', 'hochzeit': 'wedding', 'matrimonio': 'wedding', 'casamento': 'wedding',
    'playa': 'beach', 'plage': 'beach', 'strand': 'beach', 'spiaggia': 'beach', 'praia': 'beach',
    'trabajo': 'office', 'oficina': 'office', 'travail': 'office', 'bureau': 'office', 'buro': 'office',
    'arbeit': 'office', 'lavoro': 'office', 'ufficio': 'office', 'trabalho': 'office', 'escritorio': 'office',
    'invierno': 'winter', 'hiver': 'winter', 'winter': 'winter', 'inverno': 'winter',
    'verano': 'summer', 'ete': 'summer', 'sommer': 'summer', 'estate': 'summer', 'verao': 'summer',
    'lana': 'wool', 'laine': 'wool', 'wolle': 'wool', 'algodon': 'cotton', 'coton': 'cotton', 'baumwolle': 'cotton',
    'cotone': 'cotton', 'algodao': 'cotton', 'lino': 'linen', 'lin': 'linen', 'leinen': 'linen', 'linho': 'linen',
    'seda': 'silk', 'soie': 'silk', 'seide': 'silk', 'seta': 'silk', 'cuero': 'leather', 'cuir': 'leather',
    'leder': 'leather', 'pelle': 'leather', 'couro': 'leather',
    'caliente': 'warm', 'calido': 'warm', 'chaud': 'warm', 'chaude': 'warm', 'warm': 'warm', 'caldo': 'warm',
    'quente': 'warm', 'frio': 'cold', 'froid': 'cold', 'kalt': 'cold', 'freddo': 'cold',
    'barato': 'cheap', 'barata': 'cheap', 'pas cher': 'cheap', 'billig': 'cheap', 'economico': 'cheap',
    'regalo': 'gift', 'cadeau': 'gift', 'geschenk': 'gift', 'presente': 'gift',
    'nina': 'girl', 'nino': 'boy', 'fille': 'girl', 'garcon': 'boy', 'madchen': 'girl', 'junge': 'boy',
    'bebe': 'baby', 'hombre': 'men', 'homme': 'men', 'herren': 'men', 'uomo': 'men', 'homem': 'men',
    'mujer': 'women', 'femme': 'women', 'damen': 'women', 'donna': 'women', 'mulher': 'women',
}
# Common shopping words in Devanagari (Hindi), applied before accents are folded away.
DEVANAGARI_WORDS = {
    'लड़कों': 'boys', 'लड़को': 'boys', 'लड़के': 'boys', 'लड़का': 'boy', 'बेटे': 'son', 'बेटा': 'son',
    'लड़कियों': 'girls', 'लड़कियो': 'girls', 'लड़की': 'girl', 'बेटी': 'daughter',
    'बच्चों': 'kids', 'बच्चे': 'kids', 'बच्चा': 'kid', 'शिशु': 'baby',
    'महिलाओं': 'women', 'महिला': 'women', 'औरतों': 'women', 'औरत': 'women', 'स्त्री': 'women',
    'पुरुषों': 'men', 'पुरुष': 'men', 'आदमी': 'men', 'मर्द': 'men',
    'के लिए': 'for', 'के लिये': 'for', 'लिए': 'for', 'और': 'and', 'या': 'or', 'के साथ': 'with',
    'नीली': 'blue', 'नीला': 'blue', 'नीले': 'blue', 'लाल': 'red', 'काली': 'black', 'काला': 'black',
    'काले': 'black', 'सफेद': 'white', 'सफ़ेद': 'white', 'हरी': 'green', 'हरा': 'green', 'हरे': 'green',
    'पीली': 'yellow', 'पीला': 'yellow', 'गुलाबी': 'pink', 'भूरा': 'brown', 'भूरी': 'brown', 'स्लेटी': 'grey',
    'जींस': 'jeans', 'कमीज़': 'shirt', 'कमीज': 'shirt', 'शर्ट': 'shirt', 'टी-शर्ट': 't-shirt',
    'टीशर्ट': 't-shirt', 'पैंट': 'trousers', 'पतलून': 'trousers', 'जूते': 'shoes', 'जूता': 'shoes',
    'चप्पल': 'sandals', 'साड़ी': 'saree', 'कुर्ता': 'kurta', 'कुर्ती': 'kurti', 'ड्रेस': 'dress',
    'पोशाक': 'dress', 'जैकेट': 'jacket', 'स्वेटर': 'sweater', 'टोपी': 'hat', 'बैग': 'bag', 'थैला': 'bag',
    'घड़ी': 'watch', 'चश्मा': 'sunglasses', 'मोज़े': 'socks', 'मोजे': 'socks', 'दुपट्टा': 'scarf',
    'सस्ता': 'cheap', 'सस्ती': 'cheap', 'सस्ते': 'cheap', 'से कम': 'under', 'से ज़्यादा': 'over',
    'रुपये': 'inr', 'रुपए': 'inr', 'रु': 'inr', 'शादी': 'wedding', 'पार्टी': 'party', 'सर्दी': 'winter',
    'गर्मी': 'summer', 'बारिश': 'rain', 'ऑफिस': 'office',
}
DEVANAGARI_DIGITS = str.maketrans('०१२३४५६७८९', '0123456789')

# Inflected forms ("rojos", "noires", "rote") are looked up by trying these endings.
FOREIGN_ENDINGS = ('s', 'es', 'e', 'en', 'n', 'a', 'as', 'os', 'er')
KIDS_AUDIENCES = ('Children Sizes 92-140', 'Children Sizes 134-170',
                  'Children Accessories, Swimwear', 'Baby Sizes 50-98')
WOMEN_AUDIENCES = ('Ladieswear', 'Ladies Accessories', 'Lingeries/Tights', 'Divided')


def table(name):
    return getattr(lexicon, name, None) or {}


class PhraseMatcher:
    """Greedy longest-match of lexicon phrases over a token sequence."""

    def __init__(self):
        self.entries = defaultdict(list)
        self.max_len = 1

    def add(self, phrase, kind, payload=None):
        key = tuple(tokenize(phrase))
        if not key:
            return
        entry = (kind, payload)
        if entry not in self.entries[key]:
            self.entries[key].append(entry)
        self.max_len = max(self.max_len, len(key))

    def kinds(self, phrase):
        return {kind for kind, _ in self.entries.get(tuple(tokenize(phrase)), [])}

    def scan(self, tokens):
        """[(start, end, entries)] covering matched spans; unmatched tokens get entries == []."""
        spans, i, n = [], 0, len(tokens)
        while i < n:
            for length in range(min(self.max_len, n - i), 0, -1):
                entries = self.entries.get(tuple(tokens[i:i + length]))
                if entries:
                    spans.append((i, i + length, entries))
                    i += length
                    break
            else:
                spans.append((i, i + 1, []))
                i += 1
        return spans


def edit_distance(a, b, limit=2):
    """Damerau-Levenshtein (optimal string alignment) distance, early exit above limit."""
    if abs(len(a) - len(b)) > limit:
        return limit + 1
    prev2, prev = None, list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        best = cur[0]
        for j in range(1, len(b) + 1):
            cost = a[i - 1] != b[j - 1]
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                cur[j] = min(cur[j], prev2[j - 2] + 1)
            best = min(best, cur[j])
        if best > limit:
            return limit + 1
        prev2, prev = prev, cur
    return prev[-1]


class SpellFixer:
    """Symmetric-delete typo correction toward known catalog and lexicon terms.

    Terms are stems; their plural surface forms are indexed too, so the raw
    query word ("jeens") is matched against "jeans" before stemming.
    """

    def __init__(self, term_freq, protected):
        self.freq = term_freq
        self.protected = protected      # lexicon terms win ties: they are shopper words
        self.surface = {}               # surface form -> stem
        for term in term_freq:
            for form in (term, term + 's'):
                self.surface.setdefault(form, term)
        self.deletes = defaultdict(list)
        for form in self.surface:
            if len(form) >= 3:
                variants = self._deletes(form)
                if self.surface[form] in protected and len(form) >= 6:
                    # Two deletions for shopper words only ("dauter" -> daughter).
                    variants = variants | {v2 for v in variants for v2 in self._deletes(v)}
                for variant in variants:
                    self.deletes[variant].append(form)

    @staticmethod
    def _deletes(term):
        return {term} | {term[:i] + term[i + 1:] for i in range(len(term))}

    def correct(self, token):
        """The stem of the best known word within reach of a typo, or None.

        4 letters: only a missing letter ("blak" -> black); short real words such
        as "bake" are too close to others for substitutions. 5-6 letters: one
        edit. 7 or more: two edits.
        """
        if len(token) < 4 or token.isdigit():
            return None
        limit = 1 if len(token) <= 5 else 2
        found = set()
        for variant in self._deletes(token):
            found.update(self.deletes.get(variant, ()))
        best = None
        for form in found:
            if form[0] != token[0]:
                continue          # typos almost never change the first letter ("bake" is not "fake")
            if len(token) == 4 and not (len(form) == 5 and token in self._deletes(form)
                                        and self.surface[form] in self.protected):
                continue
            d = edit_distance(token, form, limit)
            if d > limit:
                continue
            term = self.surface[form]
            if d == 2 and not (term in self.protected or self.freq.get(term, 0) >= 200):
                continue
            # A shopper-lexicon word one edit further beats a rare catalog token
            # ("swimsute" -> swimsuit, not the catalog's misspelling "swimsuite").
            rank = (d + (term not in self.protected), term not in self.protected, -self.freq.get(term, 0), term)
            if best is None or rank < best[0]:
                best = (rank, term)
        return best[1] if best else None


class Vocab:
    def __init__(self, products, index, bm25):
        self.index = index
        self.bm25 = bm25
        self.values = {f: list(index.values[f]) for f in STRING_FIELDS}
        self.canonical = {f: {fold(v): v for v in reversed(self.values[f])} for f in STRING_FIELDS}
        self._price_bands(products)
        self._group_masks()
        self.matcher = PhraseMatcher()
        self._load_lexicon()
        self._spelling()
        self.complement_map = {t: [c for c in comps if c in self.type_set]
                               for t, comps in table('COMPLEMENT_MAP').items() if t in self.type_set}
        self.complement_keywords = table('COMPLEMENT_KEYWORDS')

    # ----- catalog facts -------------------------------------------------
    def _price_bands(self, products):
        prices = defaultdict(list)
        for p in products:
            prices[p['product_type']].append(p['price']['amount'])
        self.type_band = {t: (min(v), float(np.median(v)), max(v)) for t, v in prices.items()}
        self.type_set = set(self.values['product_type'])
        counts = defaultdict(lambda: defaultdict(int))
        for p in products:
            counts[p['product_type']][p['category']] += 1
        self.type_category = {t: max(c, key=c.get) for t, c in counts.items()}

    def _group_masks(self):
        idx = self.index
        audience = idx.codes['audience']
        aud_values = idx.values['audience']
        sec_values = idx.values['section']
        section = idx.codes['section']

        def audiences(names):
            return np.isin(audience, [aud_values.index(a) for a in names if a in aud_values])

        def sections(pred):
            return np.isin(section, [i for i, s in enumerate(sec_values) if pred(s)])

        kids = audiences(KIDS_AUDIENCES) | sections(lambda s: s == 'Kids Sports')
        self.group_masks = {
            'women': audiences(WOMEN_AUDIENCES) | sections(lambda s: s == 'Ladies H&M Sport'),
            'men': audiences(('Menswear',)) | sections(lambda s: s == 'Men H&M Sport'),
            'kids': kids,
            'girls': kids & ~sections(lambda s: 'Boy' in s),
            'boys': kids & ~sections(lambda s: 'Girl' in s),
            'baby': audiences(('Baby Sizes 50-98',)) | sections(lambda s: s.startswith('Baby')),
        }
        for mask in self.group_masks.values():
            mask.flags.writeable = False

    def categories_of(self, types):
        return {self.type_category[t] for t in types if t in self.type_category}

    def group_of(self, pos):
        """The broad audience family of one product (for recommendation audience matching)."""
        for group in ('baby', 'kids', 'men', 'women'):
            if self.group_masks[group][pos]:
                return group
        return None

    # ----- lexicon -------------------------------------------------------
    def exact(self, field, value):
        return self.canonical[field].get(fold(value))

    def types(self, names):
        out = []
        for name in names:
            # "dog wear" folds to two spellings; keep every catalog spelling.
            for value in self.values['product_type']:
                if fold(value) == fold(name) and value not in out:
                    out.append(value)
        return out

    def _load_lexicon(self):
        m = self.matcher
        for value in self.values['product_type']:
            m.add(value.lower().replace('/', ' '), 'type', (self.types([value]), []))
        for word, (types, keywords) in table('TYPE_WORDS').items():
            types = self.types(types)
            if types or keywords:
                m.add(word, 'type', (types, list(keywords)))
        for word, types in table('TYPE_GROUP_WORDS').items():
            types = self.types(types)
            if types:
                m.add(word, 'type_group', (types, []))
        for value in self.values['color']:
            if value not in ('Unknown', 'Other'):
                m.add(value.lower(), 'color', [value])
        for word, colors in table('COLOR_WORDS').items():
            colors = [c for c in (self.exact('color', c) for c in colors) if c]
            if colors:
                m.add(word, 'color', colors)
        for word, group in table('AUDIENCE_WORDS').items():
            if group in GROUPS:
                m.add(word, 'audience', group)
        for word, info in table('SECTION_WORDS').items():
            sections = [s for s in (self.exact('section', s) for s in info.get('sections', [])) if s]
            m.add(word, 'section', {'sections': sections, 'group': info.get('group')})
        for word, patterns in table('PATTERN_WORDS').items():
            patterns = [p for p in (self.exact('pattern', p) for p in patterns) if p]
            if patterns:
                m.add(word, 'pattern', patterns)
        for word, info in table('OCCASIONS').items():
            m.add(word, 'occasion', {
                'types': self.types(info.get('types', [])),
                'keywords': list(info.get('keywords', [])),
                'sections': [s for s in (self.exact('section', s) for s in info.get('sections', [])) if s],
                'audience': self.exact('audience', info['audience']) if info.get('audience') else None,
            })
        for word, (types, keywords) in table('BODY_PART_WORDS').items():
            m.add(word, 'hint', {'types': self.types(types), 'keywords': list(keywords), 'sections': [],
                                 'audience': None})
        for word, words in table('MATERIAL_WORDS').items():
            m.add(word, 'material', list(words))
        for word, words in table('STYLE_WORDS').items():
            m.add(word, 'style', list(words))
        for word in table('PRICE_LOW_WORDS'):
            m.add(word, 'price_low')
        for word in table('PRICE_HIGH_WORDS'):
            m.add(word, 'price_high')
        for word in table('RATING_WORDS'):
            m.add(word, 'rating')
        for word in table('NEGATION_WORDS'):
            m.add(word, 'negation')
        for word in table('NON_CATALOG_WORDS'):
            m.add(word, 'non_catalog')
        for word in table('GENERIC_WORDS'):
            m.add(word, 'generic')
        self.stopwords = {stem(w) for w in table('STOPWORDS') for w in raw_tokens(w)}
        self.foreign = {}
        for word, english in list(table('FOREIGN_WORDS').items()) + list(FOREIGN_FUNCTION_WORDS.items()):
            key = ' '.join(fold_accents(word).lower().split())
            if key and key not in self.foreign:
                self.foreign[key] = english
        self.foreign_re = None
        if self.foreign:
            keys = sorted(self.foreign, key=len, reverse=True)
            self.foreign_re = re.compile(r'\b(?:' + '|'.join(r'[\s\-]+'.join(map(re.escape, k.split())) for k in keys)
                                         + r')\b')

    @staticmethod
    def native_to_english(query):
        """Devanagari shopping words -> English, longest phrases first; other text unchanged."""
        if not any('\u0900' <= ch <= '\u097f' for ch in query):
            return query
        query = query.translate(DEVANAGARI_DIGITS)
        for word in sorted(DEVANAGARI_WORDS, key=len, reverse=True):
            query = query.replace(word, f' {DEVANAGARI_WORDS[word]} ')
        return query

    def translate_text(self, text):
        """Replace known non-English words/phrases (whole words) with English shopper words.

        Works on the normalized query string and keeps punctuation, which the
        planner needs for splitting mixed choices.
        """
        if self.foreign_re is None:
            return text
        text = self.foreign_re.sub(lambda m: self.foreign[' '.join(re.split(r'[\s\-]+', m.group(0)))], text)
        return WORD_RE.sub(self._inflected, text)

    def _inflected(self, match):
        """An inflected foreign word ("rojos", "noires") whose base form is known; English words stay."""
        word = match.group(0)
        if len(word) < 4 or stem(word) in self.lexicon_terms or self.bm25.df(stem(word)) >= 5:
            return word
        for ending in FOREIGN_ENDINGS:
            if word.endswith(ending):
                base = word[:-len(ending)]
                for candidate in (base, base + 'o', base + 'e'):
                    english = self.foreign.get(candidate)
                    if english is not None and len(candidate) >= 3:
                        return english
        return word

    def _spelling(self):
        freq = {t: df for t, df in self.bm25.text.df.items() if df >= 3 and not t.isdigit()}
        self.lexicon_terms, positive = set(), set()
        for key, entries in self.matcher.entries.items():
            sells = any(kind != 'non_catalog' for kind, _ in entries)
            for term in key:
                self.lexicon_terms.add(term)
                if sells:
                    positive.add(term)
                freq.setdefault(term, 1000 if sells else 50)
        # Only words for things we sell get the tie-break bonus: "sandles" must
        # become sandal, not the non-catalog word "candle".
        self.spell = SpellFixer(freq, positive)

    def correction(self, raw, term):
        """Spell-corrected stem for a query word, or None to keep it.

        Shopper-lexicon words and stopwords are never touched. Unknown or very
        rare words take the best nearby known word. A known but uncommon
        catalog token ("dres" stems to "dre", a product name) yields only to a
        lexicon word that is far more common ("dress").
        """
        if term in self.lexicon_terms or term in self.stopwords or term.isdigit():
            return None
        df = self.bm25.df(term)
        fixed = self.spell.correct(raw)
        if fixed is None or fixed == term:
            return None
        if df < 5:
            # A typo is a slip on a common word: never "correct" toward an obscure name token.
            if fixed in self.spell.protected or self.bm25.df(fixed) >= 20:
                return fixed
            return None
        if fixed in self.spell.protected and self.spell.freq.get(fixed, 0) >= 50 * df:
            return fixed
        return None
