"""Rule planner: instant, deterministic query understanding with no LLM.

It is the fallback when the LLM planner is unavailable or late, the racing
path while it runs, and the safety net for budgets and colours it misses.
"""
import re
from dataclasses import dataclass, field

from engine.plan import Choice, Plan
from engine.text import normalize_query, raw_tokens, stem, tokenize

NUM = r'(\d+(?:\.\d+)?)'
CUR = r'(?:inr\s*)?'
TAIL = r'(?:\s*inr)?(?:\s*/-)?'
MAX_RES = [re.compile(p) for p in (
    rf'\b(?:under|below|beneath|less than|lesser than|lower than|up ?to|upto|till|within|max(?:imum)?'
    rf'|not more than|no more than|not over|cheaper than|at most|not exceeding|no higher than|for less than'
    rf'|for under|budget(?:\s+(?:of|is|around|upto|up to))?|in (?:a |the |my )?budget of|a budget of|spend(?:ing)?'
    rf'(?:\s+(?:of|upto|up to|max))?)\s*(?:of\s*)?(?:is\s*)?(?::\s*)?{CUR}{NUM}{TAIL}',
    rf'(?:<=?|≤)\s*{CUR}{NUM}{TAIL}',
    rf'{CUR}{NUM}{TAIL}\s*(?:or less|or under|or below|or cheaper|or lower|and under|and below|max(?:imum)?\b'
    rf'|budget|tak\b|se kam|se neeche|se niche|ke andar|ke under|ke neeche|ke niche|se sasta|ke budget'
    rf'|se (?:zyada|jyada|jada|upar|oopar) (?:nahi|not)\b)',
)]
MIN_RES = [re.compile(p) for p in (
    rf'\b(?:over|above|more than|greater than|at least|atleast|min(?:imum)?|starting (?:at|from)|upwards of'
    rf'|not less than|no less than|costlier than|pricier than)\s*{CUR}{NUM}{TAIL}',
    rf'(?:>=?|≥)\s*{CUR}{NUM}{TAIL}',
    rf'{CUR}{NUM}{TAIL}\s*(?:or more|or above|and above|and up\b|plus\b|\+|se zyada|se jyada|se upar|se oopar|ke upar)',
)]
RANGE_RE = re.compile(rf'(?:\bbetween\s*|\bfrom\s*|\bin the range (?:of\s*)?)?{CUR}{NUM}{TAIL}\s*(?:-|–|to|and|se)\s*{CUR}{NUM}{TAIL}')
AROUND_RE = re.compile(rf'\b(?:around|about|approximately|approx|roughly|close to|near|nearly|~)\s*{CUR}{NUM}{TAIL}')
BARE_RE = re.compile(rf'(?:\b(?:for|at|@|priced at|costing|cost|price)\s*{CUR}{NUM}{TAIL})|(?:inr\s*{NUM}{TAIL})|(?:{NUM}\s*inr)')
EACH_RE = re.compile(r'\b(?:each|per item|per piece|apiece|a piece|har ek|each one|both|dono|all of them|all three)\b')

# "X with a matching Y": Y is a requested accompanying item (recommendations).
COMPLEMENT_RE = re.compile(
    r'\b(?:(?:with|and|plus|along with|together with)\s+(?:a |an |some |the |pair of |a pair of )?'
    r'(?:matching|coordinating|co-ordinating|complementary|complimentary|coordinated|co-ordinated|harmonising'
    r'|harmonizing)\b'
    r'|(?:with|and)\s+(?:a |an |some )?(?:(?:nice|cute|good|suitable) )?accessor(?:y|ies) (?:to match|for it|for that)'
    r'|accessori[sz]e (?:it )?with|complete (?:the|my) look with)')
# "a bag to match": the complement follows "with" and precedes "to match".
WITH_TO_MATCH_RE = re.compile(r'\bwith\s+(?:a |an |some )?(.+?)\s+to match\b')
# "X to go with my Y": the shopper wants X; Y is context they already own.
CONTEXT_RE = re.compile(
    r'\b(?:to go with|that goes? with|which goes? with|that would go with|goes? well with|to wear with|to pair with'
    r'|that pairs? with|to match|that matches|matching (?=my |our |her |his )|for (?=my |our )(?:\w+ ){0,2}'
    r'(?:dress|outfit|suit|saree|sari|lehenga|gown|jeans|jacket))')
SPLIT_RE = re.compile(r'\band\b|&|\bplus\b(?![\s-]*sized?\b)|\bas well as\b|,|/|;|\+|\bor\b|\balso\b|\baur\b'
                      r'|\bya\b|\bthen\b')
INJECTION_RE = re.compile(
    r'ignore (?:all |any |the |your |previous |prior |above |earlier )*(?:instructions|rules|prompts?|guidelines)'
    r'|system prompt|developer (?:message|mode)|jailbreak|you are now|act as (?:a|an|the)\b|pretend (?:to be|you)'
    r'|return (?:all|every|each) (?:the )?(?:products?|items?|ids?)|list (?:all|every) (?:products?|items?|ids?)'
    r'|dump (?:the )?(?:catalog|database)|reveal (?:your|the) (?:prompt|instructions)|disregard')
# Brands that the catalog does not carry; a request for one is not satisfiable from catalog facts.
BRANDS = {
    'nike', 'adidas', 'puma', 'reebok', 'gucci', 'prada', 'zara', 'levi', 'levis', 'uniqlo', 'mango', 'gap',
    'armani', 'versace', 'chanel', 'dior', 'balenciaga', 'louis', 'vuitton', 'hermes', 'burberry', 'calvin',
    'tommy', 'hilfiger', 'ralph', 'lauren', 'lacoste', 'fila', 'asics', 'skechers', 'vans', 'converse',
    'crocs', 'birkenstock', 'timberland', 'clarks', 'bata', 'woodland', 'allen', 'solly', 'biba', 'fabindia',
    'manyavar', 'sabyasachi', 'shein', 'forever21', 'superdry', 'jordan', 'yeezy', 'rolex', 'casio', 'fossil',
    'titan', 'ray', 'rayban', 'oakley', 'samsonite', 'wildcraft', 'decathlon', 'lululemon', 'gymshark',
    'underarmour', 'patagonia', 'northface', 'columbia', 'diesel', 'boss', 'hugo', 'fendi', 'valentino',
    'givenchy', 'kenzo', 'moncler', 'canada', 'goose', 'apple', 'samsung', 'sony', 'nokia', 'xiaomi', 'oneplus',
}
STYLE_TYPE_PHRASES = {
    ('boyfriend', 'jean'): 'boyfriend', ('mom', 'jean'): 'mom', ('boyfriend', 'shirt'): 'boyfriend',
    ('boyfriend', 'blazer'): 'boyfriend', ('mom', 'short'): 'mom', ('dad', 'sneaker'): 'dad',
    ('dad', 'shoe'): 'dad', ('dad', 'cap'): 'dad', ('girlfriend', 'jean'): 'girlfriend',
}
NUMERIC_NOISE_RE = re.compile(r'\b\d+(?:\.\d+)?\s*(?:%|percent|pcs?|pieces?|pack|pk|years?|yrs?|y/o|months?|'
                              r'cm|mm|inch(?:es)?|in\b|size|sizes|xl|xxl|kg|ml|l\b)')
MIN_BUDGET = 100
AGE_RE = re.compile(r'\b(\d{1,2})\s*(?:-|\s)?\s*(years?|yrs?|y/o|yo|months?|mths?)(?:\s*-?\s*old)?\b'
                    r'|\baged?\s*(\d{1,2})\b')
# Words too generic to tell a product type apart ("football shirt": football matters, shirt doesn't).
# Sections that are a shopper requirement when named ("plus size", "maternity"), not a mood.
REQUIRED_SECTIONS = {'H&M+', 'Mama'}
# Gifts for little ones: the adult gift list (wallets, watches) has nothing for them.
CHILD_GIFT_TYPES = ['Giftbox', 'Soft Toys', 'Bodysuit', 'Garment Set', 'Sleeping sack', 'Bootie', 'Hat/beanie',
                    'Socks', 'Pyjama set', 'Dress']
# Verbs of services and actions the shop does not sell ("book a flight", "order pizza", "track my
# order"). They only count when no product type is named ("order a black dress" is shopping).
SERVICE_VERBS = {'book', 'booking', 'reserve', 'reservation', 'rent', 'rental', 'hire', 'cancel', 'refund',
                 'deliver', 'delivery', 'download', 'install', 'subscribe', 'schedule', 'cook', 'bake', 'fix',
                 'repair', 'calculate', 'translate', 'convert', 'stream', 'invest', 'pay', 'order', 'track',
                 'recipe', 'weather', 'news', 'score', 'lyrics', 'directions', 'appointment'}
QUESTION_RE = re.compile(r"^\s*(?:how|what|why|when|where|who|which|whats|what's|is|are|do|does|did|can|could|"
                         r"will|would|should)\b")
PET_TYPES = {'Dog Wear', 'Dog wear'}
ADULT_CONTEXT_RE = re.compile(r'\b(?:at work|work|office|colleague|coworker|co-worker|college|uni|university|'
                              r'girlfriend|gf|wife|woman|women|lady|ladies|girls night|her birthday party|hostel)\b')
CHILD_CONTEXT_RE = re.compile(r'\b(?:daughter|niece|little|toddler|kid|kids|child|children|school|baby|year old|'
                              r'years old|yr old|granddaughter|beti)\b')
# Words whose singular is an adjective while the plural names a garment.
ADJECTIVE_FORMS = {'short'}
IDIOM_RE = re.compile(r"\b(?:over the top|on top of|top notch|top-notch|tip top|on top)\b")
BACK_REFERENCES = {'it', 'them', 'this', 'that', 'these', 'those', 'one', 'it too', 'them too'}
# "unicorn stuff", "cute things": the object is "clothing in general".
PLACEHOLDER_OBJECTS = {'stuff', 'thing', 'item', 'clothe', 'clothing', 'garment', 'apparel', 'gear', 'kit',
                       'solution', 'option', 'idea', 'suggestion', 'recommendation', 'choice'}
GENERIC_GARMENT_WORDS = {'shirt', 'top', 'tee', 'tshirt', 'wear', 'set', 'clothe', 'clothing', 'outfit', 'item',
                         'sport', 'sports', 'style', 'piece', 'garment', 'fashion', 'thing', 'stuff'}
PREPOSITIONS = {'for', 'with', 'to', 'in', 'on', 'at', 'from', 'during', 'like', 'without', 'by', 'near', 'into'}


@dataclass
class Budget:
    min_price: float = None
    max_price: float = None
    start: int = 0
    end: int = 0
    soft_target: float = None


def _value(text):
    return float(text)


def parse_budgets(text):
    """Budgets in a normalized query: [(Budget)], each with its char span. Spans never overlap."""
    taken = []
    found = []

    def free(span):
        return all(span[1] <= a or span[0] >= b for a, b in taken)

    def plausible(match, *values):
        has_currency = 'inr' in match.group(0)
        return all(has_currency or v >= MIN_BUDGET for v in values)

    for m in RANGE_RE.finditer(text):
        low, high = _value(m.group(1)), _value(m.group(2))
        if free(m.span()) and min(low, high) >= 50 and plausible(m, low, high) and low <= high:
            if re.search(r'\b(?:years?|yrs?|months?|y/o|cm|size)\b', text[m.end():m.end() + 8]):
                continue
            found.append(Budget(min_price=low, max_price=high, start=m.start(), end=m.end()))
            taken.append(m.span())
    for patterns, kind in ((MAX_RES, 'max'), (MIN_RES, 'min')):
        for pattern in patterns:
            for m in pattern.finditer(text):
                value = _value(m.group(1))
                if not free(m.span()) or not plausible(m, value):
                    continue
                if NUMERIC_NOISE_RE.match(text, m.start(1)):
                    continue
                budget = Budget(start=m.start(), end=m.end())
                setattr(budget, f'{kind}_price', value)
                found.append(budget)
                taken.append(m.span())
    for m in AROUND_RE.finditer(text):
        value = _value(m.group(1))
        if free(m.span()) and plausible(m, value):
            found.append(Budget(start=m.start(), end=m.end(), soft_target=value))
            taken.append(m.span())
    for m in BARE_RE.finditer(text):
        raw = next(g for g in m.groups() if g)
        value = _value(raw)
        if free(m.span()) and value >= MIN_BUDGET and not NUMERIC_NOISE_RE.match(text, m.start()):
            # "jeans for 2000" / "₹999 kurta": treat a bare price as a ceiling.
            found.append(Budget(max_price=value, start=m.start(), end=m.end()))
            taken.append(m.span())
    found.sort(key=lambda b: b.start)
    return found


@dataclass
class Segment:
    start: int
    end: int
    text: str
    types: list = field(default_factory=list)
    primary_types: list = field(default_factory=list)
    needs_keyword: bool = False
    type_words: list = field(default_factory=list)
    type_keywords: list = field(default_factory=list)
    group_types: list = field(default_factory=list)
    colors: list = field(default_factory=list)          # list of colour families (one per colour word)
    exclude_colors: list = field(default_factory=list)
    patterns: list = field(default_factory=list)
    exclude_patterns: list = field(default_factory=list)
    materials: list = field(default_factory=list)
    styles: list = field(default_factory=list)
    descriptors: list = field(default_factory=list)
    exclude_words: list = field(default_factory=list)
    occasions: list = field(default_factory=list)
    sections: list = field(default_factory=list)
    unknown: list = field(default_factory=list)
    brands: list = field(default_factory=list)
    non_catalog: list = field(default_factory=list)
    generic: bool = False
    budgets: list = field(default_factory=list)
    content: list = field(default_factory=list)     # (token position, kind) of content spans
    content_words: list = field(default_factory=list)
    hint_types: list = field(default_factory=list)  # "keep my hands warm" -> Gloves
    match_terms: list = field(default_factory=list) # words a product must say to be the named type
    audience: set = field(default_factory=set)      # audience groups named inside this segment
    required_sections: list = field(default_factory=list)  # plus size / maternity: stated requirements
    prep_at: int = None

    @property
    def has_type(self):
        return bool(self.types or self.group_types)


@dataclass
class Signals:
    """Query-wide facts the rule planner found (also used by the engine for abstention)."""
    audience: set = field(default_factory=set)
    sections: list = field(default_factory=list)
    price_low: bool = False
    price_high: bool = False
    rating: bool = False
    injection: bool = False
    content_tokens: int = 0
    unknown: list = field(default_factory=list)
    brands: list = field(default_factory=list)
    non_catalog: list = field(default_factory=list)
    corrected: dict = field(default_factory=dict)
    soft_target: float = None
    context: str = ''
    service: bool = False
    question: bool = False


def _mask_spans(text, spans):
    chars = list(text)
    for start, end in spans:
        for i in range(start, end):
            chars[i] = ' '
    return ''.join(chars)


class RulePlanner:
    def __init__(self, vocab):
        self.vocab = vocab
        self.m = vocab.matcher

    # ----- segment analysis ----------------------------------------------
    def analyze(self, seg, signals):
        """Fill a Segment from its text using the lexicon matcher."""
        tokens, raws = [], []
        for raw in raw_tokens(IDIOM_RE.sub(' ', seg.text)):
            if len(raw) < 2:
                continue
            raws.append(raw)
            t = stem(raw)
            fixed = self.vocab.correction(raw, t)
            if fixed:
                signals.corrected[raw] = fixed
                t = fixed
            tokens.append(t)
        spans = self.m.scan(tokens)
        modifiers = set()
        for i in range(len(spans) - 1):
            here, there = spans[i], spans[i + 1]
            if (any(k == 'type' for k, _ in here[2]) and here[1] == there[0]
                    and any(k in ('type', 'type_group') for k, _ in there[2])):
                modifiers.add(i)
        negate_until = -1
        prep_at = next((i for i, t in enumerate(tokens) if t in PREPOSITIONS), len(tokens))
        seg.prep_at = prep_at
        for idx, (start, end, entries) in enumerate(spans):
            kinds = {k for k, _ in entries}
            phrase = tokens[start:end]
            negated = start <= negate_until
            content_kind = self._content_kind(kinds, phrase, entries)
            if content_kind:
                seg.content.append((start, content_kind))
                seg.content_words.append((start, ' '.join(phrase)))
            late_type = (start > prep_at and (seg.types or seg.group_types)
                         and any(k in ('type', 'type_group') for k, _ in entries))
            # "too short", "a short dress": singular "short" is a length, only "shorts" is the garment.
            adjective = end - start == 1 and raws[start] in ADJECTIVE_FORMS and 'type' in kinds
            pet = late_type and any(t in PET_TYPES for k, payload in entries if k == 'type' for t in payload[0])
            if pet:
                # "a warm jumper for my dog": the wearer is a pet, so the item is pet clothing
                # and the human garment word describes it.
                seg.descriptors += [w for w in seg.type_words if w not in seg.descriptors]
                seg.types, seg.primary_types, seg.type_words = [], [], []
                seg.type_keywords, seg.match_terms, seg.needs_keyword = [], [], False
                late_type = False
            if idx in modifiers or late_type or adjective:
                # "shirt dress": the modifier noun describes the head item; "boots with a heel":
                # once the item is named, a type after a preposition describes it.
                seg.descriptors.append(' '.join(phrase))
                signals.content_tokens += 1
                continue
            if 'negation' in kinds and len(kinds) == 1:
                negate_until = end + 2
                continue
            if not entries:
                word = phrase[0]
                if word in self.vocab.stopwords or word.isdigit():
                    continue
                if word in BRANDS:
                    seg.brands.append(word)
                    continue
                signals.content_tokens += 1
                if negated:
                    seg.exclude_words.append(word)
                elif self.vocab.bm25.known(word):
                    seg.descriptors.append(word)
                else:
                    seg.unknown.append(word)
                continue
            if 'stop' in kinds:
                continue
            nxt = spans[idx + 1] if idx + 1 < len(spans) else None
            if 'audience' in kinds and nxt is not None:
                pair = (phrase[-1], tokens[nxt[0]])
                if pair in STYLE_TYPE_PHRASES:
                    seg.descriptors.append(STYLE_TYPE_PHRASES[pair])
                    signals.content_tokens += 1
                    continue
            for kind, payload in entries:
                if kind == 'type':
                    types, keywords = payload
                    signals.content_tokens += 1
                    for t in types:
                        if t not in seg.types:
                            seg.types.append(t)
                    if types and types[0] not in seg.primary_types:
                        seg.primary_types.append(types[0])
                    # "jeans" -> Trousers: a trouser only counts as jeans if its text says so.
                    type_terms = {t for name in types for t in tokenize(name.replace('/', ' '))}
                    if types and not set(phrase) <= type_terms:
                        seg.needs_keyword = True
                        specific = [t for t in phrase if t not in type_terms and t not in GENERIC_GARMENT_WORDS]
                        for keyword in keywords:
                            specific += [t for t in tokenize(keyword) if t not in GENERIC_GARMENT_WORDS
                                         and t not in type_terms]
                        seg.match_terms += [t for t in dict.fromkeys(specific) if t not in seg.match_terms]
                    seg.type_words.append(' '.join(phrase))
                    seg.type_keywords += [k for k in keywords if k not in seg.type_keywords]
                elif kind == 'type_group':
                    signals.content_tokens += 1
                    seg.group_types += [t for t in payload[0] if t not in seg.group_types]
                    seg.type_words.append(' '.join(phrase))
                elif kind == 'color':
                    (seg.exclude_colors if negated else seg.colors).append(list(payload))
                elif kind == 'pattern':
                    if negated:
                        # "without lace": also drop products whose text says lace (a lace trim).
                        seg.exclude_patterns += payload
                        seg.exclude_words.append(' '.join(phrase))
                    else:
                        seg.patterns += [p for p in payload if p not in seg.patterns]
                elif kind == 'material':
                    signals.content_tokens += 1
                    if negated:
                        seg.exclude_words += [w for w in payload]
                    else:
                        seg.materials.append((' '.join(phrase), list(payload)))
                elif kind == 'style':
                    signals.content_tokens += 1
                    if not negated:
                        seg.styles.append((' '.join(phrase), list(payload)))
                elif kind == 'occasion':
                    signals.content_tokens += 1
                    seg.occasions.append(payload)
                elif kind == 'hint':
                    signals.content_tokens += 1
                    seg.hint_types += [t for t in payload['types'] if t not in seg.hint_types]
                    seg.occasions.append(payload)
                elif kind == 'audience':
                    signals.audience.add(payload)
                    seg.audience.add(payload)
                elif kind == 'section':
                    signals.content_tokens += 1
                    seg.sections += payload['sections']
                    seg.required_sections += [x for x in payload['sections'] if x in REQUIRED_SECTIONS
                                              and x not in seg.required_sections]
                    if payload.get('group'):
                        signals.audience.add(payload['group'])
                        seg.audience.add(payload['group'])
                elif kind == 'price_low':
                    signals.price_low = True
                elif kind == 'price_high':
                    signals.price_high = True
                elif kind == 'rating':
                    signals.rating = True
                elif kind == 'non_catalog':
                    seg.non_catalog.append(' '.join(phrase))
                elif kind == 'generic':
                    seg.generic = True
            if 'negation' in kinds:
                negate_until = end + 2
        return seg

    def _content_kind(self, kinds, phrase, entries):
        """What a span contributes as content (None for stopwords, colours, audience, budget words)."""
        if not entries:
            word = phrase[0]
            if word in PLACEHOLDER_OBJECTS:
                return 'generic'
            if word in self.vocab.stopwords or word.isdigit():
                return None
            if word in BRANDS:
                return 'brand'
            return 'known' if self.vocab.bm25.known(word) else 'unknown'
        for kind in ('type', 'type_group', 'non_catalog', 'hint', 'occasion', 'material', 'style', 'generic',
                     'section'):
            if kind in kinds:
                return kind
        return None

    @staticmethod
    def head_kind(seg, prep_at=None):
        """Kind of the segment's head noun: the last content word before the first
        preposition ("spaceship for underwater travel"), else the last content word
        ("travel pillow")."""
        if not seg.content:
            return None
        before = [kind for pos, kind in seg.content if prep_at is None or pos < prep_at]
        if not before:
            return 'placeholder'      # "something to keep my hands warm": the purpose defines it
        return before[-1]

    # ----- plan assembly -------------------------------------------------
    def plan(self, query):
        signals = Signals()
        text = normalize_query(self.vocab.native_to_english(query))
        text = self.vocab.translate_text(text)
        if INJECTION_RE.search(text):
            signals.injection = True
        words = {stem(t) for t in raw_tokens(text)}
        signals.service = bool(words & SERVICE_VERBS)
        signals.question = bool(QUESTION_RE.match(text))
        for age_match in AGE_RE.finditer(text):
            age = int(age_match.group(1) or age_match.group(3))
            unit = age_match.group(2) or 'year'
            if unit.startswith(('month', 'mth')) or age <= 2:
                signals.audience.add('baby')
            elif age <= 14:
                signals.audience.add('kids')
        text = AGE_RE.sub(' ', text)
        each = bool(EACH_RE.search(text))
        text = EACH_RE.sub(lambda m: ' ' * len(m.group(0)), text)
        budgets = parse_budgets(text)
        for b in budgets:
            if b.soft_target is not None:
                signals.soft_target = b.soft_target
        budget_spans = [(b.start, b.end) for b in budgets]
        clean = _mask_spans(text, budget_spans)

        main_end, comp_start, comp_end = len(clean), None, len(clean)
        context_start = None
        m = COMPLEMENT_RE.search(clean)
        wm = WITH_TO_MATCH_RE.search(clean)
        if m:
            main_end, comp_start = m.start(), m.end()
        elif wm:
            main_end, comp_start, comp_end = wm.start(), wm.start(1), wm.end(1)
        cm = CONTEXT_RE.search(clean[:main_end])
        back_reference = False
        if cm:
            context_start = cm.start()
            signals.context = clean[cm.end():main_end].strip()
            # "a tie to go with it", "socks to wear with them": the item before the marker
            # accompanies the main item; nothing new is named after it.
            first_clause = re.split(r'[,.;!?]', signals.context)[0]
            back_reference = ' '.join(first_clause.split()) in BACK_REFERENCES
            main_end = context_start
        if comp_start is None and (not cm or back_reference):
            accompany = self._with_item(clean[:main_end])
            if accompany is not None:
                main_end, comp_start, comp_end = accompany, accompany + 4, main_end

        # Split the main part into segments.
        segments, pos = [], 0
        main = clean[:main_end]
        for sm in SPLIT_RE.finditer(main):
            segments.append(Segment(pos, sm.start(), main[pos:sm.start()]))
            pos = sm.end()
        segments.append(Segment(pos, main_end, main[pos:main_end]))
        segments = [self.analyze(s, signals) for s in segments if s.text.strip()]
        complement = None
        if comp_start is not None:
            complement = self.analyze(Segment(comp_start, comp_end, clean[comp_start:comp_end]), signals)

        # Assign budgets to the segment that contains them (complement gets its own).
        for b in budgets:
            if b.soft_target is not None:
                continue
            if complement is not None and b.start >= comp_start - 1:
                complement.budgets.append(b)
                continue
            before = [s for s in segments if s.start <= b.start]
            owner = before[-1] if before else (segments[0] if segments else None)
            if owner is not None:
                owner.budgets.append(b)

        trailing = self._trailing_modifiers(segments)
        alternatives = self._alternatives(main)
        choices_segs = self._merge(segments)
        # A budget absorbed into a choice is the same budget, not a second one.
        main_budgets = list({id(b): b for s in segments for b in s.budgets}.values())
        choices_segs = self._distribute_headers(choices_segs)
        if len(choices_segs) > 1:
            for mod in trailing:
                for target in choices_segs:
                    self._share(target, mod)
            if alternatives:
                # "a dress or a top, in yellow": alternatives for one need share what was stated.
                for source in list(choices_segs):
                    for target in choices_segs:
                        if target is not source:
                            self._share(target, source, only_missing=True)
        if back_reference and complement is None and len(choices_segs) >= 2:
            # "chelsea boots for men, and some socks to wear with them": the last item accompanies.
            complement = choices_segs.pop()
            comp_start = complement.start
        if len(choices_segs) > 1 and (each or len(main_budgets) == 1):
            for s in choices_segs:
                s.budgets = list(main_budgets)
        if complement is not None and not complement.budgets and main_budgets and (each or len(main_budgets) == 1):
            complement.budgets = list(main_budgets)

        if 'girls' in signals.audience and ADULT_CONTEXT_RE.search(text) and not CHILD_CONTEXT_RE.search(text):
            signals.audience.discard('girls')
            signals.audience.add('women')
            for seg in segments:
                if 'girls' in seg.audience:
                    seg.audience.discard('girls')
                    seg.audience.add('women')
        plan = Plan(source='rules')
        plan.audience_group = self._audience(signals.audience)
        per_choice = {self._audience(s.audience) for s in choices_segs if s.audience}
        for s in choices_segs:
            choice = self._choice(s, signals)
            if len(per_choice) > 1:
                # "a swimsuit for my wife and swim shorts for my son": each item has its own wearer.
                choice.audience_group = self._audience(s.audience) if s.audience else None
            plan.choices.append(choice)
        if len(per_choice) > 1:
            plan.audience_group = None
        if complement is not None and (complement.has_type or complement.descriptors):
            plan.complement = self._choice(complement, signals)
        signals.unknown = [w for s in segments for w in s.unknown]
        signals.brands = [w for s in segments for w in s.brands]
        signals.non_catalog = [w for s in segments for w in s.non_catalog]
        signals.sections = [x for s in segments for x in s.sections]
        plan.kind = self._kind(plan, segments, signals)
        plan.notes.append(signals)
        return plan

    def _with_item(self, text):
        """Position of a bare "with" that introduces a separate product ("black dress with gold
        earrings"), or None. "dress with pockets" is a feature, not an accompanying item."""
        for w in re.finditer(r'\bwith\b', text):
            before = self.analyze(Segment(0, w.start(), text[:w.start()]), Signals())
            after = self.analyze(Segment(0, 0, text[w.end():]), Signals())
            if (before.has_type and after.types and self.head_kind(after, after.prep_at) == 'type'
                    and not set(after.types) & set(before.types + before.group_types)
                    and not self.vocab.categories_of(after.types) & self.vocab.categories_of(
                        before.types + before.group_types)):
                # "dress with gold earrings": another category is an accompanying item;
                # "boots with a heel" stays within shoes, so the heel is a feature.
                return w.start()
        return None

    @staticmethod
    def _trailing_modifiers(segments):
        """Typeless segments after the last item ("..., nothing black pls") qualify every item."""
        last_typed = max((i for i, seg in enumerate(segments) if seg.has_type), default=-1)
        return [seg for i, seg in enumerate(segments) if i > last_typed and not seg.has_type and last_typed >= 0]

    @staticmethod
    def _alternatives(text):
        """True when the items are joined only by "or"/"ya" (alternatives), not "and"."""
        seps = re.findall(r'\b(?:and|or|ya|aur|plus)\b|&|\+', text)
        return bool(seps) and all(sep in ('or', 'ya') for sep in seps)

    @staticmethod
    def _share(target, source, only_missing=False):
        """Copy colours, exclusions and budgets from source to target."""
        for name in ('colors', 'exclude_colors', 'exclude_patterns', 'exclude_words', 'budgets'):
            values = getattr(target, name)
            if only_missing and values:
                continue
            for v in getattr(source, name):
                if v not in values:
                    values.append(v)

    def _distribute_headers(self, segs):
        """"black shoes under 3000, ankle boots or trainers": a group word ("shoes") followed by
        specific items of that group is a header; its colours and budget apply to those items."""
        if len(segs) < 2:
            return segs
        out = list(segs)
        for header in segs:
            if header.types or not header.group_types:
                continue
            members = [s for s in segs if s is not header and s.types and set(s.types) <= set(header.group_types)]
            if not members:
                continue
            for s in members:
                for name in ('colors', 'exclude_colors', 'patterns', 'exclude_patterns', 'materials', 'styles',
                             'occasions', 'sections', 'exclude_words'):
                    values = getattr(s, name)
                    for v in getattr(header, name):
                        if v not in values:
                            values.append(v)
                if not s.budgets:
                    s.budgets = list(header.budgets)
            out.remove(header)
        return out

    def _merge(self, segments):
        """Segments without a type word attach to the previous choice (or the next one if first)."""
        if not segments:
            return []
        typed = [s for s in segments if s.has_type]
        if len(typed) <= 1:
            merged = Segment(segments[0].start, segments[-1].end, ' '.join(s.text for s in segments))
            for s in segments:
                self._absorb(merged, s)
            return [merged]
        out, pending = [], []
        for s in segments:
            if s.has_type:
                for p in pending:
                    self._absorb(s, p)
                pending = []
                out.append(s)
            elif out:
                self._absorb(out[-1], s)
            else:
                pending.append(s)
        return out

    @staticmethod
    def _absorb(target, other):
        if target is other:
            return
        target.needs_keyword = target.needs_keyword or other.needs_keyword
        if not target.content and other.content:
            target.content, target.prep_at = list(other.content), other.prep_at
            target.content_words = list(other.content_words)
        for name in ('types', 'primary_types', 'hint_types', 'match_terms', 'type_words', 'type_keywords', 'group_types', 'colors',
                     'exclude_colors', 'patterns',
                     'exclude_patterns', 'materials', 'styles', 'descriptors', 'exclude_words', 'occasions',
                     'sections', 'unknown', 'brands', 'non_catalog', 'budgets'):
            values = getattr(target, name)
            for v in getattr(other, name):
                if v not in values:
                    values.append(v)
        target.generic = target.generic or other.generic
        target.audience |= other.audience
        target.required_sections += [x for x in other.required_sections if x not in target.required_sections]

    @staticmethod
    def _audience(groups):
        groups = set(groups)
        if not groups:
            return None
        if groups >= {'men', 'women'}:
            return None
        for specific in ('baby', 'girls', 'boys'):
            if specific in groups:
                return specific
        if 'kids' in groups:
            return 'kids'
        return next(iter(sorted(groups)))

    def _choice(self, seg, signals):
        types = list(seg.types) or list(seg.group_types)
        keywords = list(seg.type_keywords)
        for word in seg.type_words:
            for t in tokenize(word):
                if t not in keywords:
                    keywords.append(t)
        soft_words, soft_sections = [], list(dict.fromkeys(seg.sections))
        occasion_types, occasion_sections, occasion_audiences = [], [], []
        type_sets = []
        for occ in seg.occasions:
            soft_words += [w for w in occ['keywords'] if w not in soft_words]
            soft_sections += [s for s in occ['sections'] if s not in soft_sections]
            occasion_sections += [s for s in occ['sections'] if s not in occasion_sections]
            if occ.get('audience') and occ['audience'] not in occasion_audiences:
                occasion_audiences.append(occ['audience'])
            occasion_types += [t for t in occ['types'] if t not in occasion_types]
            if occ['types']:
                type_sets.append(set(occ['types']))
        if len(type_sets) > 1:
            # "beach wedding": what both occasions call for (a dress), not everything either does.
            shared = [t for t in occasion_types if all(t in ts for ts in type_sets)]
            if shared:
                occasion_types = shared
        group = self._audience(seg.audience) or self._audience(signals.audience)
        if group in ('baby', 'kids', 'girls', 'boys') and any('gift' in occ['keywords'] for occ in seg.occasions):
            occasion_types = [t for t in CHILD_GIFT_TYPES if t in self.vocab.type_set]
        from_occasion = not types and bool(occasion_types or seg.hint_types)
        if from_occasion:
            types = list(seg.hint_types) or occasion_types
        descriptors = list(seg.descriptors)
        for word, words in seg.materials + seg.styles:
            descriptors.append(word)
            soft_words += [w for w in words if w not in soft_words]
        colors = []
        for family in seg.colors:
            colors += [c for c in family if c not in colors]
        exclude_colors = []
        for family in seg.exclude_colors:
            exclude_colors += [c for c in family if c not in exclude_colors]
        min_price = max_price = None
        for b in seg.budgets:
            if b.min_price is not None:
                min_price = b.min_price if min_price is None else max(min_price, b.min_price)
            if b.max_price is not None:
                max_price = b.max_price if max_price is None else min(max_price, b.max_price)
        label = ' '.join(seg.text.split()) or ' '.join(seg.type_words)
        choice = Choice(
            label=label[:120] or 'item', product_types=types, keywords=keywords, colors=colors,
            exclude_colors=exclude_colors, patterns=list(seg.patterns), exclude_patterns=list(seg.exclude_patterns),
            min_price=min_price, max_price=max_price, soft_sections=soft_sections, soft_words=soft_words,
            price_preference='low' if signals.price_low else ('high' if signals.price_high else None),
            rating_preference=signals.rating, exclude_words=list(seg.exclude_words))
        choice.descriptors = descriptors
        choice.unknown = list(seg.unknown)
        choice.primary_types = list(seg.primary_types) or list(seg.group_types[:0])
        choice.needs_keyword = seg.needs_keyword
        choice.match_terms = list(seg.match_terms)
        choice.occasion_types = from_occasion
        choice.occasion_sections = occasion_sections
        choice.occasion_audiences = occasion_audiences
        choice.required_sections = list(seg.required_sections)
        choice.head = self.head_kind(seg, seg.prep_at)
        before = [w for pos, w in seg.content_words if seg.prep_at is None or pos < seg.prep_at]
        choice.head_word = before[-1] if before else None
        choice.color_families = [list(f) for f in seg.colors]
        choice.type_words = list(seg.type_words)
        choice.occasion = bool(seg.occasions)
        choice.generic = seg.generic
        return choice

    @staticmethod
    def _kind(plan, segments, signals):
        has_type = any(c.product_types for c in plan.choices)
        has_signal = has_type or any(c.colors or c.descriptors or c.soft_words or c.soft_sections or c.patterns
                                     for c in plan.choices)
        if signals.injection and not has_type:
            return 'nonsense'
        if not has_signal and not signals.unknown:
            return 'nonsense'
        return 'search'


def rule_plan(query, vocab):
    return RulePlanner(vocab).plan(query)


def signals_of(plan):
    for note in plan.notes:
        if isinstance(note, Signals):
            return note
    return Signals()
