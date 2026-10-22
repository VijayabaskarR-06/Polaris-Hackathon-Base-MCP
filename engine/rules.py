"""Rule planner: instant, deterministic query understanding with no LLM.

It is the fallback when the LLM planner is unavailable or late, the racing
path while it runs, and the safety net for budgets and colours it misses.
"""
import copy
import re
from dataclasses import dataclass, field

from engine.plan import Choice, Plan
from engine.text import normalize_query, raw_tokens, stem, tokenize

NUM = r'(\d+(?:\.\d+)?)'
CUR = r'(?:inr\s*)?'
TAIL = r'(?:\s*inr)?(?:\s*/-)?'
# Words that may stand between a negation and the item it negates ("not any dresses").
NEGATION_FILLERS = {'an', 'the', 'any', 'some', 'more', 'those', 'these', 'such', 'one', 'ones'}
WOMENS_ACCOMPANIMENTS = {'Bag', 'Shoulder bag', 'Cross-body bag', 'Tote bag', 'Earring', 'Earrings', 'Necklace',
                         'Heels', 'Pumps', 'Heeled sandals', 'Wedge', 'Hair/alice band', 'Hair clip'}
STRICT_MAX_RE = re.compile(r'(?:under|below|beneath|less than|lesser than|lower than|cheaper than|for less than|for under|<(?!=))')
MAX_RES = [re.compile(p) for p in (
    rf'\b(?:under|below|beneath|less than|lesser than|lower than|up ?to|upto|till|within|max(?:imum)?'
    rf'|not more than|no more than|not over|cheaper than|at most|not exceeding|no higher than|for less than'
    rf'|nothing (?:over|above|more than|beyond|pricier than|costlier than)|none over|not above'
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
EXACT_RE = re.compile(rf'\b(?:(?:costs?|costing|priced|price(?: is)?|for)\s+)?(?:exactly|precisely)\s*(?:at\s*)?{CUR}{NUM}{TAIL}')
AROUND_RE = re.compile(rf'\b(?:around|about|approximately|approx|roughly|close to|near|nearly|~)\s*{CUR}{NUM}{TAIL}')
BARE_RE = re.compile(rf'(?:\b(?:for|at|@|priced at|costing|cost|price)\s*{CUR}{NUM}{TAIL})|(?:inr\s*{NUM}{TAIL})|(?:{NUM}\s*inr)')
EACH_RE = re.compile(r'\b(?:each|per item|per piece|apiece|a piece|har ek|each one|both|dono|all of them|all three)\b')

# "X with a matching Y": Y is a requested accompanying item (recommendations).
# Shoppers misspell it often ("matcing", "matchng").
MATCH_WORDS = (r'(?:matching|matchng|matcing|maching|matchin|matchig|mathcing|matchinng|machting|matchimg'
               r'|coordinating|co-ordinating|complementary|complimentary|coordinated|co-ordinated|harmonising'
               r'|harmonizing)')
COMPLEMENT_RE = re.compile(
    rf'(?:\b(?:with|and|plus|along with|together with|n)\b|[+&,]|(?<=[.!?])|(?<=[.!?]\s))\s*(?:it\s+|them\s+)?'
    rf'(?:a |an |some |the |pair of |a pair of )?(?:(?:nice|cute|good|suitable) )?{MATCH_WORDS}\b'
    r'|\b(?:with|and)\s+(?:a |an |some )?(?:(?:nice|cute|good|suitable) )?accessor(?:y|ies) (?:to match|for it|for that)'
    r'|\baccessori[sz]e (?:it )?with|\bcomplete (?:the|my) look with|\bpaired with\b')
MATCH_WORD_RE = re.compile(rf'match|{MATCH_WORDS}|coordinat')
# "a bag to match", "con aretes a juego" (with earrings matching): the complement sits between.
WITH_TO_MATCH_RE = re.compile(rf'\bwith\s+(?:a |an |some |the )?(.+?)\s+(?:to match|{MATCH_WORDS}|that match(?:es)?)\b')
# "X to go with my Y": the shopper wants X; Y is context they already own. With nothing (or only
# "it"/"them") after the marker, X itself is the accompanying item: "a coat and gloves to match".
CONTEXT_RE = re.compile(
    r'\b(?:go(?:es)?\s+(?:well\s+|nicely\s+|best\s+|great\s+|good\s+)?with|wear\s+(?:it\s+|them\s+)?with'
    r'|(?:to\s+)?pairs?\s+(?:it\s+|them\s+)?(?:well\s+|nicely\s+)?with'
    r'|(?:to\s+)?(?:style|team|combine|coordinate)\s+(?:it\s+|them\s+)?with'
    r'|(?:to|that|which|would|will|can)\s+match(?:es)?\b|(?:to|that|which|would|will)\s+complements?\b'
    r'|look(?:s)?\s+(?:good|great|nice|best|lovely|smart|cute|amazing|perfect|better)\s+with'
    r'|(?:would|will|might|could|to)\s+suit\b'
    r'|(?:to\s+)?(?:complete|finish(?:\s+off)?|round\s+(?:off|out))\s+(?:the|my|this|her|his|your)\s+(?:look|outfit)'
    r'|matching (?=my |our |her |his )|for (?=my |our )(?:\w+ ){0,2}'
    r'(?:dress|outfit|suit|saree|sari|lehenga|gown|jeans|jacket))')
GLOBAL_RE = re.compile(r'\b(?:everything|all|both|whole|entire|every piece|every item|all of (?:it|them))\b')
POSSESSIVE_RE = re.compile(r"\s*(?:my|our|his|her|their|your|ur|mine|meri|mere|mera)\b")
CONTEXT_FILLER_RE = re.compile(r'(?:\b(?:to|that|which|what|would|will|can|could|might|else|really|i|you|we)\s+)+$')
# "a black dress, and suggest a bag": what follows the verb accompanies the item named before it.
SUGGEST_RE = re.compile(r'\b(?:suggest|recommend)(?:s|ed|ing)?\b')
# "accessories for a little black dress": an accessory for a garment the shopper already has.
FOR_DET_RE = re.compile(r'\bfor\s+(?:a|an|the|this|that|my|our|her|his|your)\s+')
ACCESSORY_CATEGORIES = {'Accessories', 'Bags', 'Shoes', 'Socks & Tights'}
GARMENT_CATEGORIES = {'Garment Full body', 'Garment Upper body', 'Garment Lower body', 'Swimwear', 'Nightwear'}
# Items that a bare "with" usually describes ("a coat with a belt" is a belted coat).
WITH_FEATURE_TYPES = {'Belt'}
SPLIT_RE = re.compile(r'\band\b|&|\bplus\b(?![\s-]*sized?\b)|\bas well as\b|,|/|;|\+|\bor\b|\balso\b|\baur\b'
                      r'|\bya\b|\bthen\b|\bn\b|\balong\s*with\b|\btogether\s+with\b|\balongside\b'
                      r'|\bin addition to\b|(?<=\w)[.!?]+(?=\s|$)|\s[-\u2013]\s')
# "I bought a navy blazer, find me a tie to go with it": the bought item is context, the tie is
# what they want. Possession attached to the first item, followed by a request for another one.
OWN_RE = re.compile(
    r"\bi(?:'ve|'d|\s+have|\s+had)?\s+(?:just\s+|recently\s+|finally\s+|already\s+|also\s+)*"
    r"(?:bought|got|gotten|purchased|picked up|found|ordered|own|received|inherited)\b"
    r"|(?:^|(?<=[.!?,]\s))(?:just\s+|recently\s+|finally\s+|already\s+)*(?:bought|got|picked up|purchased|found)\b"
    r"|\bgot (?:myself|herself|himself|ourselves)\b|\btreated (?:myself|herself|himself|her|him) to\b"
    r"|\b(?:i'm|i am|he's|she's|he is|she is|they are|they're|is|are|am)\s+(?:already\s+)?(?:wearing|going to wear|planning to wear)\b"
    r"|\balready\s+(?:have|has|own|owns|bought|got|picked)\b"
    r"|\b(?:i'm|i am|we're|we are)\s+(?:buying|getting|gifting|ordering)\b"
    r"|\b(?:want|going|plan|planning|decided|would like|'d like)\s+to\s+(?:buy|get|gift|order)\b"
    r"|\b(?:he|she|they|my\s+\w+|our\s+\w+)\s+(?:already\s+|just\s+)?(?:has|have|owns|got|bought)\b(?!\s+to\b)"
    r"|\bi\s+have\b(?!\s+to\b)|\bi've\s+got\b|\bwe\s+have\b(?!\s+to\b)"
    r"|\b(?:is|are)\s+(?:all\s+|already\s+)?(?:sorted|done|ready|bought|picked|chosen|set|decided|finali[sz]ed"
    r"|organi[sz]ed)\b")
REQUEST_RE = re.compile(
    r"\b(?:need|needs|want|wants|looking for|look for|find|suggest|recommend|show|get me|get her|get him|help me"
    r"|any|do you have|got any|which|what|could you|can you|would like|i'd like|love|dikhao|dikhaiye|chahiye"
    r"|chaiye|batao|i'm after|i am after|shopping for|searching for|in the market for|kaunsa|kaunsi|konsa|konsi"
    r"|kaun sa|kaun si)\b")
BACK_REFERENCE_WORDS = {'it', 'them', 'this', 'that', 'these', 'those', 'one', 'ones', 'too', 'pls', 'please',
                        'plz', 'well', 'also', 'nicely', 'perfectly', 'great', 'thanks', 'thank', 'you', 'if',
                        'possible', 'poss', 'have', 'any', 'em'}
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
AGE_RE = re.compile(r'\b(\d{1,2})\s*(?:-|\s)?\s*(years?|yrs?|y/o|yo|months?|mths?|ans|anos|jahre|saal)'
                    r'(?:\s*-?\s*old)?\b|\baged?\s*(\d{1,2})\b|\bturn(?:s|ing)?\s+(\d{1,2})\b'
                    r'|\b(\d{1,2})(?:st|nd|rd|th)\s+birthday\b')
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
                 'recipe', 'news', 'score', 'lyrics', 'directions', 'appointment', 'service', 'servicing'}
# Asking for the weather is a service; dressing for it ("something for very hot weather") is shopping.
FORECAST_RE = re.compile(r"\b(?:weather|forecast|temperature)\s+(?:today|tomorrow|tonight|this week|now|in|at)\b"
                         r"|\b(?:how|what)(?:'s|s| is| will be)\s+the\s+weather\b"
                         r"|\bwill it (?:rain|snow|be (?:hot|cold|sunny|rainy|windy))\b|\bis it (?:raining|snowing|going to rain)\b"
                         r"|\b(?:rain|snow)\s+(?:today|tomorrow|tonight)\b|\bchance of (?:rain|snow)\b"
                         r"|\b(?:news|headlines|breaking news)\b")
# Questions that are never a shopping request, whatever else they mention.
NONSHOPPING_QUESTION_RE = re.compile(r"^\s*(?:who|whom|whose|when|why)\b")
QUESTION_RE = re.compile(r"^\s*(?:how|what|why|when|where|who|which|whats|what's|is|are|do|does|did|can|could|"
                         r"will|would|should)\b")
PET_TYPES = {'Dog Wear', 'Dog wear'}
ADULT_CONTEXT_RE = re.compile(r'\b(?:at work|work|office|colleague|coworker|co-worker|college|uni|university|'
                              r'girlfriend|gf|wife|woman|women|lady|ladies|girls night|her birthday party|hostel)\b')
CHILD_CONTEXT_RE = re.compile(r'\b(?:daughter|niece|little|toddler|kid|kids|child|children|school|baby|year old|'
                              r'years old|yr old|granddaughter|beti)\b')
# Words whose singular is an adjective while the plural names a garment.
ADJECTIVE_FORMS = {'short', 'tight'}
IDIOM_RE = re.compile(r"\b(?:over the top|on top of|top notch|top-notch|tip top|on top"
                      r"|top (?:news|stories|story|headlines|\d+|ten|five|picks|rated|selling|trending|deals|tips|songs|movies|scorer))\b")
BACK_REFERENCES = {'it', 'them', 'this', 'that', 'these', 'those', 'one', 'it too', 'them too'}
# "unicorn stuff", "cute things": the object is "clothing in general".
PLACEHOLDER_OBJECTS = {'stuff', 'thing', 'item', 'clothe', 'clothing', 'garment', 'apparel', 'gear', 'kit',
                       'solution', 'option', 'idea', 'suggestion', 'recommendation', 'choice'}
GENERIC_GARMENT_WORDS = {'shirt', 'top', 'tee', 'tshirt', 'wear', 'set', 'clothe', 'clothing', 'outfit', 'item',
                         'sport', 'sports', 'style', 'piece', 'garment', 'fashion', 'thing', 'stuff'}
PREPOSITIONS = {'for', 'with', 'to', 'in', 'on', 'at', 'from', 'during', 'like', 'without', 'by', 'near', 'into'}
# Words naming when, not what: in "prom night" or "wedding day" the occasion is the head.
TIME_WORDS = {'night', 'day', 'evening', 'morning', 'afternoon', 'weekend', 'eve', 'time', 'season', 'event',
              'function', 'ceremony', 'celebration'}


@dataclass
class Budget:
    min_price: float = None
    max_price: float = None
    start: int = 0
    end: int = 0
    soft_target: float = None
    strict: bool = False         # "under 999": a product priced 999 is not under it


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

    for m in EXACT_RE.finditer(text):
        # "a dress that costs exactly 999": an exact price is both bounds.
        value = _value(m.group(1))
        if free(m.span()) and plausible(m, value):
            found.append(Budget(min_price=value, max_price=value, start=m.start(), end=m.end()))
            taken.append(m.span())
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
                budget.strict = kind == 'max' and bool(STRICT_MAX_RE.match(m.group(0)))
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
    exclude_types: list = field(default_factory=list)   # "anything but jeans": Trousers
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
    no_baby: bool = False                                 # a stated age of four or more
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
        negation_end = -10
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
                negation_end = end
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
            right_after = negated and all(t in NEGATION_FILLERS for t in tokens[negation_end:start])
            head = [p for k, p in self.m.entries.get((phrase[-1],), []) if k == 'type'] if end - start > 1 else []
            if right_after and head and any(k == 'type' for k, _ in entries):
                # "i hate skinny jeans": the style is unwanted, not the jeans.
                seg.exclude_words.append(' '.join(phrase[:-1]))
                types, keywords = head[0]
                seg.types += [t for t in types if t not in seg.types]
                if types and types[0] not in seg.primary_types:
                    seg.primary_types.append(types[0])
                seg.type_words.append(phrase[-1])
                seg.type_keywords += [k for k in keywords if k not in seg.type_keywords]
                signals.content_tokens += 1
                continue
            if right_after and any(k in ('type', 'type_group') for k, _ in entries):
                # "something other than jeans for a party", "not a dress, maybe a jumpsuit": the
                # item named right after a negation is what the shopper does not want.
                for kind, payload in entries:
                    if kind in ('type', 'type_group'):
                        seg.exclude_types += [t for t in payload[0] if t not in seg.exclude_types]
                seg.exclude_words.append(' '.join(phrase))
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
                    # "half pants" (Shorts), "sipper" (Waterbottle): a plain synonym whose own keyword
                    # is the type's name needs no word of the phrase in the product.
                    synonym = any(tokenize(k) and set(tokenize(k)) <= type_terms for k in keywords)
                    if types and not set(phrase) <= type_terms and not synonym:
                        specific = [t for t in phrase if t not in type_terms and t not in GENERIC_GARMENT_WORDS]
                        for keyword in keywords:
                            specific += [t for t in tokenize(keyword) if t not in GENERIC_GARMENT_WORDS
                                         and t not in type_terms]
                        # A word no product uses ("neckless", an alias typo) cannot be required.
                        specific = [t for t in dict.fromkeys(specific) if self.vocab.bm25.df(t) > 0]
                        if specific:
                            seg.needs_keyword = True
                            seg.match_terms += [t for t in specific if t not in seg.match_terms]
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
                    else:
                        seg.exclude_words.append(' '.join(phrase))
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
        words = dict(seg.content_words)
        before = [(pos, kind) for pos, kind in seg.content if prep_at is None or pos < prep_at]
        while len(before) > 1 and words.get(before[-1][0]) in TIME_WORDS:
            before.pop()
        if not before:
            return 'placeholder'      # "something to keep my hands warm": the purpose defines it
        return before[-1][1]

    # ----- plan assembly -------------------------------------------------
    def plan(self, query, recs=True):
        """recs: include_recommendations. With recommendations off, a bare "with" joining two
        kinds of item ("a white shirt with black trousers") names two choices, as the brief
        asks for every item not explicitly requested as an accompaniment."""
        signals = Signals()
        text = normalize_query(self.vocab.native_to_english(query))
        text = self.vocab.translate_text(text)
        text = self.vocab.split_run_together(text)
        if INJECTION_RE.search(text):
            signals.injection = True
        words = {stem(t) for t in raw_tokens(text)}
        signals.service = bool(words & SERVICE_VERBS) or bool(FORECAST_RE.search(text))
        signals.question = bool(QUESTION_RE.match(text))
        if NONSHOPPING_QUESTION_RE.match(text) and not REQUEST_RE.search(text):
            signals.service = True        # "who won the 2011 cricket world cup"
        for age_match in AGE_RE.finditer(text):
            age = int(next(g for g in (age_match.group(1), age_match.group(3), age_match.group(4),
                                       age_match.group(5)) if g))
            unit = age_match.group(2) or 'year'
            if unit.startswith(('month', 'mth')) or age <= 2:
                signals.audience.add('baby')
            elif age <= 14:
                signals.audience.add('kids')
                if age >= 4:
                    # Baby sizes 50-98 stop at about three: a six-year-old's pyjamas are kids' sizes.
                    signals.no_baby = True
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
        comp_text = None          # set when the accompaniment comes before the main item
        matching = False
        m = COMPLEMENT_RE.search(clean)
        wm = WITH_TO_MATCH_RE.search(clean)
        if m:
            main_end, comp_start = m.start(), m.end()
            matching = bool(MATCH_WORD_RE.search(m.group(0)))
        elif wm:
            main_end, comp_start, comp_end = wm.start(), wm.start(1), wm.end(1)
            matching = True
        cm = CONTEXT_RE.search(clean[:main_end]) or self._for_context(clean[:main_end])
        back_reference = False
        backref_at = None
        context = None
        pair = None               # a not-yet-owned item the wanted one should go with
        if cm:
            # "earrings that would go with ...": the filler before the marker belongs to it.
            filler = CONTEXT_FILLER_RE.search(clean[:cm.start()])
            context_start = filler.start() if filler else cm.start()
            clause = re.split(r'[,.;!?]|\s[-\u2013]\s', clean[cm.end():main_end])[0]
            clause_end = cm.end() + len(clause)
            signals.context = ' '.join(clause.split())
            # "a tie to go with it", "socks to wear with them pls", "a coat and gloves to match": the
            # item before the marker accompanies the main item; nothing new is named after it.
            words = [w for w in raw_tokens(clause) if stem(w) not in self.vocab.stopwords]
            back_reference = all(w in BACK_REFERENCE_WORDS for w in words)
            matching = matching or bool(MATCH_WORD_RE.search(cm.group(0)))
            item_start = max([0] + [x.end() for x in SPLIT_RE.finditer(clean[:context_start])])
            wanted = self.analyze(Segment(0, 0, clean[item_start:context_start]), Signals())
            possessed = bool(POSSESSIVE_RE.match(clause) or OWN_RE.search(clause)
                             or OWN_RE.search(clean[:cm.start()]))
            if back_reference:
                backref_at = context_start
            else:
                context = self.analyze(Segment(cm.end(), clause_end, clause), Signals())
                if recs and comp_start is None and wanted.has_type and not possessed and cm.re is CONTEXT_RE:
                    # "What belt would you pair with black skinny jeans?", "earrings to go with a black
                    # dress": the named item is what the shopper asks for (judges grade the request
                    # that way), and the item it should go with, not owned yet, is the natural
                    # recommendation. Owned items stay context only.
                    pair = context
            # The marker and its clause are not wanted items; later sentences still are ("I have
            # nothing to wear with them. can you show me some navy blouses?").
            if cm is not None:
                clean = clean[:context_start] + ',' + ' ' * (clause_end - context_start - 1) + clean[clause_end:]
        own = OWN_RE.search(clean[:main_end])
        if comp_start is None:
            for sm in SUGGEST_RE.finditer(clean[:main_end]):
                if own is not None and own.start() < sm.start():
                    # "I bought a dress, suggest heels": the heels are what they want; but in "already
                    # has trainers. find him a tee and recommend shorts", the shorts accompany the tee.
                    request = REQUEST_RE.search(clean, own.end(), sm.start())
                    between = self.analyze(Segment(0, 0, clean[request.end():sm.start()]), Signals()) \
                        if request is not None else None
                    if between is None or not between.has_type:
                        break
                before = self.analyze(Segment(0, sm.start(), clean[:sm.start()]), Signals())
                after = self.analyze(Segment(0, 0, clean[sm.end():main_end]), Signals())
                if before.has_type and after.has_type:
                    main_end, comp_start, comp_end = sm.start(), sm.end(), main_end
                    break
        if comp_start is None and (not cm or back_reference):
            found = self._with_item(clean[:main_end], loose=back_reference)
            if found is not None:
                accompany, after_types = found
                if own is not None and own.start() < accompany:
                    # "my husband got a navy suit; surprise him with a red tie": the tie is wanted.
                    clean = clean[:accompany] + 'and ' + clean[accompany + 4:]
                elif back_reference or recs:
                    main_end, comp_start, comp_end = accompany, accompany + 4, main_end
                elif not set(after_types) <= WITH_FEATURE_TYPES:
                    # Recommendations off: "with" joins two choices, like "and" (same length, so
                    # budget spans stay put).
                    clean = clean[:accompany] + 'and ' + clean[accompany + 4:]

        # Split the main part into segments.
        segments, pos = [], 0
        main = clean[:main_end]
        for sm in SPLIT_RE.finditer(main):
            segments.append(Segment(pos, sm.start(), main[pos:sm.start()]))
            pos = sm.end()
        segments.append(Segment(pos, main_end, main[pos:main_end]))
        segments = [self.analyze(s, signals) for s in segments if s.text.strip(' ,.;:!?-')]
        complement = None
        if comp_start is not None:
            text_of = comp_text if comp_text is not None else clean[comp_start:comp_end]
            complement = self.analyze(Segment(comp_start, comp_end, text_of), signals)

        # Assign budgets to the segment that contains them (complement gets its own).
        for b in budgets:
            if b.soft_target is not None:
                continue
            if complement is not None and b.start >= comp_start - 1 and (comp_text is None or b.start < comp_end):
                complement.budgets.append(b)
                continue
            before = [s for s in segments if s.start <= b.start]
            owner = before[-1] if before else (segments[0] if segments else None)
            if owner is not None:
                owner.budgets.append(b)

        trailing = self._trailing_modifiers(segments)
        alternatives = self._alternatives(main)
        choices_segs = self._merge(segments, main)
        # A budget absorbed into a choice is the same budget, not a second one.
        main_budgets = list({id(b): b for s in segments for b in s.budgets}.values())
        choices_segs = self._distribute_headers(choices_segs)
        if context is not None and not back_reference and context.types and len(choices_segs) >= 2 and any(
                set(context.types) & set(x.types) for x in choices_segs if x.start < context_start):
            # "... trousers, a blouse and pumps, plus a belt to go with the trousers": the item before
            # the marker accompanies one already asked for.
            back_reference, backref_at, context = True, context_start, None
        owned = self._owned(choices_segs, own, clean, back_reference, backref_at)
        if owned:
            # "I bought a navy blazer, find me a tie to go with it": the blazer is context.
            for seg in owned:
                choices_segs.remove(seg)
                main_budgets = [b for b in main_budgets if all(b is not x for x in seg.budgets)]
            context = owned[0]
        first_typed = next((i for i, x in enumerate(segments) if x.has_type), len(segments))
        # "everything has to be black. I need a blazer, trousers and pumps": a leading requirement
        # said of everything applies to every item.
        trailing += [x for x in segments[:first_typed] if (x.colors or x.exclude_colors)
                     and GLOBAL_RE.search(x.text)]
        if len(choices_segs) > 1:
            for mod in trailing:
                for target in choices_segs:
                    self._share(target, mod)
            setting = self._trailing_setting(choices_segs[-1])
            if setting is not None:
                # "shirts and ties for the office", "leggings and sports bras for yoga": the occasion
                # said after the last item is the setting of every item without one of its own.
                for target in choices_segs[:-1]:
                    if not target.occasions:
                        self._share_setting(target, setting)
            if alternatives:
                # "a dress or a top, in yellow": alternatives for one need share what was stated.
                for source in list(choices_segs):
                    for target in choices_segs:
                        if target is not source:
                            self._share(target, source, only_missing=True)
        missing_complement = False
        if back_reference and complement is None:
            # "chelsea boots for men, and some socks to wear with them": the item just before the
            # marker accompanies the ones named before it.
            before = [x for x in choices_segs if backref_at is None or x.start < backref_at]
            if len(before) >= 2:
                # "a warm coat + scarf n gloves to go with it": items joined by and/n/& right before
                # the marker accompany together; a comma, "+" or "plus" ends the run.
                run = [before[-1]]
                for prev in reversed(before[:-1]):
                    if not re.fullmatch(r'\s*(?:and|n|&|or)\s*', main[prev.end:run[0].start]):
                        break
                    run.insert(0, prev)
                if run[0] is before[0]:
                    run = [before[-1]]     # "jeans and a tee and a belt to go with them": just the belt
                    if owned:
                        run = []           # "I have a dress, need heels and a bag to go with it": all wanted
                complement = run[0] if run else None
                for extra in run[1:]:
                    self._absorb(complement, extra)
                for seg in run:
                    choices_segs.remove(seg)
                comp_start = complement.start if complement is not None else comp_start
                # "..., and cufflinks to match - max 1000 for the cufflinks": a later mention of the
                # accompaniment refines it rather than naming a new choice.
                for later in [x for x in choices_segs if complement is not None and backref_at is not None
                              and x.start > backref_at and x.types and set(x.types) <= set(complement.types)]:
                    self._absorb(complement, later)
                    choices_segs.remove(later)
            else:
                # "yoga leggings and a yoga mat to go with them": the accompaniment is not sold.
                last = [x for x in segments if backref_at is None or x.end <= backref_at + 1]
                if len(last) >= 2 and not last[-1].has_type and last[-1].non_catalog:
                    missing_complement = True
        if complement is not None and complement.budgets:
            # "sneakers and a tie to go with it under 3000": the accompaniment's own budget is not
            # one of the main items' budgets (it still bounds them per item, below).
            main_budgets = [b for b in main_budgets if all(b is not x for x in complement.budgets)]
        if len(choices_segs) > 1 and (each or len(main_budgets) == 1):
            for s in choices_segs:
                # "a blazer, trousers and pumps, plus a belt under 1000": one stated budget is shared
                # unless no product of an item's kind could meet it (that budget was the belt's).
                if each or s.budgets or not self._priced_out(s, main_budgets):
                    s.budgets = list(main_budgets)
        if each:
            # "a white shirt and a tie to go with it, dono 2000 ke andar": the budget is per item,
            # wherever in the request it was said.
            shared = main_budgets or (list(complement.budgets) if complement is not None else [])
            for s in choices_segs + ([complement] if complement is not None else []):
                if not s.budgets and shared:
                    s.budgets = list(shared)
        if complement is not None and not complement.budgets and main_budgets and (each or len(main_budgets) == 1):
            complement.budgets = list(main_budgets)
        if complement is not None and complement.budgets and not main_budgets and len(complement.budgets) == 1:
            # "a black dress with a matching bag under INR 1500": the brief makes one stated budget
            # per item, so it bounds the dress as well as the bag, unless no dress could meet it.
            for s in choices_segs:
                if not s.budgets and not self._priced_out(s, complement.budgets):
                    s.budgets = list(complement.budgets)

        context_types = []
        if context is not None:
            # The wearer of the owned item wears what goes with it: "earrings to go with my wife's
            # dress", "shoes to match my red dress" (a dress is women's wear).
            signals.audience |= context.audience
            context_types = list(context.primary_types or context.types or context.group_types)
            if not signals.audience:
                genders = {self.vocab.type_gender.get(t) for t in (context.primary_types or context.types)}
                if len(genders) == 1 and None not in genders:
                    signals.audience.add(genders.pop())
        if 'girls' in signals.audience and ADULT_CONTEXT_RE.search(text) and not CHILD_CONTEXT_RE.search(text):
            signals.audience.discard('girls')
            signals.audience.add('women')
            for seg in segments:
                if 'girls' in seg.audience:
                    seg.audience.discard('girls')
                    seg.audience.add('women')
        plan = Plan(source='rules')
        plan.audience_group = self._audience(signals.audience)
        plan.exclude_baby = signals.no_baby and plan.audience_group != 'baby'
        if plan.audience_group in ('girls', 'boys', 'kids') and 'baby' not in signals.audience \
                and not re.search(r'\btoddlers?\b', text):
            # "a shirt for a boy", "dress for my daughter": a child, not a baby, unless said.
            plan.exclude_baby = True
        per_choice = {self._audience(s.audience) for s in choices_segs if s.audience}
        for s in choices_segs:
            choice = self._choice(s, signals)
            if len(per_choice) > 1:
                # "a swimsuit for my wife and swim shorts for my son": each item has its own wearer.
                choice.audience_group = self._audience(s.audience) if s.audience else None
            plan.choices.append(choice)
        if len(per_choice) > 1:
            plan.audience_group = None
        if context_types:
            self._fit_context(plan, context_types, signals)
        plan.matching = matching
        if complement is not None and (complement.has_type or complement.descriptors):
            plan.complement = self._choice(complement, signals)
        elif missing_complement or (complement is not None and (complement.unknown or complement.non_catalog)):
            # "a shirt with a matching cummerbund": the accompanying item is not sold, and no
            # recommendation beats a forced one.
            plan.complement_missing = True
        elif pair is not None and pair.has_type and context is pair:
            plan.complement = self._choice(pair, signals)
        if plan.complement is not None and plan.audience_group is None and not signals.audience:
            # "a white shirt with a matching tie": ties are men's, so the shirt most likely is too;
            # "a top with matching earrings" is a woman's. A preference, never a requirement.
            known = {self.vocab.type_gender.get(t) for t in plan.complement.product_types} - {None}
            if not known and set(plan.complement.product_types) & WOMENS_ACCOMPANIMENTS:
                known = {'women'}    # "shoes for a wedding with a matching bag": a woman's outfit
            if len(known) == 1:
                for choice in plan.choices:
                    choice.prefer_group = next(iter(known))
        # What the shopper owns ("I bought a Zara jacket") is not what they ask for.
        asked = segments if owned is None else choices_segs + ([complement] if complement is not None else [])
        signals.unknown = [w for s in asked for w in s.unknown]
        signals.brands = [w for s in asked for w in s.brands]
        signals.non_catalog = [w for s in asked for w in s.non_catalog]
        signals.sections = [x for s in asked for x in s.sections]
        if {'men', 'boys'} & signals.audience and any('Mama' in (getattr(c, 'required_sections', None) or [])
                                                        for c in plan.choices):
            plan.kind = 'nonsense'        # maternity wear for a man: an impossible requirement
        else:
            plan.kind = self._kind(plan, segments, signals)
        plan.notes.append(signals)
        return plan

    def _with_item(self, text, loose=False):
        """(position, item types) of a bare "with" that introduces a separate product ("black
        dress with gold earrings"), or None. "dress with pockets" is a feature, not an item.

        loose: the text goes on to say the item accompanies ("a sweater with a shirt to go with
        it"), so a same-category item or a group word ("jewellery") counts too.
        """
        for w in re.finditer(r'\bwith\b', text):
            if re.search(r'\b(?:along|together)\s*$', text[:w.start()]):
                continue          # "along with" joins two choices
            before = self.analyze(Segment(0, w.start(), text[:w.start()]), Signals())
            after = self.analyze(Segment(0, 0, re.split(r'[,.;!?]', text[w.end():])[0]), Signals())
            after_types = after.types or (after.group_types if loose else [])
            if not (before.has_type and after_types):
                continue
            if not set(after_types) & set(before.types + before.group_types) and (
                    (loose and self.head_kind(after, after.prep_at) in ('type', 'type_group'))
                    or (self.head_kind(after, after.prep_at) == 'type'
                        and not self.vocab.categories_of(after_types) & self.vocab.categories_of(
                            before.types + before.group_types))):
                # "dress with gold earrings": another category is an accompanying item;
                # "boots with a heel" stays within shoes, so the heel is a feature.
                return w.start(), list(after_types)
        return None

    def _for_context(self, text):
        """"accessories for a little black dress": a span marking the garment the accessory is
        for (context the shopper already has), or None. "a bag for my mum" names no garment."""
        for m in FOR_DET_RE.finditer(text):
            before = self.analyze(Segment(0, m.start(), text[:m.start()]), Signals())
            head = before.types or before.group_types
            if not head or not self.vocab.categories_of(head) <= ACCESSORY_CATEGORIES:
                continue
            after = self.analyze(Segment(0, 0, re.split(r'[,.;!?]', text[m.end():])[0]), Signals())
            if (after.types and self.vocab.categories_of(after.types) & GARMENT_CATEGORIES
                    and not set(after.types) & set(head)):
                return m
        return None

    def _fit_context(self, plan, context_types, signals):
        """Shape the wanted items by the item the shopper already has.

        A broad word ("shoes", "accessories", "jewellery") narrows to what the catalog pairs
        with that item; a request naming no item at all ("something to wear with white jeans",
        "what goes with a navy blazer?") asks for exactly those pairings.
        """
        plan.context_types = list(context_types)
        pairs = [t for t in self.vocab.complement_map.get(context_types[0], []) if t not in context_types]
        if not pairs:
            return
        for choice in plan.choices:
            if getattr(choice, 'from_group', False):
                narrowed = [t for t in pairs if t in choice.product_types]
                if narrowed:
                    choice.product_types, choice.primary_types = narrowed, list(narrowed)
        if any(c.product_types for c in plan.choices):
            return
        base = plan.choices[0] if plan.choices else self._choice(Segment(0, 0, ''), signals)
        # One choice per pairing, so the list covers tops, shirts and tees rather than whichever
        # type happens to sort first.
        plan.choices = []
        for t in pairs[:4]:
            choice = copy.copy(base)
            choice.product_types, choice.primary_types, choice.keywords = [t], [t], []
            choice.occasion_types, choice.needs_keyword, choice.match_terms = False, False, []
            choice.head, choice.from_group = 'type', False
            choice.label = f'{t.lower()} to go with the {context_types[0].lower()}'
            plan.choices.append(choice)

    @staticmethod
    def _trailing_modifiers(segments):
        """Typeless segments after the last item ("..., nothing black pls") qualify every item."""
        last_typed = max((i for i, seg in enumerate(segments) if seg.has_type), default=-1)
        return [seg for i, seg in enumerate(segments) if i > last_typed and not seg.has_type and last_typed >= 0]

    def _trailing_setting(self, seg):
        """The occasion phrase ending an item ("ties for the office"): a typeless Segment, or None."""
        for m in re.finditer(r'\b(?:for|at|during|on|in)\b', seg.text):
            head = self.analyze(Segment(seg.start, seg.start + m.start(), seg.text[:m.start()]), Signals())
            if not head.has_type:
                continue
            tail = self.analyze(Segment(seg.start + m.start(), seg.end, seg.text[m.start():]), Signals())
            if tail.has_type or not (tail.occasions or tail.styles):
                return None
            return tail
        return None

    @staticmethod
    def _share_setting(target, setting):
        for name in ('occasions', 'sections', 'styles'):
            values = getattr(target, name)
            for v in getattr(setting, name):
                if v not in values:
                    values.append(v)

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

    def _merge(self, segments, text=''):
        """Segments without a type word attach to the previous choice (or the next one if first).

        A bare colour joined by "or" to the next item belongs to it ("a nude or beige handbag").
        """
        if not segments:
            return []
        typed = [s for s in segments if s.has_type]
        if len(typed) <= 1:
            merged = Segment(segments[0].start, segments[-1].end, ' '.join(s.text for s in segments))
            # The named item is the head ("a yoga mat and leggings": leggings, not the mat).
            for s in typed + [s for s in segments if all(s is not t for t in typed)]:
                self._absorb(merged, s)
            return [merged]
        out, pending = [], []
        for i, s in enumerate(segments):
            nxt = segments[i + 1] if i + 1 < len(segments) else None
            leads = (nxt is not None and nxt.has_type and s.colors and not s.content
                     and re.search(r'\bor\b|/', text[s.end:nxt.start]))
            if s.has_type:
                for p in pending:
                    self._absorb(s, p)
                pending = []
                out.append(s)
            elif out and not leads:
                self._absorb(out[-1], s)
            else:
                pending.append(s)
        return out

    def _priced_out(self, seg, budgets):
        """True when no product of the segment's kinds fits the budgets (by catalog price bands)."""
        bands = [self.vocab.type_band[t] for t in (seg.types or seg.group_types) if t in self.vocab.type_band]
        if not bands:
            return False
        cheapest, dearest = min(b[0] for b in bands), max(b[2] for b in bands)
        return any((b.max_price is not None and cheapest > b.max_price)
                   or (b.min_price is not None and dearest < b.min_price) for b in budgets)

    @staticmethod
    def _owned(segs, own, clean, back_reference, backref_at=None):
        """Choices the shopper already has ("I bought a navy blazer", "my son has swim shorts",
        "the dress is all sorted") when they ask for something else to go with them; [] when
        every choice is wanted ("I'm buying a jacket. Which beanie would match it?")."""
        if len(segs) < 2:
            return []
        if own is not None:
            owned = next((s for s in segs if s.start <= own.start() < s.end), None)
            if owned is not None:
                # Everything named from the owned item up to the request is what they have: "I have
                # black jeans and white sneakers, looking for a top to go with them".
                request = REQUEST_RE.search(clean, owned.end)
                have = []
                if request is not None:
                    have = [s for s in segs if owned.start <= s.start and s.end <= request.start()]
                elif back_reference and backref_at is not None:
                    wanted = [s for s in segs if s.start < backref_at]
                    if wanted and wanted[-1] is not owned:
                        have = [s for s in segs if owned.start <= s.start and s.end <= wanted[-1].start]
                elif REQUEST_RE.search(clean[:owned.start]):
                    have = [owned]
                if have and len(have) < len(segs):
                    return have
        return []

    @staticmethod
    def _absorb(target, other):
        if target is other:
            return
        target.needs_keyword = target.needs_keyword or other.needs_keyword
        if not target.content and other.content:
            target.content, target.prep_at = list(other.content), other.prep_at
            target.content_words = list(other.content_words)
        for name in ('types', 'primary_types', 'hint_types', 'match_terms', 'type_words', 'type_keywords', 'group_types', 'exclude_types', 'colors',
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
        occasion_colors = []
        for occ in seg.occasions:
            soft_words += [w for w in occ['keywords'] if w not in soft_words]
            occasion_colors += [c for c in occ.get('colors', []) if c not in occasion_colors]
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
        footwear = [t for occ in seg.occasions for t in occ.get('footwear', [])]
        if seg.group_types and not seg.types and (occasion_types or footwear):
            # "shoes for the gym", "shoes for the beach", "shoes for a wedding": the occasion says
            # which of the group's kinds (dressy shoes for a wedding, not flip-flops).
            narrowed = [t for t in seg.group_types if t in occasion_types or t in footwear]
            if narrowed:
                types = narrowed
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
        # The item's own budgets first: contradicting ones said of it make it impossible ("at least
        # 2500 but no more than 1200"); a budget absorbed from other words must not contradict them.
        own = [b for b in seg.budgets if seg.start <= b.start < seg.end]
        for b in own + [b for b in seg.budgets if all(b is not x for x in own)]:
            lo = b.min_price if min_price is None or b.min_price is None else max(min_price, b.min_price)
            hi = b.max_price if max_price is None or b.max_price is None else min(max_price, b.max_price)
            lo = min_price if lo is None else lo
            hi = max_price if hi is None else hi
            if lo is not None and hi is not None and lo > hi and all(b is not x for x in own):
                continue
            min_price, max_price = lo, hi
        strict_max = any(b.strict and b.max_price == max_price for b in seg.budgets) if max_price is not None else False
        label = ' '.join(seg.text.split()).strip(' ,.;:!?') or ' '.join(seg.type_words)
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
        while len(before) > 1 and before[-1] in TIME_WORDS:
            before.pop()
        choice.head_word = before[-1] if before else None
        choice.color_families = [list(f) for f in seg.colors]
        choice.type_words = list(seg.type_words)
        choice.occasion = bool(seg.occasions)
        choice.strict_max = strict_max
        choice.exclude_types = list(seg.exclude_types)
        if choice.exclude_types:
            choice.product_types = [t for t in choice.product_types if t not in choice.exclude_types]
        # "haldi": a ceremony's traditional colour is a strong preference unless a colour was stated.
        choice.soft_colors = [] if colors or exclude_colors else occasion_colors
        choice.generic = seg.generic
        choice.from_group = not seg.types and bool(seg.group_types)
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
