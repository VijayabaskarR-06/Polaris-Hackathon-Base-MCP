"""Text normalization shared by the catalog index (BM25) and query understanding.

Both sides must tokenize identically, so every rule lives here: lowercase,
accents folded, compounds such as "t-shirt" joined, tokens of [a-z0-9] with at
least two characters, and a light plural stemmer.
"""
import re
import unicodedata
from functools import lru_cache

TOKEN_RE = re.compile(r'[a-z0-9]+')
# Compounds whose parts are meaningless alone ("t", "v") or whose spacing varies.
COMPOUNDS = [
    (re.compile(r'\bt[\s\-]?shirts?\b'), ' tshirt '),
    (re.compile(r'\btees?[\s\-]?shirts?\b'), ' tshirt '),
    (re.compile(r'\bv[\s\-]neck(s|ed)?\b'), ' vneck '),
    (re.compile(r'\bpolo[\s\-]neck(s|ed)?\b'), ' poloneck '),
    (re.compile(r'\bflip[\s\-]?flops?\b'), ' flipflop '),
    (re.compile(r'\bswim[\s\-]?suits?\b'), ' swimsuit '),
    (re.compile(r'\bswim[\s\-]?wear\b'), ' swimwear '),
    (re.compile(r'\bactive[\s\-]?wear\b'), ' activewear '),
    (re.compile(r'\bsports?[\s\-]?wear\b'), ' sportswear '),
    (re.compile(r'\bethnic[\s\-]?wear\b'), ' ethnicwear '),
    (re.compile(r'\bsweat[\s\-]?shirts?\b'), ' sweatshirt '),
    (re.compile(r'\bsweat[\s\-]?pants\b'), ' sweatpants '),
    (re.compile(r'\bhoodies?\b|\bhoody\b'), ' hoodie '),
    (re.compile(r'\bpy?jamas?\b|\bpajamas?\b'), ' pyjama '),
    (re.compile(r'\bpj\'?s\b'), ' pyjama '),
    (re.compile(r'\bbody[\s\-]?suits?\b'), ' bodysuit '),
    (re.compile(r'\bjump[\s\-]?suits?\b'), ' jumpsuit '),
    (re.compile(r'\bplay[\s\-]?suits?\b'), ' playsuit '),
    (re.compile(r'\bover[\s\-]?alls?\b'), ' overall '),
    (re.compile(r'\bhand[\s\-]?bags?\b'), ' handbag '),
    (re.compile(r'\bback[\s\-]?packs?\b'), ' backpack '),
    (re.compile(r'\bcross[\s\-]?body\b'), ' crossbody '),
    (re.compile(r'\bsun[\s\-]?glass(es)?\b'), ' sunglasses '),
    (re.compile(r'\bear[\s\-]?rings?\b'), ' earring '),
    (re.compile(r'\bwater[\s\-]?bottles?\b'), ' waterbottle '),
    (re.compile(r'\bra?in[\s\-]?coats?\b'), ' raincoat '),
    (re.compile(r'\bknit[\s\-]?wear\b'), ' knitwear '),
    (re.compile(r'\bnight[\s\-]?wear\b'), ' nightwear '),
    (re.compile(r'\bnight[\s\-]?(gown|dress|ie|y)s?\b'), ' nightgown '),
    (re.compile(r'\bsleep[\s\-]?wear\b'), ' sleepwear '),
    (re.compile(r'\bunder[\s\-]?wear\b'), ' underwear '),
    (re.compile(r'\bfoot[\s\-]?wear\b'), ' footwear '),
    (re.compile(r'\bout[\s\-]?wear\b'), ' outerwear '),
    (re.compile(r'\bhigh[\s\-]?waist(ed)?\b'), ' highwaist '),
    (re.compile(r'\bhigh[\s\-]?heel(s|ed)?\b'), ' heel '),
    (re.compile(r'\boff[\s\-]?white\b'), ' offwhite '),
    (re.compile(r'\boff[\s\-]?(the[\s\-]?)?shoulder\b'), ' offshoulder '),
    (re.compile(r'\bplus[\s\-]?size(d)?\b'), ' plussize '),
    (re.compile(r'\bdenim[\s\-]?jeans\b'), ' jeans '),
    (re.compile(r'\bmonkey[\s\-]?caps?\b'), ' balaclava beanie '),
]


def fold_accents(text):
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')


def join_compounds(text):
    for pattern, replacement in COMPOUNDS:
        text = pattern.sub(replacement, text)
    return text


@lru_cache(maxsize=200_000)
def stem(word):
    """Light plural stemmer: shoes -> shoe, dresses -> dress, accessories -> accessory."""
    if len(word) <= 3 or word.isdigit():
        return word
    if word.endswith('sses'):
        return word[:-2]
    if word.endswith('ies') and len(word) > 4:
        return word[:-3] + 'y'
    if word.endswith(('ches', 'shes', 'xes')):
        return word[:-2]
    if word.endswith('s') and not word.endswith(('ss', 'us', 'is')):
        return word[:-1]
    return word


def raw_tokens(text):
    """Lowercased, accent-folded, compound-joined [a-z0-9]+ tokens (no stemming, no length filter)."""
    return TOKEN_RE.findall(join_compounds(fold_accents(text).lower()))


def tokenize(text):
    """Index/query tokens: raw tokens of length >= 2, stemmed."""
    return [stem(t) for t in raw_tokens(text) if len(t) >= 2]


NUMBER_K = re.compile(r'(?<!\by)(\d+(?:\.\d+)?)\s*k\b')    # "2k" -> 2000, but "y2k" is a style
NUMBER_COMMA = re.compile(r'(?<=\d),(?=\d{3}\b)')
CURRENCY = re.compile(r'₹|\brs\.?(?=\s|\d|$)|\brupees?\b|\binr\b|\binr\.')
WITH_ABBREVIATION = re.compile(r'\bw/(o\b)?')    # "top w/ matching bottoms", "dress w/o sleeves"
STRETCHED = re.compile(r'([a-z])\1{2,}')          # "hooooodie", "sooo cute": at most a double letter
TSHIRT_SLIP = re.compile(r'\bt[\s\-]?(?:shrit|shrt|sirt|shirtt|shrti|sjirt|shitr|shirrt)(s?)\b')   # "t-shrit"
# Texting "4" for "for" between words ("top 4 new yrs eve party", "shoes 4 men"); never a count,
# size or age ("pack of 4 socks", "size 4", "4 year old").
SMS_FOR = re.compile(r'\b([a-z]+) 4 (?=([a-z]+)\b)')
SMS_FOR_KEEP_BEFORE = {'size', 'of', 'pack', 'set', 'under', 'below', 'inr', 'rs', 'uk', 'us', 'eu', 'age', 'aged',
                       'only', 'just', 'buy', 'get', 'than', 'upto', 'to', 'and', 'or', 'about', 'around'}
SMS_FOR_KEEP_AFTER = ('year', 'yr', 'yo', 'month', 'mth', 'pack', 'pk', 'pair', 'pc', 'piece', 'inr', 'rs', 'rupee',
                      'thousand', 'hundred', 'item', 'set', 'people', 'person', 'day', 'week', 'times', 'colour', 'color')


def _sms_for(match):
    before, after = match.group(1), match.group(2)
    if before in SMS_FOR_KEEP_BEFORE or after.startswith(SMS_FOR_KEEP_AFTER) or after in ('k', 'x'):
        return match.group(0)
    return f'{before} for '


EMOJI_WORDS = {
    '👶': 'baby', '🍼': 'baby', '👧': 'girl', '👦': 'boy', '👩': 'women', '👨': 'men', '👗': 'dress', '👖': 'jeans',
    '👟': 'sneakers', '👞': 'shoes', '👠': 'heels', '👡': 'sandals', '👢': 'boots', '🥾': 'boots', '🧥': 'jacket',
    '👒': 'hat', '🎩': 'hat', '🧢': 'cap', '🕶': 'sunglasses', '👓': 'glasses', '👜': 'bag', '👛': 'purse', '🎒': 'backpack',
    '🧦': 'socks', '🧤': 'gloves', '🧣': 'scarf', '👔': 'tie', '👕': 'tshirt', '👘': 'robe', '👙': 'bikini', '🩳': 'shorts',
    '🩱': 'swimsuit', '🩲': 'underwear', '💍': 'ring', '📿': 'necklace', '⌚': 'watch', '☔': 'rain', '🌂': 'umbrella',
    '🐶': 'dog', '🐕': 'dog', '🎃': 'halloween', '🎄': 'christmas', '🏖': 'beach', '🏝': 'beach', '❄': 'winter', '⛄': 'winter',
    '☀': 'summer', '🌧': 'rain', '🌞': 'summer', '💪': 'gym', '🏋': 'gym', '🧘': 'yoga', '🏃': 'running', '💃': 'party',
    '🕺': 'party', '👰': 'wedding', '💒': 'wedding', '🎓': 'graduation', '🎁': 'gift', '🎂': 'birthday', '💼': 'office',
    '🏢': 'office', '✈': 'travel', '🧳': 'travel', '🤰': 'maternity', '🐱': 'cat', '🧸': 'toy', '🔥': '', '😍': '', '🥰': '',
    '😎': '', '✨': '', '💯': '', '🙏': '', '👍': '', '❤': '', '💕': '', '😊': '', '🙂': '', '😀': '', '😁': '', '🤗': '',
}
EMOJI_RE = re.compile('|'.join(re.escape(e) for e in sorted(EMOJI_WORDS, key=len, reverse=True)))


NUMBER_WORDS = {
    'a': 1, 'an': 1, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8,
    'nine': 9, 'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15,
    'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19, 'twenty': 20, 'thirty': 30, 'forty': 40,
    'fifty': 50, 'sixty': 60, 'seventy': 70, 'eighty': 80, 'ninety': 90,
    # Hindi, as written in Hinglish: "do hazaar", "dedh hazaar" (1.5 thousand), "paanch sau".
    'ek': 1, 'do': 2, 'teen': 3, 'char': 4, 'chaar': 4, 'paanch': 5, 'panch': 5, 'chhe': 6, 'saat': 7,
    'aath': 8, 'nau': 9, 'das': 10, 'dedh': 1.5, 'dhai': 2.5, 'dhaai': 2.5, 'adhai': 2.5,
}
NUMBER_SCALES = {'hundred': 100, 'sau': 100, 'thousand': 1000, 'grand': 1000, 'hazaar': 1000, 'hazar': 1000,
                 'hajar': 1000, 'hazaar': 1000}
# A run of number words or digits that contains a scale word ("two thousand five hundred", "3 grand",
# "a thousand", "dedh hazaar"); runs without one ("one shoulder", "two piece") are left alone.
NUMBER_RUN = re.compile(r'\b(?:(?:{w}|\d+(?:\.\d+)?|and)\s+)*(?:{w}|\d+(?:\.\d+)?)?\s*(?:{s})\b(?:\s+(?:and\s+)?(?:{w}|{s}|\d+)\b)*'.format(
    w='|'.join(sorted((k for k in NUMBER_WORDS if k not in ('a', 'an', 'do', 'das')), key=len, reverse=True))
      + r'|a|an|do|das',
    s='|'.join(sorted(NUMBER_SCALES, key=len, reverse=True))))


def _number_value(run):
    total, current, seen_scale = 0.0, 0.0, False
    for word in run.split():
        if word == 'and':
            continue
        if word in NUMBER_SCALES:
            scale = NUMBER_SCALES[word]
            seen_scale = True
            if scale == 100:
                current = (current or 1) * 100
            else:
                total += (current or 1) * scale
                current = 0.0
        elif word in NUMBER_WORDS:
            current += NUMBER_WORDS[word]
        else:
            try:
                current += float(word)
            except ValueError:
                return None
    value = total + current
    return int(round(value)) if seen_scale and value > 0 else None


GRAND_FAMILY = re.compile(r'\s*(?:daughters?|sons?|child(?:ren)?|kids?|ma|pa|mothers?|fathers?|parents?|dad|mum|mom'
                          r'|opening|prix|piano|finale?)\b')


def numbers_from_words(text):
    """'under two thousand' -> 'under 2000', 'fifteen hundred' -> '1500', 'a grand' -> '1000'."""
    def replace(m):
        run = m.group(0)
        words = run.split()
        if words[-1] == 'grand' and (len(words) < 2 or GRAND_FAMILY.match(m.string, m.end())):
            return run           # "grand daughter", "grand opening": not a thousand
        value = _number_value(run)
        return f' {value} ' if value is not None else run
    return NUMBER_RUN.sub(replace, text)


def normalize_query(query):
    """Lowercase, fold accents, currency to " inr ", "2,000" -> "2000", "2k" -> "2000"."""
    text = query.replace('₹', ' inr ')
    text = EMOJI_RE.sub(lambda m: ' ' + EMOJI_WORDS[m.group(0)] + ' ', text)
    text = fold_accents(text).lower()
    text = STRETCHED.sub(r'\1\1', text)
    text = TSHIRT_SLIP.sub(r'tshirt\1', text)
    text = WITH_ABBREVIATION.sub(lambda m: ' without ' if m.group(1) else ' with ', text)
    text = CURRENCY.sub(' inr ', text)
    text = NUMBER_COMMA.sub('', text)
    text = numbers_from_words(text)
    text = NUMBER_K.sub(lambda m: str(int(float(m.group(1)) * 1000)), text)
    text = re.sub(r'(?<=\d)(inr)\b', r' \1', text)
    text = re.sub(r'\binr(?=\d)', 'inr ', text)
    text = SMS_FOR.sub(_sms_for, text)
    return ' '.join(text.split())
