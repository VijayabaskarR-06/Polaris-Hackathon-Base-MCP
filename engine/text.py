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


def normalize_query(query):
    """Lowercase, fold accents, currency to " inr ", "2,000" -> "2000", "2k" -> "2000"."""
    text = query.replace('₹', ' inr ')
    text = EMOJI_RE.sub(lambda m: ' ' + EMOJI_WORDS[m.group(0)] + ' ', text)
    text = fold_accents(text).lower()
    text = WITH_ABBREVIATION.sub(lambda m: ' without ' if m.group(1) else ' with ', text)
    text = CURRENCY.sub(' inr ', text)
    text = NUMBER_COMMA.sub('', text)
    text = NUMBER_K.sub(lambda m: str(int(float(m.group(1)) * 1000)), text)
    text = re.sub(r'(?<=\d)(inr)\b', r' \1', text)
    text = re.sub(r'\binr(?=\d)', 'inr ', text)
    return ' '.join(text.split())
