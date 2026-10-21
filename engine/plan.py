"""Search plans: what the shopper wants, produced by the rule planner or the LLM planner.

Hard fields (colors, exclusions, prices, audience_group) filter products; soft
fields only shape retrieval and ranking.
"""
from dataclasses import asdict, dataclass, field


@dataclass
class Choice:
    label: str
    product_types: list = field(default_factory=list)     # strong preference, not a hard filter
    keywords: list = field(default_factory=list)          # catalog wording, for BM25 and name hits
    colors: list = field(default_factory=list)            # HARD (strict-first family)
    exclude_colors: list = field(default_factory=list)    # HARD
    patterns: list = field(default_factory=list)          # soft boost
    exclude_patterns: list = field(default_factory=list)  # HARD
    min_price: float = None                               # HARD
    max_price: float = None                               # HARD
    soft_sections: list = field(default_factory=list)
    soft_words: list = field(default_factory=list)
    price_preference: str = None                          # 'low' | 'high' | None
    rating_preference: bool = False
    exclude_words: list = field(default_factory=list)     # HARD: "no lace", "without sequins"

    def infeasible_price(self):
        return (self.min_price is not None and self.max_price is not None
                and self.min_price > self.max_price)


@dataclass
class Plan:
    kind: str = 'search'                                  # 'search' | 'nonsense'
    audience_group: str = None                            # HARD when set
    choices: list = field(default_factory=list)           # 1..4 Choice
    complement: Choice = None                             # accompanying item, recommendations only
    exclude_baby: bool = False                            # HARD: a child too old for baby sizes
    source: str = 'rules'                                 # 'llm' | 'rules' | 'merged'
    notes: list = field(default_factory=list)             # diagnostics only

    @property
    def mixed(self):
        return len(self.choices) > 1

    def to_dict(self):
        return asdict(self)
