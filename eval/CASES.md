# Local benchmark cases

A local copy of the organizer's hidden functional suite, in the same proportions. Tune on
`eval/cases.jsonl`; leave `eval/holdout.jsonl` alone until attempts 3 to 5 (CLAUDE.md section 17).
Run them with `python eval/run_eval.py --url URL [--holdout]`.

| Category | Official | cases.jsonl | holdout.jsonl | Total |
|---|---|---|---|---|
| validity (of which invalid calls) | 30 (10) | 34 (11) | 13 (4) | 47 (15) |
| ranking | 25 | 28 | 12 | 40 |
| unusual | 10 | 11 | 5 | 16 |
| irrelevant / impossible | 10 | 11 | 5 | 16 |
| mixed | 15 | 17 | 7 | 24 |
| recs | 10 | 11 | 6 | 17 |
| **total** | 100 | 112 | 48 | 160 |

Ids are unique across both files. Holdout queries are not paraphrases of main queries.

## Line format

`{"id", "category", "args", "expect"}`. `args` holds the tool arguments exactly as the organizer would send them,
including integral floats, whitespace and a 1,001-character query. `expect` fields:

- `error: true`: an invalid call that must produce an MCP error.
- `empty`: expected presence (`false`) or absence (`true`) of results.
- `exact: true`: an empty-query page that must equal `eval/reference.py`. Coerce integral floats to int before
  calling the oracle, because the untouched starter crashes on `limit: 100.0`.
- `probe: true`: run the repeat probe and the larger-prefix probe. Probe cases keep `offset + limit < 100`.
- `constraints`: hard requirements stated in the query that every primary result must meet. Explicit
  `args.filters` are checked separately and are never repeated here. Only these keys are used: `colors`,
  `exclude_colors`, `min_price`, `max_price`, `audience_group`, `exclude_patterns` and `exclude_product_types`.
  All of them are in run_eval's vocabulary.
  - **Colours** map a shopper word to its catalog family (CLAUDE.md section 5). For example, white is
    White + Off White, navy is Dark Blue, brown and tan are Yellowish Brown + Dark Beige, cream is Off White +
    Light Beige, charcoal is Dark Grey, and olive and khaki are Greenish Khaki. Two-word catalog colours are
    exact.
  - **Audience groups** use the CLAUDE.md section 5 masks.
- `choices` / `choice_constraints`: the mixed-case labels, with one constraint object per choice (`{}` means
  none). `constraints` still holds requirements that apply to every choice, such as a shared audience or a
  budget with "each".
- `complement`: the accompanying item the query asks for. `null` means implicit or unspecified (rec-02,
  rec-12). `"none"` means no complement is expected (rec-09, where recommendations are disabled), so an empty
  list earns full credit.
- `rec_constraints`: hard requirements on recommendations. These are complement budgets and colours, plus the
  stated audience, which carries over. They never include `args.filters`.
- `notes`: why the case exists, plus its grounding count.

## Grounding

Grounding was run against `data/products.json` by scripts kept outside the repo. A case's count is the number
of products that meet its declared hard constraints and filters and that also match an intent predicate (an
explicit product_type plus a name or description pattern). Every count is recorded in `notes`. The thresholds
are:

- at least 10 for each ranking case, unusual case and recs primary;
- at least 5 for each mixed choice and each recs complement;
- exactly 0 for impossible cases, whose notes also give the count that exists without the constraint.

Unrelated requests are not grounded. Their notes list catalog look-alikes instead.

Validation checks that:
- both files parse;
- ids are unique;
- category and invalid counts are as above;
- every non-error `args` passes `schemas/input.schema.json`, with no blank string filter and no
  `min_price > max_price`;
- every error case fails those checks;
- the oracle agrees with `empty` on exact cases;
- `run_eval.load_cases` loads both files with 0 warnings.

## Phenomena covered

- **Budgets in prose**: under 2500, `₹999 max`, `below INR 1,500`, `between 1000 and 2000`, `under 1,500`,
  `2k`, `2000 se kam`, `each` (mix-01, mix-21), per-choice budgets (mix-04, mix-24), complement-only budgets
  (rec-01, rec-06, rec-13, rec-16).
- **Colours missing from the catalog**: navy, brown, tan, burgundy, cream, charcoal, olive, khaki, gray.
  Negation: `not white` (rank-08), `without lace` (rank-10), `not pink` across the whole family (rank-38).
- **British/American words**: trainers/sneakers, jumper/sweater, trousers/pants, wellies, dressing gown, midi
  (calf-length in the catalog).
- **Audience phrases**: for her, men's, for my son/daughter/husband/wife/boyfriend, baby girl, newborn, toddler,
  8 year old, i'm a guy. Special sections: maternity (rank-25), plus size (rank-26).
- **Product range**: sport (sports bra, yoga, swim, football, gym), accessories, footwear and kids/baby items.
  Non-clothing items: water bottle, umbrella, dog jumper, lip balm, phone case.
- **Unusual queries**:
  - vague occasions: lazy sunday, job interview, beach wedding, bike commute, cold feet;
  - typos: unu-05, unu-06, unu-13;
  - other languages: Spanish, French, German, Hindi (Devanagari) and Hinglish (unu-09, mix-06, mix-19);
  - a 720-character rambling query, a gift query, ALL CAPS + emoji, and an unsupported size claim.
- **Irrelevant requests**:
  - unrelated: spaceship, bread, laptop, tyres, iPhone, pizza;
  - noise: gibberish, `!!!`, two prompt injections;
  - price-floor/ceiling impossibles: irr-04, irr-05, irr-08, irr-14;
  - impossible through an explicit filter: irr-11, irr-16.
- **Mixed queries**:
  - separators: and, `,`, `&`, plus, `/`, or, aur;
  - three-way requests: mix-03, mix-09, mix-23;
  - a global filter (mix-01, which is the brief's call verbatim, and mix-05), shared and per-choice audiences,
    and per-choice colours;
  - contrast pairs with recs: mix-02/rec-01 and mix-08/rec-11 ("X and Y" vs "X with matching Y").
- **Recommendations**:
  - the brief's call verbatim (rec-01) and implicit or unspecified complements;
  - markers: with matching, to go with, to wear with, to match, to pair it with;
  - complement alternatives (rec-05);
  - filters that must not reach recommendations (rec-08 colour, rec-17 audience);
  - recommendations disabled (rec-09).
- **Validity**:
  - all 10 `sort_by x sort_order` pairs on empty queries;
  - case and whitespace variants of filter values;
  - the `Dog Wear` / `Dog wear` collision and an unknown value (`Navy`);
  - inclusive bounds at 199, 8999 and 9000;
  - null ratings under `min_rating` 1 and 5 and under rating sorts;
  - `min_review_count` 0;
  - offsets near the end, at the end and at 10000;
  - limit 1 and 100, and integral floats;
  - a whitespace-only query, and an empty query with recommendations;
  - 6 nonempty-query field sorts with probes, including filters beating conflicting query wording (val-20);
  - 15 invalid calls.

## Catalog traps found while grounding

- **Label noise makes product_type price bands unsafe for abstention.** CLAUDE.md section 2 calls "dress
  under 900" impossible, but it is not, so irr-14 uses "under 300" instead. These cases are answerable even
  though the type's price band says otherwise:

  | Product | Typed as | Price (INR) |
  |---|---|---|
  | Real dresses | Blouse, Shirt, Top, Vest top | 499-899 |
  | Padded parkas and jackets | Hoodie | 599-1799 |
  | "Jackie Coat" (wool) | Top | 1899 |
  | Gold hoop earrings | Other accessories | 1399, 2599 |
  | Sock packs | Bag, Unknown | 2399-2999 |
  | Ankle boots | Wedge, Heeled sandals | 699-799 |
  | Smartphone cases | Other accessories | 499-999 (Mobile case starts at 2499; rank-17) |

- **Look-alikes that must not be returned**:
  - "Italian Earring coat" (irr-08);
  - sneaker socks (irr-05);
  - "Car Shoe" moccasins (irr-10);
  - "Pizza Costume" (irr-13);
  - laptop sleeves and USB cables (irr-03);
  - cases that only fit iPhone 5-8 (irr-12);
  - aqua swim shoes (irr-16);
  - dogtooth jumpers (rank-15);
  - women's "Boyfriend" jeans (unu-13).
- **No running shoes exist**: "running shoes" means trainers with "runner" in the name (rec-03).
