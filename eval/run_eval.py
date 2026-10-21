"""Local exam simulator: python eval/run_eval.py --url URL [options].

Runs eval cases against an MCP endpoint with the real MCP client (one session,
sequential calls, 10 s per call) and scores them as hackathon-brief.md says:

* Every valid call gets four binary checks: (1) structure: schema, length,
  unique and known ids, recommendation references; (2) hard constraints:
  explicit filters and expect.constraints on primary results,
  expect.rec_constraints on recommendations; (3) sorting: the exact oracle page
  for empty queries, field order for other sorts, plus repeat and
  larger-prefix probes; (4) presence or absence against expect.empty. Invalid
  calls pass on isError or a JSON-RPC error. A timeout, transport error or
  malformed response zeroes the case, and a failed check (1) zeroes any
  semantic case.
* validity scores the mean of the checks, irrelevant scores correct-empty, and
  ranking, unusual, mixed and recs are judged over fact cards and scored with
  eval/metrics.py.

Modes:
  --judge none               deterministic checks only (default)
  --judge packet             also write OUT/packet.json for an external grader
  --grades FILE --out OUT    score OUT/results.json with a grades file
  --judge bedrock|anthropic  grade with eval/local_judge.py, then score
                             (without --url: grade an existing OUT/packet.json)

Outputs go to --out (default eval/runs/<UTC timestamp>/): results.json,
packet.json, grades.json and score.json.
"""
import argparse
import asyncio
import fnmatch
import json
import math
import statistics
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import anyio  # noqa: E402
from jsonschema import Draft202012Validator  # noqa: E402
from mcp import ClientSession  # noqa: E402
from mcp.client.streamable_http import streamable_http_client  # noqa: E402
from mcp.shared.exceptions import McpError  # noqa: E402
from pydantic import ValidationError  # noqa: E402

from eval import local_judge, metrics, reference  # noqa: E402

TOOL = 'search_products'
PROTOCOL = '2025-11-25'
CALL_TIMEOUT = 10.0
PROBE_CASE_TIMEOUT = 30.0
DISCOVERY_TIMEOUT = 30.0
MAX_LIMIT = 100
SEMANTIC = ('ranking', 'unusual', 'mixed', 'recs')
ERROR_KINDS = ('tool_error', 'rpc_error')
CHECKS = ('structure', 'constraints', 'sorting', 'presence')
DEFAULT_CASES = ROOT / 'eval/cases.jsonl'
HOLDOUT_CASES = ROOT / 'eval/holdout.jsonl'
RUNS_DIR = ROOT / 'eval/runs'
CATALOG_PATH = ROOT / 'data/products.json'
OUTPUT_VALIDATOR = Draft202012Validator(reference.OUTPUT_SCHEMA)
SCHEMA_ANNOTATIONS = ('title', 'description', '$comment', 'examples')

EXPECT_KEYS = {'error', 'empty', 'probe', 'exact', 'constraints', 'choices', 'choice_constraints',
               'complement', 'rec_constraints', 'notes'}
# Constraint key -> (catalog field, True if the value must be listed, False if it must not be).
LIST_CONSTRAINTS = {
    'colors': ('color', True), 'exclude_colors': ('color', False),
    'product_types': ('product_type', True), 'exclude_product_types': ('product_type', False),
    'patterns': ('pattern', True), 'exclude_patterns': ('pattern', False),
    'categories': ('category', True), 'audiences': ('audience', True), 'sections': ('section', True),
}
CONSTRAINT_KEYS = set(LIST_CONSTRAINTS) | {'min_price', 'max_price', 'audience_group'}
AUDIENCE_GROUPS = ('women', 'men', 'kids', 'girls', 'boys', 'baby')
WOMEN_AUDIENCES = {'Ladieswear', 'Ladies Accessories', 'Lingeries/Tights', 'Divided'}
KIDS_AUDIENCES = {'Children Sizes 92-140', 'Children Sizes 134-170', 'Children Accessories, Swimwear',
                  'Baby Sizes 50-98'}


# ---------------------------------------------------------------- cases

def constraint_problems(constraints, where):
    if constraints is None:
        return []
    if not isinstance(constraints, dict):
        return [f'{where} must be an object']
    problems = []
    for key, value in constraints.items():
        if key not in CONSTRAINT_KEYS:
            problems.append(f'{where}: unknown constraint {key!r} (known: {sorted(CONSTRAINT_KEYS)})')
        elif key in LIST_CONSTRAINTS:
            if not (isinstance(value, list) and all(isinstance(v, str) for v in value)):
                problems.append(f'{where}.{key} must be a list of strings')
        elif key == 'audience_group':
            if value is not None and value not in AUDIENCE_GROUPS:
                problems.append(f'{where}.audience_group must be one of {AUDIENCE_GROUPS}')
        elif value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
            problems.append(f'{where}.{key} must be a number')
    return problems


def case_problems(case):
    """Format errors in one case; an empty list means it is well formed."""
    if not isinstance(case, dict):
        return ['a case must be a JSON object']
    problems = []
    if not isinstance(case.get('id'), str) or not case['id']:
        problems.append('id must be a non-empty string')
    category = case.get('category')
    if category not in metrics.WEIGHTS:
        problems.append(f'category must be one of {list(metrics.WEIGHTS)}')
    if not isinstance(case.get('args'), dict):
        problems.append('args must be an object')
    expect = case.get('expect', {})
    if not isinstance(expect, dict):
        return problems + ['expect must be an object']
    problems += constraint_problems(expect.get('constraints'), 'constraints')
    problems += constraint_problems(expect.get('rec_constraints'), 'rec_constraints')
    choices = expect.get('choices')
    if choices is not None and not (isinstance(choices, list)
                                    and all(isinstance(c, str) and c.strip() for c in choices)):
        problems.append('choices must be a list of non-empty strings')
        choices = None
    if category == 'mixed' and not choices:
        problems.append('a mixed case needs expect.choices')
    choice_constraints = expect.get('choice_constraints')
    if choice_constraints is not None:
        if not isinstance(choice_constraints, list) or len(choice_constraints) != len(choices or []):
            problems.append('choice_constraints needs exactly one object per choice')
        else:
            for i, constraints in enumerate(choice_constraints):
                problems += constraint_problems(constraints, f'choice_constraints[{i}]')
    if expect.get('complement') is not None and not isinstance(expect['complement'], str):
        problems.append('complement must be a string or null')
    if category == 'irrelevant' and expect.get('empty') is False:
        problems.append('an irrelevant case expects an empty response')
    return problems


def load_cases(path):
    """Cases from a JSON-lines file (blank and # lines skipped); exits listing every format error."""
    cases, errors = [], []
    for number, line in enumerate(Path(path).read_text(encoding='utf-8').splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        try:
            case = json.loads(line)
        except ValueError as exc:
            errors.append(f'line {number}: invalid JSON ({exc})')
            continue
        name = case.get('id') if isinstance(case, dict) else None
        errors += [f'line {number} ({name}): {problem}' for problem in case_problems(case)]
        cases.append(case)
    counts = Counter(c.get('id') for c in cases if isinstance(c, dict))
    errors += [f'duplicate case id {cid!r}' for cid, n in counts.items() if n > 1]
    if errors:
        raise SystemExit(f'{path}: invalid case file\n  ' + '\n  '.join(errors))
    return cases


def select_cases(cases, only):
    """Cases named by --only: comma-separated categories or id patterns (fnmatch, e.g. rank-*)."""
    if not only:
        return cases
    patterns = [p.strip() for p in only.split(',') if p.strip()]
    chosen = [c for c in cases if any(c['category'] == p or fnmatch.fnmatchcase(c['id'], p)
                                      for p in patterns)]
    if not chosen:
        raise SystemExit(f'--only {only!r} matches no case')
    return chosen


def case_warnings(case, catalog):
    """Suspicious but legal case content: unknown expect keys, values the catalog never uses."""
    expect = case.get('expect') or {}
    warnings = [f'{case["id"]}: unknown expect key {key!r}' for key in expect if key not in EXPECT_KEYS]
    if expect.get('exact') and not is_empty_query(case['args']):
        warnings.append(f'{case["id"]}: exact only applies to empty queries; field order is checked instead')
    groups = [('constraints', expect.get('constraints')), ('rec_constraints', expect.get('rec_constraints'))]
    groups += [(f'choice_constraints[{i}]', c) for i, c in enumerate(expect.get('choice_constraints') or [])]
    for where, constraints in groups:
        for key, values in (constraints or {}).items():
            if key in LIST_CONSTRAINTS and values:
                field = LIST_CONSTRAINTS[key][0]
                unknown = [v for v in values if fold(v) not in catalog.values[field]]
                if unknown:
                    warnings.append(f'{case["id"]}: {where}.{key} values not in the catalog: {unknown}')
    return warnings


# ---------------------------------------------------------------- catalog and constraints

def fold(value):
    """The contract's string comparison: outer whitespace stripped, case-insensitive."""
    return str(value or '').strip().casefold()


class Catalog:
    """The frozen catalog, loaded once: facts by id plus the starter oracle for exact pages."""

    def __init__(self, products):
        self.oracle = reference.Catalog(products)
        self.by_id = {p['id']: p for p in self.oracle.products}
        self.values = {field: {fold(p.get(field)) for p in self.oracle.products}
                       for field in reference.STRING_FILTERS}

    @classmethod
    def load(cls, path=CATALOG_PATH):
        return cls(json.loads(Path(path).read_text(encoding='utf-8')))


def in_audience_group(product, group):
    """CLAUDE.md section 5 audience-group masks over audience and section."""
    audience, section = product.get('audience') or '', product.get('section') or ''
    kids = audience in KIDS_AUDIENCES or section == 'Kids Sports'
    if group == 'women':
        return audience in WOMEN_AUDIENCES or section == 'Ladies H&M Sport'
    if group == 'men':
        return audience == 'Menswear' or section == 'Men H&M Sport'
    if group == 'kids':
        return kids
    if group == 'girls':
        return kids and 'Boy' not in section
    if group == 'boys':
        return kids and 'Girl' not in section
    if group == 'baby':
        return audience == 'Baby Sizes 50-98' or section.startswith('Baby')
    raise ValueError(f'unknown audience group {group!r}')


def violations(product, constraints):
    """Readable reasons a product breaks a constraint set; empty when it complies."""
    found = []
    price = product['price']['amount']
    for key, value in (constraints or {}).items():
        if value is None or value == []:
            continue
        if key in LIST_CONSTRAINTS:
            field, inside = LIST_CONSTRAINTS[key]
            if (fold(product.get(field)) in {fold(v) for v in value}) != inside:
                relation = 'not in' if inside else 'is excluded by'
                found.append(f'{field} {product.get(field)!r} {relation} {value}')
        elif key == 'min_price' and price < value:
            found.append(f'price {price} below min {value:g}')
        elif key == 'max_price' and price > value:
            found.append(f'price {price} above max {value:g}')
        elif key == 'audience_group' and not in_audience_group(product, value):
            found.append(f'audience {product.get("audience")!r} / section {product.get("section")!r} '
                         f'is not {value}')
    return found


def primary_violations(product, args, expect):
    """Explicit filters (exactly as the oracle applies them) plus expect.constraints."""
    found = []
    filters = args.get('filters') or {}
    if filters and not reference.Catalog.matches(product, filters):
        found.append(f'fails explicit filters {filters}')
    return found + violations(product, expect.get('constraints'))


def facts(product):
    return {'id': product['id'], 'name': product['name'], 'type': product['product_type'],
            'color': product['color'], 'price': product['price']['amount']}


def describe(output, args, expect, catalog):
    """Per-product summaries with constraint flags for the results and recommendations of a page."""
    choice_constraints = expect.get('choice_constraints') or []
    results = []
    for item in output['results']:
        product = catalog.by_id.get(item['id'])
        if product is None:
            results.append({'id': item['id'], 'ok': False, 'violations': ['unknown id'],
                            'choice_ok': [False] * len(choice_constraints)})
            continue
        found = primary_violations(product, args, expect)
        per_choice = [violations(product, c) for c in choice_constraints]
        if per_choice and all(per_choice):
            found.append('fits no requested choice: ' + ' / '.join('; '.join(v) for v in per_choice))
        results.append(dict(facts(product), ok=not found, violations=found,
                            choice_ok=[not v for v in per_choice]))
    recs = []
    for item in output['recommendations']:
        product = catalog.by_id.get(item['id'])
        if product is None:
            recs.append({'id': item['id'], 'for_product_id': item['for_product_id'], 'ok': False,
                         'violations': ['unknown id']})
            continue
        found = violations(product, expect.get('rec_constraints'))
        recs.append(dict(facts(product), for_product_id=item['for_product_id'], reason=item['reason'],
                         ok=not found, violations=found))
    return results, recs


# ---------------------------------------------------------------- deterministic checks

def duplicates(values):
    return sorted(v for v, n in Counter(values).items() if n > 1)


def schema_problems(output):
    return [f'schema: {e.message}' for e in list(OUTPUT_VALIDATOR.iter_errors(output))[:3]]


def structure_problems(output, args, known):
    """Check (1) on a schema-valid response: length, unique and known ids, recommendation references."""
    results = [r['id'] for r in output['results']]
    recs = output['recommendations']
    rec_ids = [r['id'] for r in recs]
    on_page = set(results)
    problems = []
    limit = int(args.get('limit', 10))
    if len(results) > limit:
        problems.append(f'{len(results)} results exceed limit {limit}')
    if duplicates(results):
        problems.append(f'duplicate result ids {duplicates(results)[:3]}')
    unknown = [pid for pid in results + rec_ids if pid not in known]
    if unknown:
        problems.append(f'ids not in the catalog {unknown[:3]}')
    if recs and not args.get('include_recommendations', False):
        problems.append('recommendations returned although include_recommendations is false')
    if recs and not results:
        problems.append('recommendations without primary results')
    if duplicates(rec_ids):
        problems.append(f'duplicate recommendation ids {duplicates(rec_ids)[:3]}')
    repeated = [pid for pid in rec_ids if pid in on_page]
    if repeated:
        problems.append(f'recommendations repeat primary results {repeated[:3]}')
    orphans = [r['for_product_id'] for r in recs if r['for_product_id'] not in on_page]
    if orphans:
        problems.append(f'for_product_id not among this page\'s results {orphans[:3]}')
    return problems


def is_empty_query(args):
    """Only a whitespace-only query is empty; punctuation is a real query."""
    return not str(args.get('query', '')).strip()


def oracle_args(args):
    """The starter oracle slices with offset and limit as given, so integral floats become ints."""
    fixed = dict(args)
    for key in ('offset', 'limit'):
        if key in fixed:
            fixed[key] = int(fixed[key])
    filters = fixed.get('filters')
    if isinstance(filters, dict) and 'min_review_count' in filters:
        fixed['filters'] = dict(filters, min_review_count=int(filters['min_review_count']))
    return fixed


def field_value(product, sort_by):
    if sort_by == 'price':
        return product['price']['amount']
    if sort_by == 'rating':
        return product['rating']['average']
    if sort_by == 'review_count':
        return product['rating']['count']
    return product['id']


def precedes(a, b, sort_by, descending):
    """Whether product a may come directly before b: field order, id-ascending ties, null ratings last."""
    if sort_by == 'id':
        return a['id'] > b['id'] if descending else a['id'] < b['id']
    va, vb = field_value(a, sort_by), field_value(b, sort_by)
    if va is None or vb is None:  # only ratings are nullable
        return vb is None and (va is not None or a['id'] < b['id'])
    if va != vb:
        return va > vb if descending else va < vb
    return a['id'] < b['id']


def order_problems(ids, sort_by, sort_order, by_id):
    """The first adjacent pair that breaks the contract's field order, if any."""
    descending = sort_order == 'desc'
    for i in range(len(ids) - 1):
        a, b = by_id.get(ids[i]), by_id.get(ids[i + 1])
        if a is not None and b is not None and not precedes(a, b, sort_by, descending):
            return [f'{sort_by} {sort_order} order broken at ranks {i + 1}-{i + 2}: {a["id"]} '
                    f'({field_value(a, sort_by)}) before {b["id"]} ({field_value(b, sort_by)})']
    return []


def prefix_request(args):
    """A larger request containing this page: (its args, the page's position in it), or (None, None)."""
    offset, limit = int(args.get('offset', 0)), int(args.get('limit', 10))
    if offset + limit <= MAX_LIMIT and (offset > 0 or limit < MAX_LIMIT):
        return dict(args, offset=0, limit=min(MAX_LIMIT, offset + 2 * limit)), offset
    if limit < MAX_LIMIT:
        return dict(args, limit=min(MAX_LIMIT, 2 * limit)), 0
    return None, None


def result_ids(output):
    return [r['id'] for r in output['results']]


def first_difference(got, want):
    for i, (a, b) in enumerate(zip(got, want)):
        if a != b:
            return f'rank {i + 1}: got {a}, expected {b}'
    return f'got {len(got)} ids, expected {len(want)}'


def probe_problems(calls, shift, limit):
    """A repeated request must be identical; a larger request must contain this exact page."""
    problems = []
    main = calls['main']['output']
    if 'repeat' in calls and calls['repeat']['output'] != main:
        problems.append('repeated identical request returned a different response')
    if 'prefix' in calls:
        prefix = calls['prefix']
        page = result_ids(main)
        window = result_ids(prefix['output'])[shift:shift + limit]
        if window != page:
            problems.append(f'larger request (offset {prefix["args"].get("offset", 0)}, limit '
                            f'{prefix["args"]["limit"]}) disagrees with this page: '
                            f'{first_difference(window, page)}')
    return problems


def sorting_problems(args, expect, calls, shift, catalog):
    """Check (3): exact oracle page for empty queries, field order otherwise, then the probes."""
    output = calls['main']['output']
    problems = []
    if is_empty_query(args):
        try:
            want = catalog.oracle.search(oracle_args(args))
        except ValueError as exc:
            return [f'the oracle rejects these arguments ({exc}); is the case really valid?']
        if output != want:
            got_ids, want_ids = result_ids(output), result_ids(want)
            detail = (first_difference(got_ids, want_ids) if got_ids != want_ids
                      else 'recommendations should be empty')
            problems.append(f'empty-query page differs from the catalog oracle: {detail}')
    elif args.get('sort_by', 'relevance') != 'relevance':
        problems += order_problems(result_ids(output), args['sort_by'], args.get('sort_order', 'desc'),
                                   catalog.by_id)
    if expect.get('probe'):
        problems += probe_problems(calls, shift, int(args.get('limit', 10)))
    return problems


def presence_problems(output, expect, category):
    """Check (4): results present or absent as expect.empty says (irrelevant cases expect empty)."""
    want_empty = expect.get('empty', True if category == 'irrelevant' else None)
    if want_empty is None or (not output['results']) == bool(want_empty):
        return []
    if want_empty:
        return [f'expected no results, got {len(output["results"])}']
    return ['expected results, got none']


def blank_record(case, calls, shift, case_ms):
    return {'id': case['id'], 'category': case['category'], 'args': case['args'],
            'expect': case.get('expect') or {}, 'calls': calls, 'prefix_shift': shift,
            'case_ms': round(case_ms, 1), 'checks': None, 'problems': [], 'validity_score': 0.0,
            'structural_ok': False, 'correct_empty': False, 'results': [], 'recommendations': []}


def evaluate(case, calls, shift, case_ms, catalog):
    """Every deterministic check for one executed case, as a JSON-ready record."""
    record = blank_record(case, calls, shift, case_ms)
    args, expect, category = case['args'], record['expect'], case['category']
    main = calls['main']
    if expect.get('error'):
        got_error = main['kind'] in ERROR_KINDS
        record['validity_score'] = metrics.invalid_case_score(got_error)
        if not got_error:
            record['problems'] = [f'expected an MCP error, got {main["kind"]}']
        return record
    failed = [f'{name}: {c["kind"]}' + (f' ({c["error"]})' if c['error'] else '')
              for name, c in calls.items() if c['kind'] != 'result']
    if expect.get('probe') and case_ms > PROBE_CASE_TIMEOUT * 1000:
        failed.append(f'case took {case_ms / 1000:.1f} s, over the {PROBE_CASE_TIMEOUT:g} s probe deadline')
    failed += [f'{name}: malformed, {p}' for name, c in calls.items() if c['kind'] == 'result'
               for p in schema_problems(c['output'])]
    if failed:
        record['problems'] = failed
        return record

    output = main['output']
    record['results'], record['recommendations'] = describe(output, args, expect, catalog)
    problems = {name: [] for name in CHECKS}
    for name, c in calls.items():
        problems['structure'] += [f'{name}: {p}' for p in structure_problems(c['output'], c['args'],
                                                                            catalog.by_id)]
        if not c['text_ok']:
            problems['structure'].append(f'{name}: text block missing or differs from structuredContent')
    problems['constraints'] = ([f'result {r["id"]}: {v}' for r in record['results'] for v in r['violations']]
                               + [f'recommendation {r["id"]}: {v}' for r in record['recommendations']
                                  for v in r['violations']])
    problems['sorting'] = sorting_problems(args, expect, calls, shift, catalog)
    problems['presence'] = presence_problems(output, expect, category)

    checks = {name: not problems[name] for name in CHECKS}
    record.update(checks=checks, validity_score=metrics.validity_case_score(*checks.values()),
                  problems=[f'{name}: {p}' for name in CHECKS for p in problems[name]],
                  structural_ok=checks['structure'],
                  correct_empty=checks['structure'] and not output['results'] and not output['recommendations'])
    return record


# ---------------------------------------------------------------- MCP client

def schema_shape(schema, in_properties=False):
    """A schema without prose annotations: the brief lets descriptions differ."""
    if isinstance(schema, dict):
        return {k: schema_shape(v, k == 'properties' and not in_properties) for k, v in schema.items()
                if in_properties or k not in SCHEMA_ANNOTATIONS}
    if isinstance(schema, list):
        return [schema_shape(v) for v in schema]
    return schema


async def discover(session):
    """initialize + tools/list, checked like the organizer's discovery step."""
    start = time.perf_counter()
    init = await session.initialize()
    listed = await session.list_tools()
    problems = []
    if init.protocolVersion != PROTOCOL:
        problems.append(f'protocol {init.protocolVersion}, expected {PROTOCOL}')
    names = [tool.name for tool in listed.tools]
    if names != [TOOL]:
        problems.append(f'tools/list returned {names}, expected exactly [{TOOL!r}]')
    tool = next((t for t in listed.tools if t.name == TOOL), None)
    if tool is not None:
        if not tool.description:
            problems.append('the tool has no description')
        if schema_shape(tool.inputSchema) != schema_shape(reference.INPUT_SCHEMA):
            problems.append('inputSchema differs from schemas/input.schema.json')
        output_shape = None if tool.outputSchema is None else schema_shape(tool.outputSchema)
        if output_shape != schema_shape(reference.OUTPUT_SCHEMA):
            problems.append('outputSchema differs from schemas/output.schema.json')
    return {'ok': not problems, 'problems': problems, 'protocol': init.protocolVersion,
            'server': f'{init.serverInfo.name} {init.serverInfo.version}',
            'ms': round((time.perf_counter() - start) * 1000, 1)}


def text_of(result):
    return next((block.text for block in result.content if getattr(block, 'type', None) == 'text'), '')


def text_matches(result):
    """The success contract: the same object serialized in a text content block."""
    try:
        return json.loads(text_of(result)) == result.structuredContent
    except ValueError:
        return False


async def call(session, args, timeout):
    """One tools/call as {'kind', 'ms', 'args', 'output', 'text_ok', 'error'}.

    kind is result, tool_error (isError), rpc_error (JSON-RPC error), timeout or
    malformed (the SDK rejected the result). Anything else means the session is
    broken; it propagates to run_cases, which reconnects.
    """
    record = {'kind': 'timeout', 'ms': 0.0, 'args': args, 'output': None, 'text_ok': False, 'error': None}
    if timeout <= 0:
        record['error'] = 'case deadline used up before this call'
        return record
    start = time.perf_counter()
    try:
        with anyio.fail_after(timeout):
            result = await session.call_tool(TOOL, args)
    except TimeoutError:
        record['error'] = f'no response within {timeout:.3g} s'
    except McpError as exc:
        record.update(kind='rpc_error', error=exc.error.message)
    except ValidationError as exc:
        record.update(kind='malformed', error=str(exc)[:300])
    except RuntimeError as exc:
        if 'structured content' not in str(exc) and 'Invalid schema' not in str(exc):
            raise
        record.update(kind='malformed', error=str(exc)[:300])
    else:
        if result.isError:
            record.update(kind='tool_error', error=text_of(result)[:300])
        else:
            record.update(kind='result', output=result.structuredContent, text_ok=text_matches(result))
    record['ms'] = round((time.perf_counter() - start) * 1000, 1)
    return record


async def run_case(session, case, catalog, timeout):
    """The case's call, plus the repeat and larger-prefix probes when expect.probe is set."""
    args, expect = case['args'], case.get('expect') or {}
    start = time.perf_counter()
    calls, shift = {'main': await call(session, args, timeout)}, None
    if expect.get('probe') and not expect.get('error') and calls['main']['kind'] == 'result':
        calls['repeat'] = await call(session, args,
                                     min(timeout, PROBE_CASE_TIMEOUT - (time.perf_counter() - start)))
        prefix_args, shift = prefix_request(args)
        if prefix_args is not None and calls['repeat']['kind'] == 'result':
            calls['prefix'] = await call(session, prefix_args,
                                         min(timeout, PROBE_CASE_TIMEOUT - (time.perf_counter() - start)))
    return evaluate(case, calls, shift, (time.perf_counter() - start) * 1000, catalog)


def exception_text(exc):
    if isinstance(exc, BaseExceptionGroup):
        return '; '.join(exception_text(e) for e in exc.exceptions)
    return f'{type(exc).__name__}: {exc}'[:300]


def failed_record(case, detail, catalog):
    call_record = {'kind': 'transport', 'ms': 0.0, 'args': case['args'], 'output': None,
                   'text_ok': False, 'error': detail}
    return evaluate(case, {'main': call_record}, None, 0.0, catalog)


def progress(done, total, record):
    main = record['calls'].get('main', {})
    print(f'[{done:>3}/{total}] {record["id"]:<16} {record["category"]:<10} {main.get("kind", "?"):<10} '
          f'{main.get("ms", 0.0):>7.0f} ms {len(record["results"]):>4} results', file=sys.stderr, flush=True)


async def run_cases(url, cases, catalog, timeout=CALL_TIMEOUT):
    """Run the cases in order over one MCP session; a session broken by a transport error is reopened."""
    records, discovery, pending = [], None, list(cases)
    while pending:
        current = None
        try:
            async with streamable_http_client(url) as (read, write, _):
                async with ClientSession(read, write) as session:
                    with anyio.fail_after(DISCOVERY_TIMEOUT):
                        info = await discover(session)
                    discovery = discovery or info
                    while pending:
                        current = pending[0]
                        records.append(await run_case(session, current, catalog, timeout))
                        pending.pop(0)
                        current = None
                        progress(len(records), len(cases), records[-1])
        except Exception as exc:  # the transport tore the session down
            detail = exception_text(exc)
            if current is None:  # could not (re)connect: the rest fails
                discovery = discovery or {'ok': False,
                                          'problems': [f'connection or discovery failed: {detail}']}
                records += [failed_record(case, detail, catalog) for case in pending]
                pending = []
            else:
                records.append(failed_record(current, detail, catalog))
                pending.pop(0)
                progress(len(records), len(cases), records[-1])
    return records, discovery


# ---------------------------------------------------------------- packet and grades

def build_packet(records, catalog, meta):
    """Everything an external grader needs for the semantic cases, with an empty grades template."""
    cases, skipped, template = [], [], {}
    for record in records:
        if record['category'] not in SEMANTIC:
            continue
        if not record['structural_ok']:
            skipped.append({'id': record['id'], 'reason': 'structurally invalid response (scores 0)'})
            continue
        if not record['results']:
            skipped.append({'id': record['id'], 'reason': 'no results, nothing to grade'})
            continue
        expect = record['expect']
        choices = (expect.get('choices') or []) if record['category'] == 'mixed' else []
        recs = record['recommendations'] if record['category'] == 'recs' else []
        cases.append({
            'id': record['id'], 'category': record['category'], 'query': record['args'].get('query', ''),
            'args': record['args'], 'choices': choices, 'complement': expect.get('complement'),
            'results': [{'id': r['id'], 'card': local_judge.fact_card(catalog.by_id[r['id']]),
                         'constraint_ok': r['ok']} for r in record['results']],
            'recommendations': [{'id': r['id'], 'for_product_id': r['for_product_id'],
                                 'card': local_judge.fact_card(catalog.by_id[r['id']]),
                                 'anchor_card': local_judge.fact_card(catalog.by_id[r['for_product_id']]),
                                 'constraint_ok': r['ok']} for r in recs],
        })
        template[record['id']] = {
            'primary': {r['id']: None for r in record['results']},
            'choices': {r['id']: dict.fromkeys(choices) for r in record['results']} if choices else {},
            'recs': {r['id']: None for r in recs},
        }
    needed = sum(len(c['results']) * (1 + len(c['choices'])) + len(c['recommendations']) for c in cases)
    return {'meta': dict(meta, judge_prompt_sha=local_judge.PROMPT_SHA, grades_needed=needed),
            'instructions': local_judge.PACKET_INSTRUCTIONS, 'rubric': local_judge.RUBRIC,
            'cases': cases, 'skipped': skipped, 'grades_template': template}


def load_grades(path):
    grades = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(grades, dict):
        raise SystemExit(f'{path}: grades must be a JSON object keyed by case id')
    return {k: v for k, v in grades.items() if not k.startswith('_')}


class GradeBook:
    """Grade lookups for one run; absent grades count as 0 and are tallied as missing."""

    def __init__(self, grades):
        self.grades = grades
        self.missing = Counter()

    def get(self, case_id, part, pid, label=None):
        table = (self.grades.get(case_id) or {}).get(part) or {}
        value = table.get(pid)
        if label is not None:
            value = value.get(label) if isinstance(value, dict) else None
        if value is None:
            self.missing[case_id] += 1
            return 0
        grade = local_judge.as_grade(value)
        if grade is None:
            raise SystemExit(f'invalid grade {value!r} for {case_id} {part} {pid} {label or ""}')
        return grade


def fits_choice(result, c):
    """Whether a result meets choice c's own constraints (always, when the case declares none)."""
    return result['choice_ok'][c] if c < len(result['choice_ok']) else True


def score_case(record, book):
    """(score, parts, graded) for one case; graded holds the grades used, for the report.

    A product breaking a choice's own constraints gets choice grade 0 for that
    choice, so it cannot cover it.
    """
    category, case_id = record['category'], record['id']
    if category == 'validity':
        return record['validity_score'], {}, {}
    if category == 'irrelevant':
        return float(record['correct_empty']), {}, {}
    if not record['structural_ok']:
        return 0.0, {'structurally_invalid': True}, {}
    results = record['results']
    primary = [book.get(case_id, 'primary', r['id']) for r in results]
    ok = [r['ok'] for r in results]
    graded = {'primary': primary}
    if category in ('ranking', 'unusual'):
        return metrics.ndcg(metrics.effective(primary, ok)), {}, graded
    if category == 'mixed':
        labels = record['expect']['choices']
        rows = [[book.get(case_id, 'choices', r['id'], label) if fits_choice(r, c) else 0
                 for c, label in enumerate(labels)] for r in results]
        graded['choices'] = rows
        coverage = metrics.choice_coverage(primary, rows, len(labels), ok)
        ndcg = metrics.ndcg(metrics.effective(primary, ok))
        return metrics.mixed_score(primary, rows, len(labels), ok), {'coverage': coverage, 'ndcg': ndcg}, graded
    effective = metrics.effective(primary, ok)
    by_id = {r['id']: g for r, g in zip(results, effective)}
    recs = record['recommendations']
    rec_grades = [book.get(case_id, 'recs', r['id']) for r in recs]
    anchors = [by_id.get(r['for_product_id'], 0) for r in recs]
    rec_ok = [r['ok'] for r in recs]
    no_complement = record['expect'].get('complement') == 'none'
    graded['recs'] = rec_grades
    parts = {'primary_ndcg': metrics.ndcg(effective),
             'rec_quality': metrics.rec_quality(rec_grades, anchors, rec_ok, no_complement)}
    return metrics.rec_case_score(effective, rec_grades, anchors, rec_ok, no_complement), parts, graded


def twin_pairs(records, grades, log_path):
    """(twin grade, judge grade) pairs for single-intent primary results.

    Joins the server's LOG_GRADES=1 log (lines of {"query", "pid", "twin_grades"},
    one grade per twin pass) with a graded run on (query, pid); the latest log
    line wins. Mixed cases are left out: the twin grades their products per choice.
    """
    twin = {}
    for line in Path(log_path).read_text(encoding='utf-8').splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            twin[(' '.join(str(row.get('query', '')).split()), row.get('pid'))] = row.get('twin_grades') or []
    pairs = []
    for record in records:
        if record['category'] not in ('ranking', 'unusual', 'recs') or not record['structural_ok']:
            continue
        query = ' '.join(str(record['args'].get('query', '')).split())
        table = (grades.get(record['id']) or {}).get('primary') or {}
        for result in record['results']:
            judge = local_judge.as_grade(table.get(result['id']))
            for predicted in twin.get((query, result['id']), []):
                predicted = local_judge.as_grade(predicted)
                if judge is not None and predicted is not None:
                    pairs.append({'case': record['id'], 'pid': result['id'], 'twin': predicted, 'judge': judge})
    return pairs


def print_agreement(pairs):
    if not pairs:
        print('\nTwin calibration: no (twin, judge) pairs joined')
        return
    exact = sum(p['twin'] == p['judge'] for p in pairs) / len(pairs)
    close = sum(abs(p['twin'] - p['judge']) <= 1 for p in pairs) / len(pairs)
    bias = statistics.fmean(p['twin'] - p['judge'] for p in pairs)
    print(f'\nTwin calibration: {len(pairs)} pairs, exact agreement {exact:.1%}, within one {close:.1%}, '
          f'mean twin - judge {bias:+.2f}')
    print('  twin \\ judge     0     1     2     3')
    for t in range(4):
        row = [sum(p['twin'] == t and p['judge'] == g for p in pairs) for g in range(4)]
        print(f'  {t:>12}' + ''.join(f'{n:>6}' for n in row))


def score_run(records, grades, perf=None):
    """Per-case scores, category means and totals (eval/metrics.py) for a graded run."""
    book = GradeBook(grades)
    cases = []
    for record in records:
        score, parts, graded = score_case(record, book)
        cases.append({'id': record['id'], 'category': record['category'], 'score': score,
                      'parts': parts, 'graded': graded})
    by_category = {}
    for case in cases:
        by_category.setdefault(case['category'], []).append(case['score'])
    means = {category: statistics.fmean(scores) for category, scores in by_category.items()}
    summary = metrics.total(means, perf)
    summary['categories'] = {
        category: {'cases': len(by_category.get(category, [])), 'mean': means.get(category),
                   'points': summary['points'][category], 'max': metrics.WEIGHTS[category]}
        for category in metrics.CATEGORIES}
    summary['missing_grades'] = dict(book.missing)
    summary['cases'] = cases
    return summary


# ---------------------------------------------------------------- reports

def nearest_rank(values, pct):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(pct / 100 * len(ordered)) - 1)]


def latency_line(label, values):
    if not values:
        return f'  {label}: no calls'
    return (f'  {label}: n={len(values)} p50 {statistics.median(values):.0f} ms  '
            f'p95 {nearest_rank(values, 95):.0f} ms  max {max(values):.0f} ms')


def print_discovery(discovery):
    if discovery.get('ok'):
        print(f'Discovery OK: protocol {discovery["protocol"]}, server {discovery["server"]}, '
              f'{discovery["ms"]:.0f} ms')
    else:
        print('DISCOVERY FAILED (the official attempt would score 0): '
              + '; '.join(discovery.get('problems', [])))


def print_validity(records):
    mean = statistics.fmean(r['validity_score'] for r in records)
    weight = metrics.WEIGHTS['validity']
    print(f'\nValidity: {len(records)} cases, mean {mean:.3f}, {weight * mean:.2f} / {weight:g} points')
    for r in records:
        if r['validity_score'] < 1:
            print(f'  {r["id"]:<16} {r["validity_score"]:.2f}  ' + ' | '.join(r['problems'][:3]))


def print_irrelevant(records):
    correct = sum(r['correct_empty'] for r in records)
    weight = metrics.WEIGHTS['irrelevant']
    print(f'\nIrrelevant: correct-empty {correct}/{len(records)}, '
          f'{weight * correct / len(records):.2f} / {weight:g} points')
    for r in records:
        if r['correct_empty']:
            continue
        if r['results']:
            names = ', '.join(f'{p.get("name")} ({p.get("type")})' for p in r['results'][:3])
            detail = f'{len(r["results"])} results, e.g. {names}'
        else:
            detail = ' | '.join(r['problems'][:2])
        print(f'  {r["id"]:<16} {r["args"].get("query", "")!r}: {detail}')


def semantic_notes(record):
    """Judge-free warnings about one semantic case: violations, presence, missing recommendations."""
    notes = []
    bad = [p for p in record['results'] if not p['ok']]
    if bad:
        notes.append(f'{len(bad)}/{len(record["results"])} results break constraints '
                     f'(e.g. {bad[0]["id"]}: {bad[0]["violations"][0]})')
    bad_recs = [p for p in record['recommendations'] if not p['ok']]
    if bad_recs:
        notes.append(f'{len(bad_recs)} recommendations break rec_constraints')
    if record['checks'] and not record['checks']['presence']:
        notes += [p for p in record['problems'] if p.startswith('presence')]
    if record['category'] == 'recs' and record['structural_ok'] and record['results'] \
            and not record['recommendations']:
        full = record['expect'].get('complement') == 'none'
        notes.append('no recommendations' + (' (complement "none": full credit)' if full else ''))
    return notes


def print_semantic(records):
    invalid = [r for r in records if not r['structural_ok']]
    print(f'\nSemantic cases: {len(records)} run, {len(records) - len(invalid)} structurally valid')
    for r in invalid:
        print(f'  INVALID {r["id"]:<16} (scores 0) ' + ' | '.join(r['problems'][:2]))
    for r in records:
        notes = semantic_notes(r)
        if notes:
            print(f'  {r["id"]:<16} ' + '; '.join(notes))


def main_latencies(records, empty):
    """Main-call latencies of the empty-query (or semantic) cases that reached the server."""
    return [r['calls']['main']['ms'] for r in records
            if is_empty_query(r['args']) == empty and r['calls']['main']['kind'] != 'transport']


def print_checks(run):
    """Everything decidable without a judge: discovery, validity, abstention, constraints, latency."""
    records = run['cases']
    print_discovery(run['discovery'] or {})
    by_category = {}
    for record in records:
        by_category.setdefault(record['category'], []).append(record)
    if by_category.get('validity'):
        print_validity(by_category['validity'])
    if by_category.get('irrelevant'):
        print_irrelevant(by_category['irrelevant'])
    semantic = [r for r in records if r['category'] in SEMANTIC]
    if semantic:
        print_semantic(semantic)

    print('\nLatency (main calls):')
    print(latency_line('semantic queries', main_latencies(records, empty=False)))
    print(latency_line('empty queries', main_latencies(records, empty=True)))
    timeouts = [r['id'] for r in records if any(c['kind'] == 'timeout' for c in r['calls'].values())]
    if timeouts:
        print(f'  TIMEOUTS: {timeouts}')

    scores = {'validity': [r['validity_score'] for r in by_category.get('validity', [])],
              'irrelevant': [float(r['correct_empty']) for r in by_category.get('irrelevant', [])]}
    points = sum(metrics.WEIGHTS[c] * statistics.fmean(s) for c, s in scores.items() if s)
    possible = sum(metrics.WEIGHTS[c] for c, s in scores.items() if s)
    print(f'\nDeterministic points (validity + irrelevant): {points:.2f} / {possible:g}')
    return points


def product_line(label, product, grade, extra=''):
    """One graded product of the worst-case report; X marks a constraint violation (no gain)."""
    flag = '  ' if product.get('ok', True) else 'X '
    return (f'    {label:>3}. g{grade} {flag}{product.get("name", "?")} | {product.get("type", "?")} | '
            f'{product.get("color", "?")} | INR {product.get("price", "?")}{extra}'
            + ('' if product.get('ok', True) else f'   <- {product["violations"][0]}'))


def print_score(summary, records, worst):
    print(f'\n{"Category":<14}{"Cases":>6}{"Mean":>8}{"Points":>9}{"Max":>7}')
    for category, row in summary['categories'].items():
        mean = '-' if row['mean'] is None else f'{row["mean"]:.3f}'
        print(f'{category:<14}{row["cases"]:>6}{mean:>8}{row["points"]:>9.2f}{row["max"]:>7g}')
    print(f'{"F (functional)":<28}{summary["functional"]:>9.2f}{metrics.FUNCTIONAL_MAX:>7g}'
          f'   measured max {summary["measured_max"]:g}')
    perf_note = (f'eligible: {summary["perf_eligible"]}' if summary.get('perf')
                 else 'not measured (pass --p95 MS)')
    print(f'{"P (performance)":<28}{summary["performance"]:>9.2f}{metrics.PERF_POINTS:>7g}   {perf_note}')
    print(f'{"Total":<28}{summary["total"]:>9.2f}{100:>7}')
    if summary['missing_grades']:
        print(f'WARNING: {sum(summary["missing_grades"].values())} grades missing (counted as 0): '
              f'{summary["missing_grades"]}')

    by_id = {r['id']: r for r in records}
    semantic = sorted((c for c in summary['cases'] if c['category'] in SEMANTIC),
                      key=lambda c: (c['score'], c['id']))
    if semantic and worst:
        print(f'\nWorst {min(worst, len(semantic))} semantic cases:')
    for case in semantic[:worst]:
        record = by_id[case['id']]
        parts = ' '.join(f'{k} {v:.3f}' for k, v in case['parts'].items() if isinstance(v, float))
        print(f'\n  {case["id"]} [{case["category"]}] score {case["score"]:.3f} {parts}')
        print(f'    query: {record["args"].get("query", "")!r}'
              + (f'  filters: {record["args"]["filters"]}' if record['args'].get('filters') else ''))
        if case['parts'].get('structurally_invalid'):
            print('    ' + ' | '.join(record['problems'][:3]))
            continue
        if not record['results']:
            print('    (no results)')
        graded = case['graded']
        for i, product in enumerate(record['results']):
            choice = (f'  choices {graded["choices"][i]}' if graded.get('choices') else '')
            print(product_line(str(i + 1), product, graded['primary'][i], choice))
        for i, rec in enumerate(record['recommendations'] if record['category'] == 'recs' else []):
            print(product_line(f'R{i + 1}', rec, graded['recs'][i], f'  for {rec["for_product_id"]}'))


# ---------------------------------------------------------------- main

def write_json(path, data):
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'wrote {path}', file=sys.stderr)


def read_json(path):
    if not path.exists():
        raise SystemExit(f'{path} not found (pass --out with the run directory)')
    return json.loads(path.read_text(encoding='utf-8'))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument('--url', help='MCP endpoint, e.g. http://127.0.0.1:8000/mcp')
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--cases', help=f'case file (default {DEFAULT_CASES.relative_to(ROOT)})')
    source.add_argument('--holdout', action='store_true', help='use eval/holdout.jsonl')
    parser.add_argument('--judge', choices=('none', 'packet', 'bedrock', 'anthropic'), default='none')
    parser.add_argument('--grades', help='grades JSON to score OUT/results.json with')
    parser.add_argument('--only', help='comma-separated categories or case-id patterns (fnmatch)')
    parser.add_argument('--out', help='run directory (default eval/runs/<UTC timestamp>/)')
    parser.add_argument('--p95', type=float, help='measured perf p95 in ms (eval/perf.py) for the total')
    parser.add_argument('--perf-correct', type=float, default=1.0, help='perf correct-call rate (default 1.0)')
    parser.add_argument('--timeout', type=float, default=CALL_TIMEOUT, help='per-call deadline in seconds')
    parser.add_argument('--workers', type=int, default=local_judge.WORKERS, help='parallel judge calls')
    parser.add_argument('--worst', type=int, default=10, help='how many worst semantic cases to show')
    parser.add_argument('--twin-log', nargs='?', const=str(ROOT / 'logs/grades.jsonl'), metavar='FILE',
                        help='join the server\'s LOG_GRADES=1 log with the judge grades and report agreement '
                             '(default file logs/grades.jsonl); writes OUT/calibration_pairs.json')
    opts = parser.parse_args(argv)
    if not opts.url and not opts.grades and opts.judge not in ('bedrock', 'anthropic'):
        parser.error('--url is required unless scoring an existing run (--grades or --judge bedrock|anthropic)')
    if not opts.url and not opts.out:
        parser.error('--out RUN_DIR is required without --url')
    return opts


def execute(opts, out):
    """Run the cases against the endpoint; returns (results, catalog)."""
    path = Path(opts.cases) if opts.cases else HOLDOUT_CASES if opts.holdout else DEFAULT_CASES
    cases = select_cases(load_cases(path), opts.only)
    print(f'{len(cases)} cases from {path}; loading the catalog...', file=sys.stderr, flush=True)
    catalog = Catalog.load()
    for warning in [w for case in cases for w in case_warnings(case, catalog)]:
        print(f'case warning: {warning}', file=sys.stderr)
    started = datetime.now(timezone.utc)
    records, discovery = asyncio.run(run_cases(opts.url, cases, catalog, opts.timeout))
    meta = {'url': opts.url, 'cases_file': str(path), 'only': opts.only, 'started_utc': started.isoformat(),
            'seconds': round((datetime.now(timezone.utc) - started).total_seconds(), 1),
            'timeout': opts.timeout}
    results = {'meta': meta, 'discovery': discovery, 'cases': records}
    write_json(out / 'results.json', results)
    return results, catalog


def main(argv=None):
    opts = parse_args(argv)
    out = Path(opts.out) if opts.out else RUNS_DIR / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out.mkdir(parents=True, exist_ok=True)

    packet = None
    if opts.url:
        results, catalog = execute(opts, out)
        if opts.judge != 'none':
            packet = build_packet(results['cases'], catalog, results['meta'])
            write_json(out / 'packet.json', packet)
        del catalog
    else:
        results = read_json(out / 'results.json')
        selected = {c['id'] for c in select_cases(results['cases'], opts.only)}
        results['cases'] = [r for r in results['cases'] if r['id'] in selected]
    print_checks(results)

    judge_meta = None
    if opts.grades:
        grades = load_grades(opts.grades)
        judge_meta = {'grades_file': str(opts.grades)}
    elif opts.judge in ('bedrock', 'anthropic'):
        packet = packet or read_json(out / 'packet.json')
        print(f'grading {packet["meta"]["grades_needed"]} products with the {opts.judge} judge...',
              file=sys.stderr, flush=True)
        llm, model = local_judge.make_judge(opts.judge)
        grades = local_judge.grade_packet(packet, llm, model, workers=opts.workers)
        write_json(out / 'grades.json', grades)
        judge_meta = {'provider': opts.judge, 'model': model, 'prompt_sha': local_judge.PROMPT_SHA}
    else:
        if opts.judge == 'packet' and packet is not None:
            print(f'\nPacket: {len(packet["cases"])} semantic cases, {packet["meta"]["grades_needed"]} grades '
                  f'needed ({len(packet["skipped"])} skipped). Grade it, then score with:\n'
                  f'  python eval/run_eval.py --grades GRADES.json --out {out}')
        return 0 if (results['discovery'] or {}).get('ok') else 2

    perf = None if opts.p95 is None else (opts.perf_correct, opts.p95)
    summary = score_run(results['cases'], grades, perf)
    summary.update(meta=results['meta'], judge=judge_meta, perf=None if perf is None else
                   {'correct_rate': perf[0], 'p95_ms': perf[1]},
                   discovery_ok=bool((results['discovery'] or {}).get('ok')))
    print_score(summary, results['cases'], opts.worst)
    write_json(out / 'score.json', summary)
    if opts.twin_log and not Path(opts.twin_log).exists():
        print(f'\nTwin calibration: {opts.twin_log} not found (run the server with LOG_GRADES=1)')
    elif opts.twin_log:
        pairs = twin_pairs(results['cases'], grades, opts.twin_log)
        print_agreement(pairs)
        write_json(out / 'calibration_pairs.json', {
            'twin_log': opts.twin_log, 'pairs': pairs,
            'refit': 'engine.points.fit_calibration([(p["twin"], p["judge"]) for p in pairs])'})
    return 0 if summary['discovery_ok'] else 2


if __name__ == '__main__':
    sys.exit(main())
