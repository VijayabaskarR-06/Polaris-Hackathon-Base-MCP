"""LLM planner + Judge Twin orchestration with scripted fake LLMs (no network)."""
import json
import time
import unittest

from engine.cache import LRU
from engine.judge import JudgeTwin, parse_grades
from engine.llm import FakeLLM
from engine.planner import LLMPlanner, merge_safety_net, validate_llm_plan
from engine.rules import rule_plan
from tests.shared import catalog

ARGS = {'filters': {}, 'sort_by': 'relevance', 'sort_order': 'desc', 'offset': 0, 'limit': 10,
        'include_recommendations': False}


def plan_json(**choice):
    base = {'label': 'item', 'product_types': [], 'keywords': [], 'colors': [], 'exclude_colors': [],
            'patterns': [], 'exclude_patterns': [], 'min_price': None, 'max_price': None, 'soft_sections': [],
            'soft_words': [], 'price_preference': None, 'rating_preference': False}
    base.update(choice)
    return json.dumps({'kind': 'search', 'audience_group': None, 'choices': [base], 'complement': None})


def grade_all(value):
    """A judge responder that grades every numbered card with the same grade."""
    def respond(system, messages, model):
        text = messages[-1]['content']
        if 'Shopper message:' in text:
            return None
        count = sum(1 for line in text.splitlines() if line.startswith('['))
        return json.dumps({'grades': [{'n': i + 1, 'g': value} for i in range(count)]})
    return respond


class LLMPathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = catalog().engine
        cls.products = cls.engine.products

    def setUp(self):
        self.saved = (self.engine.planner, self.engine.twin, self.engine.cache, self.engine.rec_cache)
        self.engine.cache, self.engine.rec_cache = LRU(), LRU()

    def tearDown(self):
        self.engine.planner, self.engine.twin, self.engine.cache, self.engine.rec_cache = self.saved

    def use(self, llm):
        self.engine.planner = LLMPlanner(self.engine.vocab, llm, 'fake-planner')
        self.engine.twin = JudgeTwin(self.products, llm, 'fake-judge')

    def run_query(self, query, **extra):
        start = time.monotonic()
        out = self.engine.search(dict(ARGS, query=query, **extra))
        return out, time.monotonic() - start

    def test_bad_json_falls_back_to_rules(self):
        self.use(FakeLLM(lambda system, messages, model: 'sorry, I cannot help {not json'))
        out, _ = self.run_query('black jeans')
        self.assertEqual(len(out['results']), 10)
        self.assertTrue(all(self.products[self.engine.index.pos_of[r['id']]]['color'] == 'Black'
                            for r in out['results']))

    def test_slow_planner_still_answers_in_time(self):
        def respond(system, messages, model):
            if 'Shopper message:' in messages[-1]['content']:
                return plan_json(label='jeans', product_types=['Trousers'], keywords=['jeans'], colors=['Black'])
            return grade_all(3)(system, messages, model)
        self.use(FakeLLM(respond, delay=lambda system, messages, model:
                         3.0 if 'Shopper message:' in messages[-1]['content'] else 0.05))
        out, elapsed = self.run_query('black jeans for a slow planner')
        self.assertLess(elapsed, 8.5)
        self.assertTrue(out['results'])

    def test_slow_judge_falls_back_to_estimates(self):
        self.use(FakeLLM(grade_all(3), delay=lambda system, messages, model:
                         0.0 if 'Shopper message:' in messages[-1]['content'] else 9.0))
        out, elapsed = self.run_query('white sneakers for men')
        self.assertLess(elapsed, 8.5)
        self.assertEqual(len(out['results']), 10)

    def test_twin_grades_drive_ranking_and_abstention(self):
        self.use(FakeLLM(grade_all(0)))
        out, _ = self.run_query('black jeans graded zero')
        self.assertEqual(out, {'results': [], 'recommendations': []})

    def test_invalid_vocabulary_values_are_dropped(self):
        raw = json.loads(plan_json(label='x', product_types=['Trousers', 'Jeggings', 'dress'],
                                   colors=['Navy', 'Dark Blue'], soft_sections=['Nowhere', 'Mama'],
                                   keywords=['jeans']))
        plan = validate_llm_plan(raw, self.engine.vocab)
        choice = plan.choices[0]
        self.assertEqual(choice.product_types, ['Trousers', 'Dress'])
        self.assertEqual(choice.colors, ['Dark Blue'])
        self.assertEqual(choice.soft_sections, ['Mama'])
        self.assertIsNone(validate_llm_plan({'kind': 'search', 'choices': 'nope'}, self.engine.vocab))
        self.assertIsNone(validate_llm_plan(None, self.engine.vocab))

    def test_safety_net_restores_budget_and_colour(self):
        query = 'black jeans under 2000'
        rules = rule_plan(query, self.engine.vocab)
        llm = validate_llm_plan(json.loads(plan_json(label='jeans', product_types=['Trousers'],
                                                     keywords=['jeans'])), self.engine.vocab)
        merged = merge_safety_net(llm, rules)
        self.assertEqual(merged.choices[0].max_price, 2000)
        self.assertIn('Black', merged.choices[0].colors)

    def test_parse_grades_validates(self):
        self.assertEqual(parse_grades({'grades': [{'n': 1, 'g': 3}, {'n': 2, 'g': 7}, {'n': 9, 'g': 1},
                                                  {'n': True, 'g': 2}, 'x']}, 3), {1: 3})
        self.assertEqual(parse_grades(None, 3), {})


if __name__ == '__main__':
    unittest.main()
