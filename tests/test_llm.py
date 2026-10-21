"""Offline tests for engine/llm.py: no network, no credentials, no catalog; a few seconds in all.

Providers are exercised through injected stand-ins (a stub boto3 client, an
httpx.MockTransport) or real clients that cannot reach anything: credentials,
region and instance metadata are stripped from the environment and botocore's
HTTP send is blocked.
"""
import json
import logging
import os
import threading
import time
import unittest
from contextlib import contextmanager
from unittest import mock

import httpx
from botocore.exceptions import ClientError, EndpointConnectionError

from engine.llm import (AnthropicLLM, BedrockLLM, FakeLLM, NullLLM, deadline_remaining, extract_json,
                        make_llm, model_for, run_parallel, warm_up)

SLACK = 0.3  # how far past its timeout a call may return
LOGGER = logging.getLogger('polaris.llm')
QUIET = logging.NullHandler()
ENV_PREFIXES = ('AWS_', 'ANTHROPIC_', 'LLM_', 'BOTO')
MODEL_VARS = ('PLANNER_MODEL_ID', 'JUDGE_MODEL_ID', 'JUDGE2_MODEL_ID', 'LOCAL_JUDGE_MODEL_ID')
BEDROCK_URL = 'https://bedrock-runtime.us-east-1.amazonaws.com'


def setUpModule():
    # Failure paths warn by design; keep the test output readable (assertLogs still sees them).
    LOGGER.addHandler(QUIET)
    LOGGER.propagate = False


def tearDownModule():
    LOGGER.removeHandler(QUIET)
    LOGGER.propagate = True


@contextmanager
def offline_env(**overrides):
    """No AWS or Anthropic credentials, region, model ids or LLM settings; no instance metadata."""
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(ENV_PREFIXES) and k not in MODEL_VARS}
    env.update(AWS_EC2_METADATA_DISABLED='true', AWS_CONFIG_FILE='/nonexistent/aws/config',
               AWS_SHARED_CREDENTIALS_FILE='/nonexistent/aws/credentials')
    env.update(overrides)
    with mock.patch.dict(os.environ, env, clear=True):
        yield


@contextmanager
def no_network():
    """Block every HTTP request botocore tries to send; yields the URLs it attempted."""
    attempts = []

    def blocked(session, request):
        attempts.append(request.url)
        raise EndpointConnectionError(endpoint_url=request.url)

    with mock.patch('botocore.httpsession.URLLib3Session.send', blocked):
        yield attempts


class StubBedrock:
    """Stands in for the bedrock-runtime client: reply(request) returns a response or raises."""

    def __init__(self, reply):
        self.reply = reply
        self.requests = []

    def converse(self, **request):
        self.requests.append(request)
        return self.reply(request)


def converse_reply(text, stop='end_turn', reasoning=False):
    content = [{'reasoningContent': {'reasoningText': {'text': 'thinking'}}}] if reasoning else []
    content.append({'text': text})
    return {'output': {'message': {'role': 'assistant', 'content': content}}, 'stopReason': stop,
            'usage': {'inputTokens': 11, 'outputTokens': 3, 'totalTokens': 14},
            'metrics': {'latencyMs': 42}}


def bedrock_error(code, message, status=400):
    return ClientError({'Error': {'Code': code, 'Message': message},
                        'ResponseMetadata': {'HTTPStatusCode': status}}, 'Converse')


def raiser(exc):
    def reply(request):
        raise exc
    return reply


def anthropic_reply(text='hello', stop='end_turn'):
    content = [{'type': 'thinking', 'thinking': '', 'signature': 'sig'}, {'type': 'text', 'text': text}]
    body = {'id': 'msg_1', 'type': 'message', 'role': 'assistant', 'content': content,
            'stop_reason': stop, 'usage': {'input_tokens': 9, 'output_tokens': 2}}
    return httpx.Response(200, json=body)


def anthropic_error(status, kind, message, **headers):
    return httpx.Response(status, json={'type': 'error', 'error': {'type': kind, 'message': message}},
                          headers=headers)


class ExtractJsonTests(unittest.TestCase):
    def test_plain_object(self):
        text = '{"kind": "search", "choices": []}'
        self.assertEqual(extract_json(text), {'kind': 'search', 'choices': []})

    def test_code_fence_and_prose(self):
        text = 'Here is the plan:\n```json\n{"a": {"b": [1, 2]}}\n```\nHope this helps! {not json}'
        self.assertEqual(extract_json(text), {'a': {'b': [1, 2]}})

    def test_nested_braces(self):
        text = '{"a": {"b": {"c": {}}}, "d": [{"e": 1}, {"f": {"g": 2}}]}'
        self.assertEqual(extract_json(text), json.loads(text))

    def test_braces_quotes_and_escapes_inside_strings(self):
        text = r'{"label": "curly } and { braces", "quote": "say \"}\" now", "path": "C:\\"} tail }'
        self.assertEqual(extract_json(text), {'label': 'curly } and { braces', 'quote': 'say "}" now',
                                              'path': 'C:\\'})

    def test_skips_a_closed_invalid_object_before_the_json(self):
        self.assertEqual(extract_json('Schema {kind, choices}: {"kind": "nonsense", "choices": []}'),
                         {'kind': 'nonsense', 'choices': []})

    def test_first_of_two_objects_and_raw_newlines(self):
        self.assertEqual(extract_json('{"a": 1} {"b": 2}'), {'a': 1})
        self.assertEqual(extract_json('{"a": "line1\nline2"}'), {'a': 'line1\nline2'})

    def test_garbage_is_none(self):
        truncated = '{"grades": [{"n": 1, "g": 3}, {"n": 2, "g"'
        for text in (None, '', 'no json here', '{', '}{', '{"a": }', '[1, 2]', 'null', truncated,
                     'sorry, I cannot help {not json', 42, b'{"a": 1}'):
            self.assertIsNone(extract_json(text), text)


class ParallelTests(unittest.TestCase):
    def test_keeps_task_order(self):
        tasks = [lambda d=d, i=i: (time.sleep(d), i)[1] for i, d in enumerate([0.15, 0.0, 0.08])]
        self.assertEqual(run_parallel(tasks, 2.0), [0, 1, 2])

    def test_returns_by_the_timeout_when_tasks_hang(self):
        release = threading.Event()
        self.addCleanup(release.set)

        def hang(value):
            return lambda: (release.wait(5.0), value)[1]
        start = time.monotonic()
        self.assertEqual(run_parallel([lambda: 'a', hang('b'), lambda: 'c', hang('d')], 0.3),
                         ['a', None, 'c', None])
        self.assertLess(time.monotonic() - start, 0.3 + SLACK)

    def test_more_hanging_tasks_than_workers(self):
        release = threading.Event()
        self.addCleanup(release.set)
        start = time.monotonic()
        self.assertEqual(run_parallel([lambda: (release.wait(5.0), 1)[1]] * 40, 0.3), [None] * 40)
        self.assertLess(time.monotonic() - start, 0.3 + SLACK)

    def test_failed_task_is_none(self):
        self.assertEqual(run_parallel([lambda: 1, lambda: 1 / 0, lambda: 3], 1.0), [1, None, 3])

    def test_no_tasks_or_no_time(self):
        ran = []
        self.assertEqual(run_parallel([], 1.0), [])
        self.assertEqual(run_parallel([lambda: ran.append(1)], 0), [None])
        self.assertEqual(run_parallel([lambda: ran.append(1)], -1.0), [None])
        time.sleep(0.05)
        self.assertEqual(ran, [])

    def test_nested_fan_outs_do_not_starve_the_pool(self):
        # 24 outer tasks fill all 16 workers; their inner tasks must still run.
        def outer():
            return run_parallel([lambda: (time.sleep(0.05), 1)[1], lambda: 2], 2.0)
        start = time.monotonic()
        self.assertEqual(run_parallel([outer] * 24, 3.0), [[1, 2]] * 24)
        self.assertLess(time.monotonic() - start, 1.5)

    def test_deadline_remaining(self):
        self.assertEqual(deadline_remaining(time.monotonic() - 1.0), 0.0)
        self.assertTrue(4.9 < deadline_remaining(time.monotonic() + 5.0) <= 5.0)


class FakeLLMTests(unittest.TestCase):
    def test_responder_and_recorded_call(self):
        fake = FakeLLM(lambda system, messages, model: f'{system}|{messages[-1]["content"]}|{model}')
        self.assertEqual(fake.complete('S', 'U', max_tokens=5, timeout=1.0, model='M'), 'S|U|M')
        call = fake.calls[0]
        self.assertEqual([call[k] for k in ('system', 'user', 'model', 'max_tokens', 'timeout')],
                         ['S', 'U', 'M', 5, 1.0])
        self.assertEqual(call['messages'], [{'role': 'user', 'content': 'U'}])
        self.assertEqual(call['result'], 'S|U|M')

    def test_scripted_constant_and_json_responses(self):
        fake = FakeLLM(['a', {'kind': 'search'}])
        answers = [fake.complete('s', 'u', max_tokens=5, timeout=1.0) for _ in range(3)]
        self.assertEqual(answers, ['a', '{"kind": "search"}', None])
        self.assertEqual(FakeLLM('same').complete('s', 'u', max_tokens=5, timeout=1.0), 'same')
        self.assertIsNone(FakeLLM().complete('s', 'u', max_tokens=5, timeout=1.0))

    def test_delay_beyond_timeout_costs_the_timeout_and_returns_none(self):
        fake = FakeLLM('late', delay=5.0)
        start = time.monotonic()
        self.assertIsNone(fake.complete('s', 'u', max_tokens=5, timeout=0.2))
        elapsed = time.monotonic() - start
        self.assertGreaterEqual(elapsed, 0.19)
        self.assertLess(elapsed, 0.2 + SLACK)
        self.assertEqual(fake.stats['timeout'], 1)
        self.assertIsNone(fake.complete('s', 'u', max_tokens=5, timeout=0))

    def test_callable_delay(self):
        def delay(system, messages, model):
            return 0.3 if model == 'slow' else 0.0
        fake = FakeLLM(lambda system, messages, model: model, delay=delay)
        self.assertEqual(fake.complete('s', 'u', max_tokens=5, timeout=1.0, model='fast'), 'fast')
        self.assertIsNone(fake.complete('s', 'u', max_tokens=5, timeout=0.1, model='slow'))
        start = time.monotonic()
        self.assertEqual(fake.complete('s', 'u', max_tokens=5, timeout=1.0, model='slow'), 'slow')
        self.assertGreaterEqual(time.monotonic() - start, 0.29)

    def test_responder_errors_return_none(self):
        with self.assertLogs(LOGGER, 'WARNING'):
            self.assertIsNone(FakeLLM(lambda *a: 1 / 0).complete('s', 'u', max_tokens=5, timeout=1.0))

    def test_concurrent_calls_are_all_recorded(self):
        fake = FakeLLM(lambda s, m, model: m[-1]['content'], delay=0.01)
        tasks = [lambda i=i: fake.complete('s', str(i), max_tokens=5, timeout=1.0) for i in range(40)]
        self.assertEqual(run_parallel(tasks, 5.0), [str(i) for i in range(40)])
        self.assertEqual(sorted(int(c['user']) for c in fake.calls), list(range(40)))


class FactoryTests(unittest.TestCase):
    def test_none_provider(self):
        with offline_env():
            self.assertIsInstance(make_llm('none'), NullLLM)
            self.assertIsInstance(make_llm(), NullLLM)
        with offline_env(LLM_PROVIDER='none'):
            self.assertIsInstance(make_llm(), NullLLM)
        null = NullLLM()
        self.assertEqual(null.name, 'none')
        self.assertFalse(null.enabled)
        self.assertIsNone(null.complete('s', 'u', max_tokens=5, timeout=1.0, model='m'))
        self.assertIsNone(null.complete_messages('s', [], max_tokens=5, timeout=1.0))

    def test_unknown_provider_falls_back_to_none(self):
        with offline_env(), self.assertLogs(LOGGER, 'WARNING'):
            self.assertIsInstance(make_llm('gpt'), NullLLM)

    def test_providers_construct_and_fail_fast_without_credentials(self):
        with offline_env(LLM_PROVIDER='Bedrock', AWS_REGION='us-east-1'), no_network() as attempts:
            bedrock = make_llm()
            anthropic = make_llm('anthropic')
            self.addCleanup(anthropic.close)
            self.assertIsInstance(bedrock, BedrockLLM)
            self.assertIsInstance(anthropic, AnthropicLLM)
            for llm in (bedrock, anthropic):
                start = time.monotonic()
                self.assertIsNone(llm.complete('s', 'u', max_tokens=5, timeout=2.0, model='m'))
                self.assertLess(time.monotonic() - start, 1.0)
        self.assertEqual(attempts, [])

    def test_model_for_roles(self):
        with offline_env(PLANNER_MODEL_ID=' planner ', JUDGE_MODEL_ID='judge',
                         LOCAL_JUDGE_MODEL_ID='  '):
            self.assertEqual(model_for('planner'), 'planner')
            self.assertEqual(model_for('judge'), 'judge')
            self.assertEqual(model_for('judge2'), 'judge')
            self.assertIsNone(model_for('local_judge'))
            self.assertIsNone(model_for('chef'))
        with offline_env(JUDGE_MODEL_ID='judge', JUDGE2_MODEL_ID='judge2', LOCAL_JUDGE_MODEL_ID='big'):
            self.assertEqual(model_for('judge2'), 'judge2')
            self.assertEqual(model_for('local_judge'), 'big')
            self.assertIsNone(model_for('planner'))

    def test_warm_up(self):
        self.assertTrue(warm_up(FakeLLM('OK'), 'm'))
        self.assertFalse(warm_up(FakeLLM(None), 'm'))
        self.assertFalse(warm_up(NullLLM(), 'm'))


class BedrockTests(unittest.TestCase):
    def bedrock(self, reply, **options):
        stub = StubBedrock(reply)
        with offline_env():
            llm = BedrockLLM(region='us-east-1', client=stub, **options)
        return llm, stub

    def test_request_shape_and_text(self):
        llm, stub = self.bedrock(lambda r: converse_reply('{"ok": true}'))
        text = llm.complete('sys', 'hi', max_tokens=50, timeout=2.0, model='m1')
        self.assertEqual(text, '{"ok": true}')
        self.assertEqual(stub.requests, [{
            'modelId': 'm1', 'system': [{'text': 'sys'}],
            'messages': [{'role': 'user', 'content': [{'text': 'hi'}]}],
            'inferenceConfig': {'maxTokens': 50, 'temperature': 0.0}}])
        self.assertEqual([llm.stats[k] for k in ('ok', 'input_tokens', 'output_tokens')], [1, 11, 3])

    def test_few_shot_messages_and_reasoning_blocks(self):
        llm, stub = self.bedrock(lambda r: converse_reply('B', reasoning=True))
        messages = [{'role': 'user', 'content': 'q1'}, {'role': 'assistant', 'content': 'a1'},
                    {'role': 'user', 'content': 'q2'}, {'role': 'user', 'content': 'q3'}]
        self.assertEqual(llm.complete_messages('', messages, max_tokens=5, timeout=2.0, model='m'), 'B')
        sent = stub.requests[0]
        self.assertNotIn('system', sent)
        self.assertEqual(sent['messages'], [
            {'role': 'user', 'content': [{'text': 'q1'}]},
            {'role': 'assistant', 'content': [{'text': 'a1'}]},
            {'role': 'user', 'content': [{'text': 'q2'}, {'text': 'q3'}]}])

    def test_rejected_temperature_is_dropped_and_remembered(self):
        def reply(request):
            if request['modelId'] == 'new' and 'temperature' in request['inferenceConfig']:
                raise bedrock_error('ValidationException',
                                    '`temperature` is not supported for this model.')
            return converse_reply('ok')
        llm, stub = self.bedrock(reply)
        with self.assertLogs(LOGGER, 'WARNING'):
            self.assertEqual(llm.complete('s', 'u', max_tokens=5, timeout=2.0, model='new'), 'ok')
        self.assertEqual(llm.complete('s', 'u', max_tokens=5, timeout=2.0, model='new'), 'ok')
        sent = ['temperature' in r['inferenceConfig'] for r in stub.requests]
        self.assertEqual(sent, [True, False, False])
        # Other models keep temperature 0.
        llm.complete('s', 'u', max_tokens=5, timeout=2.0, model='other')
        self.assertEqual(stub.requests[-1]['inferenceConfig'].get('temperature'), 0.0)
        self.assertEqual(llm.stats['resent'], 1)

    def test_other_validation_errors_are_not_resent(self):
        error = bedrock_error('ValidationException', 'The provided model identifier is invalid.')
        llm, stub = self.bedrock(raiser(error))
        self.assertIsNone(llm.complete('s', 'u', max_tokens=5, timeout=2.0, model='bad'))
        self.assertEqual(len(stub.requests), 1)

    def test_settings_from_the_environment(self):
        stub = StubBedrock(lambda r: converse_reply('ok'))
        with offline_env(LLM_TEMPERATURE='none', LLM_EXTRA_FIELDS='{"thinking": {"type": "disabled"}}',
                         LLM_MAX_INFLIGHT='3'):
            llm = BedrockLLM(region='us-east-1', client=stub)
        llm.complete('s', 'u', max_tokens=5, timeout=2.0, model='m')
        self.assertEqual(stub.requests[0]['inferenceConfig'], {'maxTokens': 5})
        self.assertEqual(stub.requests[0]['additionalModelRequestFields'],
                         {'thinking': {'type': 'disabled'}})
        self.assertEqual(llm.max_inflight, 3)
        bad = dict(LLM_EXTRA_FIELDS='not json', LLM_MAX_INFLIGHT='many', LLM_TEMPERATURE='warm')
        with offline_env(**bad), self.assertLogs(LOGGER, 'WARNING'):
            llm = BedrockLLM(region='us-east-1', client=stub)
        self.assertEqual((llm.extra_fields, llm.max_inflight, llm.temperature), ({}, 10, 0.0))

    def test_provider_errors_return_none(self):
        errors = [bedrock_error('AccessDeniedException', 'no access to this model', 403),
                  bedrock_error('ThrottlingException', 'Too many requests', 429),
                  EndpointConnectionError(endpoint_url=BEDROCK_URL),
                  RuntimeError('boom')]
        for error in errors:
            llm, _ = self.bedrock(raiser(error))
            with self.assertLogs(LOGGER, 'WARNING') as logs:
                self.assertIsNone(llm.complete('s', 'u', max_tokens=5, timeout=2.0, model='m'))
            self.assertEqual(len(logs.output), 1)
            self.assertEqual(llm.stats['failed'], 1)

    def test_refusal_is_none_and_truncation_is_returned(self):
        llm, _ = self.bedrock(lambda r: converse_reply('', stop='guardrail_intervened'))
        self.assertIsNone(llm.complete('s', 'u', max_tokens=5, timeout=2.0, model='m'))
        llm, _ = self.bedrock(lambda r: converse_reply('{"grades": [', stop='max_tokens'))
        with self.assertLogs(LOGGER, 'WARNING'):
            text = llm.complete('s', 'u', max_tokens=5, timeout=2.0, model='m')
        self.assertEqual(text, '{"grades": [')

    def test_timeout_is_enforced(self):
        release = threading.Event()
        self.addCleanup(release.set)
        llm, _ = self.bedrock(lambda r: (release.wait(5.0), converse_reply('late'))[1])
        start = time.monotonic()
        self.assertIsNone(llm.complete('s', 'u', max_tokens=5, timeout=0.2, model='m'))
        self.assertLess(time.monotonic() - start, 0.2 + SLACK)
        self.assertEqual(llm.stats['timeout'], 1)

    def test_in_flight_limit(self):
        entered, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)

        def reply(request):
            entered.set()
            release.wait(5.0)
            return converse_reply('done')
        llm, _ = self.bedrock(reply, max_inflight=1)
        first = []
        worker = threading.Thread(target=lambda: first.append(
            llm.complete('s', 'u', max_tokens=5, timeout=3.0, model='m')))
        worker.start()
        self.assertTrue(entered.wait(2.0))
        start = time.monotonic()
        self.assertIsNone(llm.complete('s', 'u', max_tokens=5, timeout=0.2, model='m'))
        self.assertLess(time.monotonic() - start, 0.2 + SLACK)
        release.set()
        worker.join(3.0)
        self.assertEqual(first, ['done'])
        self.assertEqual(llm.stats['busy'], 1)
        # The slot came back.
        self.assertEqual(llm.complete('s', 'u', max_tokens=5, timeout=1.0, model='m'), 'done')

    def test_bad_arguments_never_raise_or_send(self):
        llm, stub = self.bedrock(lambda r: converse_reply('x'))
        with offline_env():
            calls = [
                lambda: llm.complete('s', None, max_tokens=5, timeout=1.0, model='m'),
                lambda: llm.complete('s', '  ', max_tokens=5, timeout=1.0, model='m'),
                lambda: llm.complete(123, 'u', max_tokens=5, timeout=1.0, model='m'),
                lambda: llm.complete('s', 'u', max_tokens=0, timeout=1.0, model='m'),
                lambda: llm.complete('s', 'u', max_tokens=5, timeout=0, model='m'),
                lambda: llm.complete('s', 'u', max_tokens=5, timeout=float('nan'), model='m'),
                lambda: llm.complete('s', 'u', max_tokens=5, timeout=1.0),  # no model configured
                lambda: llm.complete_messages('s', 'not a list', max_tokens=5, timeout=1.0, model='m'),
                lambda: llm.complete_messages('s', [{'role': 'system', 'content': 'x'}], max_tokens=5,
                                              timeout=1.0, model='m'),
            ]
            for call in calls:
                self.assertIsNone(call())
        self.assertEqual(stub.requests, [])
        self.assertEqual(llm.complete(None, 'u', max_tokens=5, timeout=1.0, model='m'), 'x')

    def test_model_defaults_to_the_configured_planner_model(self):
        llm, stub = self.bedrock(lambda r: converse_reply('x'))
        with offline_env(PLANNER_MODEL_ID='planner-model'):
            llm.complete('s', 'u', max_tokens=5, timeout=1.0)
        self.assertEqual(stub.requests[0]['modelId'], 'planner-model')

    def test_real_client_without_credentials_fails_fast_offline(self):
        with offline_env(AWS_REGION='us-east-1'), no_network() as attempts:
            llm = BedrockLLM()
            self.assertTrue(llm.enabled)
            start = time.monotonic()
            with self.assertLogs(LOGGER, 'WARNING') as logs:
                self.assertIsNone(llm.complete('s', 'u', max_tokens=5, timeout=2.0, model='m'))
            self.assertLess(time.monotonic() - start, 1.0)
        self.assertEqual(attempts, [])
        self.assertIn('NoCredentialsError', ''.join(logs.output))

    def test_no_region_does_not_raise(self):
        with offline_env(), no_network() as attempts:
            llm = BedrockLLM()
            start = time.monotonic()
            self.assertIsNone(llm.complete('s', 'u', max_tokens=5, timeout=2.0, model='m'))
            self.assertLess(time.monotonic() - start, 0.5)
        self.assertFalse(llm.enabled)
        self.assertEqual(attempts, [])


class AnthropicTests(unittest.TestCase):
    def anthropic(self, handler, **options):
        requests = []

        def record(request):
            requests.append(request)
            return handler(request)
        with offline_env():
            transport = httpx.MockTransport(record)
            llm = AnthropicLLM(api_key='sk-test-secret', transport=transport, **options)
        self.addCleanup(llm.close)
        return llm, requests

    def test_request_shape_and_text_after_thinking_block(self):
        llm, requests = self.anthropic(lambda r: anthropic_reply('hello'))
        text = llm.complete('sys', 'hi', max_tokens=40, timeout=2.0, model='claude-x')
        self.assertEqual(text, 'hello')
        request = requests[0]
        self.assertEqual(request.method, 'POST')
        self.assertEqual(str(request.url), 'https://api.anthropic.com/v1/messages')
        self.assertEqual(request.headers['x-api-key'], 'sk-test-secret')
        self.assertEqual(request.headers['anthropic-version'], '2023-06-01')
        self.assertEqual(request.headers['content-type'], 'application/json')
        self.assertEqual(json.loads(request.content), {
            'model': 'claude-x', 'max_tokens': 40, 'temperature': 0.0, 'system': 'sys',
            'messages': [{'role': 'user', 'content': 'hi'}]})

    def test_throttle_is_retried_once(self):
        replies = [anthropic_error(429, 'rate_limit_error', 'slow down', **{'retry-after': '0'}),
                   anthropic_reply('ok')]
        llm, requests = self.anthropic(lambda r: replies.pop(0))
        self.assertEqual(llm.complete('s', 'u', max_tokens=5, timeout=3.0, model='m'), 'ok')
        self.assertEqual((len(requests), llm.stats['retries']), (2, 1))

    def test_persistent_overload_is_none_after_one_retry(self):
        llm, requests = self.anthropic(lambda r: anthropic_error(529, 'overloaded_error', 'Overloaded'))
        with self.assertLogs(LOGGER, 'WARNING') as logs:
            self.assertIsNone(llm.complete('s', 'u', max_tokens=5, timeout=3.0, model='m'))
        self.assertEqual(len(requests), 2)
        self.assertIn('529', ''.join(logs.output))

    def test_rejected_temperature_is_dropped(self):
        def handler(request):
            if 'temperature' in json.loads(request.content):
                return anthropic_error(400, 'invalid_request_error',
                                       'temperature: only the default is supported')
            return anthropic_reply('fine')
        llm, requests = self.anthropic(handler)
        self.assertEqual(llm.complete('s', 'u', max_tokens=5, timeout=3.0, model='m'), 'fine')
        self.assertEqual(len(requests), 2)

    def test_client_errors_are_not_retried_and_the_key_is_redacted(self):
        llm, requests = self.anthropic(
            lambda r: anthropic_error(401, 'authentication_error', 'invalid x-api-key sk-test-secret'))
        with self.assertLogs(LOGGER, 'WARNING') as logs:
            self.assertIsNone(llm.complete('s', 'u', max_tokens=5, timeout=3.0, model='m'))
        self.assertEqual(len(requests), 1)
        output = ''.join(logs.output)
        self.assertIn('authentication_error', output)
        self.assertNotIn('sk-test-secret', output)

    def test_hung_response_costs_only_the_timeout(self):
        release = threading.Event()
        self.addCleanup(release.set)
        llm, _ = self.anthropic(lambda r: (release.wait(5.0), anthropic_reply('late'))[1])
        start = time.monotonic()
        self.assertIsNone(llm.complete('s', 'u', max_tokens=5, timeout=0.25, model='m'))
        self.assertLess(time.monotonic() - start, 0.25 + SLACK)

    def test_missing_key_fails_fast_without_requests(self):
        requests = []
        transport = httpx.MockTransport(lambda r: requests.append(r) or anthropic_reply())
        with offline_env():
            llm = AnthropicLLM(transport=transport)
        self.addCleanup(llm.close)
        self.assertFalse(llm.enabled)
        start = time.monotonic()
        self.assertIsNone(llm.complete('s', 'u', max_tokens=5, timeout=2.0, model='m'))
        self.assertLess(time.monotonic() - start, 0.1)
        self.assertEqual(requests, [])

    def test_unreachable_endpoint(self):
        with offline_env():  # 127.0.0.1:9 is local and closed: refused at once
            llm = AnthropicLLM(api_key='k', base_url='http://127.0.0.1:9')
        self.addCleanup(llm.close)
        start = time.monotonic()
        self.assertIsNone(llm.complete('s', 'u', max_tokens=5, timeout=1.0, model='m'))
        self.assertLess(time.monotonic() - start, 1.0 + SLACK)


if __name__ == '__main__':
    unittest.main()
