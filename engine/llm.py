"""Provider-neutral LLM access: Amazon Bedrock, the Anthropic API, none, and a fake for tests.

The server must answer every call within about 8.5 s, so the contract is strict.
complete() and complete_messages() return the assistant text, or None on any
error, timeout, throttle or bad argument. They never raise, and they never wait
longer than their timeout. A failure logs one warning line on 'polaris.llm'
naming the provider, model and error, never credentials or prompts. Token usage
is logged at debug level.

How a call runs:

* A semaphore per instance caps in-flight requests (LLM_MAX_INFLIGHT, default
  10). Waiting for a slot counts against the call's timeout.
* The blocking request runs on a private I/O pool while the caller waits at
  most `timeout`. A late request is abandoned; it keeps its slot until the
  socket timeouts end it. This pool is deliberately not EXECUTOR: a task running
  on EXECUTOR (the planner, judge batches) that sent its request through
  EXECUTOR could end up waiting for work queued behind itself.
* Optional request fields are temperature (LLM_TEMPERATURE, default 0; "none"
  omits it) and LLM_EXTRA_FIELDS, a JSON object sent as extra Anthropic body
  fields or as Bedrock additionalModelRequestFields, for example
  '{"thinking": {"type": "disabled"}}'. When a model answers 400 naming one of
  them, it is dropped for that model and the request is sent once more. Current
  Claude models (Haiku 5.5, Sonnet 5.5, Opus 5.5) accept only the default
  temperature; warm_up() at startup absorbs that first 400.
* The answer is read from the response's text blocks. Models that think by
  default put a thinking or reasoning block first, so content[0] is not always
  the text.

run_parallel() fans work out on EXECUTOR under the same deadline rules.

Environment: LLM_PROVIDER (bedrock | anthropic | none), AWS_REGION or
AWS_DEFAULT_REGION, ANTHROPIC_API_KEY, ANTHROPIC_BASE_URL, PLANNER_MODEL_ID,
JUDGE_MODEL_ID, JUDGE2_MODEL_ID, LOCAL_JUDGE_MODEL_ID, LLM_MAX_INFLIGHT,
LLM_TEMPERATURE, LLM_EXTRA_FIELDS.
"""
import concurrent.futures
import json
import logging
import os
import random
import re
import threading
import time
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor

log = logging.getLogger('polaris.llm')

ROLE_ENV = {
    'planner': ('PLANNER_MODEL_ID',),
    'judge': ('JUDGE_MODEL_ID',),
    'judge2': ('JUDGE2_MODEL_ID', 'JUDGE_MODEL_ID'),
    'local_judge': ('LOCAL_JUDGE_MODEL_ID',),
}
ANTHROPIC_URL = 'https://api.anthropic.com'
ANTHROPIC_VERSION = '2023-06-01'
RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504, 529})
FAILED_STOPS = frozenset({'refusal', 'guardrail_intervened', 'content_filtered'})
RETRY_WINDOW = 1.0  # seconds that must remain for a second request to be worth sending
STAT_KEYS = ('calls', 'ok', 'failed', 'timeout', 'busy', 'retries', 'resent',
             'input_tokens', 'output_tokens')
_ENV = object()  # constructor default: read the setting from the environment


def _env_int(name, default):
    raw = os.getenv(name, '').strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        log.warning('ignoring %s=%r: not an integer', name, raw)
        return default


def _env_temperature():
    raw = os.getenv('LLM_TEMPERATURE', '0').strip().lower()
    if raw in ('', 'none', 'off', 'default'):
        return None
    try:
        return float(raw)
    except ValueError:
        log.warning('ignoring LLM_TEMPERATURE=%r; using 0', raw)
        return 0.0


def _env_json(name):
    raw = os.getenv(name, '').strip()
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except ValueError:
        value = None
    if isinstance(value, dict):
        return value
    log.warning('ignoring %s: not a JSON object', name)
    return {}


EXECUTOR = ThreadPoolExecutor(max_workers=16, thread_name_prefix='polaris-par')
# Blocking provider requests only, never orchestration (see the module docstring).
_CALLS = ThreadPoolExecutor(max_workers=max(32, 2 * _env_int('LLM_MAX_INFLIGHT', 10)),
                            thread_name_prefix='polaris-llm-io')
# Threads already inside a fan-out: their own fan-outs get fresh threads (see run_parallel).
_NESTED = ('polaris-par', 'polaris-nested')


def deadline_remaining(deadline):
    """Seconds left until a time.monotonic() deadline, never negative."""
    return max(0.0, deadline - time.monotonic())


def model_for(role):
    """Model id for a role from the environment, or None.

    Roles: 'planner', 'judge', 'judge2' (falls back to JUDGE_MODEL_ID) and 'local_judge'.
    """
    for name in ROLE_ENV.get(role, ()):
        value = os.getenv(name, '').strip()
        if value:
            return value
    return None


def extract_json(text):
    """The first JSON object in model output, as a dict, or None.

    Code fences and prose around the object are ignored. Each top-level '{' is
    matched to its '}' under JSON string and escape rules and parsed; an object
    that never closes (output cut off at max_tokens) ends the scan instead of
    yielding one of its inner objects. Last resort: first '{' to last '}'.
    """
    if not isinstance(text, str):
        return None
    first = start = text.find('{')
    while start != -1:
        end = _closing_brace(text, start)
        if end is None:
            break
        value = _loads_dict(text[start:end + 1])
        if value is not None:
            return value
        start = text.find('{', end + 1)
    last = text.rfind('}')
    return _loads_dict(text[first:last + 1]) if -1 < first < last else None


def _closing_brace(text, start):
    """Index of the '}' that closes the '{' at start, ignoring braces inside JSON strings."""
    depth, in_string, escaped = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == '\\':
                escaped = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return i
    return None


def _loads_dict(candidate):
    try:
        value = json.loads(candidate, strict=False)  # strict=False: raw newlines inside strings
    except (ValueError, RecursionError):
        return None
    return value if isinstance(value, dict) else None


def run_parallel(tasks, timeout):
    """Run zero-argument callables concurrently; their results in task order.

    Waits at most `timeout` seconds. A task that raised, or is still running at
    the deadline, yields None; late tasks are abandoned (queued ones cancelled)
    and never block the caller. A non-positive timeout starts nothing. Called
    from inside a pool task, the tasks get their own threads instead of queueing
    behind the caller, so nested fan-outs cannot starve EXECUTOR.
    """
    tasks = list(tasks)
    timeout = float(timeout or 0)
    if not tasks or not timeout > 0:
        return [None] * len(tasks)
    deadline = time.monotonic() + timeout
    nested = threading.current_thread().name.startswith(_NESTED)
    submit = _spawn if nested else EXECUTOR.submit
    futures = []
    for task in tasks:
        try:
            futures.append(submit(task))
        except Exception as exc:  # interpreter shutdown, thread limit
            log.warning('parallel task not started: %s', _brief(exc))
            futures.append(None)
    concurrent.futures.wait([f for f in futures if f is not None],
                            timeout=deadline_remaining(deadline))
    results, late = [], 0
    for index, future in enumerate(futures):
        if future is None or future.cancelled():
            results.append(None)
        elif not future.done():
            future.cancel()
            late += 1
            results.append(None)
        elif future.exception() is not None:
            log.warning('parallel task %d failed: %s', index, _brief(future.exception()))
            results.append(None)
        else:
            results.append(future.result())
    if late:
        log.warning('%d of %d parallel tasks unfinished after %.2f s', late, len(tasks), timeout)
    return results


def _spawn(task):
    """Run task on a fresh daemon thread; returns a Future for it."""
    future = Future()

    def run():
        if not future.set_running_or_notify_cancel():
            return
        try:
            result = task()
        except BaseException as exc:  # reported through the future, as ThreadPoolExecutor does
            future.set_exception(exc)
        else:
            future.set_result(result)

    threading.Thread(target=run, name='polaris-nested', daemon=True).start()
    return future


def warm_up(llm, model=None, timeout=10.0):
    """One tiny call before traffic arrives: opens connections and learns rejected fields.

    True if the model answered.
    """
    if not getattr(llm, 'enabled', False):
        return False
    started = time.monotonic()
    text = llm.complete('Reply with the single word OK.', 'ping', max_tokens=16, timeout=timeout,
                        model=model)
    log.info('%s warm-up %s %s in %.0f ms', llm.name, llm.resolve_model(model) or '(no model)',
             'ok' if text is not None else 'failed', (time.monotonic() - started) * 1000)
    return text is not None


def _brief(exc, limit=240):
    """One log-safe line describing an exception."""
    text = ' '.join(str(exc).split())[:limit]
    if isinstance(exc, LLMError):
        return text or 'LLMError'
    return f'{type(exc).__name__}: {text}' if text else type(exc).__name__


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _bad_request(system, messages, max_tokens, timeout, model):
    """Why these arguments cannot be sent, or None."""
    if not isinstance(model, str) or not model.strip():
        return 'no model id (set PLANNER_MODEL_ID / JUDGE_MODEL_ID or pass model=)'
    if system is not None and not isinstance(system, str):
        return 'system prompt must be a string'
    if not isinstance(messages, (list, tuple)) or not messages:
        return 'messages must be a non-empty list'
    for message in messages:
        if (not isinstance(message, dict) or message.get('role') not in ('user', 'assistant')
                or not isinstance(message.get('content'), str) or not message['content'].strip()):
            return 'each message needs role "user" or "assistant" and non-empty string content'
    if not _is_number(max_tokens) or not 1 <= max_tokens <= 1e6:
        return 'max_tokens must be a positive number'
    if not _is_number(timeout) or not 0 < timeout <= 3600:
        return 'timeout must be a positive number of seconds'
    return None


def _last_user(messages):
    try:
        return next(m.get('content') for m in reversed(messages) if m.get('role') == 'user')
    except Exception:
        return None


def _seconds(value):
    """A retry-after header in seconds, or None."""
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    return seconds if 0 <= seconds < 3600 else None


class LLMError(Exception):
    """A failed provider request, with what the retry and fallback logic needs."""

    def __init__(self, message, *, status=None, code=None, retryable=False, retry_after=None):
        super().__init__(message)
        self.status = status
        self.code = code
        self.retryable = retryable
        self.retry_after = retry_after


class LLM:
    """Base class: the never-raise, deadline-bounded call path. Providers implement _call()."""

    name = 'base'

    def __init__(self, *, default_model=None, max_inflight=None, temperature=_ENV,
                 extra_fields=None):
        self.default_model = default_model
        self.max_inflight = max(1, int(max_inflight or _env_int('LLM_MAX_INFLIGHT', 10)))
        self.temperature = _env_temperature() if temperature is _ENV else temperature
        if extra_fields is None:
            extra_fields = _env_json('LLM_EXTRA_FIELDS')
        self.extra_fields = dict(extra_fields)
        self.unavailable = None  # why this instance cannot reach its provider, if it cannot
        self.stats = Counter(dict.fromkeys(STAT_KEYS, 0))
        self._slots = threading.Semaphore(self.max_inflight)
        self._lock = threading.Lock()
        self._rejected = {}  # model -> frozenset of optional fields it answered 400 to
        self._warned = set()

    @property
    def enabled(self):
        return self.unavailable is None

    def resolve_model(self, model=None):
        """The model to call: the argument, the instance default, else the planner or judge model."""
        return model or self.default_model or model_for('planner') or model_for('judge')

    def complete(self, system, user, *, max_tokens, timeout, model=None):
        """Send one user message; the assistant text or None. Never raises."""
        return self.complete_messages(system, [{'role': 'user', 'content': user}],
                                      max_tokens=max_tokens, timeout=timeout, model=model)

    def complete_messages(self, system, messages, *, max_tokens, timeout, model=None):
        """Send a conversation, e.g. few-shot pairs.

        messages: [{"role": "user" | "assistant", "content": str}, ...]. Returns
        the assistant text ('' if the model wrote none) or None on any error,
        timeout, throttle or bad argument. Never raises, and returns within
        `timeout` seconds.
        """
        started = time.monotonic()
        self._count(calls=1)
        try:
            model = self.resolve_model(model)
            problem = self.unavailable or _bad_request(system, messages, max_tokens, timeout, model)
            if problem:
                self._count(failed=1)
                self._warn_once(f'{self.name} {model or "(no model)"} not called: {problem}')
                return None
            deadline = started + timeout
            if not self._slots.acquire(timeout=deadline_remaining(deadline)):
                self._fail('busy', model, 'no free slot within %.2f s (max_inflight %d)',
                           timeout, self.max_inflight)
                return None
            future = self._start(system or '', messages, int(max_tokens), model, deadline)
            try:
                text = future.result(timeout=deadline_remaining(deadline))
            except concurrent.futures.TimeoutError:
                if future.cancel():  # never started, so its slot is still ours to return
                    self._slots.release()
                self._fail('timeout', model, 'timed out after %.2f s', timeout)
                return None
            self._count(ok=1)
            return text
        except Exception as exc:
            elapsed_ms = (time.monotonic() - started) * 1000
            self._fail('failed', model, 'failed after %.0f ms: %s', elapsed_ms, _brief(exc))
            return None

    def optional_fields(self, model):
        """Temperature and LLM_EXTRA_FIELDS, minus those this model has rejected."""
        fields = {} if self.temperature is None else {'temperature': self.temperature}
        fields.update(self.extra_fields)
        rejected = self._rejected.get(model, frozenset())
        return {key: value for key, value in fields.items() if key not in rejected}

    def close(self):
        """Release network resources (tests; the server keeps one instance for its lifetime)."""
        close = getattr(getattr(self, 'client', None), 'close', None)
        if callable(close):
            close()

    def _start(self, system, messages, max_tokens, model, deadline):
        """Run the request on the I/O pool; the slot is released when the request really ends."""
        messages = [{'role': m['role'], 'content': m['content']} for m in messages]

        def run():
            try:
                return self._run(system, messages, max_tokens, model, deadline)
            finally:
                self._slots.release()

        try:
            return _CALLS.submit(run)
        except Exception:
            self._slots.release()
            raise

    def _run(self, system, messages, max_tokens, model, deadline):
        """The provider request, sent once more without any optional field the model rejected."""
        for attempt in (1, 2):
            fields = self.optional_fields(model)
            try:
                return self._call(system, messages, max_tokens, model, fields, deadline)
            except LLMError as exc:
                dropped = self._learn_rejection(model, fields, exc)
                if attempt == 2 or not dropped or deadline_remaining(deadline) < RETRY_WINDOW:
                    raise
                self._count(resent=1)

    def _call(self, system, messages, max_tokens, model, fields, deadline):
        """Blocking provider request: the assistant text, or raise (LLMError for provider errors)."""
        raise NotImplementedError

    def _learn_rejection(self, model, fields, exc):
        """Remember the optional fields a 400 names, for this model; True if any were dropped."""
        if exc.status != 400 and exc.code != 'ValidationException':
            return False
        message = str(exc).lower()
        named = {key for key in fields if re.search(r'\b%s\b' % re.escape(key.lower()), message)}
        if not named:
            return False
        with self._lock:
            self._rejected[model] = self._rejected.get(model, frozenset()) | named
        log.warning('%s %s rejected %s; omitting it for this model from now on',
                    self.name, model, ', '.join(sorted(named)))
        return True

    def _finish(self, model, text, stop, tokens_in, tokens_out, started, server_ms=None):
        """Shared success path: refusal check, truncation warning, usage accounting."""
        if stop in FAILED_STOPS:
            raise LLMError(f'stop reason {stop}')
        if stop == 'max_tokens':
            log.warning('%s %s output cut off at max_tokens (%d chars)', self.name, model, len(text))
        tokens_in = tokens_in if isinstance(tokens_in, int) else 0
        tokens_out = tokens_out if isinstance(tokens_out, int) else 0
        self._count(input_tokens=tokens_in, output_tokens=tokens_out)
        elapsed_ms = (time.monotonic() - started) * 1000
        log.debug('%s %s answered in %.0f ms (server %s ms): %d tokens in, %d out, stop %s',
                  self.name, model, elapsed_ms, server_ms, tokens_in, tokens_out, stop)
        return text

    def _count(self, **deltas):
        with self._lock:
            self.stats.update(deltas)

    def _fail(self, kind, model, message, *args):
        self._count(**{kind: 1})
        log.warning('%s %s ' + message, self.name, model or '(no model)', *args)

    def _warn_once(self, message):
        """Lasting problems (no key, no model id) log a warning once, then at debug level."""
        with self._lock:
            first = message not in self._warned
            self._warned.add(message)
        log.log(logging.WARNING if first else logging.DEBUG, '%s', message)


class NullLLM(LLM):
    """LLM_PROVIDER=none: every call returns None at once, and the rule path runs alone."""

    name = 'none'

    def __init__(self, **options):
        super().__init__(**options)
        self.unavailable = 'LLM_PROVIDER is none'

    def complete_messages(self, system, messages, *, max_tokens, timeout, model=None):
        return None


class FakeLLM(LLM):
    """Scripted LLM for tests: no network and no threads of its own; safe to call concurrently.

    responder: a callable (system, messages, model) -> str | dict | None, a fixed
    string, or a list of responses returned in order (None once used up). dict
    and list answers are serialized as JSON. delay: seconds, or a callable
    (system, messages, model) -> seconds. A delay longer than the call's timeout
    costs the timeout and returns None, as a late provider would. Each call is
    recorded in self.calls as {system, messages, user, model, max_tokens,
    timeout, result}.
    """

    name = 'fake'

    def __init__(self, responder=None, delay=0.0, **options):
        super().__init__(**options)
        self.responder = responder
        self.delay = delay
        self.calls = []
        self._script = list(responder) if isinstance(responder, (list, tuple)) else None

    def complete_messages(self, system, messages, *, max_tokens, timeout, model=None):
        copied = list(messages) if isinstance(messages, (list, tuple)) else messages
        record = {'system': system, 'messages': copied, 'user': _last_user(messages), 'model': model,
                  'max_tokens': max_tokens, 'timeout': timeout, 'result': None}
        with self._lock:
            self.calls.append(record)
        self._count(calls=1)
        try:
            delay = self.delay(system, messages, model) if callable(self.delay) else self.delay
            delay = float(delay or 0.0)
            if not timeout > 0:
                return None
            if delay > timeout:
                time.sleep(timeout)
                self._count(timeout=1)
                log.debug('fake %s timed out after %.2f s', model, timeout)
                return None
            if delay > 0:
                time.sleep(delay)
            record['result'] = self._respond(system, messages, model)
        except Exception as exc:
            self._fail('failed', model, 'responder failed: %s', _brief(exc))
            return None
        if record['result'] is not None:
            self._count(ok=1)
        return record['result']

    def _respond(self, system, messages, model):
        if self._script is not None:
            with self._lock:
                out = self._script.pop(0) if self._script else None
        elif callable(self.responder):
            out = self.responder(system, messages, model)
        else:
            out = self.responder
        if isinstance(out, (dict, list)):
            return json.dumps(out)
        return out if out is None or isinstance(out, str) else str(out)


class BedrockLLM(LLM):
    """Amazon Bedrock Runtime converse() through one shared, thread-safe boto3 client.

    botocore retries throttles once (standard mode); its read timeout is per
    read, so the call's own deadline is enforced by complete*().
    """

    name = 'bedrock'

    def __init__(self, region=None, client=None, **options):
        super().__init__(**options)
        self.region = region or os.getenv('AWS_REGION') or os.getenv('AWS_DEFAULT_REGION') or None
        self.client = client
        if client is None:
            try:
                self.client = _bedrock_runtime(self.region)
            except Exception as exc:  # NoRegionError, ProfileNotFound, boto3 missing
                self.unavailable = f'no Bedrock client ({_brief(exc)})'
                log.warning('bedrock disabled: %s', self.unavailable)

    def _call(self, system, messages, max_tokens, model, fields, deadline):
        from botocore.exceptions import BotoCoreError, ClientError
        if deadline_remaining(deadline) <= 0:
            raise LLMError('no time left')
        started = time.monotonic()
        request = self._request(system, messages, max_tokens, model, fields)
        try:
            response = self.client.converse(**request)
        except ClientError as exc:
            error = exc.response.get('Error') or {}
            status = (exc.response.get('ResponseMetadata') or {}).get('HTTPStatusCode')
            raise LLMError(f"{error.get('Code', 'ClientError')}: {error.get('Message', '')}",
                           status=status, code=error.get('Code')) from None
        except BotoCoreError as exc:
            raise LLMError(_brief(exc)) from None
        blocks = ((response.get('output') or {}).get('message') or {}).get('content') or []
        text = ''.join(b['text'] for b in blocks
                       if isinstance(b, dict) and isinstance(b.get('text'), str))
        usage = response.get('usage') or {}
        server_ms = (response.get('metrics') or {}).get('latencyMs')
        return self._finish(model, text, response.get('stopReason'), usage.get('inputTokens'),
                            usage.get('outputTokens'), started, server_ms)

    @staticmethod
    def _request(system, messages, max_tokens, model, fields):
        """converse() arguments: temperature goes in inferenceConfig, other fields to the model."""
        fields = dict(fields)
        inference = {'maxTokens': max_tokens}
        if 'temperature' in fields:
            inference['temperature'] = float(fields.pop('temperature'))
        request = {'modelId': model, 'messages': _converse_messages(messages),
                   'inferenceConfig': inference}
        if system:
            request['system'] = [{'text': system}]
        if fields:
            request['additionalModelRequestFields'] = fields
        return request


def _bedrock_runtime(region):
    import boto3
    from botocore.config import Config
    config = Config(connect_timeout=2, read_timeout=6, retries={'max_attempts': 2, 'mode': 'standard'},
                    max_pool_connections=32)
    return boto3.session.Session().client('bedrock-runtime', region_name=region, config=config)


def _converse_messages(messages):
    """Converse turns; consecutive messages of one role become one turn with several text blocks."""
    turns = []
    for message in messages:
        block = {'text': message['content']}
        if turns and turns[-1]['role'] == message['role']:
            turns[-1]['content'].append(block)
        else:
            turns.append({'role': message['role'], 'content': [block]})
    return turns


class AnthropicLLM(LLM):
    """Anthropic Messages API over one shared httpx.Client (keep-alive connection pool).

    Throttles (429, 529), server errors and network errors are retried once when
    the deadline leaves room; each request's socket timeouts end at the deadline.
    """

    name = 'anthropic'

    def __init__(self, api_key=None, base_url=None, transport=None, **options):
        super().__init__(**options)
        import httpx
        self.api_key = (os.getenv('ANTHROPIC_API_KEY', '') if api_key is None else api_key).strip()
        self.base_url = (base_url or os.getenv('ANTHROPIC_BASE_URL') or ANTHROPIC_URL).rstrip('/')
        self.client = None
        try:
            self.client = httpx.Client(
                base_url=self.base_url, transport=transport, timeout=httpx.Timeout(8.0, connect=2.0),
                headers={'x-api-key': self.api_key, 'anthropic-version': ANTHROPIC_VERSION,
                         'content-type': 'application/json'},
                limits=httpx.Limits(max_connections=32, max_keepalive_connections=16,
                                    keepalive_expiry=60.0))
        except Exception as exc:  # e.g. a malformed ANTHROPIC_BASE_URL
            self.unavailable = f'no HTTP client ({_brief(exc)})'
        if not self.api_key:
            self.unavailable = 'ANTHROPIC_API_KEY is not set'
        if self.unavailable:
            log.warning('anthropic disabled: %s', self.unavailable)

    def _call(self, system, messages, max_tokens, model, fields, deadline):
        import httpx
        body = {**fields, 'model': model, 'max_tokens': max_tokens, 'messages': messages}
        if system:
            body['system'] = system
        for attempt in (1, 2):
            remaining = deadline_remaining(deadline)
            if remaining <= 0:
                raise LLMError('no time left')
            started = time.monotonic()
            timeout = httpx.Timeout(remaining, connect=min(2.0, remaining))
            try:
                response = self.client.post('/v1/messages', json=body, timeout=timeout)
            except httpx.HTTPError as exc:
                error = LLMError(_brief(exc), retryable=True)
            else:
                if response.status_code == 200:
                    return self._parse(response.json(), model, started)
                error = self._error(response)
            pause = error.retry_after if error.retry_after is not None else 0.2 + 0.2 * random.random()
            room = deadline_remaining(deadline) - pause
            if attempt == 2 or not error.retryable or room < RETRY_WINDOW:
                raise error
            self._count(retries=1)
            time.sleep(pause)

    def _parse(self, data, model, started):
        blocks = data.get('content') or []
        text = ''.join(b.get('text') or '' for b in blocks
                       if isinstance(b, dict) and b.get('type') == 'text')
        usage = data.get('usage') or {}
        return self._finish(model, text, data.get('stop_reason'), usage.get('input_tokens'),
                            usage.get('output_tokens'), started)

    def _error(self, response):
        """LLMError for a non-200 answer; the API key is never echoed into logs."""
        status = response.status_code
        try:
            error = response.json()['error']
            detail = f"{error.get('type', 'error')}: {error.get('message', '')}"
        except Exception:
            detail = response.text[:200]
        if self.api_key:
            detail = detail.replace(self.api_key, '[redacted]')
        return LLMError(f'HTTP {status} {detail}', status=status, retryable=status in RETRYABLE_STATUS,
                        retry_after=_seconds(response.headers.get('retry-after')))


PROVIDERS = {'bedrock': BedrockLLM, 'anthropic': AnthropicLLM}


def make_llm(provider=None):
    """The LLM for `provider`, else LLM_PROVIDER, else 'none'. Never raises: falls back to NullLLM."""
    name = str(provider or os.getenv('LLM_PROVIDER') or 'none').strip().lower()
    factory = PROVIDERS.get(name)
    if factory is None:
        if name not in ('none', 'off', ''):
            log.warning('unknown LLM provider %r; using none', name)
        return NullLLM()
    try:
        return factory()
    except Exception as exc:
        log.warning('%s provider unavailable (%s); using none', name, _brief(exc))
        return NullLLM()
