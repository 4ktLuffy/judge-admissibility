"""Eval datasets, and judge-certification cases, from the traces of production agent runs.

pydantic-ai issue #4340 asks for "production traces -> eval dataset cases" in one call (a maintainer
preferred `Dataset.from_logfire`). This is that conversion, minus the fetching: it reads spans
already in hand, so it works on any OpenTelemetry backend and needs no Logfire account.
`dataset_from_spans` builds a pydantic-evals `Dataset`; `judge_cases_from_spans` builds `JudgeCase`s
from the same runs, so a judge can be certified on real traffic.

Spans are accepted as OpenTelemetry `ReadableSpan`s, `ReadableSpan.to_json()` dicts, Logfire's
`exported_spans_as_dict()` dicts, Logfire query rows (`span_name`, `attributes`, `trace_id`, ...,
as `LogfireQueryClient.query_json_rows` returns them), and OTLP/JSON (one span, or a whole
`{"resourceSpans": [...]}` document). Logfire's pending spans are skipped and a span seen twice is
read once.

What pydantic-ai emits, and what is read here (pydantic-ai `origin/main` 66951321, 2026-10-02):

- The agent run span. `capabilities/instrumentation.py` `Instrumentation.wrap_run`. Its name is
  `invoke_agent {agent_name}` (instrumentation versions 3-6) or `agent run` (version 2), from
  `InstrumentationNames.for_version` in `_instrumentation.py`. A run span is recognised by
  `gen_ai.operation.name == 'invoke_agent'` or either name. Attributes read:
    - `gen_ai.agent.name`, falling back to `agent_name` (both set; version 2 spans have `agent_name`);
    - `gen_ai.agent.call.id` (the run id) and `gen_ai.conversation.id`;
    - `model_name`;
    - `pydantic_ai.all_messages` (versions 2-6): the run's whole history as OTel GenAI messages,
      `[{role, parts: [{type: text|tool_call|tool_call_response|thinking|uri|blob, ...}]}]`, built by
      `InstrumentationSettings.messages_to_otel_messages` (`models/instrumented.py`). Version 6 gives
      tool results `role='tool'` instead of `'user'`; both are read.
    - `pydantic_ai.new_message_index`: where this run's messages start, set only when it is > 0,
      i.e. when the run was given `message_history`. Messages before it are the history.
    - `all_messages_events` (version 1, removed in "Pydantic AI V2", commit e2b661cb): a list of
      event dicts (`role`, `content`, `tool_calls`, `id`, `name`). Read best-effort; version 1 has no
      `new_message_index`, so the prompt is taken as the last user message with text.
    - `final_result`: the output. A `str` output is recorded as is; any other output as JSON. A
      string that parses as JSON is therefore ambiguous: it is kept a string when it equals the
      text of the run's last assistant message, and parsed otherwise. Pass `output_type` to say.
    - `gen_ai.system_instructions` (`[{type: 'text', content}]`) and `metadata` (JSON).
    - Status: an ERROR span status, or an `exception` event (`_record_uncaught_errors`). Logfire
      dicts carry no status, so there it is the event or `logfire.level_num >= 17`.
- Tool spans, `execute_tool {tool}` (version 3+) or `running tool` (version 2), from
  `Instrumentation._tool_span_attributes` / `_run_tool_span`: `gen_ai.tool.name`,
  `gen_ai.tool.call.id`, and an ERROR status when the tool raised. Arguments and results are read
  from the run's messages, which every version records; a tool span only adds whether that call
  failed, matched by `gen_ai.tool.call.id` within the trace.

With `include_content=False` none of the content is recorded: such runs have no prompt and are
skipped (`agent_runs` still returns them, with `problems`).

A recorded output is what the agent said, not what it should have said. `dataset_from_spans` puts
it in metadata (`recorded_output`) unless `output_as='expected_output'` is asked for, and
`judge_cases_from_spans` makes no case a known-good answer until `accepted` says it is one.
"""

from __future__ import annotations

import json
import random
import warnings
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

from ._cases import JudgeCase

__all__ = (
    'TracedRun',
    'agent_runs',
    'dataset_from_spans',
    'judge_cases_from_spans',
    'history_messages',
)

_RUN_NAMES = ('agent run',)
_TOOL_NAMES = ('running tool',)
_ERROR_LEVEL = 17  # Logfire's 'error' level number


# ---------------------------------------------------------------------------------------------
# Spans, whatever shape they arrive in
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Span:
    trace_id: str
    span_id: str
    parent_id: str | None
    name: str
    attributes: Mapping[str, Any]
    error: str | None
    start: datetime | None


def _hex(value: Any, width: int) -> str | None:
    """A trace or span id as lowercase hex of `width` digits, from an int, '0x..' or hex string."""
    if value is None or value == '':
        return None
    if isinstance(value, int):
        return format(value, f'0{width}x')
    text = str(value).lower()
    if text.startswith('0x'):
        text = text[2:]
    return text.zfill(width)


def _time(value: Any) -> datetime | None:
    if value is None or value == '':
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)) or (isinstance(value, str) and value.isdigit()):
        return datetime.fromtimestamp(int(value) / 1e9, tz=timezone.utc)
    try:
        moment = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    except ValueError:
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def _otlp_value(value: Mapping[str, Any]) -> Any:
    """One OTLP/JSON `AnyValue`."""
    for key in ('stringValue', 'boolValue', 'doubleValue'):
        if key in value:
            return value[key]
    if 'intValue' in value:
        return int(value['intValue'])
    if 'arrayValue' in value:
        return [_otlp_value(v) for v in value['arrayValue'].get('values', [])]
    if 'kvlistValue' in value:
        return {kv['key']: _otlp_value(kv.get('value', {})) for kv in value['kvlistValue'].get('values', [])}
    return None


def _event_error(events: Iterable[Any]) -> str | None:
    for event in events:
        if isinstance(event, Mapping):
            name, attributes = event.get('name'), event.get('attributes') or {}  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
        else:
            name, attributes = getattr(event, 'name', None), getattr(event, 'attributes', None) or {}
        if name == 'exception':
            kind = attributes.get('exception.type') or 'exception'  # pyright: ignore[reportUnknownMemberType]
            message = attributes.get('exception.message')  # pyright: ignore[reportUnknownMemberType]
            return f'{kind}: {message}' if message else str(kind)  # pyright: ignore[reportUnknownArgumentType]
    return None


def _from_readable(span: Any) -> _Span | None:
    context = span.context
    if context is None:
        return None
    status = getattr(span, 'status', None)
    code = getattr(getattr(status, 'status_code', None), 'name', None)
    error = _event_error(span.events or ())
    if code == 'ERROR' and error is None:
        error = getattr(status, 'description', None) or 'ERROR'
    return _Span(
        trace_id=_hex(context.trace_id, 32) or '',
        span_id=_hex(context.span_id, 16) or '',
        parent_id=_hex(span.parent.span_id, 16) if span.parent else None,
        name=span.name,
        attributes=dict(span.attributes or {}),
        error=error,
        start=_time(span.start_time),
    )


def _from_dict(span: Mapping[str, Any]) -> _Span | None:
    if 'span_name' in span:  # a Logfire query row
        attributes = span.get('attributes') or {}
        if isinstance(attributes, str):
            attributes = json.loads(attributes)
        error = None
        if span.get('otel_status_code') == 'ERROR' or span.get('is_exception'):
            kind, message = span.get('exception_type'), span.get('exception_message') or span.get('otel_status_message')
            error = f'{kind}: {message}' if kind and message else (kind or message or 'ERROR')
        return _Span(
            trace_id=_hex(span.get('trace_id'), 32) or '',
            span_id=_hex(span.get('span_id'), 16) or '',
            parent_id=_hex(span.get('parent_span_id'), 16),
            name=span['span_name'],
            attributes=attributes,
            error=error,
            start=_time(span.get('start_timestamp')),
        )
    if 'traceId' in span:  # OTLP/JSON
        attributes = {kv['key']: _otlp_value(kv.get('value', {})) for kv in span.get('attributes', [])}
        status = span.get('status') or {}
        error = _event_error(
            {
                'name': e.get('name'),
                'attributes': {kv['key']: _otlp_value(kv.get('value', {})) for kv in e.get('attributes', [])},
            }
            for e in span.get('events', [])
        )
        if status.get('code') in (2, 'STATUS_CODE_ERROR') and error is None:
            error = status.get('message') or 'ERROR'
        return _Span(
            trace_id=_hex(span['traceId'], 32) or '',
            span_id=_hex(span.get('spanId'), 16) or '',
            parent_id=_hex(span.get('parentSpanId'), 16),
            name=span.get('name', ''),
            attributes=attributes,
            error=error,
            start=_time(span.get('startTimeUnixNano')),
        )
    if 'context' in span:  # Logfire's exported_spans_as_dict, or ReadableSpan.to_json
        context = span['context'] or {}
        parent = span.get('parent') or {}
        parent_id = parent.get('span_id') if isinstance(parent, Mapping) else None
        if parent_id is None:
            parent_id = span.get('parent_id')
        attributes = span.get('attributes') or {}
        error = _event_error(span.get('events') or ())
        status = span.get('status') or {}
        if error is None and status.get('status_code') == 'ERROR':
            error = status.get('description') or 'ERROR'
        level = attributes.get('logfire.level_num')
        if error is None and isinstance(level, int) and level >= _ERROR_LEVEL:
            error = 'ERROR'
        return _Span(
            trace_id=_hex(context.get('trace_id'), 32) or '',
            span_id=_hex(context.get('span_id'), 16) or '',
            parent_id=_hex(parent_id, 16),
            name=span.get('name', ''),
            attributes=attributes,
            error=error,
            start=_time(span.get('start_time')),
        )
    return None


def _flatten(spans: Any) -> list[Any]:
    """`spans` as a flat list: OTLP/JSON documents (`resourceSpans`) and JSON text unpacked."""
    if isinstance(spans, (str, bytes)):
        spans = json.loads(spans)
    if isinstance(spans, Mapping):
        spans = [spans]
    out: list[Any] = []
    for item in spans:
        if isinstance(item, Mapping) and 'resourceSpans' in item:
            for resource in item['resourceSpans']:
                for scope in resource.get('scopeSpans', resource.get('instrumentationLibrarySpans', [])):
                    out.extend(scope.get('spans', []))
        else:
            out.append(item)
    return out


def _normalise(spans: Any) -> list[_Span]:
    """Spans in one shape, each read once, pending spans dropped. Unreadable items are skipped."""
    seen: set[tuple[str, str]] = set()
    out: list[_Span] = []
    for item in _flatten(spans):
        try:
            span = _from_dict(item) if isinstance(item, Mapping) else _from_readable(item)
        except (AttributeError, KeyError, TypeError, ValueError):
            span = None
        if span is None or span.attributes.get('logfire.span_type') == 'pending_span':
            continue
        key = (span.trace_id, span.span_id)
        if key in seen and span.span_id:
            continue
        seen.add(key)
        out.append(span)
    return out


# ---------------------------------------------------------------------------------------------
# Agent runs
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class TracedRun:
    """One agent run, read from its span.

    `prompt` is this run's user prompt: a string, or the list of OTel message parts when it was
    multimodal. `history` holds the messages the run was given (`message_history`), as OTel GenAI
    messages; `history_messages` turns them back into pydantic-ai messages. `tool_calls` are this
    run's calls, in order: `{id, name, arguments, result}`, plus `error` when the call's tool span
    failed. `output` is the recorded `final_result`, None when the run failed or content was not
    recorded. `output_valid` is False when an `output_type` was given and the recorded output does
    not fit it: `output` then holds the raw recorded value, which is never used as an expected
    output or a known-good answer. `problems` says what could not be read; a run with no prompt is
    not a case.
    `instrumentation_version` is the message format read: 1, 2, or 3 for versions 3 to 6, which
    record messages alike (6 differs only in the tool-result role).
    """

    trace_id: str
    span_id: str
    agent_name: str | None
    run_id: str | None = None
    conversation_id: str | None = None
    model: str | None = None
    start_time: datetime | None = None
    prompt: Any = None
    history: tuple[Mapping[str, Any], ...] = ()
    instructions: str | None = None
    tool_calls: tuple[Mapping[str, Any], ...] = ()
    output: Any = None
    output_recorded: bool = False
    output_valid: bool = True
    error: str | None = None
    nested: bool = False
    run_metadata: Any = None
    instrumentation_version: Literal[1, 2, 3] | None = None
    problems: tuple[str, ...] = field(default=())

    @property
    def name(self) -> str:
        """The case name: agent, trace and span, so it is unique and leads back to the span."""
        return f'{self.agent_name or "agent"}-{self.trace_id[:8]}-{self.span_id[:8]}'


def _json(value: Any) -> Any:
    if isinstance(value, (str, bytes)):
        return json.loads(value)
    return value


def _is_run(span: _Span) -> bool:
    return (
        span.attributes.get('gen_ai.operation.name') == 'invoke_agent'
        or span.name in _RUN_NAMES
        or span.name.startswith('invoke_agent ')
    )


def _is_tool(span: _Span) -> bool:
    return (
        span.attributes.get('gen_ai.operation.name') == 'execute_tool'
        or span.name in _TOOL_NAMES
        or span.name.startswith('execute_tool ')
    )


def _v1_messages(events: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Version-1 `all_messages_events` as OTel GenAI messages, so one reader serves every version."""
    out: list[dict[str, Any]] = []
    for event in events:
        role = event.get('role')
        parts: list[dict[str, Any]] = []
        content = event.get('content')
        if role == 'tool':
            parts.append(
                {'type': 'tool_call_response', 'id': event.get('id'), 'name': event.get('name'), 'result': content}
            )
        else:
            items: list[Any] = content if isinstance(content, list) else ([] if content is None else [content])  # pyright: ignore[reportUnknownVariableType]
            for item in items:
                if isinstance(item, str):
                    parts.append({'type': 'text', 'content': item})
                elif isinstance(item, Mapping) and item.get('kind') in ('text', 'thinking'):  # pyright: ignore[reportUnknownMemberType]
                    parts.append({'type': item['kind'], 'content': item.get('text')})  # pyright: ignore[reportUnknownMemberType]
                else:
                    parts.append(dict(item) if isinstance(item, Mapping) else {'type': 'unknown', 'content': item})  # pyright: ignore[reportUnknownArgumentType]
            for call in event.get('tool_calls') or []:
                function = call.get('function') or {}
                parts.append(
                    {
                        'type': 'tool_call',
                        'id': call.get('id'),
                        'name': function.get('name'),
                        'arguments': function.get('arguments'),
                    }
                )
        out.append({'role': role, 'parts': parts})
    return out


def _texts(message: Mapping[str, Any]) -> list[str]:
    return [
        p['content'] for p in message.get('parts', []) if p.get('type') == 'text' and isinstance(p.get('content'), str)
    ]


def _prompt_of(message: Mapping[str, Any]) -> Any:
    """A user message's prompt: its text, or its parts when it held anything but text."""
    parts = [p for p in message.get('parts', []) if p.get('type') not in ('tool_call_response',)]
    if parts and all(p.get('type') == 'text' and isinstance(p.get('content'), str) for p in parts):
        return '\n'.join(p['content'] for p in parts)
    return parts or None


def _has_prompt(message: Mapping[str, Any]) -> bool:
    return message.get('role') == 'user' and any(
        p.get('type') != 'tool_call_response' and ('content' in p or 'uri' in p) for p in message.get('parts', [])
    )


def _tool_calls(messages: Sequence[Mapping[str, Any]], failed: Mapping[str, str]) -> tuple[dict[str, Any], ...]:
    calls: list[dict[str, Any]] = []
    by_id: dict[Any, dict[str, Any]] = {}
    for message in messages:
        for part in message.get('parts', []):
            if part.get('type') == 'tool_call':
                call = {'id': part.get('id'), 'name': part.get('name'), 'arguments': part.get('arguments')}
                calls.append(call)
                if part.get('id') is not None:
                    by_id[part['id']] = call
            elif part.get('type') == 'tool_call_response' and part.get('id') in by_id:
                by_id[part['id']]['result'] = part.get('result')
    for call in calls:
        if call['id'] in failed:
            call['error'] = failed[call['id']]
    return tuple(calls)


def _output(raw: Any, messages: Sequence[Mapping[str, Any]], output_type: Any) -> Any:
    if output_type is not None:
        from pydantic import TypeAdapter

        adapter: TypeAdapter[Any] = TypeAdapter(output_type)
        if isinstance(raw, str) and output_type is not str:
            return adapter.validate_json(raw)
        return adapter.validate_python(raw)
    if not isinstance(raw, str):
        return raw
    last = next((m for m in reversed(messages) if m.get('role') == 'assistant'), None)
    if last is not None and ''.join(_texts(last)) == raw:
        return raw  # the agent answered in text: a str output, even if it looks like JSON
    try:
        return json.loads(raw)
    except ValueError:
        return raw


def _run(span: _Span, failed: Mapping[str, str], nested: bool, output_type: Any) -> TracedRun:
    attrs = span.attributes
    problems: list[str] = []
    version: Literal[1, 2, 3] | None = None
    messages: list[dict[str, Any]] = []
    try:
        if 'pydantic_ai.all_messages' in attrs:
            messages = list(_json(attrs['pydantic_ai.all_messages']))
            version = 2 if span.name in _RUN_NAMES else 3
        elif 'all_messages_events' in attrs:
            messages = _v1_messages(list(_json(attrs['all_messages_events'])))
            version = 1
        else:
            problems.append('no messages recorded')
    except (TypeError, ValueError) as e:
        problems.append(f'messages unreadable: {e}')
    messages = [m for m in messages if isinstance(m, Mapping)]  # pyright: ignore[reportUnnecessaryIsInstance]

    start: int | None = None
    if version in (2, 3):
        index = attrs.get('pydantic_ai.new_message_index', 0)
        index = index if isinstance(index, int) and 0 <= index <= len(messages) else 0
        start = next((i for i in range(index, len(messages)) if _has_prompt(messages[i])), None)
    elif version == 1:
        start = next((i for i in reversed(range(len(messages))) if _has_prompt(messages[i])), None)
    if messages and start is None:
        problems.append('no user prompt in the messages (content not recorded?)')

    prompt = _prompt_of(messages[start]) if start is not None else None
    history = tuple(messages[:start]) if start is not None else ()
    own = messages[start:] if start is not None else messages

    output, recorded, valid = None, 'final_result' in attrs, True
    if recorded:
        try:
            output = _output(attrs['final_result'], own, output_type)
        except ValueError as e:  # pydantic's ValidationError is a ValueError
            problems.append(f'output does not fit output_type: {e}')
            output, valid = attrs['final_result'], False
    elif span.error is None:
        problems.append('no final_result recorded')

    instructions = None
    try:
        system = _json(attrs['gen_ai.system_instructions']) if 'gen_ai.system_instructions' in attrs else []
        instructions = '\n'.join(p.get('content', '') for p in system if isinstance(p, Mapping)) or None
    except (TypeError, ValueError):
        problems.append('instructions unreadable')
    run_metadata = attrs.get('metadata')
    if isinstance(run_metadata, str):
        try:
            run_metadata = json.loads(run_metadata)
        except ValueError:
            pass

    return TracedRun(
        trace_id=span.trace_id,
        span_id=span.span_id,
        agent_name=attrs.get('gen_ai.agent.name') or attrs.get('agent_name'),
        run_id=attrs.get('gen_ai.agent.call.id') or None,
        conversation_id=attrs.get('gen_ai.conversation.id') or None,
        model=attrs.get('model_name'),
        start_time=span.start,
        prompt=prompt,
        history=history,
        instructions=instructions,
        tool_calls=_tool_calls(own, failed),
        output=output,
        output_recorded=recorded,
        output_valid=valid,
        error=span.error,
        nested=nested,
        run_metadata=run_metadata,
        instrumentation_version=version,
        problems=tuple(problems),
    )


def agent_runs(spans: Any, *, output_type: Any = None) -> list[TracedRun]:
    """Every pydantic-ai agent run in `spans`, oldest first, including runs that cannot be a case.

    `spans` is any of the shapes in the module docstring, mixed freely. `output_type`, when given,
    validates each recorded output (a string recorded for a non-`str` type is read as JSON).
    """
    normal = _normalise(spans)
    by_id = {(s.trace_id, s.span_id): s for s in normal}
    failed: dict[tuple[str, Any], str] = {}
    for s in normal:
        if _is_tool(s) and s.error is not None and s.attributes.get('gen_ai.tool.call.id') is not None:
            failed[(s.trace_id, s.attributes['gen_ai.tool.call.id'])] = s.error

    def inside_a_run(span: _Span) -> bool:
        seen: set[str] = set()
        parent = span.parent_id
        while parent is not None and parent not in seen:
            seen.add(parent)
            above = by_id.get((span.trace_id, parent))
            if above is None:
                return False
            if _is_run(above):
                return True
            parent = above.parent_id
        return False

    runs = []
    for s in normal:
        if not _is_run(s):
            continue
        trace_failed = {call_id: err for (trace, call_id), err in failed.items() if trace == s.trace_id}
        runs.append(_run(s, trace_failed, inside_a_run(s), output_type))
    order = {id(r): i for i, r in enumerate(runs)}
    epoch = datetime.min.replace(tzinfo=timezone.utc)
    return sorted(runs, key=lambda r: (r.start_time or epoch, order[id(r)]))


# ---------------------------------------------------------------------------------------------
# Selection: filter, deduplicate, sample
# ---------------------------------------------------------------------------------------------


def _key(run: TracedRun) -> str:
    return json.dumps([run.prompt, list(run.history)], sort_keys=True, default=str)


def _select(
    runs: list[TracedRun],
    *,
    agent_name: str | Collection[str] | None,
    errors: Literal['include', 'exclude', 'only'],
    nested: bool,
    where: Callable[[TracedRun], bool] | None,
    deduplicate: bool,
    sample: int | None,
    seed: int,
) -> list[tuple[TracedRun, list[TracedRun]]]:
    """The runs to make cases of, each with the later runs that had the same inputs.

    Every filter, `where` included, is applied to each run before runs are deduplicated, and
    sampling comes last: a run that is filtered out never hides a run that is not.
    """
    if errors not in ('include', 'exclude', 'only'):
        raise ValueError("errors must be 'include', 'exclude' or 'only'")
    if sample is not None and sample < 0:
        raise ValueError('sample must be >= 0')
    names = {agent_name} if isinstance(agent_name, str) else (set(agent_name) if agent_name is not None else None)
    kept = [
        r
        for r in runs
        if r.prompt is not None
        and (names is None or r.agent_name in names)
        and (errors == 'include' or (errors == 'only') == (r.error is not None))
        and (nested or not r.nested)
        and (where is None or where(r))
    ]
    skipped = [r for r in runs if r.prompt is None and (names is None or r.agent_name in names)]
    if skipped:
        warnings.warn(f'{len(skipped)} agent run(s) skipped: no user prompt recorded', stacklevel=3)
    groups: dict[str, tuple[TracedRun, list[TracedRun]]] = {}
    chosen: list[tuple[TracedRun, list[TracedRun]]] = []
    for r in kept:
        if not deduplicate:
            chosen.append((r, []))
            continue
        key = _key(r)
        if key in groups:
            groups[key][1].append(r)
        else:
            groups[key] = (r, [])
            chosen.append(groups[key])
    if sample is not None and sample < len(chosen):
        picked = sorted(random.Random(seed).sample(range(len(chosen)), sample))
        chosen = [chosen[i] for i in picked]
    return chosen


def _present(values: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in values.items() if v is not None and v != [] and v != ()}


def _metadata(run: TracedRun, duplicates: Sequence[TracedRun]) -> dict[str, Any]:
    return _present(
        {
            'source': 'trace',
            'trace_id': run.trace_id,
            'span_id': run.span_id,
            'run_id': run.run_id,
            'conversation_id': run.conversation_id,
            'agent_name': run.agent_name,
            'model': run.model,
            'start_time': run.start_time.isoformat() if run.start_time else None,
            'instructions': run.instructions,
            'tool_calls': [dict(c) for c in run.tool_calls],
            'recorded_output': run.output,
            'output_valid': None if run.output_valid else False,
            'error': run.error,
            'problems': list(run.problems),
            'run_metadata': run.run_metadata,
            'duplicates': [{'trace_id': d.trace_id, 'span_id': d.span_id} for d in duplicates],
        }
    )


def _default_inputs(run: TracedRun, include_history: bool) -> Any:
    if not include_history:
        return run.prompt
    return {'prompt': run.prompt, 'message_history': [dict(m) for m in run.history]}


# ---------------------------------------------------------------------------------------------
# Public builders
# ---------------------------------------------------------------------------------------------


def dataset_from_spans(
    spans: Any,
    *,
    name: str = 'production',
    agent_name: str | Collection[str] | None = None,
    errors: Literal['include', 'exclude', 'only'] = 'include',
    nested: bool = True,
    where: Callable[[TracedRun], bool] | None = None,
    deduplicate: bool = True,
    sample: int | None = None,
    seed: int = 0,
    output_as: Literal['metadata', 'expected_output'] = 'metadata',
    include_history: bool = True,
    inputs: Callable[[TracedRun], Any] | None = None,
    input_type: Any = None,
    output_type: Any = None,
    evaluators: Sequence[Any] = (),
) -> Any:
    """A pydantic-evals `Dataset` with one `Case` per recorded agent run.

    Each case's `inputs` is `{'prompt': ..., 'message_history': [...]}` (the history as OTel GenAI
    messages; `history_messages` turns it into pydantic-ai messages to pass back to the agent), or
    the prompt alone with `include_history=False`, or `inputs(run)`, then validated as `input_type`
    when given. Its `metadata` links back to the span and holds what the run did: `trace_id`,
    `span_id`, `run_id`, `conversation_id`, `agent_name`, `model`, `start_time`, `instructions`,
    `tool_calls`, `recorded_output`, `output_valid` (present, False, only when the output does not
    fit `output_type`), `error`, `problems`, `run_metadata` and `duplicates` (absent when empty).

    A production output is what the agent said, not ground truth: it is `recorded_output` in
    metadata, and `expected_output` only with `output_as='expected_output'`. Failed runs, and runs
    whose output does not fit `output_type`, have none.

    Args:
        spans: Spans in any shape the module docstring lists.
        name: The dataset's name.
        agent_name: Keep runs of this agent (or these agents) only.
        errors: Keep failed runs too (`'include'`), drop them, or keep only them.
        nested: Keep runs made inside another run (an agent called from a tool).
        where: Keep runs for which this returns True.
        deduplicate: One case per distinct (prompt, history); the earliest run is kept, the others
            are listed in its `duplicates`. Applied before sampling.
        sample: Keep this many cases, chosen at random with `seed`, in time order.
        seed: The sampling seed.
        output_as: Where the recorded output goes besides metadata.
        include_history: Whether the default inputs carry the run's message history.
        inputs: Builds a case's inputs from its run, instead of the default.
        input_type: Validate inputs as this type (e.g. a model with `prompt` and `message_history`).
        output_type: Validate recorded outputs as this type.
        evaluators: The dataset's evaluators.
    """
    from pydantic import TypeAdapter
    from pydantic_evals import Case, Dataset

    if output_as not in ('metadata', 'expected_output'):
        raise ValueError("output_as must be 'metadata' or 'expected_output'")
    chosen = _select(
        agent_runs(spans, output_type=output_type),
        agent_name=agent_name,
        errors=errors,
        nested=nested,
        where=where,
        deduplicate=deduplicate,
        sample=sample,
        seed=seed,
    )
    adapter: TypeAdapter[Any] | None = TypeAdapter(input_type) if input_type is not None else None
    cases: list[Case[Any, Any, Any]] = []
    for run, duplicates in chosen:
        value = inputs(run) if inputs is not None else _default_inputs(run, include_history)
        if adapter is not None:
            value = adapter.validate_python(value)
        usable = run.error is None and run.output_recorded and run.output_valid
        expected = run.output if output_as == 'expected_output' and usable else None
        cases.append(Case(name=run.name, inputs=value, metadata=_metadata(run, duplicates), expected_output=expected))
    return Dataset[Any, Any, Any](name=name, cases=cases, evaluators=list(evaluators))


def judge_cases_from_spans(
    spans: Any,
    *,
    accepted: Callable[[TracedRun], bool] | Collection[str] | Literal['all'],
    agent_name: str | Collection[str] | None = None,
    nested: bool = True,
    where: Callable[[TracedRun], bool] | None = None,
    deduplicate: bool = True,
    sample: int | None = None,
    seed: int = 0,
    include_history: bool = True,
    inputs: Callable[[TracedRun], Any] | None = None,
    expected_output: Callable[[TracedRun], Any] | None = None,
    output_type: Any = None,
) -> list[JudgeCase]:
    """`JudgeCase`s from recorded runs, so a judge is certified on the traffic it will grade.

    A `JudgeCase` is a known-good answer the judge must pass, and a recorded output is not known
    to be good. `accepted` says which are: a predicate, the case names (`TracedRun.name`, the same
    names `dataset_from_spans` gives) a person approved, or `'all'` to take every output as good,
    which only makes sense for traffic already reviewed. Failed runs, runs with no recorded output,
    and runs whose output does not fit `output_type` are never cases.

    Acceptance is decided per run, before deduplication: of several runs with the same inputs, the
    earliest accepted one is the case, and only accepted runs are listed in its `duplicates`. An
    unaccepted answer never displaces an accepted one. The controls (`must_fail`, `must_hold`) need no review: an empty
    output, or another customer's answer, is wrong for any input.

    Case names match `dataset_from_spans`'s, so `HumanLabel`s keyed by them line up. `inputs` and
    metadata are built as there; `expected_output(run)`, when given, sets each case's reference.
    """
    if isinstance(accepted, str) and accepted != 'all':
        raise ValueError("accepted must be a predicate, a collection of case names, or 'all'")
    names = None if callable(accepted) or accepted == 'all' else set(accepted)

    def eligible(run: TracedRun) -> bool:
        if not (run.output_recorded and run.output_valid) or (where is not None and not where(run)):
            return False
        if callable(accepted):
            return accepted(run)
        return names is None or run.name in names

    # Every per-run decision is made before `_select` deduplicates, so a duplicate that is not a
    # known-good answer cannot take the place of one that is.
    chosen = _select(
        agent_runs(spans, output_type=output_type),
        agent_name=agent_name,
        errors='exclude',
        nested=nested,
        where=eligible,
        deduplicate=deduplicate,
        sample=sample,
        seed=seed,
    )
    out: list[JudgeCase] = []
    for run, duplicates in chosen:
        out.append(
            JudgeCase(
                name=run.name,
                inputs=inputs(run) if inputs is not None else _default_inputs(run, include_history),
                output=run.output,
                expected_output=expected_output(run) if expected_output is not None else None,
                metadata=_metadata(run, duplicates),
            )
        )
    return out


def history_messages(history: Sequence[Mapping[str, Any]]) -> list[Any]:
    """OTel GenAI messages (a case's `message_history`) as pydantic-ai `ModelMessage`s.

    Text, thinking, tool calls and tool results are converted; other parts (media, provider-native
    tools) are dropped, since their content may not have been recorded. Pass the result as
    `message_history` to re-run the agent on a recorded conversation.
    """
    from pydantic_ai.messages import (
        ModelMessage,
        ModelRequest,
        ModelRequestPart,
        ModelResponse,
        ModelResponsePart,
        SystemPromptPart,
        TextPart,
        ThinkingPart,
        ToolCallPart,
        ToolReturnPart,
        UserPromptPart,
    )

    out: list[ModelMessage] = []
    for message in history:
        role, parts = message.get('role'), message.get('parts', [])
        if role == 'assistant':
            response: list[ModelResponsePart] = []
            for p in parts:
                if p.get('type') == 'text':
                    response.append(TextPart(p.get('content', '')))
                elif p.get('type') == 'thinking':
                    response.append(ThinkingPart(p.get('content', '')))
                elif p.get('type') == 'tool_call':
                    response.append(ToolCallPart(p.get('name', ''), p.get('arguments'), tool_call_id=p.get('id') or ''))
            if response:
                out.append(ModelResponse(parts=response))
            continue
        request: list[ModelRequestPart] = []
        for p in parts:
            if p.get('type') == 'tool_call_response':
                request.append(ToolReturnPart(p.get('name', ''), p.get('result'), tool_call_id=p.get('id') or ''))
            elif p.get('type') == 'text':
                cls = SystemPromptPart if role == 'system' else UserPromptPart
                request.append(cls(p.get('content', '')))
        if request and out and isinstance(out[-1], ModelRequest):
            out[-1] = ModelRequest(parts=[*out[-1].parts, *request])  # tool results then the next prompt
        elif request:
            out.append(ModelRequest(parts=request))
    return out
