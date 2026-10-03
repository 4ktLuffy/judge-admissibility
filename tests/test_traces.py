"""Datasets and judge cases from the spans of real pydantic-ai agent runs, in every span shape and version."""

# Span contexts and case metadata are Optional in their stubs; here they are always set.
# pyright: reportOptionalMemberAccess=false, reportOptionalSubscript=false, reportOperatorIssue=false

from __future__ import annotations

import json
import warnings
from collections.abc import Sequence
from typing import Any

import pytest
from judges import judge, oracle
from logfire.testing import CaptureLogfire
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import BaseModel
from pydantic_ai import Agent, ModelRetry
from pydantic_ai.capabilities import Instrumentation
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.models.test import TestModel
from pydantic_evals import Dataset

from pydantic_evals_admissibility import certify_judge
from pydantic_evals_admissibility._traces import (
    agent_runs,
    dataset_from_spans,
    history_messages,
    judge_cases_from_spans,
)


class Capture:
    """A tracer provider with an in-memory exporter, and agents instrumented to it."""

    def __init__(self, version: int = 5, include_content: bool = True) -> None:
        self.exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(self.exporter))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # versions 2-4 are deprecated
            self.settings = InstrumentationSettings(
                tracer_provider=provider,
                version=version,  # pyright: ignore[reportArgumentType]
                include_content=include_content,
            )

    def instrument(self) -> Instrumentation:
        return Instrumentation(settings=self.settings)

    @property
    def spans(self) -> tuple[ReadableSpan, ...]:
        return self.exporter.get_finished_spans()


def last_prompt(messages: Sequence[ModelMessage]) -> str:
    return next(
        p.content
        for m in reversed(messages)
        for p in reversed(getattr(m, 'parts', []))
        if isinstance(p, UserPromptPart) and isinstance(p.content, str)
    )


def support_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Looks up the order named in the prompt, then answers with what the tool said."""
    last = messages[-1]
    returns = [p for p in getattr(last, 'parts', []) if isinstance(p, ToolReturnPart)]
    if returns:
        return ModelResponse(parts=[TextPart(f'Your order: {returns[0].content}.')])
    order = int(''.join(c for c in last_prompt(messages) if c.isdigit()) or 0)
    return ModelResponse(parts=[ToolCallPart('lookup', {'order': order}, tool_call_id=f'call-{order}')])


def support_agent(capture: Capture, name: str = 'support') -> Agent[None, str]:
    agent = Agent(
        FunctionModel(support_model, model_name='support-model'),
        name=name,
        instructions='Answer order questions.',
        capabilities=[capture.instrument()],
    )

    @agent.tool_plain
    def lookup(order: int) -> str:
        return f'order {order} shipped'

    return agent


def run_span(spans: Sequence[ReadableSpan]) -> ReadableSpan:
    [span] = [s for s in spans if (s.attributes or {}).get('gen_ai.operation.name') == 'invoke_agent']
    return span


def test_a_run_with_a_tool_call_becomes_a_case_with_every_field() -> None:
    capture = Capture()
    agent = support_agent(capture)
    result = agent.run_sync('Where is order 7?')
    dataset = dataset_from_spans(capture.spans, name='support-prod')
    span = run_span(capture.spans)
    trace_id, span_id = format(span.context.trace_id, '032x'), format(span.context.span_id, '016x')

    assert isinstance(dataset, Dataset) and dataset.name == 'support-prod'
    [case] = dataset.cases
    assert case.name == f'support-{trace_id[:8]}-{span_id[:8]}'
    assert case.inputs == {'prompt': 'Where is order 7?', 'message_history': []}
    assert case.expected_output is None  # a production output is not ground truth
    meta = case.metadata
    assert meta['source'] == 'trace'
    assert meta['trace_id'] == trace_id and meta['span_id'] == span_id
    assert meta['run_id'] == result.run_id and meta['conversation_id'] == result.conversation_id
    assert meta['agent_name'] == 'support' and meta['model'] == 'support-model'
    assert meta['instructions'] == 'Answer order questions.'
    assert meta['tool_calls'] == [
        {'id': 'call-7', 'name': 'lookup', 'arguments': {'order': 7}, 'result': 'order 7 shipped'}
    ]
    assert meta['recorded_output'] == result.output == 'Your order: order 7 shipped.'
    assert meta['start_time'].startswith('20')
    assert 'error' not in meta and 'duplicates' not in meta


@pytest.mark.parametrize('version', [2, 3, 4, 5, 6])
def test_every_instrumentation_version_gives_the_same_case(version: int) -> None:
    capture = Capture(version)
    support_agent(capture).run_sync('Where is order 3?')
    [run] = agent_runs(capture.spans)
    assert run.instrumentation_version == (2 if version == 2 else 3)
    assert run.agent_name == 'support' and run.prompt == 'Where is order 3?' and run.history == ()
    assert [dict(c) for c in run.tool_calls] == [
        {'id': 'call-3', 'name': 'lookup', 'arguments': {'order': 3}, 'result': 'order 3 shipped'}
    ]
    assert run.output == 'Your order: order 3 shipped.' and run.error is None and run.problems == ()


def test_a_continued_conversation_keeps_its_history_and_replays() -> None:
    capture = Capture(6)  # tool results under role='tool'
    agent = support_agent(capture)
    first = agent.run_sync('Where is order 1?')
    agent.run_sync('And order 2?', message_history=first.all_messages())
    dataset = dataset_from_spans(capture.spans)
    first_case, second_case = dataset.cases
    assert first_case.inputs['message_history'] == []
    assert second_case.inputs['prompt'] == 'And order 2?'
    history = second_case.inputs['message_history']
    assert [m['role'] for m in history] == ['user', 'assistant', 'tool', 'assistant']
    assert second_case.metadata['tool_calls'][0]['arguments'] == {'order': 2}  # only this run's calls
    assert second_case.metadata['conversation_id'] == first_case.metadata['conversation_id']

    replayed = history_messages(history)
    assert [type(m).__name__ for m in replayed] == ['ModelRequest', 'ModelResponse', 'ModelRequest', 'ModelResponse']
    again = Agent(TestModel(call_tools=[], custom_output_text='ok')).run_sync('And order 2?', message_history=replayed)
    assert again.all_messages()[: len(replayed)] == replayed and again.output == 'ok'
    assert last_prompt(replayed) == 'Where is order 1?'


def test_include_history_false_makes_the_prompt_the_input() -> None:
    capture = Capture()
    support_agent(capture).run_sync('Where is order 4?')
    [case] = dataset_from_spans(capture.spans, include_history=False).cases
    assert case.inputs == 'Where is order 4?'


class Refund(BaseModel):
    order: int
    approved: bool


def test_structured_output_round_trips_and_can_be_the_expected_output() -> None:
    capture = Capture()
    agent = Agent(
        TestModel(custom_output_args={'order': 9, 'approved': True}),
        output_type=Refund,
        name='refunds',
        capabilities=[capture.instrument()],
    )
    agent.run_sync('Refund order 9')
    [plain] = dataset_from_spans(capture.spans).cases
    assert plain.metadata['recorded_output'] == {'order': 9, 'approved': True}
    [typed] = dataset_from_spans(capture.spans, output_type=Refund, output_as='expected_output').cases
    assert typed.expected_output == Refund(order=9, approved=True)
    assert typed.metadata['recorded_output'] == Refund(order=9, approved=True)


def test_a_text_output_that_looks_like_json_stays_text() -> None:
    capture = Capture()
    Agent(TestModel(custom_output_text='{"a": 1}'), capabilities=[capture.instrument()]).run_sync('json please')
    [run] = agent_runs(capture.spans)
    assert run.output == '{"a": 1}'


def failing_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    if 'crash' in last_prompt(messages):
        raise RuntimeError('provider down')
    return ModelResponse(parts=[TextPart('fine')])


def test_failed_runs_are_kept_dropped_or_selected() -> None:
    capture = Capture()
    agent = Agent(FunctionModel(failing_model), name='flaky', capabilities=[capture.instrument()])
    agent.run_sync('hello')
    with pytest.raises(RuntimeError):
        agent.run_sync('please crash')
    every = dataset_from_spans(capture.spans)
    assert len(every.cases) == 2
    [failed] = dataset_from_spans(capture.spans, errors='only', output_as='expected_output').cases
    assert failed.inputs['prompt'] == 'please crash'
    assert failed.metadata['error'] == 'RuntimeError: provider down'
    assert failed.expected_output is None and 'recorded_output' not in failed.metadata
    [ok] = dataset_from_spans(capture.spans, errors='exclude').cases
    assert ok.inputs['prompt'] == 'hello' and ok.metadata['recorded_output'] == 'fine'
    with pytest.raises(ValueError, match='errors must be'):
        dataset_from_spans(capture.spans, errors='some')  # pyright: ignore[reportArgumentType]


def test_a_retried_tool_call_is_marked_failed() -> None:
    capture = Capture()
    attempts: list[int] = []

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        last = messages[-1]
        if any(isinstance(p, ToolReturnPart) for p in getattr(last, 'parts', [])):
            return ModelResponse(parts=[TextPart('done')])
        n = len(attempts)
        return ModelResponse(parts=[ToolCallPart('charge', {'amount': 5}, tool_call_id=f'charge-{n}')])

    agent = Agent(FunctionModel(model), name='billing', capabilities=[capture.instrument()])

    @agent.tool_plain
    def charge(amount: int) -> str:
        attempts.append(amount)
        if len(attempts) == 1:
            raise ModelRetry('card declined, retry')
        return 'charged'

    agent.run_sync('charge me')
    [case] = dataset_from_spans(capture.spans).cases
    first, second = case.metadata['tool_calls']
    assert first['id'] == 'charge-0' and 'card declined' in first['error']
    assert second == {'id': 'charge-1', 'name': 'charge', 'arguments': {'amount': 5}, 'result': 'charged'}


def test_filters_by_agent_nesting_and_predicate() -> None:
    capture = Capture()
    inner = Agent(TestModel(custom_output_text='inner answer'), name='inner', capabilities=[capture.instrument()])

    def outer_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if any(isinstance(p, ToolReturnPart) for p in getattr(messages[-1], 'parts', [])):
            return ModelResponse(parts=[TextPart('outer answer')])
        return ModelResponse(parts=[ToolCallPart('ask_inner', {'question': 'sub'}, tool_call_id='c1')])

    outer = Agent(FunctionModel(outer_model), name='outer', capabilities=[capture.instrument()])

    @outer.tool_plain
    async def ask_inner(question: str) -> str:
        return (await inner.run(question)).output

    outer.run_sync('top question')
    names = {c.metadata['agent_name'] for c in dataset_from_spans(capture.spans).cases}
    assert names == {'outer', 'inner'}
    assert [c.metadata['agent_name'] for c in dataset_from_spans(capture.spans, nested=False).cases] == ['outer']
    [inner_case] = dataset_from_spans(capture.spans, agent_name='inner').cases
    assert inner_case.inputs['prompt'] == 'sub'
    picked = dataset_from_spans(capture.spans, where=lambda r: r.prompt == 'top question').cases
    assert [c.metadata['agent_name'] for c in picked] == ['outer']
    both = agent_runs(capture.spans)
    assert {r.agent_name: r.nested for r in both} == {'outer': False, 'inner': True}


def test_identical_inputs_are_one_case_that_lists_the_others() -> None:
    capture = Capture()
    agent = support_agent(capture)
    for prompt in ['Where is order 1?', 'Where is order 2?', 'Where is order 1?', 'Where is order 1?']:
        agent.run_sync(prompt)
    runs = agent_runs(capture.spans)
    dataset = dataset_from_spans(capture.spans)
    assert [c.inputs['prompt'] for c in dataset.cases] == ['Where is order 1?', 'Where is order 2?']
    duplicates = dataset.cases[0].metadata['duplicates']
    assert duplicates == [{'trace_id': r.trace_id, 'span_id': r.span_id} for r in (runs[2], runs[3])]
    assert dataset.cases[0].metadata['span_id'] == runs[0].span_id  # the earliest is kept
    assert len(dataset_from_spans(capture.spans, deduplicate=False).cases) == 4


def test_sampling_is_seeded_and_keeps_time_order() -> None:
    capture = Capture()
    agent = support_agent(capture)
    for i in range(10):
        agent.run_sync(f'Where is order {i}?')
    first = [c.name for c in dataset_from_spans(capture.spans, sample=4, seed=1).cases]
    assert first == [c.name for c in dataset_from_spans(capture.spans, sample=4, seed=1).cases]
    assert len(first) == 4
    order = [c.name for c in dataset_from_spans(capture.spans).cases]
    assert first == [n for n in order if n in first]
    assert first != [c.name for c in dataset_from_spans(capture.spans, sample=4, seed=2).cases]
    assert len(dataset_from_spans(capture.spans, sample=50).cases) == 10
    with pytest.raises(ValueError, match='sample'):
        dataset_from_spans(capture.spans, sample=-1)


class SupportInputs(BaseModel):
    prompt: str
    message_history: list[dict[str, Any]] = []


def test_input_type_and_custom_inputs() -> None:
    capture = Capture()
    support_agent(capture).run_sync('Where is order 5?')
    [typed] = dataset_from_spans(capture.spans, input_type=SupportInputs).cases
    assert typed.inputs == SupportInputs(prompt='Where is order 5?')
    [custom] = dataset_from_spans(capture.spans, inputs=lambda r: {'q': r.prompt, 'tools': len(r.tool_calls)}).cases
    assert custom.inputs == {'q': 'Where is order 5?', 'tools': 1}


def test_the_dataset_round_trips_through_a_file(tmp_path: Any) -> None:
    capture = Capture()
    agent = support_agent(capture)
    for i in range(3):
        agent.run_sync(f'Where is order {i}?')
    dataset = dataset_from_spans(capture.spans, output_as='expected_output')
    path = tmp_path / 'prod.yaml'
    dataset.to_file(path)
    loaded = Dataset[Any, Any, Any].from_file(path)
    assert [(c.name, c.inputs, c.expected_output, c.metadata) for c in loaded.cases] == [
        (c.name, c.inputs, c.expected_output, c.metadata) for c in dataset.cases
    ]


# --- span shapes -----------------------------------------------------------------------------


def otlp_json(spans: Sequence[ReadableSpan]) -> dict[str, Any]:
    """The spans as an OTLP/JSON document, as an OTLP file exporter writes it (hex ids)."""

    def value(v: Any) -> dict[str, Any]:
        if isinstance(v, bool):
            return {'boolValue': v}
        if isinstance(v, int):
            return {'intValue': str(v)}
        if isinstance(v, float):
            return {'doubleValue': v}
        return {'stringValue': str(v)}

    return {
        'resourceSpans': [
            {
                'scopeSpans': [
                    {
                        'spans': [
                            {
                                'traceId': format(s.context.trace_id, '032x'),
                                'spanId': format(s.context.span_id, '016x'),
                                'parentSpanId': format(s.parent.span_id, '016x') if s.parent else '',
                                'name': s.name,
                                'startTimeUnixNano': str(s.start_time),
                                'attributes': [{'key': k, 'value': value(v)} for k, v in (s.attributes or {}).items()],
                                'events': [
                                    {
                                        'name': e.name,
                                        'attributes': [
                                            {'key': k, 'value': value(v)} for k, v in (e.attributes or {}).items()
                                        ],
                                    }
                                    for e in s.events
                                ],
                                'status': {'code': {'UNSET': 0, 'OK': 1, 'ERROR': 2}[s.status.status_code.name]},
                            }
                            for s in spans
                        ]
                    }
                ]
            }
        ]
    }


def query_rows(spans: Sequence[ReadableSpan]) -> list[dict[str, Any]]:
    """The spans as Logfire `records` rows (`query_json_rows`), JSON attributes already parsed."""
    rows = []
    for s in spans:
        attributes = dict(s.attributes or {})
        for key in ('pydantic_ai.all_messages', 'gen_ai.system_instructions'):
            if key in attributes:
                attributes[key] = json.loads(str(attributes[key]))
        exception = next((dict(e.attributes or {}) for e in s.events if e.name == 'exception'), {})
        rows.append(
            {
                'trace_id': format(s.context.trace_id, '032x'),
                'span_id': format(s.context.span_id, '016x'),
                'parent_span_id': format(s.parent.span_id, '016x') if s.parent else None,
                'span_name': s.name,
                'start_timestamp': '2026-10-03T10:00:00.000000Z',
                'attributes': attributes,
                'otel_status_code': s.status.status_code.name,
                'is_exception': bool(exception),
                'exception_type': exception.get('exception.type'),
                'exception_message': exception.get('exception.message'),
            }
        )
    return rows


def comparable(dataset: Any) -> list[tuple[Any, ...]]:
    return [
        (c.name, c.inputs, c.metadata.get('tool_calls'), c.metadata.get('recorded_output'), c.metadata.get('error'))
        for c in dataset.cases
    ]


def test_every_span_shape_gives_the_same_dataset() -> None:
    capture = Capture()
    agent = support_agent(capture)
    agent.run_sync('Where is order 8?')
    flaky = Agent(FunctionModel(failing_model), name='flaky', capabilities=[capture.instrument()])
    with pytest.raises(RuntimeError):
        flaky.run_sync('crash now')
    spans = capture.spans
    expected = comparable(dataset_from_spans(spans))
    assert len(expected) == 2 and expected[1][4] == 'RuntimeError: provider down'
    shapes = {
        'to_json': [json.loads(s.to_json()) for s in spans],
        'otlp_document': otlp_json(spans),
        'otlp_text': json.dumps(otlp_json(spans)),
        'otlp_spans': otlp_json(spans)['resourceSpans'][0]['scopeSpans'][0]['spans'],
        'query_rows': query_rows(spans),
    }
    for shape, given in shapes.items():
        assert comparable(dataset_from_spans(given)) == expected, shape
    mixed = [*spans[:2], *shapes['to_json'][2:]]
    assert comparable(dataset_from_spans(mixed)) == expected


def test_logfire_capture_dicts_and_raw_spans_with_pending_ones(capfire: CaptureLogfire) -> None:
    agent = Agent(
        FunctionModel(support_model),
        name='support',
        capabilities=[Instrumentation(settings=InstrumentationSettings())],  # the global, Logfire's, provider
    )

    @agent.tool_plain
    def lookup(order: int) -> str:
        return f'order {order} shipped'

    agent.run_sync('Where is order 6?')
    raw = capfire.exporter.exported_spans  # includes Logfire's pending spans
    assert any((s.attributes or {}).get('logfire.span_type') == 'pending_span' for s in raw)
    as_dicts = capfire.exporter.exported_spans_as_dict(parse_json_attributes=True)
    for given in (raw, as_dicts, capfire.exporter.exported_spans_as_dict()):
        [case] = dataset_from_spans(given).cases
        assert case.inputs['prompt'] == 'Where is order 6?'
        assert case.metadata['tool_calls'][0]['result'] == 'order 6 shipped'
        assert case.metadata['recorded_output'] == 'Your order: order 6 shipped.'
    assert len(agent_runs([*raw, *raw])) == 1  # a span seen twice is read once


# --- malformed and partial spans ----------------------------------------------------------------


def test_content_not_recorded_is_skipped_with_a_warning() -> None:
    capture = Capture(include_content=False)
    support_agent(capture).run_sync('Where is order 1?')
    [run] = agent_runs(capture.spans)
    assert run.prompt is None and run.output is None and not run.output_recorded
    assert any('no user prompt' in p for p in run.problems)
    with pytest.warns(UserWarning, match='1 agent run'):
        assert dataset_from_spans(capture.spans).cases == []


def test_malformed_and_partial_spans() -> None:
    capture = Capture()
    support_agent(capture).run_sync('Where is order 2?')
    good = [json.loads(s.to_json()) for s in capture.spans]
    run = next(s for s in good if s['attributes'].get('gen_ai.operation.name') == 'invoke_agent')

    def variant(**attributes: Any) -> dict[str, Any]:
        changed = json.loads(json.dumps(run))
        for key, val in attributes.items():
            if val is None:
                changed['attributes'].pop(key, None)
            else:
                changed['attributes'][key] = val
        return changed

    garbage = variant(**{'pydantic_ai.all_messages': '[{"role": "user", "parts": [trunc'})
    [r] = agent_runs([garbage])
    assert r.prompt is None and any('messages unreadable' in p for p in r.problems)

    no_messages = variant(**{'pydantic_ai.all_messages': None})
    [r] = agent_runs([no_messages])
    assert r.problems == ('no messages recorded',)
    assert r.output == 'Your order: order 2 shipped.'  # what was recorded is still read

    no_output = variant(final_result=None)
    [r] = agent_runs([no_output])
    assert r.output is None and r.problems == ('no final_result recorded',)
    [case] = dataset_from_spans([no_output]).cases
    assert 'recorded_output' not in case.metadata

    bad_index = variant(**{'pydantic_ai.new_message_index': 99, 'gen_ai.system_instructions': 'not json'})
    [r] = agent_runs([bad_index])
    assert r.prompt == 'Where is order 2?' and r.problems == ('instructions unreadable',)

    bad_type = agent_runs([run], output_type=int)
    assert bad_type[0].output == 'Your order: order 2 shipped.'
    assert 'output does not fit output_type' in bad_type[0].problems[0]

    junk: list[Any] = [None, 42, {'unrelated': True}, {'context': None}, {'span_name': 'x', 'trace_id': 1}, object()]
    assert agent_runs([*junk, run]) == agent_runs([run])
    assert agent_runs([]) == []
    assert len(dataset_from_spans([run], deduplicate=False).cases) == 1


def test_version_1_message_events_are_read() -> None:
    """A version-1 run span (pydantic-ai before its V2 release), shaped as `event_to_dict` wrote it."""
    events = [
        {'event.name': 'gen_ai.system.message', 'role': 'system', 'content': 'Be brief.'},
        {'event.name': 'gen_ai.user.message', 'role': 'user', 'content': 'Where is order 4?'},
        {
            'event.name': 'gen_ai.assistant.message',
            'role': 'assistant',
            'tool_calls': [{'id': 'c4', 'type': 'function', 'function': {'name': 'lookup', 'arguments': {'order': 4}}}],
        },
        {'event.name': 'gen_ai.tool.message', 'role': 'tool', 'id': 'c4', 'name': 'lookup', 'content': 'shipped'},
        {'event.name': 'gen_ai.assistant.message', 'role': 'assistant', 'content': 'It shipped.'},
    ]
    span = {
        'name': 'agent run',
        'context': {'trace_id': 7, 'span_id': 9},
        'parent': None,
        'start_time': 1_700_000_000_000_000_000,
        'attributes': {'agent_name': 'old', 'all_messages_events': json.dumps(events), 'final_result': 'It shipped.'},
    }
    [run] = agent_runs([span])
    assert run.instrumentation_version == 1 and run.agent_name == 'old'
    assert run.prompt == 'Where is order 4?'
    assert [m['role'] for m in run.history] == ['system']
    assert [dict(c) for c in run.tool_calls] == [
        {'id': 'c4', 'name': 'lookup', 'arguments': {'order': 4}, 'result': 'shipped'}
    ]
    assert run.output == 'It shipped.' and run.trace_id == '0' * 31 + '7'


# --- judge cases ---------------------------------------------------------------------------------


def test_judge_cases_need_an_explicit_acceptance() -> None:
    capture = Capture()
    agent = support_agent(capture)
    for i in range(3):
        agent.run_sync(f'Where is order {i}?')
    flaky = Agent(FunctionModel(failing_model), name='flaky', capabilities=[capture.instrument()])
    with pytest.raises(RuntimeError):
        flaky.run_sync('crash')
    names = [c.name for c in dataset_from_spans(capture.spans).cases]
    every = judge_cases_from_spans(capture.spans, accepted='all')
    assert [c.name for c in every] == names[:3]  # the failed run is never a known-good answer
    reviewed = judge_cases_from_spans(capture.spans, accepted={names[1]})
    assert [c.name for c in reviewed] == [names[1]]
    assert reviewed[0].output == 'Your order: order 1 shipped.'
    assert reviewed[0].inputs == {'prompt': 'Where is order 1?', 'message_history': []}
    assert reviewed[0].metadata['tool_calls'][0]['result'] == 'order 1 shipped'
    by_rule = judge_cases_from_spans(capture.spans, accepted=lambda r: '2' in str(r.prompt))
    assert [c.inputs['prompt'] for c in by_rule] == ['Where is order 2?']
    with pytest.raises(ValueError, match='accepted'):
        judge_cases_from_spans(capture.spans, accepted='some')  # pyright: ignore[reportArgumentType]


async def test_a_judge_is_certified_on_cases_from_traces() -> None:
    capture = Capture()
    agent = support_agent(capture)
    for i in range(20):
        await agent.run(f'Where is order {i}?')
    cases = judge_cases_from_spans(capture.spans, accepted='all', expected_output=lambda r: r.output)
    assert len(cases) == 20
    certificate = await certify_judge(judge(oracle), cases)
    assert certificate.verdict == 'ADMISSIBLE'


# --- regressions (night review 9) ---------------------------------------------------------------


def row(span_id: str, output: str, prompt: str = 'same prompt') -> dict[str, Any]:
    """A Logfire query row for one run that answered `prompt` with `output` in text."""
    return {
        'span_name': 'invoke_agent demo',
        'trace_id': 'a' * 32,
        'span_id': span_id,
        'attributes': {
            'gen_ai.agent.name': 'demo',
            'pydantic_ai.all_messages': [
                {'role': 'user', 'parts': [{'type': 'text', 'content': prompt}]},
                {'role': 'assistant', 'parts': [{'type': 'text', 'content': output}]},
            ],
            'final_result': output,
        },
    }


def test_an_output_that_does_not_fit_output_type_is_never_expected_or_accepted() -> None:
    """T1: the invalid value was kept as `expected_output`, and the diagnostic dropped."""
    spans = [row('1111111122222222', 'not an integer')]
    [run] = agent_runs(spans, output_type=int)
    assert run.output_valid is False and run.output == 'not an integer'
    assert 'output does not fit output_type' in run.problems[0]
    [case] = dataset_from_spans(spans, output_type=int, output_as='expected_output').cases
    assert case.expected_output is None
    assert case.metadata['output_valid'] is False and case.metadata['recorded_output'] == 'not an integer'
    assert 'output does not fit output_type' in case.metadata['problems'][0]
    assert judge_cases_from_spans(spans, accepted='all', output_type=int) == []
    [valid] = dataset_from_spans([row('3333333344444444', '42')], output_type=int, output_as='expected_output').cases
    assert valid.expected_output == 42 and 'output_valid' not in valid.metadata and 'problems' not in valid.metadata


def test_an_unaccepted_duplicate_does_not_displace_an_accepted_answer() -> None:
    """T2: deduplication ran before acceptance, so an earlier rejected run hid a later approved one."""
    spans = [row('1111111122222222', 'bad'), row('3333333344444444', 'good'), row('5555555566666666', 'good too')]
    runs = agent_runs(spans)
    by_name = judge_cases_from_spans(spans, accepted=[runs[1].name])
    assert [(c.name, c.output) for c in by_name] == [(runs[1].name, 'good')]
    assert 'duplicates' not in by_name[0].metadata  # the rejected and unlisted runs are not its duplicates
    by_rule = judge_cases_from_spans(spans, accepted=lambda r: r.output != 'bad')
    assert [c.output for c in by_rule] == ['good']
    assert by_rule[0].metadata['duplicates'] == [{'trace_id': 'a' * 32, 'span_id': '5555555566666666'}]
    bad_output = [row('1111111122222222', 'x'), row('3333333344444444', '7')]
    [typed] = judge_cases_from_spans(bad_output, accepted='all', output_type=int)
    assert typed.output == 7  # an invalid earlier output does not hide a valid later one
    no_output = [{**row('1111111122222222', 'x'), 'attributes': {**row('1', 'x')['attributes']}}, row('2', 'y')]
    del no_output[0]['attributes']['final_result']
    assert [c.output for c in judge_cases_from_spans(no_output, accepted='all')] == ['y']
    picked = judge_cases_from_spans(spans, accepted='all', where=lambda r: r.output == 'good too')
    assert [c.output for c in picked] == ['good too']
