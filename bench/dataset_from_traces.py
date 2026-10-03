"""An eval dataset built from the traces of a locally run agent, round-tripped through a file.

    PYTHONPATH=.:bench <venv>/bin/python bench/dataset_from_traces.py

No model is called and no Logfire account is needed. A pydantic-ai `Agent` with a `FunctionModel`
and one tool answers order questions under the `Instrumentation` capability; its spans go to an
OpenTelemetry in-memory exporter. The traffic is shaped like production: repeated questions, a
few follow-ups that carry message history, and a few runs that fail. It is run once with
instrumentation version 5 (the default) and once with version 2, and the two datasets compared.

The dataset is written with `Dataset.to_file` (YAML and JSON), read back with `Dataset.from_file`,
and compared case by case. Writes `results/dataset_from_traces.json`: counts, timings, the
round-trip result, and three sample cases.
"""

from __future__ import annotations

import json
import tempfile
import time
import warnings
from pathlib import Path
from typing import Any

from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic_ai import Agent
from pydantic_ai.capabilities import Instrumentation
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_evals import Dataset

from pydantic_evals_admissibility._traces import agent_runs, dataset_from_spans, judge_cases_from_spans

ROOT = Path(__file__).parent.parent
OUT = ROOT / 'results' / 'dataset_from_traces.json'
DISTINCT = 25  # distinct first questions
ASKED = 40  # first questions asked, so 15 repeats
FOLLOW_UPS = 5
FAILURES = 5


def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Looks the order up, then answers with what the tool said; fails on 'outage'."""
    last = messages[-1]
    returns = [p for p in getattr(last, 'parts', []) if isinstance(p, ToolReturnPart)]
    if returns:
        return ModelResponse(parts=[TextPart(f'Your order: {returns[0].content}.')])
    prompt = next(
        p.content
        for m in reversed(messages)
        for p in getattr(m, 'parts', [])
        if isinstance(p, UserPromptPart) and isinstance(p.content, str)
    )
    if 'outage' in prompt:
        raise RuntimeError('upstream order service unavailable')
    order = int(''.join(c for c in prompt if c.isdigit()) or 0)
    return ModelResponse(parts=[ToolCallPart('lookup', {'order': order}, tool_call_id=f'lookup-{order}')])


def traffic(version: int) -> tuple[tuple[ReadableSpan, ...], float]:
    """Run the agent over the prompts under instrumentation `version`; the spans and seconds taken."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')  # version 2 is deprecated
        settings = InstrumentationSettings(tracer_provider=provider, version=version)  # pyright: ignore[reportArgumentType]
    agent = Agent(
        FunctionModel(model, model_name='orders-v1'),
        name='orders',
        instructions='Answer questions about orders.',
        capabilities=[Instrumentation(settings=settings)],
    )

    @agent.tool_plain
    def lookup(order: int) -> str:
        return f'order {order} shipped on day {order % 7 + 1}'

    started = time.perf_counter()
    results = [agent.run_sync(f'Where is order {i % DISTINCT}?') for i in range(ASKED)]
    for i in range(FOLLOW_UPS):
        agent.run_sync(f'And order {100 + i}?', message_history=results[i].all_messages())
    for i in range(FAILURES):
        try:
            agent.run_sync(f'outage test {i}')
        except RuntimeError:
            pass
    return exporter.get_finished_spans(), time.perf_counter() - started


def comparable(dataset: Any) -> list[Any]:
    return [[c.name, c.inputs, c.expected_output, c.metadata] for c in dataset.cases]


def main() -> None:
    out: dict[str, Any] = {
        'traffic': {
            'first_questions_asked': ASKED,
            'distinct_first_questions': DISTINCT,
            'follow_ups_with_history': FOLLOW_UPS,
            'failing_runs': FAILURES,
        }
    }
    datasets: dict[int, Any] = {}
    for version in (5, 2):
        spans, run_seconds = traffic(version)
        started = time.perf_counter()
        runs = agent_runs(spans)
        dataset = dataset_from_spans(spans, name='orders-production')
        convert_seconds = time.perf_counter() - started
        datasets[version] = dataset
        cases = dataset.cases
        out[f'version_{version}'] = {
            'spans': len(spans),
            'agent_runs': len(runs),
            'runs_with_problems': sum(bool(r.problems) for r in runs),
            'cases': len(cases),
            'cases_with_history': sum(bool(c.inputs['message_history']) for c in cases),
            'cases_with_error': sum('error' in c.metadata for c in cases),
            'cases_with_tool_calls': sum('tool_calls' in c.metadata for c in cases),
            'duplicate_runs_folded': sum(len(c.metadata.get('duplicates', [])) for c in cases),
            'cases_without_dedup': len(dataset_from_spans(spans, deduplicate=False).cases),
            'cases_errors_excluded': len(dataset_from_spans(spans, errors='exclude').cases),
            'sample_10_seed_0': len(dataset_from_spans(spans, sample=10).cases),
            'judge_cases_accepted_all': len(judge_cases_from_spans(spans, accepted='all')),
            'agent_seconds': round(run_seconds, 3),
            'conversion_seconds': round(convert_seconds, 4),
        }

    def key(dataset: Any) -> list[Any]:
        """The cases without the ids and times that differ between two runs of the traffic."""
        drop = ('trace_id', 'span_id', 'run_id', 'conversation_id', 'start_time', 'duplicates')
        return [
            [c.inputs, {k: v for k, v in c.metadata.items() if k not in drop}, len(c.metadata.get('duplicates', []))]
            for c in dataset.cases
        ]

    out['version_2_and_5_same_cases'] = key(datasets[2]) == key(datasets[5])

    dataset = datasets[5]
    round_trip: dict[str, Any] = {}
    with tempfile.TemporaryDirectory() as tmp:
        for suffix in ('yaml', 'json'):
            path = Path(tmp) / f'orders.{suffix}'
            dataset.to_file(path)
            loaded = Dataset[Any, Any, Any].from_file(path)
            round_trip[suffix] = {
                'bytes': path.stat().st_size,
                'cases_loaded': len(loaded.cases),
                'identical': comparable(loaded) == comparable(dataset),
            }
    out['round_trip'] = round_trip

    picks = [
        next(c for c in dataset.cases if c.metadata.get('duplicates')),
        next(c for c in dataset.cases if c.inputs['message_history']),
        next(c for c in dataset.cases if 'error' in c.metadata),
    ]
    out['sample_cases'] = [
        {'name': c.name, 'inputs': c.inputs, 'expected_output': c.expected_output, 'metadata': c.metadata}
        for c in picks
    ]
    OUT.write_text(json.dumps(out, indent=2, default=str) + '\n')
    print(json.dumps({k: v for k, v in out.items() if k != 'sample_cases'}, indent=2))
    print(f'\nwrote {OUT.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
