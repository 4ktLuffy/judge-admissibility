"""What a certificate looks like in Logfire: the span tree `log_certificate` and `certify_judge_traced` emit.

    PYTHONPATH=.:bench <venv>/bin/python bench/trace_export_demo.py

No model is called and no Logfire account is needed: spans go to Logfire's in-memory test exporter.

1. Saved certificates, rebuilt with `Certificate.from_dict` and exported as they are:
   - `results/apprentice.json` baseline: Codex with the default `include_input=False`, INADMISSIBLE
     on `mismatched_output` (it passed answers borrowed from other customers' questions);
   - `results/evidence_contract.first_run.json`, reply only: failed acceptance and two rejection families.
2. A scripted `LLMJudge` (a `FunctionModel` that is right on real answers but passes an empty one),
   certified with `certify_judge_traced`, its model wrapped by `logfire.instrument_pydantic_ai`, so
   each judge call's model span sits under the same span as the certificate.

Writes `results/trace_export_demo.json`: each span tree, names and key attributes only.
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any

import logfire
from logfire.testing import TestExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_evals.evaluators import LLMJudge

from pydantic_evals_admissibility import Certificate, JudgeCase
from pydantic_evals_admissibility._trace_export import certify_judge_traced, log_certificate

ROOT = Path(__file__).parent.parent
OUT = ROOT / 'results' / 'trace_export_demo.json'
KEEP = (
    'verdict', 'fingerprint', 'calls', 'looks', 'failed_checks', 'dataset', 'covers_judge', 'judge.model',
    'judge.include_input', 'status', 'rate', 'low', 'high', 'trials', 'errors', 'families',
    'wrong', 'judged', 'examples', 'examples_omitted', 'cases',
)  # fmt: skip


def lenient_on_empty() -> LLMJudge:
    """Passes an answer that contains the expected one, and any empty answer."""

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        prompt = ''.join(
            part.content
            for message in messages
            for part in getattr(message, 'parts', [])
            if isinstance(part, UserPromptPart) and isinstance(part.content, str)
        )
        output = re.search(r'<Output>\n(.*?)\n</Output>', prompt, re.S)
        expected = re.search(r'<ExpectedOutput>\n(.*?)\n</ExpectedOutput>', prompt, re.S)
        text = output.group(1) if output else ''
        passed = not text.strip() or bool(expected and expected.group(1) in text)
        reason = 'nothing contradicts the expected answer' if not text.strip() else 'scripted'
        tool = info.output_tools[0]
        return ModelResponse(
            parts=[ToolCallPart(tool.name, {'reason': reason, 'pass': passed, 'score': float(passed)})]
        )

    return LLMJudge(
        rubric='The output answers the question correctly.',
        model=logfire.instrument_pydantic_ai(FunctionModel(model, model_name='lenient-on-empty')),
        include_expected_output=True,
    )


def tree(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Spans as nested nodes; consecutive model-request spans folded into a count after two examples."""
    nodes = {
        s['context']['span_id']: {
            'name': s['name'],
            'message': s['attributes'].get('logfire.msg'),
            'attributes': {k: s['attributes'][k] for k in KEEP if k in s['attributes']},
            'children': [],
        }
        for s in spans
    }
    roots = []
    for s in sorted(spans, key=lambda s: s['start_time']):
        node, parent = nodes[s['context']['span_id']], s['parent']
        (nodes[parent['span_id']]['children'] if parent and parent['span_id'] in nodes else roots).append(node)
    for node in nodes.values():
        node['attributes'] = node['attributes'] or None
        calls = [c for c in node['children'] if not c['attributes']]
        if len(calls) > 2:
            rest = [c for c in node['children'] if c['attributes']]
            node['children'] = calls[:2] + [{'name': f'... {len(calls) - 2} more judge model calls'}] + rest
    return roots


async def main() -> None:
    exporter = TestExporter()
    logfire.configure(send_to_logfire=False, console=False, additional_span_processors=[SimpleSpanProcessor(exporter)])
    out: dict[str, Any] = {}

    saved = json.loads((ROOT / 'results' / 'apprentice.json').read_text())['result']['baseline_certificate']
    log_certificate(Certificate.from_dict(saved), dataset='support_eval held-out questions', max_examples=3)
    out['saved: apprentice baseline'] = tree(exporter.exported_spans_as_dict(parse_json_attributes=True))
    exporter.clear()

    saved = json.loads((ROOT / 'results' / 'evidence_contract.first_run.json').read_text())
    saved = saved['reply only (default include_input=False), no reasoning']
    log_certificate(Certificate.from_dict(saved), dataset='evidence contract', max_examples=3)
    out['saved: evidence contract, reply only'] = tree(exporter.exported_spans_as_dict(parse_json_attributes=True))
    exporter.clear()

    cases = [
        JudgeCase(f'case-{i:02d}', f'Which code is item {i}?', f'item-{i:02d}', expected_output=f'item-{i:02d}')
        for i in range(12)
    ]
    certificate = await certify_judge_traced(lenient_on_empty(), cases, dataset='items', max_examples=2)
    out['scripted: certify_judge_traced'] = tree(exporter.exported_spans_as_dict(parse_json_attributes=True))
    out['scripted: certificate table'] = certificate.table().splitlines()

    OUT.write_text(json.dumps(out, indent=2, default=str) + '\n')
    for name, roots in out.items():
        if not name.startswith('scripted: certificate'):
            print(f'\n{name}')
            show(roots, 1)
    print(f'\nwrote {OUT.relative_to(ROOT)}')


def show(nodes: list[dict[str, Any]], depth: int) -> None:
    for node in nodes:
        print('  ' * depth + (node.get('message') or node['name']))
        show(node.get('children', []), depth + 1)


if __name__ == '__main__':
    asyncio.run(main())
