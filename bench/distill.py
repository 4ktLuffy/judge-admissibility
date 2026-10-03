"""Executable rubric distillation on the support task: a `HybridJudge` against a plain `LLMJudge`.

    PYTHONPATH=.:bench <venv>/bin/python bench/distill.py --backend scripted|codex

The rubric has three clauses. Two are facts code decides exactly, with the task's own checker:

- the reply ends with a line starting with "Answer:" (the checker's answer-line pattern);
- the answer is the one the policy gives (`is_correct`);

and one is not: the reply is polite and respectful. The plain `LLMJudge` is asked all three; the
`HybridJudge` checks the first two in code and asks the model only about tone. Both are certified
with the same cases, controls and seed: another case's reply (must fail), an empty reply (must
fail), the right reply made rude (must fail, and only the model can see it), and the reply
re-spaced (must hold).

The rude control puts the rude sentence on its own first line, above the unchanged reply, so the
answer line and the answer are untouched: the code checks pass and only the tone can fail it. The
first version prefixed the reply instead, and on a reply that is one line ("Answer: no", the
fallback where none of the agent's replies was right, or a terse "Answer: $8.50") the prefix landed
on the answer line, so the `HybridJudge` failed it in code for a missing answer line rather than in
the model for its tone (2 of 10 in that Codex run, 6 of 40 in the scripted one): a correct
rejection for the wrong reason. `results/distill.json` holds the runs on the fixed control.

Replies are the support agent's real replies, read from `results/support-cache.json` (no agent
calls; a missing reply is an error, not a new call). `scripted`: every case, two repeats, the 80
labelled replies, and a scripted model that reads LLMJudge's real prompt. `codex`: 10 cases, one
repeat, no labels, Codex `gpt-5.6-luna` with no reasoning, at most 80 calls.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_evals.evaluators import EvaluatorContext, LLMJudge
from support_task import _ANSWER, SupportCase, cases, is_correct

from pydantic_evals_admissibility import (
    DEFAULT_CONTROLS,
    HumanLabel,
    JudgeCase,
    Rewrite,
    certify_judge,
)
from pydantic_evals_admissibility._distill import Clause, HybridJudge, compare_hybrid, model_calls

ROOT = Path(__file__).parent.parent
CACHE = ROOT / 'results' / 'support-cache.json'
OUT = ROOT / 'results' / 'distill.json'
AGENT_PROMPT = (  # the prompt `bench/support_eval.py` cached the replies under
    'You are a customer support agent. Answer the customer using only the store policy given. '
    'End your reply with a line "Answer: <answer>", where the answer is yes or no, or a dollar amount.'
)
RUDE = 'Honestly, this is all in the policy, which you could have read yourself before bothering us.'
BY_NAME = {c.name: c for c in cases()}


def has_answer_line(ctx: EvaluatorContext[Any, Any, Any]) -> bool:
    return isinstance(ctx.output, str) and bool(_ANSWER.findall(ctx.output))


def answer_is_right(ctx: EvaluatorContext[Any, Any, Any]) -> bool:
    case = BY_NAME.get(ctx.name or '')
    return case is not None and isinstance(ctx.output, str) and is_correct(case, ctx.output)


CLAUSES = (
    Clause("The reply ends with a line starting with 'Answer:' that states the final answer.", has_answer_line),
    Clause(
        'The answer is correct according to the store policy: the right yes or no, or the right dollar amount.',
        answer_is_right,
    ),
    Clause('The reply is polite and respectful to the customer.'),
)


def rude(output: Any) -> str | None:
    """The reply with a rude first line of its own; the answer line below it is untouched."""
    return f'{RUDE}\n\n{output}' if isinstance(output, str) and output else None


CONTROLS = (*DEFAULT_CONTROLS, Rewrite(rude, 'rude_tone'))


def replies() -> dict[str, list[str]]:
    """The agent's two cached replies per case; raises rather than calling the agent."""
    cache: dict[str, str] = json.loads(CACHE.read_text())
    out: dict[str, list[str]] = {}
    for c in cases():
        keys = [hashlib.sha256(f'{AGENT_PROMPT}|{c.inputs}|{r}'.encode()).hexdigest() for r in range(2)]
        out[c.name] = [cache[k] for k in keys]
    return out


def scripted_model() -> FunctionModel:
    """Reads LLMJudge's prompt: fails rude replies, and, when asked about correctness, wrong ones."""
    by_inputs = {c.inputs: c for c in cases()}

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        prompt = ''.join(
            p.content
            for m in messages
            for p in getattr(m, 'parts', [])
            if isinstance(p, UserPromptPart) and isinstance(p.content, str)
        )

        def section(tag: str) -> str:
            found = re.search(rf'<{tag}>\n(.*?)\n</{tag}>', prompt, re.S)
            return found.group(1) if found else ''

        output, rubric, case = section('Output'), section('Rubric'), by_inputs.get(section('Input'))
        passed = bool(output.strip()) and 'bothering us' not in output
        if 'correct according to the store policy' in rubric:
            passed = passed and case is not None and is_correct(case, output)
        tool = info.output_tools[0]
        return ModelResponse(parts=[ToolCallPart(tool.name, {'reason': 'scripted', 'pass': passed, 'score': 1.0})])

    return FunctionModel(model, model_name='scripted')


def judge_cases(cs: list[SupportCase], rs: dict[str, list[str]]) -> list[JudgeCase]:
    good = {c.name: next((r for r in rs[c.name] if is_correct(c, r)), f'Answer: {c.answer}') for c in cs}
    return [JudgeCase(c.name, c.inputs, good[c.name], expected_output=c.answer) for c in cs]


async def run(backend: str) -> dict[str, Any]:
    rs = replies()
    if backend == 'codex':
        from codex_judge import codex_model

        model: Any = codex_model(effort='none')
        cs = [cases()[i] for i in (0, 1, 2, 10, 11, 20, 21, 30, 31, 32)]  # every kind; yes and no both ways
        repeats, labels, concurrency = 1, [], 2
    else:
        model = scripted_model()
        cs = cases()
        repeats, concurrency = 2, 8
        labels = [HumanLabel(c.name, r, is_correct(c, r)) for c in cs for r in rs[c.name]]
    jcs = judge_cases(cs, rs)
    hybrid = HybridJudge(CLAUSES, model=model)
    plain = LLMJudge(rubric=hybrid.full_rubric, model=model, include_input=True)
    certs = {}
    for label, judge in (('plain LLMJudge, full rubric', plain), ('HybridJudge, tone only to the model', hybrid)):
        cert = await certify_judge(
            judge, jcs, controls=CONTROLS, human_labels=labels, repeats=repeats, max_concurrency=concurrency
        )
        print(f'\n== {label}: {cert.verdict}, {model_calls(cert)} model calls of {cert.calls}\n{cert.table()}')
        certs[label] = cert
    plain_cert, hybrid_cert = certs.values()
    comparison = compare_hybrid(plain_cert, hybrid_cert)
    print(
        f'\nagreement {comparison.agreed}/{comparison.compared}, '
        f'calls {comparison.calls}, saved {comparison.calls_saved}'
    )
    disagreements = [
        {
            'case': case,
            'role': role,
            'plain': a,
            'hybrid': b,
            'plain_reason': next(j.reason for j in plain_cert.judgments if j.case == case and j.role == role),
            'hybrid_reason': next(j.reason for j in hybrid_cert.judgments if j.case == case and j.role == role),
        }
        for case, role, a, b in comparison.disagreements
    ]
    for d in disagreements:
        print('  differs:', d)
    return {
        'cases': len(jcs),
        'repeats': repeats,
        'labels': len(labels),
        'clauses': [repr(c) for c in CLAUSES],
        'model_rubric': hybrid.model_rubric,
        'comparison': {**comparison.to_dict(), 'disagreements': disagreements},
        'certificates': {k: {**c.to_dict(), 'model_calls': model_calls(c)} for k, c in certs.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', choices=('scripted', 'codex'), default='scripted')
    backend = parser.parse_args().backend
    results: dict[str, Any] = json.loads(OUT.read_text()) if OUT.exists() else {}
    results[backend] = asyncio.run(run(backend))
    OUT.write_text(json.dumps(results, indent=2, default=str))


if __name__ == '__main__':
    main()
