"""Which defects in the support agent would its evals catch? A mutation report, offline.

    PYTHONPATH=.:bench <venv>/bin/python bench/mutation_report.py

Known-good replies: for each of the 40 support questions (`bench/support_task.py`), the first
real agent reply that `is_correct` accepts, read from `results/support-cache.json` (Codex
`gpt-5.6-luna`, recorded by `support_eval.py`), else `Answer: <the right answer>`. Nothing is
called: no agent, no judge model, no network.

Mutants: the built-in ones (`truncated`, `last_sentence_dropped`, `number_changed`,
`yes_no_flipped`, `emptied`), plus the support agent's own mistakes (`plausible_wrong`: the
refund without the restocking fee, the other side of the shipping threshold, the opposite
yes/no), `explanation_cut` (only the `Answer:` line left), and `policy_changed`, an evidence
mutant: the policy the agent was given now says otherwise (another restocking fee, return
window, shipping threshold or warranty length), the reply is untouched. `is_correct` is the
oracle, on the policy each case now carries.

Evaluators: pydantic-evals `EqualsExpected`, `Contains` (each case's expected answer), and
`IsInstance(str)`, and three `LLMJudge`s whose models are scripted `FunctionModel`s (the prompt
is the real one `LLMJudge` builds; only the decision is scripted): one that reads the policy and
checks the answer (`include_input=True`), one lenient one that passes any reply with an
`Answer:` line (`include_input=True`), and one shown only the reply and the expected answer
(`include_input=False`, the default).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_evals.evaluators import Contains, EqualsExpected, IsInstance, LLMJudge
from support_eval import AGENT_PROMPT, REPEATS, RUBRIC, fmt, plausible_wrong
from support_task import SupportCase, cases, is_correct

from pydantic_evals_admissibility import DEFAULT_MUTANTS, JudgeCase, Mutant, mutation_report

ROOT = Path(__file__).parent.parent
CACHE = ROOT / 'results' / 'support-cache.json'
OUT = ROOT / 'results' / 'mutation_report.json'
_ANSWER_LINE = re.compile(r'^\s*\**answer\**\s*:.*$', re.I | re.M)


def known_good(c: SupportCase, cache: dict[str, str]) -> tuple[str, str]:
    """The first cached real reply that is right, and where it came from."""
    for r in range(REPEATS):
        reply = cache.get(hashlib.sha256(f'{AGENT_PROMPT}|{c.inputs}|{r}'.encode()).hexdigest())
        if reply is not None and is_correct(c, reply):
            return reply, 'agent'
    return f'Answer: {fmt(c, c.answer)}', 'template'


def policy_changed(c: SupportCase) -> SupportCase | None:
    """The same question under a policy whose right answer differs: the reply is now wrong."""
    p = c.policy
    if c.kind == 'refund':
        pct = next(x for x in (20, 10, 15) if x != p.restocking_pct)
        price = float(re.search(r'paid \$([\d.]+)', c.question).group(1))  # type: ignore[union-attr]
        return replace(c, policy=replace(p, restocking_pct=pct), answer=f'{price * (100 - pct) / 100:.2f}')
    if c.kind == 'shipping':
        total = float(re.search(r'\$([\d.]+)', c.question).group(1))  # type: ignore[union-attr]
        if c.answer == '0.00':
            return replace(
                c, policy=replace(p, free_shipping_over=math.ceil(total) + 10), answer=f'{p.shipping_fee:.2f}'
            )
        if total - 10 < 1:
            return None
        return replace(c, policy=replace(p, free_shipping_over=math.floor(total) - 10), answer='0.00')
    flipped = 'no' if c.answer == 'yes' else 'yes'
    if c.kind == 'return':
        dates = re.findall(r'on (\w+ \d+, \d{4})|Today is (\w+ \d+, \d{4})', c.question)
        received, today = (datetime.strptime(a or b, '%B %d, %Y') for a, b in dates)
        elapsed = (today - received).days
        days = elapsed - 5 if c.answer == 'yes' else elapsed + 5
        return replace(c, policy=replace(p, return_days=days), answer=flipped) if days >= 1 else None
    months = int(re.search(r'(\d+) months', c.question).group(1))  # type: ignore[union-attr]
    warranty = months // 2 if c.answer == 'yes' else months + 6
    return replace(c, policy=replace(p, warranty_months=warranty), answer=flipped) if warranty >= 1 else None


def scripted(decide: Any, include_input: bool) -> LLMJudge:
    """A real `LLMJudge` whose model reads the `<Input>`, `<Output>` and `<ExpectedOutput>` sections."""

    def section(prompt: str, tag: str) -> str:
        found = re.search(rf'<{tag}>\n(.*?)\n</{tag}>', prompt, re.S)
        return found.group(1) if found else ''

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        prompt = ''.join(p.content for m in messages for p in getattr(m, 'parts', []) if isinstance(p, UserPromptPart))
        passed = decide(section(prompt, 'Input'), section(prompt, 'Output'), section(prompt, 'ExpectedOutput'))
        tool = info.output_tools[0]
        return ModelResponse(parts=[ToolCallPart(tool.name, {'reason': 'scripted', 'pass': passed, 'score': 1.0})])

    return LLMJudge(
        rubric=RUBRIC, model=FunctionModel(model), include_input=include_input, include_expected_output=True
    )


async def main() -> None:
    cache: dict[str, str] = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    support = cases()
    good = {c.name: known_good(c, cache) for c in support}
    changed = {c.name: policy_changed(c) for c in support}
    by_inputs = {c.inputs: c for c in support} | {c.inputs: c for c in changed.values() if c is not None}
    judge_cases = [JudgeCase(c.name, c.inputs, good[c.name][0], expected_output=c.answer, metadata=c) for c in support]

    def oracle(case: JudgeCase, output: Any) -> bool:
        return isinstance(output, str) and is_correct(case.metadata, output)

    def answer_swapped(case: JudgeCase) -> JudgeCase | None:
        c: SupportCase = case.metadata
        lines = _ANSWER_LINE.findall(case.output)
        if not lines:
            return None
        end = case.output.rfind(lines[-1])
        wrong = f'Answer: {fmt(c, plausible_wrong(c))}'
        return replace(case, output=case.output[:end] + wrong + case.output[end + len(lines[-1]) :])

    def explanation_cut(case: JudgeCase) -> JudgeCase | None:
        lines = _ANSWER_LINE.findall(case.output)
        return replace(case, output=lines[-1].strip()) if lines else None

    def evidence(case: JudgeCase) -> JudgeCase | None:
        new = changed[case.name]
        return None if new is None else replace(case, inputs=new.inputs, metadata=new)

    mutants = (
        *DEFAULT_MUTANTS,
        Mutant('plausible_wrong', answer_swapped),
        Mutant('explanation_cut', explanation_cut),
        Mutant('policy_changed', evidence, changes='evidence'),
    )

    def checks_the_answer(inputs: str, output: str, expected: str) -> bool:
        c = by_inputs.get(inputs)
        return c is not None and is_correct(c, output)

    def lenient(inputs: str, output: str, expected: str) -> bool:
        return bool(_ANSWER_LINE.search(output))

    def against_expected(inputs: str, output: str, expected: str) -> bool:
        c = next(c for c in support if c.answer == expected)  # is_correct reads only the kind and the answer
        return is_correct(replace(c, answer=expected), output)

    evaluators = {
        'EqualsExpected': EqualsExpected(),
        'Contains(expected)': lambda case: Contains(value=case.expected_output),
        'IsInstance(str)': IsInstance('str'),
        'judge: checks the answer': scripted(checks_the_answer, include_input=True),
        'judge: lenient': scripted(lenient, include_input=True),
        'judge: output only': scripted(against_expected, include_input=False),
    }
    report = await mutation_report(evaluators, judge_cases, mutants, oracle=oracle)
    sources = [source for _, source in good.values()]
    print(f'{sources.count("agent")} real agent replies, {sources.count("template")} templated\n')
    print(report.table())
    for name in evaluators:
        for outcome in report.missed(name)[:1]:
            print(f'\n{name} missed {outcome.mutant} on {outcome.case}: {outcome.output!r}')
    OUT.write_text(
        json.dumps(
            {'replies': {'agent': sources.count('agent'), 'template': sources.count('template')}, **report.to_dict()},
            indent=2,
            default=str,
        )
    )


if __name__ == '__main__':
    asyncio.run(main())
