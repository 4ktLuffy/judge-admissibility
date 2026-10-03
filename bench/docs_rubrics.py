"""The LLM-judge guide's "Better" rubric, verbatim, with and without the question.

    PYTHONPATH=.:bench <venv>/bin/python bench/docs_rubrics.py

pydantic-evals' guide (`docs/evals/evaluators/llm-judge.md`, "Best Practices", "1. Be Specific in
Rubrics") shows a progression: "Bad" `LLMJudge(rubric='Good answer')`; "Better"
`LLMJudge(rubric='Response accurately answers the question without hallucinating facts')`, with
the default `include_input=False`; "Best", a longer rubric with `include_input=True`. The
"Better" rubric asks about the question, which that configuration does not show the judge.

The guide ships no cases for it, so this measures it on this repository's support task (40
questions, the agent's correct replies cached in `results/support-cache.json`, so the agent is
not called): every correct reply, and every reply moved to another customer's question
(`MismatchedOutput`), which a judge that answers "does it answer the question" must fail. The
same rubric runs twice, as the guide writes it and with `include_input=True`. Codex
`gpt-5.6-luna`, no reasoning, one judgment each. Writes `results/docs_rubrics.json`.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from codex_judge import TOKENS, codex_model
from pydantic_evals.evaluators import LLMJudge
from support_eval import agent_replies, fmt
from support_task import cases, is_correct

from pydantic_evals_admissibility import JudgeCase, MismatchedOutput, certify_judge

ROOT = Path(__file__).parent.parent
OUT = ROOT / 'results' / 'docs_rubrics.json'
BETTER = 'Response accurately answers the question without hallucinating facts'
SOURCE = 'docs/evals/evaluators/llm-judge.md, Best Practices, "1. Be Specific in Rubrics", Better'


async def main() -> None:
    cs = cases()
    replies = await agent_replies(cs)  # cached: no agent calls
    good = {c.name: next((r for r in replies[c.name] if is_correct(c, r)), f'Answer: {fmt(c, c.answer)}') for c in cs}
    judge_cases = [JudgeCase(c.name, c.inputs, good[c.name], expected_output=c.answer) for c in cs]
    out: dict[str, object] = {'rubric': BETTER, 'source': SOURCE, 'cases': len(judge_cases)}
    for label, include_input in (('as the guide writes it (include_input=False)', False),
                                 ('with include_input=True', True)):  # fmt: skip
        before = len(TOKENS)
        judge = LLMJudge(rubric=BETTER, model=codex_model(effort='none'), include_input=include_input)
        cert = await certify_judge(judge, judge_cases, controls=(MismatchedOutput(),), repeats=1, max_concurrency=6)
        rows = {c.name: (c.successes, c.trials) for c in cert.checks}
        wrong = [j for j in cert.judgments if j.role.startswith('must_fail') and j.passed is not None]
        out[label] = {
            'verdict': cert.verdict,
            'correct_replies_passed': rows.get('acceptance'),
            'other_questions_answers_passed': [sum(bool(j.passed) for j in wrong), len(wrong)],
            'calls': len(TOKENS) - before,
            **cert.to_dict(),
        }
        print(f'{label}: {cert.verdict}; passed {sum(bool(j.passed) for j in wrong)} of {len(wrong)} replies '
              f'that answer another customer\'s question\n{cert}\n', flush=True)  # fmt: skip
        OUT.write_text(json.dumps(out, indent=2, default=str))


if __name__ == '__main__':
    asyncio.run(main())
