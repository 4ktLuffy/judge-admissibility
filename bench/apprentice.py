"""Judge apprenticeship on the support task: can reviewed disagreements fix a judge that cannot see the question?

    PYTHONPATH=.:bench <venv>/bin/python bench/apprentice.py

The judge: `LLMJudge` with the support rubric and the default `include_input=False`, Codex
`gpt-5.6-luna` with no reasoning. Its reasoning-high twin was certified INADMISSIBLE in
`bench/support_eval.py`: it failed correct replies it could not check ("the question is not
provided") and passed answers borrowed from other customers' questions.

The reviewers: ground truth. The 80 agent replies in `results/support_eval.json` carry
code-computed labels (`human:1` / `human:0`), which stand in for a person's review.

1. Split the 40 questions by case, 70% training. Judge the first reply to each training question
   with the blind judge (28 calls); its verdicts that differ from ground truth are the reviewed
   disagreements.
2. Codex (no reasoning, 1 call) proposes a one-sentence rubric clarification and picks up to three
   of those disagreements as examples.
3. `apprentice` certifies the original and the revised judge on the 12 held-out questions (known-good
   reply, every reviewed reply as a label, a borrowed answer as the rejection control) and gates
   the agreement with ground truth, case by case.

The known cause of the failure is `include_input=False`, which no rubric text changes; the point
is to see whether the gate and the certificate keep a rubric "fix" from being promoted when it
does not fix anything, or fixes the number while breaking the judge.

About 125 Codex calls. Training verdicts and the proposal are saved in `results/apprentice.json`
as they are made, so a rerun does not pay for them twice.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from codex_judge import TOKENS, codex_model
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_evals.evaluators import LLMJudge
from support_task import cases as support_cases

from pydantic_evals_admissibility import GateRules, JudgeCase, MismatchedOutput, assertion_of
from pydantic_evals_admissibility._apprentice import Proposal, ReviewedCase, apprentice, split_cases
from pydantic_evals_admissibility._certify import _context

ROOT = Path(__file__).parent.parent
SOURCE = ROOT / 'results' / 'support_eval.json'
OUT = ROOT / 'results' / 'apprentice.json'
RUBRIC = "The reply correctly answers the customer's question according to the store policy."
SPLIT, SEED = 0.7, 0
MAX_EXAMPLES = 3


class Proposed(BaseModel):
    reason: str = Field(description='What the reviewers saw that the judge did not, in one or two sentences.')
    clarification: str = Field(description='One sentence to append to the rubric.')
    examples: list[int] = Field(description=f'Up to {MAX_EXAMPLES} disagreement numbers to show the judge as examples.')


PROPOSER = (
    'You improve the rubric of an LLM judge. You are shown its rubric and outputs on which a reviewer overruled '
    "it, with the judge's verdict and reason and the reviewer's verdict. Write one sentence to append to the "
    'rubric so the judge agrees with the reviewer on outputs like these in general, not only on these. You may '
    'also pick a few of the disagreements to show the judge as examples.'
)


def reviewed_cases() -> list[ReviewedCase]:
    """The 80 agent replies, each with its ground-truth label, in the order the certificate judged them."""
    data = json.loads(SOURCE.read_text())
    by_name = {c.name: c for c in support_cases()}
    labelled = [
        j for j in data['certificates']['weak (no reasoning, sees the question)']['judgments']
        if j['role'].startswith('human:')
    ]  # fmt: skip
    return [
        ReviewedCase(
            j['case'],
            by_name[j['case']].inputs,
            j['output'],
            j['role'] == 'human:1',
            expected_output=by_name[j['case']].answer,
        )
        for j in labelled
    ]


def save(results: dict[str, Any]) -> None:
    OUT.write_text(json.dumps(results, indent=2, default=str))


async def judge_training(judge: LLMJudge, reviews: Sequence[ReviewedCase], cached: dict[str, Any]) -> None:
    """The blind judge's verdict on each training reply not judged yet, into `cached` (key: case)."""
    limit = asyncio.Semaphore(2)

    async def one(r: ReviewedCase) -> None:
        async with limit:
            try:
                raw = await judge.evaluate_async(
                    _context(JudgeCase(r.case, r.inputs, r.output, r.expected_output), r.output)
                )
                passed, reason = assertion_of(raw)
                cached[r.case] = {'output': r.output, 'passed': passed, 'reason': reason}
            except Exception as error:  # a verdict that errored is not a disagreement
                cached[r.case] = {'output': r.output, 'passed': None, 'reason': f'{type(error).__name__}: {error}'}

    await asyncio.gather(*(one(r) for r in reviews if r.case not in cached))


def proposer(results: dict[str, Any]) -> Any:
    async def propose(rubric: str, shown: Sequence[ReviewedCase]) -> Proposal:
        if 'proposed' not in results:
            listing = '\n\n'.join(
                f'Disagreement {i}:\nCustomer message given to the agent:\n{r.inputs}\nOutput judged:\n{r.output}\n'
                f'Judge: {"PASS" if r.judge_passed else "FAIL"} ({r.judge_reason})\n'
                f'Reviewer: {"PASS" if r.reviewer_passed else "FAIL"}'
                for i, r in enumerate(shown)
            )
            agent = Agent(codex_model(effort='none', timeout=300), output_type=Proposed, instructions=PROPOSER)
            out = (await agent.run(f'Rubric:\n{rubric}\n\n{listing}')).output
            results['proposed'] = out.model_dump()
            save(results)
        proposed = results['proposed']
        picks = [i for i in proposed['examples'] if 0 <= i < len(shown)][:MAX_EXAMPLES]
        return Proposal(proposed['clarification'], tuple(shown[i] for i in dict.fromkeys(picks)))

    return propose


async def main() -> None:
    results: dict[str, Any] = json.loads(OUT.read_text()) if OUT.exists() else {}
    reviews = reviewed_cases()
    train, held_out = split_cases([r.case for r in reviews], SPLIT, SEED)
    judge = LLMJudge(rubric=RUBRIC, model=codex_model(effort='none', timeout=300))

    # Step 1: the blind judge's verdict on the first reply to each training question.
    first = {name: next(r for r in reviews if r.case == name) for name in train}
    cached: dict[str, Any] = results.setdefault('training_verdicts', {})
    before = len(TOKENS)
    await judge_training(judge, list(first.values()), cached)
    results['training_calls'] = results.get('training_calls', 0) + (len(TOKENS) - before)
    save(results)

    with_verdicts = [
        ReviewedCase(
            r.case, r.inputs, r.output, r.reviewer_passed,
            judge_passed=cached[r.case]['passed'], judge_reason=cached[r.case]['reason'],
            expected_output=r.expected_output,
        )
        if r.case in first and r is first[r.case] and cached.get(r.case, {}).get('output') == r.output
        else r
        for r in reviews
    ]  # fmt: skip
    trained = [r for r in with_verdicts if r.judge_passed is not None]
    print(
        f'training: {len(trained)} replies judged, {sum(r.disagrees for r in trained)} disagree with ground truth '
        f'({sum(r.disagrees and r.reviewer_passed for r in trained)} correct replies failed, '
        f'{sum(r.disagrees and not r.reviewer_passed for r in trained)} wrong replies passed)',
        flush=True,
    )

    # Steps 2 and 3.
    before = len(TOKENS)
    result = await apprentice(
        judge,
        with_verdicts,
        propose=proposer(results),
        split=SPLIT,
        seed=SEED,
        controls=(MismatchedOutput(),),
        repeats=1,
        rules=GateRules(),
        max_examples=MAX_EXAMPLES,
        max_concurrency=2,
    )
    print('\n' + result.table(), flush=True)
    print(f'\nclarification: {result.proposal.clarification}')
    print(f'\n{result.baseline_certificate.table()}\n\n{result.revised_certificate.table()}')
    results['result'] = result.to_dict()
    results['training'] = {
        'judged': len(trained),
        'disagreements': sum(r.disagrees for r in trained),
        'correct_failed': sum(r.disagrees and r.reviewer_passed for r in trained),
        'wrong_passed': sum(r.disagrees and not r.reviewer_passed for r in trained),
    }
    results['apprentice_calls'] = len(TOKENS) - before
    results['judge'] = 'LLMJudge(codex gpt-5.6-luna, effort none, include_input=False)'
    save(results)
    print(f'\n{len(TOKENS)} Codex calls with a token count this run')


if __name__ == '__main__':
    asyncio.run(main())
