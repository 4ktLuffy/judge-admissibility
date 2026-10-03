"""A jury of three cheap judges against each of them alone, on support replies that are wrong but plausible.

    PYTHONPATH=.:bench <venv>/bin/python bench/jury.py

The task is the support task (`bench/support_task.py`): 16 cases, the first four of each kind
(return, shipping, refund, warranty), each with a known-good agent reply saved in
`results/support_eval.json`. Single cheap judges were measured near perfect on mismatched and
empty answers there, so those controls cannot show a jury fixing anything. The controls here are
the mistakes a support agent makes, where a judge has to check the reply against the policy:

- `plausible_wrong` (must fail): `support_eval.plausible_wrong`, as a bare `Answer: ...` line
  (the refund without the restocking fee, the other side of the shipping threshold, the opposite
  yes/no).
- `plausible_wrong_explained` (must fail): the same kind of wrong answer in a fluent reply with
  a reason that sounds right and is not: the days since delivery miscounted, the warranty length,
  shipping threshold or restocking percentage misquoted, the arithmetic consistent with the
  misquote.
- `whitespace_reformat` (must hold).

Members, all `LLMJudge(include_input=True)` on Codex, diverse in model and reasoning:
`gpt-5.6-luna` no reasoning, `gpt-reserve` no reasoning, `gpt-5.6-luna` low reasoning.

Each member is certified once (`certify_judge`, repeats 2, seed 0, concurrency 3): 80 calls each,
240 in all. The jury's certificate is computed from those verdicts (`jury_certificate`), not by
calling the members again: the jury's verdict is a function of its members' verdicts, so the
same verdicts give the jury's certificate for free and make the comparison paired. Every rule
(`majority`, `unanimous_pass`, `any_fail`) is computed from the same verdicts.

`codex_judge.run_codex` is wrapped to count real Codex invocations (pydantic-ai retries
included) and refuse any past 300. Each member's certificate is saved to `results/jury.json` as
it finishes, with the `evidence_fingerprint` of the plan it was made on, and a rerun reuses saved
members instead of calling them again.

    PYTHONPATH=.:bench <venv>/bin/python bench/jury.py --replay

recomputes every jury from the member verdicts saved in `results/jury.json`, with no Codex call:
the comparison is redone (case-level statistics) and the juries are rewritten. Certificates saved
before fingerprints were recorded get this plan's fingerprint, marked as attached on replay (and
re-attached on every replay, so it follows the fingerprint's current form); their outputs are
still checked against the plan judgment by judgment.
"""

from __future__ import annotations

import asyncio
import json
import random
import re
import sys
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import codex_judge
from codex_judge import TOKENS, codex_model
from pydantic_evals.evaluators import LLMJudge
from support_eval import RUBRIC, fmt, plausible_wrong
from support_task import SupportCase, cases

from pydantic_evals_admissibility import Certificate, JudgeCase, WhitespaceReformat, certify_judge
from pydantic_evals_admissibility._controls import ControlKind
from pydantic_evals_admissibility._jury import Jury, compare_jury, evidence_fingerprint, votes_of
from pydantic_evals_admissibility._routing import truth_of

ROOT = Path(__file__).parent.parent
OUT = ROOT / 'results' / 'jury.json'
CALL_CAP = 300
REPEATS = 2
SEED = 0
PER_KIND = 4

MEMBERS = {
    'luna@none': ('gpt-5.6-luna', 'none'),
    'reserve@none': ('gpt-reserve', 'none'),
    'luna@low': ('gpt-5.6-luna', 'low'),
}


def _days(question: str) -> int:
    found = re.findall(r'(\w+ \d+, \d{4})', question)
    delivered, today = (datetime.strptime(d, '%B %d, %Y') for d in found[:2])
    return (today - delivered).days


def explained_wrong(case: SupportCase) -> str:
    """A fluent reply with a wrong answer and a reason that sounds right: what a judge must check."""
    p, q = case.policy, case.question
    if case.kind == 'return':
        elapsed = _days(q)
        if case.answer == 'yes':  # within the window: claim a month more than it was
            claimed = elapsed + 30 if elapsed + 30 > p.return_days else p.return_days + 3
            return (
                f'Your order was delivered {claimed} days ago, which is outside the {p.return_days}-day return '
                'window, so unfortunately it can no longer be returned, even unopened.\n\nAnswer: no'
            )
        claimed = elapsed - 30 if 0 < elapsed - 30 < p.return_days else p.return_days - 3
        return (
            f'Your order was delivered {claimed} days ago, which is within the {p.return_days}-day return window. '
            'Since it is unopened, you will get a full refund.\n\nAnswer: yes'
        )
    if case.kind == 'warranty':
        months = int(re.search(r'(\d+) months after', q).group(1))  # type: ignore[union-attr]
        if case.answer == 'yes':
            shorter = max((m for m in (3, 6, 12) if m < months), default=months - 1)
            return (
                f'Our warranty covers defects for {shorter} months from purchase. At {months} months it is past '
                'the warranty period, so it is not covered.\n\nAnswer: no'
            )
        longer = min(m for m in (12, 24, 36, 48) if m > months)
        return (
            f'Our warranty covers defects for {longer} months from purchase, and it stopped working on its own '
            f'at {months} months, so it is covered.\n\nAnswer: yes'
        )
    if case.kind == 'shipping':
        total = float(re.search(r'\$(\d+\.\d{2})', q).group(1))  # type: ignore[union-attr]
        if case.answer == '0.00':
            above = min(t for t in (35, 50, 75, 100, 150) if t > total)
            return (
                f'Orders of ${above} or more ship free. Your cart is ${total:.2f}, so standard shipping is the flat '
                f'${p.shipping_fee:.2f}.\n\nAnswer: ${p.shipping_fee:.2f}'
            )
        below = max(t for t in (10, 20, 25, 35, 50, 75) if t <= total)
        return (
            f'Orders of ${below} or more ship free, and your cart is ${total:.2f}, so standard shipping is free.'
            '\n\nAnswer: $0.00'
        )
    price = float(re.search(r'paid \$(\d+\.\d{2})', q).group(1))  # type: ignore[union-attr]
    pct = min((x for x in (10, 15, 20, 25) if x != p.restocking_pct), key=lambda x: abs(x - p.restocking_pct))
    fee = round(price * pct / 100, 2)
    return (
        f'Opened items are refunded minus a {pct}% restocking fee: ${price:.2f} - ${fee:.2f} = '
        f'${price - fee:.2f}.\n\nAnswer: ${price - fee:.2f}'
    )


@dataclass(frozen=True)
class SupportWrong:
    """A must-fail control from the support case behind each judge case."""

    by_name: dict[str, SupportCase]
    explained: bool
    kind: ControlKind = 'must_fail'

    @property
    def name(self) -> str:
        return 'plausible_wrong_explained' if self.explained else 'plausible_wrong'

    def make(self, case: JudgeCase, cases: Sequence[JudgeCase], rng: random.Random) -> str | None:
        support = self.by_name[case.name]
        if self.explained:
            return explained_wrong(support)
        return f'Answer: {fmt(support, plausible_wrong(support))}'


class Budget:
    """Counts real Codex calls (retries included) and refuses any past the cap."""

    def __init__(self, used: int) -> None:
        self.used = used
        self._run = codex_judge.run_codex

    async def __call__(self, *args: Any, **kwargs: Any) -> str:
        if self.used >= CALL_CAP:
            raise RuntimeError(f'call budget of {CALL_CAP} spent')
        self.used += 1
        return await self._run(*args, **kwargs)


def save(results: dict[str, Any]) -> None:
    OUT.write_text(json.dumps(results, indent=2, default=str))


def by_family(comparison: Any, labels: Sequence[str]) -> dict[str, dict[str, Any]]:
    """Errors per control family, for each member and the jury: where a vote helps and where it cannot."""
    out: dict[str, dict[str, Any]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for j in comparison.jury.judgments:
        right = truth_of(j.role)
        votes = votes_of(j.reason or j.error)
        if right is None or votes is None:
            continue
        family = j.role.split(':', 1)[-1] if ':' in j.role else 'known_good'
        for label, vote in zip(labels, votes, strict=True):
            if vote is not None:
                out[family][label][0] += vote == right
                out[family][label][1] += 1
        if j.passed is not None:
            out[family]['jury'][0] += j.passed == right
            out[family]['jury'][1] += 1
    return {f: {k: f'{a}/{b}' for k, (a, b) in v.items()} for f, v in out.items()}


ATTACHED = 'attached on replay: the run predates evidence fingerprints; outputs checked against this plan'


async def main(replay: bool = False) -> None:
    chosen = [c for c in cases() if int(c.name.split('-')[1]) < PER_KIND]
    by_name = {c.name: c for c in chosen}
    saved = json.loads((ROOT / 'results' / 'support_eval.json').read_text())['certificates']
    good = {
        j['case']: j['output']
        for j in saved['weak (no reasoning, sees the question)']['judgments']
        if j['role'] == 'reference#0'
    }
    judge_cases = [
        JudgeCase(c.name, c.inputs, good[c.name], expected_output=c.answer, metadata={'kind': c.kind}) for c in chosen
    ]
    controls = (SupportWrong(by_name, explained=False), SupportWrong(by_name, explained=True), WhitespaceReformat())
    results: dict[str, Any] = json.loads(OUT.read_text()) if OUT.exists() else {}
    results['plan'] = {
        'cases': [c.name for c in chosen],
        'controls': [c.name for c in controls],
        'repeats': REPEATS,
        'seed': SEED,
        'members': {label: f'LLMJudge(codex:{m}@{e}, include_input=True)' for label, (m, e) in MEMBERS.items()},
        'rubric': RUBRIC,
        'call_cap': CALL_CAP,
    }
    evidence = evidence_fingerprint(judge_cases, controls=controls, repeats=REPEATS, seed=SEED)
    results['plan']['evidence_fingerprint'] = evidence
    budget = Budget(results.get('calls_used', 0))
    if replay:
        missing = [label for label in MEMBERS if label not in results.get('members', {})]
        if missing:
            raise SystemExit(f'--replay needs every member saved in {OUT}; missing {missing}')
        budget.used = CALL_CAP  # a replay makes no Codex call: any attempt is refused
    codex_judge.run_codex = budget  # codex_model's adapter looks run_codex up at call time

    judges = {
        label: LLMJudge(rubric=RUBRIC, model=codex_model(model, effort), include_input=True)
        for label, (model, effort) in MEMBERS.items()
    }
    certs: dict[str, Certificate] = {}
    tokens: dict[str, int] = {}
    plans: dict[str, str] = {}
    for label, judge in judges.items():
        done = results.get('members', {}).get(label)
        if done is not None:
            certs[label], tokens[label] = Certificate.from_dict(done['certificate']), done['tokens']
            if 'evidence_fingerprint' not in done or done.get('evidence_source') == ATTACHED:
                # Never recorded at certification: attached from this plan on replay, so re-attached
                # on each replay (the fingerprint's form changed when it began keeping order and type).
                done['evidence_fingerprint'], done['evidence_source'] = evidence, ATTACHED
            plans[label] = done['evidence_fingerprint']
            continue
        before, calls_before = len(TOKENS), budget.used
        cert = await certify_judge(judge, judge_cases, controls=controls, repeats=REPEATS, max_concurrency=3, seed=SEED)
        errored = [j.error for j in cert.judgments if j.error]
        print(f'\n== {label}: {cert.verdict}, {budget.used - calls_before} codex calls\n{cert.table()}', flush=True)
        results.setdefault('members', {})[label] = {
            'certificate': cert.to_dict(),
            'tokens': sum(TOKENS[before:]),
            'codex_calls': budget.used - calls_before,
            'errored': len(errored),
            'evidence_fingerprint': evidence,
        }
        results['calls_used'] = budget.used
        save(results)
        if len(errored) > len(cert.judgments) // 4:
            results['stopped'] = f'{label}: {len(errored)} of {len(cert.judgments)} judgments errored: {errored[:3]}'
            save(results)
            print('stopped:', results['stopped'])
            return
        certs[label], tokens[label], plans[label] = cert, sum(TOKENS[before:]), evidence

    members = [judges[label] for label in MEMBERS]
    results['juries'] = {}
    for rule in ('majority', 'unanimous_pass', 'any_fail'):
        jury = Jury(members, rule=rule)  # type: ignore[arg-type]
        comparison = await compare_jury(
            jury, judge_cases, controls=controls, repeats=REPEATS, seed=SEED,
            member_certificates=[certs[label] for label in MEMBERS],
            member_plans=[plans[label] for label in MEMBERS],
            member_tokens=[tokens[label] for label in MEMBERS],
        )  # fmt: skip
        print(f'\n== jury, {rule}: {comparison.jury.verdict}\n{comparison.table()}', flush=True)
        results['juries'][rule] = {
            **comparison.to_dict(judgments=False),
            'jury': comparison.jury.to_dict(),  # the jury's verdicts, each with every member's vote
            'members': None,  # saved once, under 'members'
            'by_family': by_family(comparison, jury.labels),
            'fingerprint': comparison.jury.fingerprint,
        }
        save(results)
    if replay:
        results['replayed'] = (
            'juries recomputed from the saved member verdicts (bench/jury.py --replay), no Codex call: '
            'accuracy by case, each member compared case by case (sign-flip, level split over the members)'
        )
    else:
        results['calls_used'] = budget.used
    results['tokens_total'] = sum(tokens.values())
    save(results)
    print(f'\ncodex calls used: {results.get("calls_used")} of {CALL_CAP}; tokens {sum(tokens.values()):,}')


if __name__ == '__main__':
    asyncio.run(main(replay='--replay' in sys.argv[1:]))
