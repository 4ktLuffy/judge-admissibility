"""Selective routing: a cheap judge where it agrees with itself, an expensive one where it does not.

    PYTHONPATH=.:bench <venv>/bin/python bench/routing.py [--codex]

1. Replay, no judge calls. Saved certificates hold each known-good answer judged twice (or three
   times) by each judge. The cheap judge's repeats are its k samples; where they disagree, the
   policy takes the expensive judge's saved verdict on the same answer. Three saved runs:
   - `support_eval.json`: Codex no reasoning (and `gpt-reserve`) routed to reasoning high, on 40
     support cases; and, on all 280 judgments with a known verdict (controls and human labels
     included), the two cheap judges' agreement with each other as the signal.
   - `evidence_contract.json`: a judge shown only the reply routed to one shown the tool calls.
   - `pydantic_example_judge.json`: Pydantic's own example rubric, no reasoning routed to high.
2. With `--codex`, a live run: `certify_routing` on 8 support cases (2 of each kind, the first
   two), with `MismatchedOutput` controls and, as labelled failures, the plausible wrong answer of
   each case (`support_eval.plausible_wrong`): 24 judgments. The expensive judge alone is
   certified first on the same plan; its verdicts are cached and reused when the policy escalates
   the same judgment, so the comparison is paired and each expensive call is paid once. Tokens
   are measured per phase (phase 2 makes only cheap calls), and so is wall time.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic_evals.evaluators import Evaluator, EvaluatorContext

from pydantic_evals_admissibility import Certificate, HumanLabel, JudgeCase, MismatchedOutput, certify_judge
from pydantic_evals_admissibility._routing import certify_routing, keyed, replay_routing, summarize_routing, truth_of

ROOT = Path(__file__).parent.parent
OUT = ROOT / 'results' / 'routing.json'
CALL_CAP = 100  # judge calls this bench may make live; the cap for the whole task is 150
EXPENSIVE_CAP = 40


def _load(name: str) -> Any:
    return json.loads((ROOT / 'results' / name).read_text())


def references(cert: dict[str, Any]) -> dict[str, list[bool | None]]:
    out: dict[str, list[bool | None]] = defaultdict(list)
    for j in sorted((j for j in cert['judgments'] if j['role'].startswith('reference#')), key=lambda j: j['role']):
        out[j['case']].append(j['passed'])
    return dict(out)


def tokens_per_call(cert: dict[str, Any]) -> float | None:
    return cert['tokens'] / cert['calls'] if cert.get('tokens') and cert.get('calls') else None


def replay_references(label: str, cheap: dict[str, Any], expensive: dict[str, Any], *, k: int) -> dict[str, Any]:
    """Self-consistency over the saved repeats of each known-good answer (truth: pass)."""
    samples = references(cheap)
    strong = {case: verdicts[0] for case, verdicts in references(expensive).items()}
    truth = dict.fromkeys(samples, True)
    cheap_cost, expensive_cost = tokens_per_call(cheap) or 1.0, tokens_per_call(expensive) or 1.0
    summary = replay_routing(samples, strong, truth, k=k, cheap_cost=cheap_cost, expensive_cost=expensive_cost)
    print(f'\n-- {label} (k={k})\n{summary.table()}')
    out = summary.to_dict()
    out['cost_unit'] = 'tokens per call (saved)' if tokens_per_call(cheap) else 'calls'
    return out


def replay_panel(
    label: str, cheap_a: dict[str, Any], cheap_b: dict[str, Any], expensive: dict[str, Any]
) -> dict[str, Any]:  # fmt: skip
    """Two cheap judges' agreement as the signal, on every judgment with a known verdict."""
    a, b, e = keyed_saved(cheap_a), keyed_saved(cheap_b), keyed_saved(expensive)
    samples = {key: [a[key], b[key]] for key in a if key in b}
    truth = {key: t for key in samples if (t := truth_of(key[1])) is not None}
    summary = replay_routing(samples, {key: e.get(key) for key in samples if key in e}, truth, k=2)
    missing = sum(1 for key in truth if key not in e)
    print(f'\n-- {label}\n{summary.table()}')
    return {**summary.to_dict(), 'unmatched_in_expensive': missing, 'cost_unit': 'calls'}


def keyed_saved(cert: dict[str, Any]) -> dict[tuple[str, str, str, int], bool | None]:
    from pydantic_evals_admissibility import Judgment

    judgments = [
        Judgment(j['case'], j['role'], j['output'], j['passed'], j['reason'], j['error']) for j in cert['judgments']
    ]
    return {key: j.passed for key, j in keyed(judgments).items()}


def replay() -> dict[str, Any]:
    support = _load('support_eval.json')['certificates']
    weak = support['weak (no reasoning, sees the question)']
    strong = support['strong (reasoning high, sees the question)']
    reserve = support['gpt-reserve (no reasoning, sees the question)']
    evidence = _load('evidence_contract.json')
    example = _load('pydantic_example_judge.json')['judges']
    runs = {
        'support: no reasoning -> reasoning high': (weak, strong, 2),
        'support: gpt-reserve -> reasoning high': (reserve, strong, 2),
        'evidence: reply only (cannot see the tool calls) -> reply and tool calls': (
            evidence['reply only (default include_input=False), no reasoning'],
            evidence['reply and tool calls, no reasoning'],
            2,
        ),
        'pydantic example (truth: the dataset expected outputs): no reasoning -> high': (
            example['no reasoning'], example['reasoning high'], 2,
        ),
        'pydantic example (truth: the dataset expected outputs): no reasoning -> high, k=3': (
            example['no reasoning'], example['reasoning high'], 3,
        ),
    }  # fmt: skip
    out = {label: replay_references(label, cheap, expensive, k=k) for label, (cheap, expensive, k) in runs.items()}
    panel = 'support panel, every role: no reasoning and gpt-reserve agree, else reasoning high'
    out[panel] = replay_panel(panel, weak, reserve, strong)
    return out


@dataclass
class Metered(Evaluator[Any, Any, Any]):
    """Counts calls against a shared cap, and with `memo`, judges each (inputs, output) once."""

    judge: Evaluator[Any, Any, Any]
    tier: str
    ledger: dict[str, int]
    cap: int
    memo: bool = False
    _cache: dict[str, Any] = field(default_factory=dict, repr=False)

    def get_default_evaluation_name(self) -> str:
        return self.judge.get_default_evaluation_name()

    async def evaluate(self, ctx: EvaluatorContext[Any, Any, Any]) -> Any:
        key = repr((ctx.inputs, ctx.output))
        if self.memo and key in self._cache:
            return self._cache[key]
        if self.ledger['total'] >= CALL_CAP or self.ledger[self.tier] >= self.cap:
            raise RuntimeError(f'call budget spent ({self.ledger})')
        self.ledger['total'] += 1
        self.ledger[self.tier] += 1
        result = await self.judge.evaluate_async(ctx)
        if self.memo:
            self._cache[key] = result
        return result


async def live() -> dict[str, Any]:
    from codex_judge import TOKENS, codex_model
    from pydantic_evals.evaluators import LLMJudge
    from support_eval import RUBRIC, fmt, plausible_wrong
    from support_task import cases

    saved = _load('support_eval.json')['certificates']['weak (no reasoning, sees the question)']
    good = {j['case']: j['output'] for j in saved['judgments'] if j['role'] == 'reference#0'}
    chosen = [c for c in cases() if int(c.name.split('-')[1]) < 2]
    judge_cases = [JudgeCase(c.name, c.inputs, good[c.name], expected_output=c.answer) for c in chosen]
    labels = [HumanLabel(c.name, f'Answer: {fmt(c, plausible_wrong(c))}', False) for c in chosen]
    plan: dict[str, Any] = dict(controls=(MismatchedOutput(),), human_labels=labels, max_concurrency=2, seed=0)
    ledger: dict[str, int] = defaultdict(int)
    cheap = Metered(LLMJudge(rubric=RUBRIC, model=codex_model(effort='none'), include_input=True), 'cheap', ledger, 80)
    expensive = Metered(
        LLMJudge(rubric=RUBRIC, model=codex_model(effort='high', timeout=300), include_input=True),
        'expensive', ledger, EXPENSIVE_CAP, memo=True,
    )  # fmt: skip

    start, before = time.monotonic(), len(TOKENS)
    alone: Certificate = await certify_judge(expensive, judge_cases, repeats=1, **plan)
    expensive_tokens, expensive_seconds = TOKENS[before:], time.monotonic() - start
    errors = [j.error for j in alone.judgments if j.error]
    print(f'\n== expensive alone (reasoning high): {alone.verdict}, {len(expensive_tokens)} calls\n{alone.table()}')
    if errors:
        print('errors:', errors[:3])
    if len(errors) > len(alone.judgments) // 2:
        return {'stopped': 'the expensive judge errored on most judgments', 'errors': errors[:5], 'ledger': ledger}

    start, before = time.monotonic(), len(TOKENS)
    routed = await certify_routing(
        cheap, expensive, judge_cases, k=2, repeats=1,
        baselines={'expensive alone': alone}, expensive_baseline='expensive alone', **plan,
    )  # fmt: skip
    cheap_tokens, cheap_seconds = TOKENS[before:], time.monotonic() - start
    per_cheap = sum(cheap_tokens) / len(cheap_tokens) if cheap_tokens else 1.0
    per_expensive = sum(expensive_tokens) / len(expensive_tokens) if expensive_tokens else 1.0
    by_tokens = summarize_routing(
        routed.certificate, k=2, expensive_alone=alone, cheap_cost=per_cheap, expensive_cost=per_expensive
    )
    n = len(alone.judgments)
    by_seconds = summarize_routing(
        routed.certificate, k=2, expensive_alone=alone,
        cheap_cost=cheap_seconds / max(1, ledger['cheap']), expensive_cost=expensive_seconds / n,
    )  # fmt: skip
    print(f'\n== routed (k=2): {routed.certificate.verdict}\n{routed.certificate.table()}\n{by_tokens.table()}')
    print(f'calls: {dict(ledger)}; tokens/call cheap {per_cheap:.0f}, expensive {per_expensive:.0f}')
    return {
        'cases': [c.name for c in chosen],
        'judgments': n,
        'routed': routed.to_dict(),
        'expensive_alone': alone.to_dict(),
        'cost_in_tokens': by_tokens.to_dict(),
        'cost_in_seconds': {
            **by_seconds.to_dict(),
            'note': 'wall time per call at concurrency 2: phase seconds / calls in the phase',
        },
        'calls': dict(ledger),
        'tokens': {'cheap': sum(cheap_tokens), 'expensive': sum(expensive_tokens)},
        'seconds': {'cheap_phase': cheap_seconds, 'expensive_phase': expensive_seconds},
    }


async def main() -> None:
    results: dict[str, Any] = json.loads(OUT.read_text()) if OUT.exists() else {}
    results['replay'] = replay()
    OUT.write_text(json.dumps(results, indent=2, default=str))
    if '--codex' in sys.argv and 'codex' not in results:
        results['codex'] = await live()
        OUT.write_text(json.dumps(results, indent=2, default=str))


if __name__ == '__main__':
    asyncio.run(main())
