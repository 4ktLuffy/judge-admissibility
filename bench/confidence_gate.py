"""Certify a judge's confidence gate: how optimistic in-sample thresholds are, then a real Codex judge.

    PYTHONPATH=.:bench <venv>/bin/python bench/confidence_gate.py [--sim-only]

1. Simulation, no model. A judge whose probability is a noisy sigmoid of the truth (70% of
   outputs pass; score = +-1.2 + N(0, 1.5)); accuracy on what it decides rises from 0.79 ungated
   to 0.99 at confidence 0.95. True accuracy per gate from 200,000 draws. For n in {100, 390,
   1000} cases (390 as in pydantic-ai #9641), many samples each, bar 0.9, the 20 gates of
   `confidence_grid()`:
   - in-sample: choose the widest gate whose exact lower bound is >= 0.9 (or whose estimate is),
     report it on the same cases;
   - split: choose on half, certify on the other half (`calibrate_gate(method='split')`);
   - fixed sequence: `calibrate_gate(method='fixed_sequence')`.
   Reported: how often a gate is chosen, mean (reported - true) accuracy, share overstated, how
   often the true accuracy is below the reported exact lower bound (nominal 2.5%), and how often a
   PASS (or an in-sample claim) is made for a gate truly below the bar. Then a negative control,
   where no gate reaches the bar (bar 0.97, separation 0.9): every PASS there is false.
2. Real run. A Codex judge (`gpt-5.6-luna`, reasoning effort none) gives a pass/fail verdict and a
   STATED confidence 0-1 on 120 support replies: the 80 cached agent replies to the 40 questions
   (`results/support-cache.json`, no agent calls), labelled by `support_task.is_correct`, and one
   plausible wrong answer per question (labelled fail). The stated confidence is verbalized, not
   a model probability: p(pass) is read as `confidence` if it says pass, `1 - confidence` if fail.
   The fallback for deferred replies is the same model with reasoning effort high, asked only on
   replies some candidate gate defers. Judge calls are saved in the results, so a rerun makes none.
   The 120 outputs come from 40 questions, and outputs of one question are not independent (the
   two agent replies are sometimes the same text, judged by the same saved verdict), so the
   question is the group and the unit of evidence: intervals count 40 questions, and a split keeps
   each question's outputs on one side. The per-output report is kept, labelled as superseded, to
   show what counting outputs claimed.

Writes `results/confidence_gate.json`.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from pydantic_evals_admissibility._confidence_gate import (
    ConfidenceGate,
    ConfidenceRequirements,
    GateCase,
    _tally,  # pyright: ignore[reportPrivateUsage]
    calibrate_gate,
    certify_confidence_gate,
    choose_gate,
    confidence_grid,
)

ROOT = Path(__file__).parent.parent
OUT = ROOT / 'results' / 'confidence_gate.json'
BAR = 0.9


def draw(n: int, rng: random.Random, sep: float = 1.2) -> list[GateCase]:
    out: list[GateCase] = []
    for i in range(n):
        label = rng.random() < 0.7
        score = sep * (1 if label else -1) + rng.gauss(0, 1.5)
        out.append(GateCase(f'c{i}', 1 / (1 + math.exp(-score)), label))
    return out


def true_accuracy(sep: float) -> dict[ConfidenceGate, float]:
    population = draw(200_000, random.Random(424242), sep)
    out: dict[ConfidenceGate, float] = {}
    for gate in confidence_grid():
        decided, correct, _, _ = _tally(gate, population)
        out[gate] = correct / decided
    return out


def _summary(rows: list[tuple[float, float, float, bool]], trials: int, bar: float) -> dict[str, Any]:
    """rows: (reported accuracy, true accuracy, reported lower bound, claimed) per trial where a gate was chosen."""
    if not rows:
        return {'chosen': 0, 'trials': trials}
    k = len(rows)
    return {
        'trials': trials,
        'chosen': k,
        'mean_overstatement': sum(r - t for r, t, _, _ in rows) / k,
        'share_overstated': sum(r > t for r, t, _, _ in rows) / k,
        'truth_below_lower_bound': sum(t < lb for _, t, lb, _ in rows) / k,
        'claims_while_truly_below_bar': sum(c and t < bar for _, t, _, c in rows),
        'claims_while_truly_below_bar_share_of_trials': sum(c and t < bar for _, t, _, c in rows) / trials,
        'claims': sum(c for *_, c in rows),
    }


def simulate(n: int, trials: int, *, sep: float = 1.2, bar: float = BAR, seed: int = 0) -> dict[str, Any]:
    truth = true_accuracy(sep)
    requirements = ConfidenceRequirements(min_accuracy=bar)
    rows: dict[str, list[tuple[float, float, float, bool]]] = {
        'in_sample_lower_bound': [],
        'in_sample_estimate': [],
        'split_in_sample_half': [],
        'split_confirmed': [],
        'fixed_sequence': [],
    }

    def lower(report: Any) -> float:
        return report.check('auto_accuracy').interval[0]

    for t in range(trials):
        cases = draw(n, random.Random(seed * 1_000_003 + t), sep)
        for criterion in ('lower_bound', 'estimate'):
            gate = choose_gate(cases, min_accuracy=bar, criterion=criterion)  # type: ignore[arg-type]
            if gate is not None:
                r = certify_confidence_gate(cases, gate, requirements=requirements)
                # The in-sample claim: "this gate reaches the bar", made on the cases it was chosen on.
                rows[f'in_sample_{criterion}'].append((r.accuracy or 0.0, truth[gate], lower(r), True))
        s = calibrate_gate(cases, requirements=requirements, seed=t)
        if s.chosen is not None and s.confirmed is not None and s.in_sample is not None:
            g = s.chosen
            rows['split_in_sample_half'].append((s.in_sample.accuracy or 0.0, truth[g], lower(s.in_sample), True))
            c = s.confirmed
            rows['split_confirmed'].append((c.accuracy or 0.0, truth[g], lower(c), c.status == 'PASS'))
        f = calibrate_gate(cases, requirements=requirements, method='fixed_sequence')
        if f.chosen is not None and f.confirmed is not None:
            c = f.confirmed
            rows['fixed_sequence'].append((c.accuracy or 0.0, truth[f.chosen], lower(c), c.status == 'PASS'))
    return {
        'n': n,
        'separation': sep,
        'bar': bar,
        'true_accuracy_by_gate': {f'{g.low:g}/{g.high:g}': round(a, 4) for g, a in truth.items()},
        **{name: _summary(r, trials, bar) for name, r in rows.items()},
    }


# Real run ----------------------------------------------------------------------------------------

JUDGE_INSTRUCTIONS = (
    'You are grading a customer support reply. Decide whether the reply correctly answers the '
    "customer's question according to the store policy given in the input. First give your reason, "
    'then the verdict (passed: true if the reply is correct, false if not), then your confidence: '
    'the probability, from 0 to 1, that your verdict is right.'
)


class ConfidentVerdict(BaseModel):
    reason: str = Field(description='One or two sentences on what the policy says and what the reply answered.')
    passed: bool
    confidence: float = Field(description='Probability from 0 to 1 that the verdict is right.')


def _key(model: str, effort: str, inputs: str, reply: str) -> str:
    return hashlib.sha256(f'{model}|{effort}|{JUDGE_INSTRUCTIONS}|{inputs}|{reply}'.encode()).hexdigest()[:20]


async def real_run(results: dict[str, Any]) -> dict[str, Any]:
    from codex_judge import TOKENS, codex_model
    from pydantic_ai import Agent
    from support_eval import AGENT_PROMPT, REPEATS, fmt, plausible_wrong
    from support_task import cases, is_correct

    real: dict[str, Any] = results.get('real') or {}
    saved: dict[str, Any] = real.get('judgments') or {}
    cache: dict[str, str] = json.loads((ROOT / 'results' / 'support-cache.json').read_text())
    outputs: list[tuple[str, str, str, bool]] = []  # (name, inputs, reply, label)
    for c in cases():
        for r in range(REPEATS):
            reply = cache[hashlib.sha256(f'{AGENT_PROMPT}|{c.inputs}|{r}'.encode()).hexdigest()]
            outputs.append((f'{c.name}#reply{r}', c.inputs, reply, is_correct(c, reply)))
        outputs.append((f'{c.name}#wrong', c.inputs, f'Answer: {fmt(c, plausible_wrong(c))}', False))

    calls = {'made': 0}

    async def judge_all(todo: list[tuple[str, str, str, bool]], model: str, effort: str) -> None:
        agent = Agent(
            codex_model(model, effort=effort, timeout=300),
            output_type=ConfidentVerdict,
            instructions=JUDGE_INSTRUCTIONS,
        )
        limit = asyncio.Semaphore(4)

        async def one(name: str, inputs: str, reply: str) -> None:
            key = _key(model, effort, inputs, reply)
            if key in saved:
                return
            async with limit:
                try:
                    v = (await agent.run(f'<Input>\n{inputs}\n</Input>\n<Reply>\n{reply}\n</Reply>')).output
                    saved[key] = {'case': name, 'model': f'codex:{model}@{effort}', 'passed': v.passed,
                                  'confidence': v.confidence, 'reason': v.reason}  # fmt: skip
                except Exception as error:  # recorded, and the case is left out
                    saved[key] = {'case': name, 'model': f'codex:{model}@{effort}', 'error': repr(error)[:300]}
                calls['made'] += 1
            real['judgments'] = saved
            results['real'] = real
            OUT.write_text(json.dumps(results, indent=2))

        # One call per distinct (input, reply): identical replies share a verdict, never race for it.
        unique = {_key(model, effort, i, r): (n, i, r) for n, i, r, _ in todo}
        await asyncio.gather(*(one(n, i, r) for n, i, r in unique.values()))

    before = len(TOKENS)
    await judge_all(outputs, 'gpt-5.6-luna', 'none')
    primary = {n: saved[_key('gpt-5.6-luna', 'none', i, r)] for n, i, r, _ in outputs}
    labels = {n: lab for n, _, _, lab in outputs}

    def p_pass(v: dict[str, Any]) -> float:
        conf = min(1.0, max(0.0, float(v['confidence'])))
        return conf if v['passed'] else 1 - conf

    ok = {n: v for n, v in primary.items() if 'error' not in v}
    candidates = confidence_grid(top=0.9)
    widest = candidates[-1]
    deferred_somewhere = [o for o in outputs if o[0] in ok and widest.decide(p_pass(ok[o[0]])) == 'defer']
    fallback: dict[str, bool] = {}
    if len(deferred_somewhere) <= 80:  # the call budget
        await judge_all(deferred_somewhere, 'gpt-5.6-luna', 'high')
        for n, i, r, _ in deferred_somewhere:
            v = saved[_key('gpt-5.6-luna', 'high', i, r)]
            if 'error' not in v:
                fallback[n] = bool(v['passed'])
    # The question is the group: its two replies and its wrong answer share the policy, the question and,
    # for identical replies, the very same saved verdict, so the question is the unit of evidence.
    gate_cases = [GateCase(n, p_pass(v), labels[n], fallback.get(n), group=n.split('#')[0]) for n, v in ok.items()]
    complete_fallback = all(c.fallback is not None for c in gate_cases if widest.decide(c.probability) == 'defer')
    if not complete_fallback:
        gate_cases = [GateCase(c.name, c.probability, c.label, group=c.group) for c in gate_cases]
    requirements = ConfidenceRequirements(min_accuracy=BAR)

    stated = [float(v['confidence']) for v in ok.values()]
    verdict_right = sum(bool(v['passed']) == labels[n] for n, v in ok.items())
    reports = {
        'ungated (p >= 0.5)': certify_confidence_gate(gate_cases, ConfidenceGate(0.5, 0.5), requirements=requirements),
        "#9641's gate (0.2/0.8), fixed in advance": certify_confidence_gate(
            gate_cases, ConfidenceGate(0.2, 0.8), requirements=requirements
        ),
        'SUPERSEDED: ungated, counted per output (outputs of one question are not independent)': (
            certify_confidence_gate(gate_cases, ConfidenceGate(0.5, 0.5), requirements=requirements, unit='case')
        ),
    }
    split = calibrate_gate(gate_cases, requirements=requirements, candidates=candidates, seed=0)
    fixed = calibrate_gate(gate_cases, requirements=requirements, candidates=candidates, method='fixed_sequence')
    seeds = [calibrate_gate(gate_cases, requirements=requirements, candidates=candidates, seed=s) for s in range(20)]
    real.update(
        {
            'judge': 'codex:gpt-5.6-luna@none',
            'fallback_judge': 'codex:gpt-5.6-luna@high' if fallback else None,
            'models_used': sorted({v['model'] for v in saved.values()}),
            'confidence_is': 'verbalized (stated in the reply), not a model probability',
            'outputs': len(outputs),
            'questions (groups, the unit of evidence)': len({c.group for c in gate_cases}),
            'distinct saved verdicts': len({_key('gpt-5.6-luna', 'none', i, r) for _, i, r, _ in outputs}),
            'judged': len(ok),
            'errors': len(primary) - len(ok),
            'labels_pass': sum(labels[n] for n in ok),
            'labels_fail': sum(not labels[n] for n in ok),
            'verdict_accuracy': verdict_right / len(ok),
            'verdict_right': verdict_right,
            'distinct_stated_confidences': sorted(set(stated)),
            'stated_confidence_counts': {str(c): stated.count(c) for c in sorted(set(stated))},
            'fallback_asked': len(deferred_somewhere) if fallback else 0,
            'fallback_complete': complete_fallback,
            'calls_this_run': calls['made'],
            'calls_first_run': real.get('calls_first_run', real.get('calls_this_run', calls['made'])),
            'tokens_this_run': sum(TOKENS[before:]),
            'reports': {k: r.to_dict() for k, r in reports.items()},
            'split_seed0': split.to_dict(),
            'fixed_sequence': fixed.to_dict(),
            'split_over_20_seeds': {
                'status': {s: sum(x.status == s for x in seeds) for s in ('PASS', 'FAIL', 'UNVALIDATED')},
                'chosen': [None if x.chosen is None else f'{x.chosen.low:g}/{x.chosen.high:g}' for x in seeds],
            },
        }
    )
    for name, r in reports.items():
        print(f'\n== {name}\n{r.table()}')
    print(f'\n== split (seed 0): {split.status}: {split.reason}')
    print(f'== fixed sequence: {fixed.status}: {fixed.reason}')
    return real


def main() -> None:
    results: dict[str, Any] = json.loads(OUT.read_text()) if OUT.exists() else {}
    start = time.time()
    sims = [simulate(100, 400), simulate(390, 400), simulate(1000, 200)]
    control = simulate(390, 400, sep=0.9, bar=0.97, seed=7)
    results['simulation'] = {
        'judge': 'p = sigmoid(+-sep + N(0, 1.5)), 70% of outputs pass',
        'candidates': 'confidence_grid(): confidence 0, 0.05, ..., 0.95 at cutoff 0.5',
        'runs': sims,
        'negative_control': control,
        'seconds': round(time.time() - start, 1),
    }
    OUT.write_text(json.dumps(results, indent=2))
    for s in [*sims, control]:
        print(f'\nn={s["n"]} sep={s["separation"]} bar={s["bar"]}')
        for name in (
            'in_sample_lower_bound',
            'in_sample_estimate',
            'split_in_sample_half',
            'split_confirmed',
            'fixed_sequence',
        ):
            row = s[name]
            if not row['chosen']:
                print(f'  {name:<24} chose nothing in {row["trials"]}')
                continue
            print(
                f'  {name:<24} chosen {row["chosen"]:>3}/{row["trials"]}  '
                f'overstates by {row["mean_overstatement"]:+.4f} '
                f'({row["share_overstated"]:.0%} of picks)  truth<LB {row["truth_below_lower_bound"]:.3f}  '
                f'claims/PASS while truly below bar {row["claims_while_truly_below_bar"]}'
            )
    if '--sim-only' not in sys.argv:
        results['real'] = asyncio.run(real_run(results))
        OUT.write_text(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
