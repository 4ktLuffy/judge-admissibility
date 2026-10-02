"""The whole loop, end to end, on the real Codex runs: improve a prompt without trusting luck.

    PYTHONPATH=.:bench <venv>/bin/python bench/self_improving_loop.py

1. The judge's certificate decides whether its scores may be used at all.
2. Codex proposes prompts; each is scored on the training questions by the certified judge.
3. Select: the candidate the gate likes best (level split across the candidates). Candidates the
   gate rejects are dropped even if nothing better exists.
4. Confirm: the selected candidate, once, on held-out questions, with the certified judge.
5. Promote: only a confirmed candidate becomes the `production` version of a Logfire managed
   variable (Logfire's local in-memory provider here; no account needed).
6. Watch: one reply per held-out question from the promoted prompt goes through `JudgeCanary`,
   which passes each to the judge and checks the judge on an empty answer for a quarter of them.

Agent replies and judge verdicts come from `results/optimize-cache.json`, so steps 1 to 5 cost
no new calls; step 6 calls the judge directly, about 50 times (40 replies plus the controls).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import logfire
from codex_judge import TOKENS, codex_model
from logfire.variables.config import LabeledValue, LatestVersion, Rollout, VariableConfig, VariablesConfig
from optimize import (
    BASELINE_PROMPT,
    ROOT,
    RUBRIC,
    Cache,
    answers,
    certificate_from,
    rate,
    split,
    truth,
    verdicts,
)
from pydantic_evals.evaluators import LLMJudge

from pydantic_evals_admissibility import CanaryMonitor, GateRules, JudgeCanary, decide, detectable_gain, promote
from pydantic_evals_admissibility._certify import _context
from pydantic_evals_admissibility._cases import JudgeCase


def step(n: int, text: str) -> None:
    print(f'\n[{n}] {text}')


async def main() -> None:
    cache, limit = Cache(), asyncio.Semaphore(6)
    train, test = split()
    by_name = {q.name: q for q in train + test}
    run = json.loads((ROOT / 'results' / 'optimize.json').read_text())
    candidates = [c['prompt'] for c in run['candidates']]

    step(1, 'Certify the judges')
    plain = certificate_from(ROOT / 'results' / 'judge_vs_truth.json')
    reference = certificate_from(ROOT / 'results' / 'judge_vs_truth.reference.json')
    print(f'    plain judge {plain.verdict}; reference judge {reference.verdict}')
    judge_cert, use_reference = (reference, True) if reference.admissible else (plain, False)
    if not judge_cert.admissible:
        print('    no admissible judge: stop here, nothing can be decided')
        return

    step(2, f'Score the baseline and {len(candidates)} proposals on {len(train)} training questions')
    base = await verdicts(
        await answers(BASELINE_PROMPT, train, cache, limit), by_name, cache, limit, reference=use_reference
    )
    print(f'    this dataset reliably detects gains of {detectable_gain(base)} or more')
    scored = []
    for i, prompt in enumerate(candidates):
        judged = await verdicts(
            await answers(prompt, train, cache, limit), by_name, cache, limit, reference=use_reference
        )
        scored.append((i, prompt, judged))

    step(3, 'Select (level split across candidates)')
    rules = GateRules().for_candidates(len(scored))
    decisions = [
        (i, prompt, judged, decide(base, judged, certificate=judge_cert, rules=rules)) for i, prompt, judged in scored
    ]
    for i, _, _, d in decisions:
        print(f'    candidate {i}: {d.decision:12} gain {d.mean_gain:+.3f}')
    alive = [(i, p, d) for i, p, _, d in decisions if d.decision != 'REJECT' and (d.mean_gain or 0) > 0]
    if not alive:
        print('    nothing worth confirming: keep the baseline')
        return
    chosen_index, chosen_prompt, _ = max(alive, key=lambda t: t[2].mean_gain or 0)
    print(f'    selected candidate {chosen_index} for confirmation')

    step(4, f'Confirm on {len(test)} held-out questions, one test, certified judge')
    base_test = await answers(BASELINE_PROMPT, test, cache, limit)
    chosen_test = await answers(chosen_prompt, test, cache, limit)
    confirmed = decide(
        await verdicts(base_test, by_name, cache, limit, reference=use_reference),
        await verdicts(chosen_test, by_name, cache, limit, reference=use_reference),
        certificate=judge_cert,
    )
    print(f'    {confirmed.summary()}')
    referee = decide(truth(base_test, by_name), truth(chosen_test, by_name))
    print(f'    (referee, ground truth, not used to decide: {referee.decision}, '
          f'{rate(truth(base_test, by_name)):.3f} -> {rate(truth(chosen_test, by_name)):.3f})')  # fmt: skip

    step(5, 'Promote to the Logfire managed variable, only if confirmed')
    logfire.configure(
        send_to_logfire=False,
        console=False,
        variables=logfire.LocalVariablesOptions(
            config=VariablesConfig(
                variables={
                    'agent_prompt': VariableConfig(
                        name='agent_prompt',
                        labels={'production': LabeledValue(version=1, serialized_value=json.dumps(BASELINE_PROMPT))},
                        rollout=Rollout(labels={'production': 1.0}),
                        overrides=[],
                        latest_version=LatestVersion(version=1, serialized_value=json.dumps(BASELINE_PROMPT)),
                    )
                }
            )
        ),
    )
    prompt_var = logfire.var('agent_prompt', type=str, default=BASELINE_PROMPT)
    promoted = promote(confirmed, 'agent_prompt', chosen_prompt)
    with prompt_var.get() as served:
        now_serving = served.value
    print(
        f'    promoted: {promoted}; production now serves candidate {chosen_index if now_serving == chosen_prompt else "baseline"}'
    )

    step(6, 'Watch the judge on live traffic with JudgeCanary (every 4th call)')
    judge = LLMJudge(rubric=RUBRIC, model=codex_model(), include_input=True, include_expected_output=use_reference)
    monitor = CanaryMonitor(min_checks=5)
    canary = JudgeCanary(judge, every=4, monitor=monitor)  # exactly 10 of 40 calls
    replies = await answers(now_serving, test, cache, limit)
    calls_before = len(TOKENS)
    for q in test:
        reply = replies[q.name][0]
        case = JudgeCase(q.name, q.question, '', expected_output=q.answer if use_reference else None)
        await canary.evaluate_async(_context(case, reply))
    print(f'    canary checks {monitor.checks}, rejected {monitor.rejected}: judge {monitor.health()} '
          f'({len(TOKENS) - calls_before} new judge calls)')  # fmt: skip

    result: dict[str, Any] = {
        'judge': judge_cert.judge,
        'certificates': {'plain': plain.verdict, 'reference': reference.verdict},
        'selection': {i: d.decision for i, _, _, d in decisions},
        'selected': chosen_index,
        'confirmation': confirmed.summary(),
        'referee': referee.summary(),
        'promoted': promoted,
        'canary': {
            'checks': monitor.checks,
            'rejected': monitor.rejected,
            'health': monitor.health(),
            'history': [{k: v for k, v in h.items() if k != 'inputs'} for h in monitor.history],
        },
    }
    (ROOT / 'results' / 'self_improving_loop.json').write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    asyncio.run(main())
