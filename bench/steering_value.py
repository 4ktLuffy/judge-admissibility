"""Does each scripted trajectory judge's advice help the agent? Offline, no model is called.

    PYTHONPATH=.:bench <venv>/bin/python bench/steering_value.py

Every judge in `steering_sim.JUDGES` steers the scripted agent from the same 24 saved states, under
three arms (silent, the judge's message, a neutral message where the judge spoke), 3 repeats each.
One more row runs the helpful judge on `attentive_agent`, which improves after any message: there
steer beats silent, and only the neutral arm shows the advice adds nothing. Hypotheses, written
before running: helpful HELPS; over-interventionist hurts the states already on a good path; stale
INCONCLUSIVE or HURTS. Writes results/steering_value.json.

`--model NAME` is the seam for a real agent and judge; it is documented here and not executed
(no model credits). To plug them in:

- a state becomes a saved pydantic-ai message history: the conversation up to the moment to
  resume, with the workflow's tools (`lookup_account`, `lookup_policy`, `request_approval`,
  `issue_refund`, `decline`) backed by an in-memory store like `steering_sim`'s, reset per run;
- `resume(state, message, rng)` runs the agent with `message_history=state` and `message` as the
  next user prompt (None: a plain continuation), then returns the messages and the store's
  side effects; `rng` is unused, so arms vary by the model's own sampling;
- `steer(state)` asks the LLM trajectory judge to review the same history and returns its Steer
  message text, or None when it lets the agent continue;
- `outcome` is `steering_sim.correct` applied to the tool calls, never a judge's opinion of them.

The Pydantic AI Harness `TrajectoryJudge` API was not checked offline; adapt the call to it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from steering_sim import JUDGES, agent, attentive_agent, correct, helpful_judge, states, steps

from pydantic_evals_admissibility import SteeringResult, assess_steering

ROOT = Path(__file__).resolve().parent.parent
REPEATS = 3


def _by_kind(result: SteeringResult[Any], kinds: dict[str, str]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for kind in sorted(set(kinds.values())):
        names = [n for n, k in kinds.items() if k == kind]
        out[kind] = {arm: sum(sum(o[n]) / len(o[n]) for n in names) / len(names) for arm, o in result.outcomes.items()}
    return out


def _record(result: SteeringResult[Any], kinds: dict[str, str]) -> dict[str, Any]:
    return {
        'verdict': result.verdict,
        'reason': result.reason,
        'control': result.control,
        'completion': result.completion,
        'completion_by_kind': _by_kind(result, kinds),
        'steps_per_run': result.steps,
        'intervened': len(result.intervened),
        'comparisons': {
            name: {
                'decision': c.decision,
                'mean_gain': c.mean_gain,
                'interval': c.interval,
                'p_better': c.p_better,
                'p_worse': c.p_worse,
                'improved': c.improved,
                'regressed': c.regressed,
            }
            for name, c in result.comparisons.items()
        },
        'good_path': {
            'states': len(result.good_path),
            'kinds': {k: sum(kinds[n] == k for n in result.good_path) for k in sorted(set(kinds.values()))},
            'decision': result.harm.decision if result.harm else None,
            'mean_gain': result.harm.mean_gain if result.harm else None,
            'p_worse': result.harm.p_worse if result.harm else None,
            'harmed': len(result.harmed),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', help='documented seam for a real agent and judge; not executed (see the docstring)')
    args = parser.parse_args()
    if args.model:
        raise SystemExit(
            f'--model {args.model}: not run here (no model credits); see the module docstring for the seam'
        )

    saved = states()
    kinds = {n: s.kind for n, s in saved.items()}
    runs = {name: (agent(), judge) for name, judge in JUDGES.items()}
    runs['helpful, agent that heeds any message'] = (attentive_agent(), helpful_judge)

    out: dict[str, Any] = {
        'states': len(saved),
        'repeats': REPEATS,
        'kinds': {k: list(kinds.values()).count(k) for k in set(kinds.values())},
    }
    header = (
        f'{"judge":40} {"silent":>7} {"steer":>7} {"neutral":>7}  '
        f'{"vs silent":>10} {"vs neutral":>11}  {"good path":>16}  verdict'
    )
    print(header)
    print('-' * len(header))
    for name, (resume, judge) in runs.items():
        result = assess_steering(saved, resume=resume, outcome=correct, steer=judge, repeats=REPEATS, steps=steps)
        out[name] = _record(result, kinds)
        c = result.completion
        vs_silent, vs_neutral = result.comparisons['steer vs silent'], result.comparisons['steer vs neutral']
        harm = f'{result.harm.decision} {result.harm.mean_gain:+.2f}' if result.harm else 'n/a'
        print(
            f'{name:40} {c["silent"]:7.2f} {c["steer"]:7.2f} {c["neutral"]:7.2f}  '
            f'{vs_silent.mean_gain:+10.3f} {vs_neutral.mean_gain:+11.3f}  {harm:>16}  {result.verdict}'
        )
    print()
    for name in runs:
        print(
            f'{name}: by kind (silent/steer/neutral)',
            {k: ' / '.join(f'{v:.2f}' for v in arms.values()) for k, arms in out[name]['completion_by_kind'].items()},
        )
    (ROOT / 'results' / 'steering_value.json').write_text(json.dumps(out, indent=2) + '\n')


if __name__ == '__main__':
    main()
