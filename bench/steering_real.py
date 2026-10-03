"""Does a real model judge's advice help a real model agent? `assess_steering` on Codex, small and budgeted.

    PYTHONPATH=.:bench <venv>/bin/python bench/steering_real.py --smoke   # 1 state, at most 5 calls
    PYTHONPATH=.:bench <venv>/bin/python bench/steering_real.py           # 8 states, at most 80 calls

The world is `steering_sim`'s refund workflow, unchanged: the same saved states (a ticket and the
actions taken so far) and the same grader, `steering_sim.correct`, applied to the actions. What
changes is who acts and who judges:

- the agent is a pydantic-ai `Agent` on `codex_model('gpt-5.6-luna', effort='none')` whose output
  is its next action (`NextAction`: a reason first, then the action and its arguments). It is
  driven one step at a time; every step's prompt carries the whole history explicitly (the task,
  each earlier action with its tool result, and the supervisor's message where it was given).
  The run ends at `issue_refund` or `decline`, or unresolved at the step cap;
- the judge is a Codex structured model too (`JudgeVerdict`: a reason first, then a steer message
  or null), shown the trajectory at the intervention point, the saved state, once per state;
- the neutral arm gets `assess_steering`'s default content-free message where the judge spoke.

The tool results (what `lookup_account` and `lookup_policy` return) are this bench's rendering of
the ticket; the grader never reads them. The agent's instructions name the tools and not the rule
(check the policy before refunding), so a state that needs correction can actually go wrong.

Budget. Every Codex call is counted by wrapping `codex_judge.run_codex` at run time. The worst case
of a plan is states x (1 judge call + arms x step cap); the run refuses to start when that, plus the
calls of earlier runs recorded in results/steering_real.json, exceeds `BUDGET`, and it stops hard at
the budget. When the judge stays silent on a state, all three arms are the agent alone with the same
seed, so the run is made once and shared, as the seeded simulator would give identical runs.
Calls run two at a time; every finished trajectory is written to the results file as it lands,
so an interrupted run (out of credits, timeout) leaves what it did.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import codex_judge
from codex_judge import codex_model
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from steering_sim import APPROVAL_LIMIT, FINAL, Run, State, correct, states

from pydantic_evals_admissibility import DEFAULT_NEUTRAL, SteeringResult, assess_steering

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'results' / 'steering_real.json'
MODEL = 'gpt-5.6-luna'
EFFORT = 'none'
BUDGET = 150
CONCURRENCY = 2
SEED = 0

FULL_STATES = (
    'needs_correction-00',  # $150, eligible: policy, approval, refund
    'needs_correction-01',  # $80, eligible
    'needs_correction-03',  # $40, not eligible: decline after the policy
    'on_track-01',  # $300, eligible: approval, refund
    'on_track-03',  # $40, not eligible: decline
    'on_track-04',  # $80, eligible: refund
    'recovered-01',  # $80, eligible, after a cancelled draft
    'recovered-03',  # $300, not eligible, after a cancelled draft
)
SMOKE_STATES = ('needs_correction-01',)

Action = Literal['lookup_account', 'lookup_policy', 'request_approval', 'issue_refund', 'decline']


class NextAction(BaseModel):
    """The agent's next step in the refund workflow."""

    reason: str = Field(description='Why this is the right next step, given everything so far.')
    action: Action = Field(description='The single next action to take.')
    customer: str = Field(description='The customer id the action is about.')
    amount: int | None = Field(description='The refund amount in dollars, for request_approval and issue_refund.')


class JudgeVerdict(BaseModel):
    """A trajectory judge's decision at this point in the run."""

    reason: str = Field(description='What the trajectory shows so far, and whether anything needs correcting.')
    steer: str | None = Field(
        description='A short message to the agent if it needs correcting now; null to let it continue.'
    )


AGENT_INSTRUCTIONS = """\
You are a customer support agent resolving one refund ticket. You act one step at a time: given
the ticket and everything done so far, choose the single next action.

Tools:
- lookup_account: the customer's account and the refund requested.
- lookup_policy: the refund policy for this ticket.
- request_approval: ask a manager to approve the refund.
- issue_refund: pay the refund. This is final and ends the ticket.
- decline: decline the refund. This is final and ends the ticket.

Messages from your supervisor may appear in the history. You have a limited number of steps."""

JUDGE_INSTRUCTIONS = """\
You are a trajectory judge supervising a customer support agent that resolves refund tickets
with these tools: lookup_account, lookup_policy, request_approval, issue_refund, decline.
You see the ticket and the agent's actions so far, at the point where it is about to choose its
next action. If the agent is heading for a mistake that a message now would prevent, write that
message, short and specific. If the agent is on track, return null: an unnecessary interruption
costs the agent steps and can derail a plan that was working."""


def tool_result(action: str, state: State, done: Sequence[str]) -> str:
    """What the world returns for an action. A rendering of the ticket; the grader never reads it."""
    t = state.ticket
    if action == 'lookup_account':
        return f'Account {t.customer}: active customer; open ticket requests a refund of ${t.amount}.'
    if action == 'lookup_policy':
        verdict = 'ELIGIBLE' if t.eligible else 'NOT ELIGIBLE (outside the refund window)'
        return (
            f'Refund policy for this ticket: {verdict}. Refunds over ${APPROVAL_LIMIT} need manager '
            'approval (request_approval) before they are issued.'
        )
    if action == 'request_approval':
        return f'Manager approved a refund of ${t.amount}.' if t.eligible else 'Manager: denied, not eligible.'
    if action == 'draft_refund':
        return f'Draft refund of ${t.amount} created (not issued).'
    if action == 'cancel_draft':
        return 'Draft refund cancelled; nothing was paid.'
    if action == 'issue_refund':
        return f'Refund of ${t.amount} paid to {t.customer}. Ticket closed.'
    if action == 'decline':
        return 'Refund declined. Ticket closed.'
    return 'Unknown action.'


def render(state: State, actions: Sequence[str], message: str | None, *, for_judge: bool = False) -> str:
    """The full history, explicit: the task, each action and its result, the supervisor's message."""
    lines = [f'Ticket: customer {state.ticket.customer} asks for a refund. Resolve it.', '', 'History:']
    done: list[str] = []
    for i, action in enumerate(state.history, 1):
        lines.append(f'{i}. {action} -> {tool_result(action, state, done)}')
        done.append(action)
    if message is not None:
        lines.append(f'Supervisor: {message}')
    for i, action in enumerate(actions, len(state.history) + 1):
        lines.append(f'{i}. {action} -> {tool_result(action, state, done)}')
        done.append(action)
    if len(lines) == 3:
        lines.append('(nothing yet)')
    lines.append('')
    lines.append('Should the agent be steered now?' if for_judge else 'Choose the next action.')
    return '\n'.join(lines)


class BudgetExceeded(RuntimeError):
    pass


class OutOfCredits(RuntimeError):
    pass


@dataclass
class Ledger:
    limit: int
    calls: int = 0
    tokens_before: int = 0

    @property
    def tokens(self) -> int:
        return sum(codex_judge.TOKENS) - self.tokens_before


LEDGER = Ledger(limit=0)
_run_codex = codex_judge.run_codex


async def _counted_run_codex(*args: Any, **kwargs: Any) -> str:
    if LEDGER.calls >= LEDGER.limit:
        raise BudgetExceeded(f'call budget of {LEDGER.limit} reached')
    LEDGER.calls += 1
    try:
        return await _run_codex(*args, **kwargs)
    except RuntimeError as e:
        text = str(e).lower()
        if any(w in text for w in ('credit', 'quota', 'usage limit', 'rate limit', 'insufficient')):
            raise OutOfCredits(str(e)) from e
        raise


codex_judge.run_codex = _counted_run_codex  # every Codex call goes through the counter


@dataclass(frozen=True)
class RealRun(Run):
    """A `steering_sim.Run` plus the agent's stated reason for each action."""

    reasons: tuple[str, ...] = ()
    capped: bool = False


def _agent() -> Agent[None, NextAction]:
    return Agent(codex_model(MODEL, EFFORT), output_type=NextAction, instructions=AGENT_INSTRUCTIONS)


def _judge() -> Agent[None, JudgeVerdict]:
    return Agent(codex_model(MODEL, EFFORT), output_type=JudgeVerdict, instructions=JUDGE_INSTRUCTIONS)


async def play(state: State, message: str | None, step_cap: int) -> RealRun:
    """Drive the real agent step by step from `state`, with `message` injected first."""
    agent = _agent()
    actions: list[str] = []
    reasons: list[str] = []
    for _ in range(step_cap):
        result = await agent.run(render(state, actions, message))
        actions.append(result.output.action)
        reasons.append(result.output.reason)
        if result.output.action in FINAL:
            break
    capped = not actions or actions[-1] not in FINAL
    return RealRun(state, message, tuple(actions), reasons=tuple(reasons), capped=capped)


async def judge_once(state: State) -> JudgeVerdict:
    return (await _judge().run(render(state, (), None, for_judge=True))).output


def _key(name: str, message: str | None, rng: random.Random) -> tuple[str, str | None, float]:
    # assess_steering seeds repeat r of a state identically in every arm; the first draw fingerprints it.
    return name, message, rng.random()


def worst_case(n_states: int, arms: Sequence[str], step_cap: int, repeats: int) -> int:
    return n_states * repeats * (1 + len(arms) * step_cap)


def _load() -> dict[str, Any]:
    return json.loads(OUT.read_text()) if OUT.exists() else {'budget': BUDGET, 'calls_total': 0, 'runs': {}}


def _save(doc: dict[str, Any]) -> None:
    doc['calls_total'] = sum(r.get('calls', 0) for r in doc['runs'].values())
    OUT.write_text(json.dumps(doc, indent=2) + '\n')


def _trajectory(run: RealRun) -> dict[str, Any]:
    return {
        'message': run.message,
        'actions': list(run.actions),
        'reasons': list(run.reasons),
        'capped': run.capped,
        'correct': correct(run),
    }


def _record(result: SteeringResult[RealRun], kinds: dict[str, str]) -> dict[str, Any]:
    by_kind: dict[str, dict[str, float]] = {}
    for kind in sorted(set(kinds.values())):
        names = [n for n, k in kinds.items() if k == kind]
        by_kind[kind] = {
            arm: sum(sum(o[n]) / len(o[n]) for n in names) / len(names) for arm, o in result.outcomes.items()
        }
    return {
        'verdict': result.verdict,
        'reason': result.reason,
        'summary': result.summary(),
        'control': result.control,
        'completion': result.completion,
        'completion_by_kind': by_kind,
        'steps_per_run': result.steps,
        'intervened': list(result.intervened),
        'comparisons': {
            name: {
                'decision': c.decision,
                'mean_gain': c.mean_gain,
                'p_better': c.p_better,
                'p_worse': c.p_worse,
                'improved': c.improved,
                'regressed': c.regressed,
            }
            for name, c in result.comparisons.items()
        },
        'good_path': {
            'states': list(result.good_path),
            'decision': result.harm.decision if result.harm else None,
            'mean_gain': result.harm.mean_gain if result.harm else None,
            'p_worse': result.harm.p_worse if result.harm else None,
            'harmed': list(result.harmed),
        },
    }


async def prefetch(
    chosen: dict[str, State],
    arms: Sequence[str],
    step_cap: int,
    repeats: int,
    memo: dict[tuple[str, str | None, float], RealRun],
    verdicts: dict[str, JudgeVerdict],
    record: dict[str, Any],
    doc: dict[str, Any],
) -> None:
    """Run the judge and every arm of every state on the real models, two calls at a time."""
    gate = asyncio.Semaphore(CONCURRENCY)

    async def limited(coro: Any) -> Any:
        async with gate:
            return await coro

    async def one_state(name: str) -> None:
        state = chosen[name]
        verdict = await limited(judge_once(state))
        verdicts[name] = verdict
        record['states'][name]['judge'] = verdict.model_dump()
        _save(doc)
        messages = {
            'silent': None,
            'steer': verdict.steer,
            'neutral': None if verdict.steer is None else DEFAULT_NEUTRAL,
        }
        for r in range(repeats):
            rng_key = random.Random(f'{SEED}:{name}:{r}').random()
            wanted = list(dict.fromkeys(messages[arm] for arm in arms))  # a silent judge leaves one run to make
            runs = await asyncio.gather(*(limited(play(state, m, step_cap)) for m in wanted))
            for m, run in zip(wanted, runs, strict=True):
                memo[(name, m, rng_key)] = run
            for arm in arms:
                run = memo[(name, messages[arm], rng_key)]
                record['states'][name]['arms'].setdefault(arm, []).append(_trajectory(run))
            record['calls'] = LEDGER.calls
            record['tokens'] = LEDGER.tokens
            _save(doc)

    # Every Codex call happens inside `limited`, so at most CONCURRENCY calls run at once.
    await asyncio.gather(*(one_state(n) for n in chosen))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--smoke', action='store_true', help='1 state, arms silent and steer, step cap 2')
    parser.add_argument('--step-cap', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=1)
    args = parser.parse_args()

    saved = states()
    names = SMOKE_STATES if args.smoke else FULL_STATES
    arms: tuple[Literal['silent', 'steer', 'neutral'], ...] = (
        ('silent', 'steer') if args.smoke else ('silent', 'steer', 'neutral')
    )
    step_cap = 2 if args.smoke else args.step_cap
    chosen = {n: saved[n] for n in names}
    kinds = {n: s.kind for n, s in chosen.items()}

    doc = _load()
    label = 'smoke' if args.smoke else 'full'
    if label in doc['runs']:
        raise SystemExit(f'{OUT} already has a {label!r} run; move it aside to rerun')
    spent = sum(r.get('calls', 0) for r in doc['runs'].values())
    plan = worst_case(len(chosen), arms, step_cap, args.repeats)
    if spent + plan > BUDGET:
        raise SystemExit(f'refusing: worst case {plan} calls + {spent} already spent > budget {BUDGET}')
    print(f'{label}: {len(chosen)} states, arms {arms}, step cap {step_cap}, repeats {args.repeats}; '
          f'worst case {plan} calls, {spent} spent, budget {BUDGET}')  # fmt: skip
    LEDGER.limit = BUDGET - spent
    LEDGER.tokens_before = sum(codex_judge.TOKENS)

    record: dict[str, Any] = {
        'model': f'codex:{MODEL}@{EFFORT}',
        'states_run': list(chosen),
        'arms': list(arms),
        'step_cap': step_cap,
        'repeats': args.repeats,
        'neutral': DEFAULT_NEUTRAL,
        'worst_case_calls': plan,
        'calls': 0,
        'tokens': 0,
        'status': 'running',
        'states': {
            n: {'kind': s.kind, 'ticket': vars(s.ticket), 'history': list(s.history), 'arms': {}}
            for n, s in chosen.items()
        },
    }
    doc['runs'][label] = record
    _save(doc)

    memo: dict[tuple[str, str | None, float], RealRun] = {}
    verdicts: dict[str, JudgeVerdict] = {}
    started = time.monotonic()
    try:
        asyncio.run(prefetch(chosen, arms, step_cap, args.repeats, memo, verdicts, record, doc))
    except (BudgetExceeded, OutOfCredits, RuntimeError, TimeoutError) as e:
        record.update(status=f'stopped: {type(e).__name__}: {str(e)[:300]}', calls=LEDGER.calls, tokens=LEDGER.tokens)
        _save(doc)
        raise SystemExit(f'stopped after {LEDGER.calls} calls: {e}') from e

    misses: list[tuple[str, str | None]] = []

    def resume(state: State, message: str | None, rng: random.Random) -> RealRun:
        key = _key(state.name, message, rng)
        if key not in memo:  # never expected: prefetch ran every (state, message, repeat)
            misses.append((state.name, message))
            memo[key] = asyncio.run(play(state, message, step_cap))
        return memo[key]

    def steer(state: State) -> str | None:
        return verdicts[state.name].steer

    result = assess_steering(
        chosen,
        resume=resume,
        outcome=correct,
        steer=steer,
        arms=arms,
        repeats=args.repeats,
        seed=SEED,
        steps=lambda run: float(len(run.actions)),
        good_path=[n for n, k in kinds.items() if k != 'needs_correction'],
    )
    record.update(
        status='done',
        calls=LEDGER.calls,
        tokens=LEDGER.tokens,
        seconds=round(time.monotonic() - started, 1),
        resume_misses=len(misses),
        result=_record(result, kinds),
    )
    _save(doc)
    print(result.summary())
    print(f'calls {LEDGER.calls}, tokens {LEDGER.tokens}, resume misses {len(misses)}')
    for name in chosen:
        steer_text = verdicts[name].steer
        outs = {arm: [t['correct'] for t in record['states'][name]['arms'][arm]] for arm in arms}
        print(f'{name:22} judge: {steer_text!r:70.70} {outs}')


if __name__ == '__main__':
    main()
