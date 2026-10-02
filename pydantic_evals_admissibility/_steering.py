"""Does a runtime judge's advice help the agent it steers?

A trajectory judge reads an agent's recent history and, when it sees a problem, injects a message.
Being right about the problem is not enough: advice can interrupt a plan that was going to work,
repeat an issue the agent already fixed, or cost steps. `assess_steering` measures the effect
instead of assuming it. Each saved agent state (a prefix of a trajectory) is resumed under three
arms:

- **silent**: no message, what the agent does on its own;
- **steer**: the judge's message, when the judge chooses to speak;
- **neutral**: a content-free message ("Continue with the task.") sent exactly where the judge
  spoke. An agent can do better after any message, from the extra turn or attention alone; only
  steer against neutral says the advice itself helped.

The state is the unit: its repeats share its difficulty, so they are averaged within it, and the
arms are compared state by state with the paired sign-flip test of `decide`. Repeat `r` of a state
gets the same random seed in every arm, so in a simulation the arms differ only in the message.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Generic, Literal, TypeVar

from ._gate import GateResult, GateRules, decide

Arm = Literal['silent', 'steer', 'neutral']
SteeringVerdict = Literal['HELPS', 'HURTS', 'INCONCLUSIVE']
State = TypeVar('State')
Trajectory = TypeVar('Trajectory')

DEFAULT_NEUTRAL = 'Continue with the task.'
_ARMS: tuple[Arm, ...] = ('silent', 'steer', 'neutral')
_VERDICT: dict[str, SteeringVerdict] = {'PROMOTE': 'HELPS', 'REJECT': 'HURTS', 'INCONCLUSIVE': 'INCONCLUSIVE'}


@dataclass(frozen=True)
class SteeringResult(Generic[Trajectory]):
    """What steering did, state by state. `verdict` comes from one test: steer against its control."""

    verdict: SteeringVerdict
    reason: str
    control: Arm
    """The arm the verdict compares steer with: neutral when it ran, since it controls for the message itself."""
    states: int
    intervened: tuple[str, ...]
    """States where the judge spoke on at least one repeat; elsewhere every arm is the agent alone."""
    completion: dict[str, float]
    """Arm to its completion rate, the mean over states of each state's rate."""
    comparisons: dict[str, GateResult]
    """'steer vs silent', 'steer vs neutral', 'neutral vs silent': whichever the arms allow."""
    good_path: tuple[str, ...]
    """States already on a good path, where advice has nothing to fix and can only cost."""
    harm: GateResult | None
    """Steer against the control on `good_path` states alone; None when it cannot be assessed fairly."""
    harmed: tuple[str, ...]
    """Good-path states whose completion steer lowered against the control."""
    steps: dict[str, float] | None = None
    """Arm to mean steps (or calls, or cost) per run, when `steps` was given."""
    outcomes: dict[str, dict[str, list[bool]]] = field(default_factory=dict, repr=False)
    """Arm to state to the outcome of each repeat."""
    trajectories: dict[str, dict[str, list[Trajectory]]] = field(default_factory=dict, repr=False)

    def summary(self) -> str:
        rates = ', '.join(f'{arm} {rate:.2f}' for arm, rate in self.completion.items())
        main = self.comparisons[f'steer vs {self.control}']
        lines = [
            f'{self.verdict}: steer vs {self.control} {main.mean_gain:+.3f} per state '
            f'(p(better)={main.p_better:.3f}, p(worse)={main.p_worse:.3f}). {self.reason}',
            f'completion: {rates}; the judge spoke on {len(self.intervened)} of {self.states} states',
        ]
        for name, result in self.comparisons.items():
            if name != f'steer vs {self.control}':
                lines.append(f'{name}: {result.decision} {result.mean_gain:+.3f}')
        if self.harm is not None:
            lines.append(
                f'good path ({len(self.good_path)} states): {self.harm.decision} {self.harm.mean_gain:+.3f}, '
                f'{len(self.harmed)} made worse'
            )
        if self.steps is not None:
            lines.append('steps per run: ' + ', '.join(f'{arm} {n:.2f}' for arm, n in self.steps.items()))
        return '\n'.join(lines)


def assess_steering(
    states: Mapping[str, State],
    *,
    resume: Callable[[State, str | None, random.Random], Trajectory],
    outcome: Callable[[Trajectory], bool],
    steer: Callable[[State], str | None],
    arms: Sequence[Arm] = _ARMS,
    neutral: str | Callable[[str], str] = DEFAULT_NEUTRAL,
    repeats: int = 3,
    seed: int = 0,
    steps: Callable[[Trajectory], float] | None = None,
    good_path: Collection[str] | None = None,
    good_path_rate: float = 1.0,
    rules: GateRules | None = None,
) -> SteeringResult[Trajectory]:
    """Resume every state under each arm `repeats` times and decide whether the judge's advice helps.

    Args:
        states: State name to a saved agent state, the prefix of a trajectory to resume from.
        resume: Continue a state to the end, with the message injected first (None: no message).
            The `Random` is the only randomness a simulated agent should use.
        outcome: Whether a finished trajectory completed the task correctly, with no invalid side effects.
        steer: The judge: its message for a state, or None to stay silent. Called once per repeat.
        arms: Which arms to run; 'steer' and at least one control. Without 'neutral' the verdict
            cannot tell good advice from the effect of any message, and the reason says so.
        neutral: The content-free message, or a function from the judge's message to one of similar length.
        repeats: Runs per state per arm; equal in every arm, as the sign-flip test requires.
        seed: Repeat `r` of a state uses the same seed in every arm.
        steps: Steps, calls or cost of a trajectory, to report what steering spends.
        good_path: States known to be on a good path already. Left out, they are the states whose
            silent completion is at least `good_path_rate`; harm is then measured against the
            neutral arm only, because the silent arm chose those states and would flatter itself
            (a state picked for lucky silent runs regresses on any other arm).
        rules: The gate's rules for every comparison.
    """
    arms = tuple(arms)
    if not states:
        raise ValueError('there are no states to resume')
    if repeats < 1:
        raise ValueError('repeats must be at least 1')
    if len(set(arms)) != len(arms) or not set(arms) <= set(_ARMS):
        raise ValueError(f'arms must be distinct, from {_ARMS}; got {arms}')
    if 'steer' not in arms or len(arms) < 2:
        raise ValueError("arms must include 'steer' and at least one control ('silent' or 'neutral')")
    if not 0.0 <= good_path_rate <= 1.0:
        raise ValueError('good_path_rate must be between 0 and 1')
    if good_path is not None and not set(good_path) <= set(states):
        raise ValueError(f'good_path names unknown states: {sorted(set(good_path) - set(states))}')
    neutral_for: Callable[[str], str] = neutral if callable(neutral) else (lambda _: neutral)

    names = sorted(states)
    outcomes: dict[str, dict[str, list[bool]]] = {arm: {n: [] for n in names} for arm in arms}
    runs: dict[str, dict[str, list[Trajectory]]] = {arm: {n: [] for n in names} for arm in arms}
    spoke: set[str] = set()
    for name in names:
        state = states[name]
        for r in range(repeats):
            advice = steer(state)
            if advice is not None:
                spoke.add(name)
            messages: dict[str, str | None] = {
                'silent': None,
                'steer': advice,
                'neutral': None if advice is None else neutral_for(advice),
            }
            for arm in arms:
                trajectory = resume(state, messages[arm], random.Random(f'{seed}:{name}:{r}'))
                result = outcome(trajectory)
                if not isinstance(result, bool):
                    raise TypeError(f'outcome must return a bool, got {type(result).__name__} for {name!r}')
                outcomes[arm][name].append(result)
                runs[arm][name].append(trajectory)

    def rate(arm: str, name: str) -> float:
        return sum(outcomes[arm][name]) / repeats

    completion = {arm: sum(rate(arm, n) for n in names) / len(names) for arm in arms}
    pairs = [('steer', 'silent'), ('steer', 'neutral'), ('neutral', 'silent')]
    comparisons = {
        f'{a} vs {b}': decide(outcomes[b], outcomes[a], rules=rules) for a, b in pairs if a in arms and b in arms
    }
    control: Arm = 'neutral' if 'neutral' in arms else 'silent'
    main = comparisons[f'steer vs {control}']
    verdict = _VERDICT[main.decision]
    if control == 'silent':
        reason = 'without a neutral arm, the gain may come from any message, not from this advice'
    elif verdict == 'HELPS':
        reason = 'the advice beats a content-free message at the same moments'
    elif verdict == 'HURTS':
        reason = 'the agent does worse with the advice than with a content-free message'
    else:
        reason = 'the states cannot tell the advice from a content-free message'

    if good_path is not None:
        chosen, harm_control = sorted(good_path), control
    elif 'silent' in arms and 'neutral' in arms:
        chosen, harm_control = [n for n in names if rate('silent', n) >= good_path_rate], 'neutral'
    else:
        chosen, harm_control = [], control
    harm = harmed = None
    if chosen:
        harm = decide(
            {n: outcomes[harm_control][n] for n in chosen}, {n: outcomes['steer'][n] for n in chosen}, rules=rules
        )
        harmed = tuple(n for n in chosen if rate('steer', n) < rate(harm_control, n))
        if harm.decision == 'REJECT':
            reason += f'; on {len(chosen)} states already on a good path it makes things worse ({len(harmed)} states)'

    steps_by_arm = None
    if steps is not None:
        steps_by_arm = {
            arm: sum(steps(t) for n in names for t in runs[arm][n]) / (len(names) * repeats) for arm in arms
        }
    return SteeringResult(
        verdict=verdict,
        reason=reason,
        control=control,
        states=len(names),
        intervened=tuple(n for n in names if n in spoke),
        completion=completion,
        comparisons=comparisons,
        good_path=tuple(chosen),
        harm=harm,
        harmed=harmed or (),
        steps=steps_by_arm,
        outcomes=outcomes,
        trajectories=runs,
    )
