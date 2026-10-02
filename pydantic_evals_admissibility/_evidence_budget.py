"""The smallest part of a trace a judge needs: drop components while no verdict that matters moves.

A judge shown the whole trace pays for every span id, every lookup, every argument, on every
call. Most of it does not change the verdict. `evidence_budget` finds which named components of
the inputs (a message, a tool call's result, its arguments) can be left out while the judge's
verdicts stay the same on a whole set of judgments: the known-good answers and the controls the
certificate is built from. Controls are what make this safe: dropping a tool result does not move
the verdict on a known-good refund (the judge passes the confident reply anyway), but it does on
the control where that tool failed. A budget fitted on known-good cases alone would drop the
evidence the judge is supposed to read.

`_witness.minimize_witness` shrinks one example while it still fools the judge; this shrinks the
shape of every example while the judge's behaviour on the set is unchanged. It is greedy, one
component at a time (largest first), not ddmin: each test costs a judge call per judgment, the
components are few, and one at a time says which component mattered.

It preserves verdicts, right or wrong. A judge that ignores the evidence gets a budget of nothing
and is no better for it: certify the projected judge before trusting its verdicts. And a model
judge that disagrees with itself makes a harmless drop look harmful, so noise keeps components;
it never drops one wrongly by chance unless the judge flips back by chance on every judgment.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from pydantic_core import to_json

from ._cases import HumanLabel, JudgeCase
from ._certify import Judgment, _judge, _plan
from ._controls import Control
from ._stats import wilson

Parts = Callable[[Any], Sequence[str]]
Project = Callable[[Any, frozenset[str]], Any]


def size(value: Any) -> int:
    """Characters the judge is shown for `value`, as `LLMJudge` serializes it (JSON unless a string)."""
    if isinstance(value, str):
        return len(value)
    try:
        return len(to_json(value).decode())
    except Exception:
        return len(repr(value))


@dataclass
class DropTrial:
    """One attempt to drop a component: kept when any judgment's verdict moved."""

    component: str
    dropped: bool
    calls: int
    changed: list[str] = field(default_factory=list)
    """The judgments (`case/role`) whose verdict moved; empty when the component was dropped."""
    chars: int = 0
    """Characters this component takes over all judgments' inputs."""


@dataclass
class EvidenceBudget:
    """Which components the judge needs, what leaving out the rest saves, and what it cost to find."""

    components: tuple[str, ...]
    kept: tuple[str, ...]
    judgments: int
    chars_full: int
    chars_kept: int
    calls: int
    complete: bool
    """False when `max_calls` ran out before every component was tried."""
    trials: list[DropTrial]
    baseline: list[Judgment] = field(repr=False)
    """The verdicts on the full inputs, which every projection had to reproduce."""

    @property
    def dropped(self) -> tuple[str, ...]:
        return tuple(c for c in self.components if c not in self.kept)

    @property
    def saved(self) -> float:
        """Share of input characters left out (about the share of input tokens)."""
        return 1 - self.chars_kept / self.chars_full if self.chars_full else 0.0

    def summary(self) -> str:
        return (
            f'kept {len(self.kept)} of {len(self.components)} components ({", ".join(self.kept) or "none"}); '
            f'inputs {self.chars_full:,} -> {self.chars_kept:,} chars ({self.saved:.0%} saved, '
            f'~{self.chars_full // 4:,} -> ~{self.chars_kept // 4:,} tokens) over {self.judgments} judgments; '
            f'{self.calls} judge calls'
            f'{"" if self.complete else "; budget ran out before every component was tried"}'
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            'components': list(self.components),
            'kept': list(self.kept),
            'dropped': list(self.dropped),
            'judgments': self.judgments,
            'chars_full': self.chars_full,
            'chars_kept': self.chars_kept,
            'saved': self.saved,
            'calls': self.calls,
            'complete': self.complete,
            'trials': [vars(t) for t in self.trials],
            'baseline': [{'case': j.case, 'role': j.role, 'passed': j.passed, 'error': j.error} for j in self.baseline],
        }


def _items(
    cases: Sequence[JudgeCase], controls: Sequence[Control], human_labels: Sequence[HumanLabel], seed: int
) -> list[tuple[JudgeCase, Any, str]]:
    """The judgments a certificate rests on, each once: known-good answers, controls, labels."""
    return _plan(cases, controls, human_labels, 1, random.Random(seed))


async def evidence_budget(
    judge: Any,
    cases: Sequence[JudgeCase],
    controls: Sequence[Control] = (),
    *,
    parts: Parts,
    project: Project,
    human_labels: Sequence[HumanLabel] = (),
    order: Sequence[str] | None = None,
    max_calls: int = 500,
    max_concurrency: int = 4,
    assertion: str | None = None,
    seed: int = 0,
) -> EvidenceBudget:
    """Drop components of every judgment's inputs while the judge's verdicts all stay the same.

    Args:
        judge: The evaluator, as it will be deployed (shown the inputs).
        cases: Known-good answers.
        controls: Built from `cases` as `certify_judge` builds them. Pass the evidence controls:
            they are what stop the budget from dropping the evidence the verdict should rest on.
        parts: The named components of one judgment's inputs; their union is what is tried.
        project: `inputs` with only the components named in `kept`, for any subset.
        human_labels: Labelled outputs, also held to their full-input verdict.
        order: Components to try, in order; by default largest (in characters) first.
        max_calls: Judge calls to spend, the full-input verdicts included.
        max_concurrency: Judge calls in flight; each component's trial stops at its first moved verdict.
    """
    items = _items(cases, controls, human_labels, seed)
    if not items:
        raise ValueError('no judgments: pass cases (and controls)')
    names = list(dict.fromkeys(name for case, _, _ in items for name in parts(case.inputs)))
    everything = frozenset(names)
    if max_calls < len(items):
        raise ValueError(f'max_calls={max_calls} cannot cover the {len(items)} full-input verdicts')
    limit = asyncio.Semaphore(max_concurrency)
    calls = 0

    async def verdicts(indices: Sequence[int], kept: frozenset[str]) -> list[Judgment]:
        nonlocal calls
        calls += len(indices)
        todo = []
        for i in indices:
            case, output, role = items[i]
            inputs = case.inputs if kept == everything else project(case.inputs, kept)
            todo.append(_judge(judge, replace(case, inputs=inputs), output, role, assertion, limit))
        return list(await asyncio.gather(*todo))

    baseline = await verdicts(range(len(items)), everything)
    chars = {
        name: sum(size(c.inputs) - size(project(c.inputs, everything - {name})) for c, _, _ in items) for name in names
    }
    tried = list(order) if order is not None else sorted(names, key=lambda n: (-chars[n], n))
    unknown = set(tried) - everything
    if unknown:
        raise ValueError(f'order names components that parts() never returns: {sorted(unknown)}')

    kept, trials, complete = everything, [], True
    # Judgments whose verdict moved before are asked first: a component the judge needs usually
    # shows on the same few controls, so a doomed trial stops after a call or two.
    # A judgment the judge errored on in full has no verdict to keep; it cannot veto a drop.
    priority: list[int] = [i for i in range(len(items)) if baseline[i].passed is not None]
    if not priority:
        raise ValueError('the judge gave no verdict on any judgment with the full inputs')
    for name in tried:
        candidate = kept - {name}
        changed: list[int] = []
        spent = calls
        for start in range(0, len(priority), max_concurrency):
            batch = priority[start : start + max_concurrency]
            if calls + len(batch) > max_calls:
                complete = False
                changed = changed or [-1]
                break
            for i, j in zip(batch, await verdicts(batch, candidate), strict=True):
                if j.passed is None or j.passed != baseline[i].passed:
                    changed.append(i)
            if changed:
                break
        moved = [i for i in changed if i >= 0]
        trials.append(
            DropTrial(
                name, not changed, calls - spent, [f'{items[i][0].name}/{items[i][2]}' for i in moved], chars[name]
            )
        )
        if not changed:
            kept = candidate
        priority = moved + [i for i in priority if i not in moved]
        if not complete:
            break
    chars_full = sum(size(c.inputs) for c, _, _ in items)
    chars_kept = sum(size(project(c.inputs, kept)) for c, _, _ in items)
    return EvidenceBudget(
        tuple(tried), tuple(n for n in tried if n in kept), len(items), chars_full, chars_kept, calls, complete,
        trials, baseline,
    )  # fmt: skip


@dataclass
class BudgetCheck:
    """Agreement of the judge's verdicts on projected and full inputs, on judgments the budget never saw."""

    agree: int
    judgments: int
    disagreements: list[str]
    chars_full: int
    chars_kept: int
    calls: int

    @property
    def interval(self) -> tuple[float, float]:
        return wilson(self.agree, self.judgments)

    def summary(self) -> str:
        low, high = self.interval
        return (
            f'{self.agree}/{self.judgments} verdicts unchanged [{low:.2f}, {high:.2f}] on held-out judgments; '
            f'inputs {self.chars_full:,} -> {self.chars_kept:,} chars; {self.calls} judge calls'
        )

    def to_dict(self) -> dict[str, Any]:
        return {**vars(self), 'interval': self.interval}


async def check_budget(
    judge: Any,
    cases: Sequence[JudgeCase],
    controls: Sequence[Control] = (),
    *,
    kept: Sequence[str],
    project: Project,
    human_labels: Sequence[HumanLabel] = (),
    max_concurrency: int = 4,
    assertion: str | None = None,
    seed: int = 0,
) -> BudgetCheck:
    """Judge new cases on full and on projected inputs and count the verdicts that stay the same.

    A budget is fitted to make every verdict agree on the cases it was found on, so its agreement
    there is 100% by construction. This is the number to report: two calls per judgment.
    """
    items = _items(cases, controls, human_labels, seed)
    keep = frozenset(kept)
    limit = asyncio.Semaphore(max_concurrency)
    full = await asyncio.gather(*(_judge(judge, c, o, r, assertion, limit) for c, o, r in items))
    projected = await asyncio.gather(
        *(_judge(judge, replace(c, inputs=project(c.inputs, keep)), o, r, assertion, limit) for c, o, r in items)
    )
    moved = [
        f'{c.name}/{r}'
        for (c, _, r), a, b in zip(items, full, projected, strict=True)
        if a.passed is None or a.passed != b.passed
    ]
    return BudgetCheck(
        len(items) - len(moved),
        len(items),
        moved,
        sum(size(c.inputs) for c, _, _ in items),
        sum(size(project(c.inputs, keep)) for c, _, _ in items),
        2 * len(items),
    )
