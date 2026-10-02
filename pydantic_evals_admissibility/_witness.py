"""Shrink an example that fools a judge to the smallest one that still does.

A judge fooled by a long reply, or by a long trace, is hard to fix from the example alone: which
sentence, which tool call, did it? `minimize_witness` removes parts (sentences, words, tool calls,
messages) for as long as the judge still gives the wrong verdict and the example is still a real
failure, by delta debugging (Zeller and Hildebrandt's ddmin). What is left is a witness an
engineer can read in a line, and a regression case to keep.

Two conditions hold at every step: `fools(passed)` (by default, the judge passes it) and
`valid(case, output)` (by default, anything: pass an oracle that says the example is still wrong,
or removing the wrong number would "fix" the answer and prove nothing). A model judge is not
deterministic: with `confirm=n`, a candidate counts only if it fools the judge on all n tries.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

from ._cases import JudgeCase
from ._certify import _judge


def sentences(text: str) -> list[str]:
    """Split prose into sentences and lines, keeping each piece's trailing space."""
    return [p for p in re.split(r'(?<=[.!?:])\s+|\n+', text) if p]


@dataclass
class Witness:
    """The smallest example found that still fools the judge."""

    case: JudgeCase
    output: Any
    original_size: int
    size: int
    calls: int
    minimal: bool
    """True when removing any one remaining part no longer fools the judge (1-minimal)."""

    @property
    def reduction(self) -> float:
        return 1 - self.size / self.original_size if self.original_size else 0.0

    def summary(self) -> str:
        what = 'output' if self.output is not None else 'case'
        return (
            f'{self.original_size} -> {self.size} parts ({self.reduction:.0%} removed) in {self.calls} judge calls; '
            f'{"1-minimal" if self.minimal else "budget ran out before 1-minimal"}. Witness {what}: {self.output!r}'
        )


async def minimize_witness(
    judge: Any,
    case: JudgeCase,
    output: Any,
    *,
    target: Literal['output', 'inputs'] = 'output',
    split: Callable[[Any], Sequence[Any]] | None = None,
    join: Callable[[Sequence[Any]], Any] | None = None,
    fools: Callable[[bool | None], bool] = lambda passed: passed is True,
    valid: Callable[[JudgeCase, Any], bool] = lambda case, output: True,
    confirm: int = 1,
    max_calls: int = 200,
    assertion: str | None = None,
) -> Witness:
    """Remove parts of `output` (or of `case.inputs`) while the judge is still fooled.

    Args:
        judge: The evaluator that was fooled.
        case: The case, with the inputs the judge is shown.
        output: The output that fooled it.
        target: Shrink the output, or the inputs (the trace, the conversation) with the output fixed.
        split: Parts of the target; by default sentences for a string and items for a list.
        join: Rebuilds the target from parts; by default the inverse of the default `split`.
        fools: Whether a verdict is the failure being reproduced; by default, passing.
        valid: Whether a smaller example is still a real failure, for example an oracle saying the
            output is still wrong. Candidates that are not are skipped without calling the judge.
        confirm: Judge calls a candidate must fool, every time, to count.
        max_calls: Judge calls to spend; the result says whether it got to 1-minimal.
    """
    whole = output if target == 'output' else case.inputs
    split, join = split or _default_split(whole), join or _default_join(whole)
    parts = list(split(whole))
    calls = 0
    limit = asyncio.Semaphore(1)

    def build(kept: Sequence[Any]) -> tuple[JudgeCase, Any]:
        value = join(kept)
        return (case, value) if target == 'output' else (replace(case, inputs=value), output)

    async def still_fools(kept: Sequence[Any]) -> bool:
        nonlocal calls
        candidate_case, candidate_output = build(kept)
        if not valid(candidate_case, candidate_output):
            return False
        for _ in range(confirm):
            if calls >= max_calls:
                return False
            calls += 1
            judgment = await _judge(judge, candidate_case, candidate_output, 'witness', assertion, limit)
            if not fools(judgment.passed):
                return False
        return True

    if not await still_fools(parts):
        raise ValueError('the example does not fool the judge as given: nothing to minimize')

    # ddmin: try removing chunks, then halve the chunk size, until no single part can go.
    n = 2
    while len(parts) >= 2 and calls < max_calls:
        size = max(1, len(parts) // n)
        chunks = [parts[i : i + size] for i in range(0, len(parts), size)]
        removed = False
        for i in range(len(chunks)):
            rest = [p for j, chunk in enumerate(chunks) if j != i for p in chunk]
            if rest and await still_fools(rest):
                parts, n, removed = rest, max(n - 1, 2), True
                break
        if not removed:
            if n >= len(parts):
                break
            n = min(len(parts), n * 2)
    # ddmin stops at the finest granularity having failed to drop any single part: 1-minimal,
    # unless it stopped because the budget ran out.
    final_case, final_output = build(parts)
    return Witness(final_case, final_output, len(list(split(whole))), len(parts), calls, calls < max_calls)


def _default_split(value: Any) -> Callable[[Any], Sequence[Any]]:
    if isinstance(value, str):
        return sentences
    if isinstance(value, list | tuple):
        return list
    if isinstance(value, dict):
        lists = [k for k, v in value.items() if isinstance(v, list)]
        if len(lists) == 1:  # a trace: one list of tool calls or messages beside fixed fields
            key = lists[0]
            return lambda d: list(d[key])
    raise TypeError(f'pass split= and join= for a {type(value).__name__}')


def _default_join(value: Any) -> Callable[[Sequence[Any]], Any]:
    if isinstance(value, str):
        return lambda parts: ' '.join(parts)
    if isinstance(value, list | tuple):
        return list
    key = next(k for k, v in value.items() if isinstance(v, list))
    return lambda parts: {**value, key: list(parts)}
