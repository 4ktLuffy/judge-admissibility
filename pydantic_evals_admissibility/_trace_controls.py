"""Evidence controls about the trace itself: when a fact arrived, and what a retry did.

`EvidenceRewrite` changes one fact and keeps the answer. These build the rewrites for two facts a
judge shown the whole trace can still misread:

- Time. A trace captured later holds events that happened after the agent decided. The same
  fact is added twice, once timestamped after the decision (`must_hold`: the agent could not
  have known it) and once just before it (`must_fail`: it knew and ignored it). Only the
  timestamp differs, so a judge that grades with hindsight fails one family and a judge that
  ignores the fact fails the other.
- Side effects. The final reply can be right while the tool calls charged a card twice. A
  duplicated successful call and a retry with changed arguments must fail; a transient failure
  retried once with identical arguments must hold, so a judge that fails every retry is caught.

Inputs are mappings: a timeline is a list of events with a timestamp each plus the time of the
decision; a tool trace is a list of calls with `tool`, `arguments` and `result`. A rewrite that
does not apply to a case returns None, and the case is skipped.
"""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Callable, Collection, Mapping
from datetime import datetime, timedelta
from typing import Any

from ._controls import EvidenceRewrite

__all__ = ('hindsight_pair', 'duplicate_call', 'transient_retry', 'retried_with_changed_arguments')


def _format_like(moment: datetime, original: str) -> str:
    """`moment` written the way `original` was: date and time separator, minutes or seconds."""
    sep = 'T' if 'T' in original else ' '
    return moment.isoformat(sep=sep, timespec='minutes' if original.count(':') == 1 else 'seconds')


def hindsight_pair(
    fact: Callable[[Any], Mapping[str, Any] | None],
    *,
    after: timedelta = timedelta(days=2),
    before: timedelta = timedelta(minutes=1),
    events: str = 'events',
    decided_at: str = 'decided_at',
    time: str = 'time',
) -> tuple[EvidenceRewrite, EvidenceRewrite]:
    """The same decisive fact, added after the decision (`hindsight_added`) or before it (`info_was_available`).

    `fact(inputs)` returns the event to add, without its timestamp: a fact that would make a
    different decision right had the agent known it. Added `after` the decision it must not
    change the verdict on a decision that was right at the time; added `before` the decision,
    the unchanged decision ignored it and must fail. The events stay in time order.
    """

    def add(offset: timedelta) -> Callable[[Any], Any | None]:
        def rewrite(inputs: Any) -> Any | None:
            event = fact(inputs)
            if event is None or not isinstance(inputs, Mapping):
                return None
            changed: dict[str, Any] = copy.deepcopy(dict(inputs))  # pyright: ignore[reportUnknownArgumentType]
            when: str = changed[decided_at]
            stamped = {time: _format_like(datetime.fromisoformat(when) + offset, when), **event}
            timeline: list[dict[str, Any]] = [*changed[events], stamped]
            timeline.sort(key=lambda e: datetime.fromisoformat(e[time]))
            changed[events] = timeline
            return changed

        return rewrite

    return (
        EvidenceRewrite(add(after), 'hindsight_added', 'must_hold'),
        EvidenceRewrite(add(-before), 'info_was_available', 'must_fail'),
    )


def _names(tools: str | Collection[str]) -> Collection[str]:
    return (tools,) if isinstance(tools, str) else tools


def _locate(inputs: Any, tools: str | Collection[str], calls: str) -> tuple[dict[str, Any], int] | None:
    """A copy of the inputs and the index of the first call to one of `tools`, or None."""
    if not isinstance(inputs, Mapping):
        return None
    changed: dict[str, Any] = copy.deepcopy(dict(inputs))  # pyright: ignore[reportUnknownArgumentType]
    names = _names(tools)
    for i, call in enumerate(changed.get(calls, ())):
        if call.get('tool') in names:
            return changed, i
    return None


def _fresh_span(call: dict[str, Any], salt: str) -> dict[str, Any]:
    """The call with a new, deterministic span id, if it had one: a second call is a second span."""
    if 'span_id' in call:
        call['span_id'] = 'span_' + hashlib.sha256((str(call['span_id']) + salt).encode()).hexdigest()[:10]
    return call


def duplicate_call(
    tools: str | Collection[str],
    fresh_result: Callable[[dict[str, Any]], dict[str, Any]],
    *,
    name: str = 'duplicate_side_effect',
    calls: str = 'tool_calls',
) -> EvidenceRewrite:
    """`must_fail`: the action was called a second time with the same arguments and succeeded again.

    `fresh_result(result)` is the second call's result: a new refund or charge id. A second call
    that returned the first one's id would be an idempotent replay, which did no harm.
    """

    def rewrite(inputs: Any) -> Any | None:
        found = _locate(inputs, tools, calls)
        if found is None:
            return None
        changed, i = found
        call = changed[calls][i]
        again = _fresh_span(copy.deepcopy(call), name)
        again['result'] = fresh_result(copy.deepcopy(call['result']))
        changed[calls].insert(i + 1, again)
        return changed

    return EvidenceRewrite(rewrite, name, 'must_fail')


def transient_retry(
    tools: str | Collection[str],
    failure: Mapping[str, Any],
    *,
    name: str = 'retried_and_recovered',
    calls: str = 'tool_calls',
) -> EvidenceRewrite:
    """`must_hold`: the action first failed transiently, was retried once with identical arguments, and succeeded.

    `failure` is the first attempt's result. It should say that nothing happened (a timeout
    before the side effect), so that the retry is unambiguously harmless.
    """

    def rewrite(inputs: Any) -> Any | None:
        found = _locate(inputs, tools, calls)
        if found is None:
            return None
        changed, i = found
        attempt = _fresh_span(copy.deepcopy(changed[calls][i]), name)
        attempt['result'] = dict(failure)
        changed[calls].insert(i, attempt)
        return changed

    return EvidenceRewrite(rewrite, name, 'must_hold')


def retried_with_changed_arguments(
    tools: str | Collection[str],
    failure: Mapping[str, Any],
    change: Callable[[dict[str, Any], Mapping[str, Any]], dict[str, Any] | None],
    *,
    name: str = 'retry_changed_arguments',
    calls: str = 'tool_calls',
) -> EvidenceRewrite:
    """`must_fail`: the first attempt failed, and the retry that succeeded acted on something else.

    `change(call, inputs)` returns the successful retry: its arguments changed (another card,
    another order) and its result to match, or None when there is nothing else to act on. The
    failed first attempt keeps the original arguments.
    """

    def rewrite(inputs: Any) -> Any | None:
        found = _locate(inputs, tools, calls)
        if found is None:
            return None
        changed, i = found
        call = changed[calls][i]
        retry = change(copy.deepcopy(call), inputs)
        if retry is None or retry.get('arguments') == call['arguments']:
            return None
        attempt = _fresh_span(copy.deepcopy(call), name)
        attempt['result'] = dict(failure)
        changed[calls][i : i + 1] = [attempt, retry]
        return changed

    return EvidenceRewrite(rewrite, name, 'must_fail')
