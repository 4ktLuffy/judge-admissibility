"""Conversations: does the judge hold the agent to what was said earlier, and credit a repair?

A final reply can be right for the last message and wrong for the conversation: the customer
withdrew consent to charge their card three turns ago, changed the delivery address, or asked a
question nobody answered. A judge that reads only the last turn passes it. The opposite mistake
is as real: an agent that said something wrong and then corrected itself has done its job, and a
judge that fails every trajectory containing an error punishes the repair, so an optimizer
learns to hide mistakes rather than fix them.

Both are tested by changing the conversation and keeping the final reply, with `EvidenceRewrite`:
`insert_turns` builds such a control from the turns to insert. `recovery_profile` reads the two
error controls of a certificate together and says which of the two mistakes, if either, the judge
makes.

A conversation here is a list of turns (any mapping, typically `{'role': ..., 'text': ...}`)
under `key` in the case's inputs. The final reply is the case's output.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from ._certify import DEFAULT_THRESHOLDS, Certificate, Check, Thresholds, _check
from ._controls import ControlKind, EvidenceRewrite

Turn = Mapping[str, Any]
RecoveryKind = Literal['recovery-aware', 'unforgiving', 'blind to unrepaired errors', 'backwards', 'inconclusive']


def insert_turns(
    turns: Callable[[Any], Sequence[Turn] | None],
    name: str,
    kind: ControlKind,
    *,
    at: int = -1,
    key: str = 'conversation',
) -> EvidenceRewrite:
    """A control that inserts `turns(inputs)` into the conversation at index `at`, final reply unchanged.

    `at=-1` inserts before the customer's last message, so the inserted turns are never the last
    thing the judge reads before the reply: a judge that looks only at the last turn cannot see
    them, which is the point of an obligation control. `turns` returns None for an episode the
    control does not apply to (consent cannot be withdrawn twice).
    """

    def rewrite(inputs: Any) -> Any | None:
        if not isinstance(inputs, Mapping) or not isinstance(inputs.get(key), Sequence):  # pyright: ignore[reportUnknownMemberType]
            return None
        new = turns(inputs)
        if not new:
            return None
        changed: dict[str, Any] = copy.deepcopy(dict(inputs))  # pyright: ignore[reportUnknownArgumentType]
        conversation = list(changed[key])
        index = at if at >= 0 else max(0, len(conversation) + at)
        changed[key] = conversation[:index] + [dict(t) for t in new] + conversation[index:]
        return changed

    return EvidenceRewrite(rewrite, name, kind)


@dataclass(frozen=True)
class RecoveryProfile:
    """How a judge treats an agent's mistakes: forgives the repaired ones, catches the rest?"""

    kind: RecoveryKind
    forgives_repaired: Check
    """Repaired trajectories passed, among episodes whose original the judge passed."""
    catches_unrepaired: Check
    """Unrepaired trajectories failed."""

    def to_dict(self) -> dict[str, Any]:
        def row(c: Check) -> dict[str, Any]:
            return {'status': c.status, 'successes': c.successes, 'trials': c.trials, 'interval': list(c.interval)}

        return {
            'kind': self.kind,
            'forgives_repaired': row(self.forgives_repaired),
            'catches_unrepaired': row(self.catches_unrepaired),
        }


def recovery_profile(
    certificate: Certificate,
    *,
    repaired: str = 'error_then_repaired',
    unrepaired: str = 'error_unrepaired',
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
) -> RecoveryProfile:
    """Read a certificate's repaired and unrepaired error controls together.

    Each is decided like one control family of a certificate (Wilson PASS, exact FAIL, at the
    invariance and rejection bars), at the plain 95% level: this describes the judge, it is not
    another chance to fail its certificate, so it does not share the certificate's error budget. A
    repaired trajectory counts only where the judge passed the original episode's first judgment,
    so a judge that fails an episode for some other reason is not called unforgiving for it.

    `unforgiving`: shown to fail repaired trajectories.
    `blind to unrepaired errors`: shown to pass trajectories whose error stands. Both: `backwards`
    (it fails the repair and passes the error, so it is likely reacting to the correction, not the
    error). `inconclusive` when the cases cannot tell.
    """
    first = {j.case: j.passed for j in certificate.judgments if j.role == 'reference#0'}
    fixed = [
        j
        for j in certificate.judgments
        if j.role == f'must_hold:{repaired}' and first.get(j.case) and j.passed is not None
    ]
    standing = [j for j in certificate.judgments if j.role == f'must_fail:{unrepaired}' and j.passed is not None]

    def errored(role: str) -> int:
        return sum(j.passed is None for j in certificate.judgments if j.role == role)

    forgives = _check(
        'forgives_repaired', sum(bool(j.passed) for j in fixed), len(fixed), thresholds.min_invariance,
        thresholds.min_trials, detail=repaired, errors=errored(f'must_hold:{repaired}'),
    )  # fmt: skip
    catches = _check(
        'catches_unrepaired', sum(j.passed is False for j in standing), len(standing), thresholds.min_rejection,
        thresholds.min_trials, detail=unrepaired, errors=errored(f'must_fail:{unrepaired}'),
    )  # fmt: skip
    statuses = forgives.status, catches.status
    kind: RecoveryKind
    if statuses == ('PASS', 'PASS'):
        kind = 'recovery-aware'
    elif statuses == ('FAIL', 'FAIL'):
        kind = 'backwards'
    elif forgives.status == 'FAIL':
        kind = 'unforgiving'
    elif catches.status == 'FAIL':
        kind = 'blind to unrepaired errors'
    else:
        kind = 'inconclusive'
    return RecoveryProfile(kind, forgives, catches)
