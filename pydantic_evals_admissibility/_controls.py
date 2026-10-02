"""Controls: outputs whose correct verdict is known before the judge sees them.

A `must_fail` control is an output no reasonable reading of the rubric accepts: another
case's answer, or nothing at all. A `must_hold` control changes nothing the rubric could be
about, such as whitespace, so the verdict on it must equal the verdict on the original.

Controls are generated from the cases the user already has. A control that cannot apply to a
case (an empty-string control on a non-string output) returns None and is skipped, never
counted as passed.
"""

from __future__ import annotations

import random
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from ._cases import JudgeCase

ControlKind = Literal['must_fail', 'must_hold']


class Control(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def kind(self) -> ControlKind: ...

    def make(self, case: JudgeCase, cases: Sequence[JudgeCase], rng: random.Random) -> Any | None:
        """The control output for `case`, or None when this control does not apply to it."""
        ...


@dataclass(frozen=True)
class MismatchedOutput:
    """Another case's known-good answer: right for some question, wrong for this one.

    The donor must give a different answer. Its output text has to differ, and when both cases
    carry an `expected_output`, those have to differ too: on a task whose answers are often "yes"
    or "no", another case's "Answer: yes" is as likely to be right as wrong, and a sound judge
    that passes it is not at fault.
    """

    name: str = 'mismatched_output'
    kind: ControlKind = 'must_fail'

    def make(self, case: JudgeCase, cases: Sequence[JudgeCase], rng: random.Random) -> Any | None:
        donors = [
            other
            for other in cases
            if other is not case
            and other.output != case.output
            and (
                case.expected_output is None
                or other.expected_output is None
                or other.expected_output != case.expected_output
            )
        ]
        if not donors:
            return None
        return rng.choice(donors).output


@dataclass(frozen=True)
class EmptyOutput:
    """An empty answer. Applies to string outputs only."""

    name: str = 'empty_output'
    kind: ControlKind = 'must_fail'

    def make(self, case: JudgeCase, cases: Sequence[JudgeCase], rng: random.Random) -> Any | None:
        return '' if isinstance(case.output, str) and case.output.strip() else None


@dataclass(frozen=True)
class WhitespaceReformat:
    """The same answer with its whitespace changed: runs collapsed, a trailing newline added.

    Applies to string outputs that actually change, so an answer with nothing to reformat is
    skipped rather than compared with itself.
    """

    name: str = 'whitespace_reformat'
    kind: ControlKind = 'must_hold'

    def make(self, case: JudgeCase, cases: Sequence[JudgeCase], rng: random.Random) -> Any | None:
        if not isinstance(case.output, str):
            return None
        reformatted = re.sub(r'[ \t]+', ' ', case.output.strip()).replace('\n', '\n\n') + '\n'
        return reformatted if reformatted != case.output else '  ' + case.output + '\n'


DEFAULT_CONTROLS: tuple[Control, ...] = (MismatchedOutput(), EmptyOutput(), WhitespaceReformat())
