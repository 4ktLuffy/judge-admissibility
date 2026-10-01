"""What a judge is certified against."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class JudgeCase:
    """One known-good answer: an output the judge should pass for these inputs.

    `name` is how human labels are matched to cases. `metadata` is passed through to the
    judge's `EvaluatorContext` unchanged.
    """

    name: str
    inputs: Any
    output: Any
    expected_output: Any = None
    metadata: Any = None


@dataclass(frozen=True)
class HumanLabel:
    """A person's verdict on one output, for example one row of a Logfire annotation export.

    `output` is the output the person judged. It need not be a case's known-good answer:
    labelled failures are what make an agreement figure mean something.
    """

    case: str
    output: Any
    passed: bool
    labels: dict[str, Any] = field(default_factory=dict)
