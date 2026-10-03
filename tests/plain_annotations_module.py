"""Classes in a module WITHOUT `from __future__ import annotations`, as most user code is written.

On Python 3.14 such classes carry an `__annotate_func__` that closes over the class namespace.
"""

import dataclasses
from typing import Any, NamedTuple

from pydantic_evals.evaluators import Evaluator, EvaluatorContext


@dataclasses.dataclass
class PlainJudge(Evaluator[Any, Any, Any]):
    threshold: float = 0.5

    async def evaluate(self, ctx: EvaluatorContext[Any, Any, Any]) -> bool:  # pragma: no cover
        return True


@dataclasses.dataclass
class PlainInput:
    question: str


class Point(NamedTuple):
    x: int
    y: int
