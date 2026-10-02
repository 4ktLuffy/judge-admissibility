"""Executable rubric distillation: check in code what code can check, and ask a model the rest.

A rubric is usually several clauses. Some are facts a few lines of Python decide exactly ("the
reply ends with an `Answer:` line", "the amount is the one the policy gives"); some are not
("the tone is polite"). An `LLMJudge` asked all of them at once pays a model call for every
verdict, and can get the checkable ones wrong. `HybridJudge` runs the checkable clauses first, in
code: a failure fails the answer without a model call. Only answers that pass every check reach
an `LLMJudge` whose rubric is the remaining clauses alone.

`HybridJudge` is an ordinary pydantic-evals `Evaluator`, so it is certified like any judge, with
the same controls. Two caveats keep the comparison honest:

- A control decided in code is decided by the check you wrote. If the check is the same code as
  your ground truth, the certificate's rejection rate on those controls is the check agreeing with
  itself. The certificate still says whether the *model* part passes what it should, and the
  calls it saves are real; whether the check is right is a question for its own tests.
- A clause moved into code changes what the model is asked. `compare_hybrid` reports where the two
  judges' verdicts differ, so a model that judged the full rubric differently from the split one is
  seen, not assumed away.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic_ai import models
from pydantic_ai.settings import ModelSettings
from pydantic_evals.evaluators import EvaluationReason, Evaluator, EvaluatorContext, LLMJudge

from ._certify import Certificate, Judgment, assertion_of
from ._stats import wilson

Check = Callable[[EvaluatorContext[Any, Any, Any]], bool]

CODE_PREFIX = 'decided in code: '
"""Every verdict `HybridJudge` reaches without a model call has a reason starting with this, so a
saved certificate still says which judgments cost a call."""


@dataclass(frozen=True)
class Clause:
    """One requirement of a rubric, with a Python `check` when code can decide it.

    `check(ctx)` returns True when the answer meets the clause. Leave it None for a clause that
    needs judgment; it goes to the model. The check's qualified name, not its address, is what a
    certificate records, so a judge's fingerprint is stable across runs.
    """

    text: str
    check: Check | None = None

    def __repr__(self) -> str:
        how = 'model' if self.check is None else getattr(self.check, '__qualname__', type(self.check).__name__)
        return f'Clause({self.text!r}, check={how})'


def split_rubric(rubric: str, checks: Mapping[str, Check] | None = None) -> list[Clause]:
    """Split `rubric` into its sentences and attach each check to the one clause it names.

    `checks` maps a phrase to a check; the phrase must occur in exactly one clause, so a typo or an
    ambiguous phrase is an error rather than a clause silently left to the model.
    """
    texts = [t.strip() for t in re.split(r'(?<=[.!?])\s+', rubric.strip()) if t.strip()]
    attached: dict[int, Check] = {}
    for phrase, check in (checks or {}).items():
        hits = [i for i, t in enumerate(texts) if phrase.lower() in t.lower()]
        if len(hits) != 1:
            raise ValueError(f'{phrase!r} matches {len(hits)} clauses of the rubric; it must match exactly one')
        if hits[0] in attached:
            raise ValueError(f'two checks name the clause {texts[hits[0]]!r}')
        attached[hits[0]] = check
    return [Clause(t, attached.get(i)) for i, t in enumerate(texts)]


@dataclass(repr=False)
class HybridJudge(Evaluator[object, object, object]):
    """Decide the checkable clauses in code, and only the rest with an `LLMJudge`.

    Checks run in order; the first that fails decides the verdict (fail) with no model call. If
    every clause has a check, an answer that passes them all passes, again with no call. Otherwise
    an `LLMJudge` built from the unchecked clauses (and the other settings here, which mean what
    they mean on `LLMJudge`) gives the verdict.
    """

    clauses: Sequence[Clause]
    model: models.Model | models.KnownModelName | str | None = None
    include_input: bool = True
    """True by default, unlike `LLMJudge`: a judge that cannot see the question cannot grade the answer."""
    include_expected_output: bool = False
    model_settings: ModelSettings | None = None

    def __post_init__(self) -> None:
        if not self.clauses:
            raise ValueError('a HybridJudge needs at least one clause')
        self.clauses = tuple(self.clauses)

    @property
    def model_rubric(self) -> str | None:
        """What the model is asked: the unchecked clauses, or None when code decides everything."""
        rest = [c.text for c in self.clauses if c.check is None]
        return ' '.join(rest) if rest else None

    @property
    def full_rubric(self) -> str:
        """Every clause, as a plain `LLMJudge` would be asked it."""
        return ' '.join(c.text for c in self.clauses)

    def llm_judge(self) -> LLMJudge | None:
        """The `LLMJudge` that judges the unchecked clauses (None if there are none)."""
        rubric = self.model_rubric
        if rubric is None:
            return None
        # Built fresh each time rather than cached on the instance: a cached field would change the
        # judge's identity after its first call, and its certificate would no longer cover it.
        return LLMJudge(
            rubric=rubric,
            model=self.model,
            include_input=self.include_input,
            include_expected_output=self.include_expected_output,
            model_settings=self.model_settings,
        )

    async def evaluate(self, ctx: EvaluatorContext[object, object, object]) -> EvaluationReason:
        for clause in self.clauses:
            if clause.check is not None and not clause.check(ctx):
                return EvaluationReason(False, f'{CODE_PREFIX}fails "{clause.text}"')
        judge = self.llm_judge()
        if judge is None:
            return EvaluationReason(True, f'{CODE_PREFIX}every clause checked')
        passed, reason = assertion_of(await judge.evaluate(ctx))
        return EvaluationReason(passed, reason)


def in_code(judgment: Judgment) -> bool:
    """Whether a judgment was decided without a model call. Errored judgments count as calls."""
    return judgment.error is None and (judgment.reason or '').startswith(CODE_PREFIX)


def model_calls(certificate: Certificate) -> int:
    """Judgments in `certificate` that cost a model call: all of them, for a plain `LLMJudge`."""
    return sum(not in_code(j) for j in certificate.judgments)


@dataclass(frozen=True)
class JudgeComparison:
    """Two judges certified on the same plan: their certificates, their cost, and where they disagree."""

    verdicts: tuple[str, str]
    checks: dict[str, tuple[str, str]]
    """Each check's status for the first and the second judge."""
    calls: tuple[int, int]
    """Model calls each certificate cost."""
    agreed: int
    compared: int
    interval: tuple[float, float]
    """95% Wilson interval on the share of paired judgments with the same verdict."""
    disagreements: tuple[tuple[str, str, bool | None, bool | None], ...] = field(repr=False)
    """(case, role, first verdict, second verdict) for every pair that differs."""

    @property
    def calls_saved(self) -> int:
        return self.calls[0] - self.calls[1]

    def to_dict(self) -> dict[str, Any]:
        return {
            'verdicts': list(self.verdicts),
            'checks': {k: list(v) for k, v in self.checks.items()},
            'calls': list(self.calls),
            'calls_saved': self.calls_saved,
            'agreement': {'agreed': self.agreed, 'compared': self.compared, 'interval': list(self.interval)},
            'disagreements': [{'case': c, 'role': r, 'first': a, 'second': b} for c, r, a, b in self.disagreements],
        }


def compare_hybrid(first: Certificate, second: Certificate) -> JudgeComparison:
    """Pair the two certificates' judgments by case, role and output, and compare verdicts and cost.

    Meant for a plain `LLMJudge` (first) and a `HybridJudge` (second) certified with the same
    cases, controls, repeats and seed, so every judgment has a partner. A judgment that errored on
    either side is left out of the agreement, not counted as a disagreement.
    """

    def key(j: Judgment) -> tuple[str, str, str]:
        return j.case, j.role, j.output if isinstance(j.output, str) else repr(j.output)

    seconds = {key(j): j for j in second.judgments}
    pairs = [(j, seconds[key(j)]) for j in first.judgments if key(j) in seconds]
    done = [(a, b) for a, b in pairs if a.passed is not None and b.passed is not None]
    agreed = sum(a.passed == b.passed for a, b in done)
    names = [c.name for c in first.checks] + [
        c.name for c in second.checks if c.name not in {x.name for x in first.checks}
    ]
    status = {c.name: c.status for c in first.checks}, {c.name: c.status for c in second.checks}
    return JudgeComparison(
        verdicts=(first.verdict, second.verdict),
        checks={n: (status[0].get(n, '-'), status[1].get(n, '-')) for n in names},
        calls=(model_calls(first), model_calls(second)),
        agreed=agreed,
        compared=len(done),
        interval=wilson(agreed, len(done)),
        disagreements=tuple((a.case, a.role, a.passed, b.passed) for a, b in done if a.passed != b.passed),
    )
