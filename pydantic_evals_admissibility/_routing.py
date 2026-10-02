"""Route each judgment: a cheap judge decides when it agrees with itself, an expensive one when not.

Most judgments are easy, and a cheap judge gets them right; paying a reasoning model, or a
person, for every one is waste. `RoutedJudge` asks the cheap judge `k` times and keeps its verdict
when all `k` agree; when they disagree (or one errors), it escalates the judgment to the expensive
judge, or, with no expensive judge, raises `Escalated` so a person decides. Self-consistency is the
confidence signal because it needs nothing from the judge but its verdict: no calibrated score, no
log-probabilities, which a Codex or any structured-output judge does not expose.

Its limit is the point to measure, not assume: a judge that is wrong the same way every time
agrees with itself, and is never escalated. `RoutingSummary.confidently_wrong` counts those.

The routing policy, not either judge, is what gets deployed, so it is what gets certified:
`certify_routing` runs `certify_judge` on the `RoutedJudge` itself. Each verdict's reason records
how it was reached (`[routed: cheap PP]`, `[routed: escalated PF]`), so the certificate is also the
record of what the policy cost. `replay_routing` computes the same summary from verdicts already
saved, with no judge calls.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeVar

from pydantic_evals.evaluators import EvaluationReason, Evaluator, EvaluatorContext

from ._cases import HumanLabel, JudgeCase
from ._certify import DEFAULT_THRESHOLDS, Certificate, Judgment, Thresholds, assertion_of, certify_judge
from ._controls import DEFAULT_CONTROLS, Control
from ._stats import wilson

_ROUTE = re.compile(r'\[routed: (cheap|escalated) ([PF?]+)\]')


class Escalated(Exception):
    """The cheap judge was unsure and there is no expensive judge: a person must decide.

    Raised rather than returned, so that `certify_judge` records it as a judgment without a
    verdict, never as a pass or a fail.
    """


def unanimous(samples: Sequence[bool | None]) -> bool | None:
    """The verdict when every sample gave it; None (escalate) on any disagreement or error."""
    if not samples or any(s is None for s in samples) or len(set(samples)) > 1:
        return None
    return samples[0]


def _code(samples: Sequence[bool | None]) -> str:
    return ''.join('?' if s is None else 'P' if s else 'F' for s in samples)


def route_of(reason: str | None) -> tuple[str, list[bool | None]] | None:
    """How a `RoutedJudge` verdict was reached: ('cheap' or 'escalated', the cheap samples)."""
    found = _ROUTE.search(reason or '')
    if found is None:
        return None
    samples: list[bool | None] = [None if c == '?' else c == 'P' for c in found.group(2)]
    return found.group(1), samples


@dataclass
class RoutedJudge(Evaluator[Any, Any, Any]):
    """The cheap judge's verdict when its `k` samples agree; otherwise the expensive judge's.

    The `k` cheap calls are made one after another, so `max_concurrency` in `certify_judge`
    bounds the calls in flight (up to one per routed judgment in flight). `k=1` never escalates
    except on an error: it is the cheap judge alone, with the bookkeeping.
    """

    cheap: Evaluator[Any, Any, Any]
    expensive: Evaluator[Any, Any, Any] | None = None
    """None routes uncertain judgments to a person: they raise `Escalated`."""
    k: int = 2
    assertion: str | None = None

    def __post_init__(self) -> None:
        if self.k < 1:
            raise ValueError(f'k must be >= 1, got {self.k}')

    def get_default_evaluation_name(self) -> str:
        return self.cheap.get_default_evaluation_name()

    async def evaluate(self, ctx: EvaluatorContext[Any, Any, Any]) -> EvaluationReason:
        samples: list[bool | None] = []
        reason: str | None = None
        for _ in range(self.k):
            try:
                passed, why = assertion_of(await self.cheap.evaluate_async(ctx), self.assertion)
            except Exception:  # an errored sample is not a vote: it can only escalate
                passed, why = None, None
            samples.append(passed)
            reason = reason or why
        decided = unanimous(samples)
        if decided is not None:
            return EvaluationReason(decided, f'[routed: cheap {_code(samples)}] {reason or ""}'.rstrip())
        tag = f'[routed: escalated {_code(samples)}]'
        if self.expensive is None:
            raise Escalated(f'{tag} the cheap judge disagreed with itself; a person must decide')
        passed, why = assertion_of(await self.expensive.evaluate_async(ctx), self.assertion)
        return EvaluationReason(passed, f'{tag} {why or ""}'.rstrip())


def _rate(successes: int, trials: int) -> dict[str, Any]:
    return {
        'successes': successes,
        'trials': trials,
        'rate': successes / trials if trials else None,
        'interval': wilson(successes, trials) if trials else None,
    }


@dataclass
class RoutingSummary:
    """What a routing policy did on a set of judgments whose right verdict is known.

    Accuracy is per judgment, against the known verdict. Costs are in the units of
    `cheap_cost` and `expensive_cost` (calls by default; pass tokens or seconds per call to get
    those), per judgment, and relative to always using the expensive judge.
    """

    k: int
    judgments: int
    escalated: int
    routed_correct: int
    cheap_correct: int
    """The cheap judge alone: its first sample on each judgment."""
    expensive_correct: int | None
    expensive_trials: int
    """Judgments with an expensive verdict alone (all of them when a baseline was given)."""
    confidently_wrong: int
    """Judgments the cheap judge got wrong on every sample: routing cannot see these."""
    escalated_cheap_wrong: int
    """Escalated judgments whose first cheap sample was wrong: what escalation caught."""
    unresolved: int = 0
    """Judgments with no verdict at all (escalated to a person, or every judge errored)."""
    cheap_cost: float = 1.0
    expensive_cost: float = 1.0
    notes: list[str] = field(default_factory=list)

    @property
    def escalation_rate(self) -> float:
        return self.escalated / self.judgments if self.judgments else 0.0

    @property
    def cost_per_judgment(self) -> float:
        if not self.judgments:
            return 0.0
        return self.k * self.cheap_cost + self.escalation_rate * self.expensive_cost

    @property
    def relative_cost(self) -> float:
        """Cost against always asking the expensive judge once per judgment."""
        return self.cost_per_judgment / self.expensive_cost

    def to_dict(self) -> dict[str, Any]:
        return {
            'k': self.k,
            'judgments': self.judgments,
            'escalated': _rate(self.escalated, self.judgments),
            'accuracy': {
                'routed': _rate(self.routed_correct, self.judgments),
                'cheap alone': _rate(self.cheap_correct, self.judgments),
                'expensive alone': (
                    _rate(self.expensive_correct, self.expensive_trials) if self.expensive_correct is not None else None
                ),
            },
            'confidently_wrong': self.confidently_wrong,
            'escalated_cheap_wrong': self.escalated_cheap_wrong,
            'unresolved': self.unresolved,
            'cost': {
                'cheap_cost': self.cheap_cost,
                'expensive_cost': self.expensive_cost,
                'cheap_calls': self.k * self.judgments,
                'expensive_calls': self.escalated,
                'per_judgment': self.cost_per_judgment,
                'relative_to_always_expensive': self.relative_cost,
            },
            'notes': self.notes,
        }

    def table(self) -> str:
        def line(name: str, correct: int | None, trials: int) -> str:
            if correct is None or not trials:
                return f'  {name:<16} not measured'
            low, high = wilson(correct, trials)
            return f'  {name:<16} {correct}/{trials} = {correct / trials:.3f} [{low:.3f}, {high:.3f}]'

        low, high = wilson(self.escalated, self.judgments) if self.judgments else (0.0, 0.0)
        return '\n'.join(
            [
                f'escalated {self.escalated}/{self.judgments} = {self.escalation_rate:.3f} '
                f'[{low:.3f}, {high:.3f}] (k={self.k})',
                'accuracy against the known verdict:',
                line('routed', self.routed_correct, self.judgments),
                line('cheap alone', self.cheap_correct, self.judgments),
                line('expensive alone', self.expensive_correct, self.expensive_trials),
                f'cheap wrong on every sample (never escalated): {self.confidently_wrong}; '
                f'escalations that caught a cheap error: {self.escalated_cheap_wrong}/{self.escalated}',
                f'cost per judgment {self.cost_per_judgment:.2f} = {self.relative_cost:.2f}x always-expensive',
            ]
        )


K = TypeVar('K', bound=Hashable)
MatchKey = tuple[str, str, str, int]


def replay_routing(
    cheap: Mapping[K, Sequence[bool | None]],
    expensive: Mapping[K, bool | None],
    truth: Mapping[K, bool],
    *,
    k: int | None = None,
    cheap_cost: float = 1.0,
    expensive_cost: float = 1.0,
) -> RoutingSummary:
    """The routing policy on verdicts already made: no judge calls.

    Args:
        cheap: Each judgment's cheap samples, at least `k` of them; the first `k` are used.
        expensive: The expensive judge's verdict on each judgment (missing or None: it errored).
        truth: The right verdict of each judgment. Only judgments in `truth` and `cheap` count.
        k: Samples the policy asks for; by default the fewest any judgment has.
    """
    keys = [key for key in cheap if key in truth]
    if not keys:
        raise ValueError('no judgment has both cheap samples and a known verdict')
    k = k or min(len(cheap[key]) for key in keys)
    if any(len(cheap[key]) < k for key in keys):
        raise ValueError(f'every judgment needs at least k={k} cheap samples')
    escalated = routed = alone = confident = caught = unresolved = 0
    expensive_correct = expensive_trials = 0
    for key in keys:
        samples, right = list(cheap[key])[:k], truth[key]
        alone += samples[0] == right
        confident += all(s is not None and s != right for s in samples)
        decided = unanimous(samples)
        if decided is None:
            escalated += 1
            caught += samples[0] != right
            decided = expensive.get(key)
            unresolved += decided is None
        routed += decided == right
        if key in expensive:
            expensive_trials += 1
            expensive_correct += expensive[key] == right
    return RoutingSummary(
        k, len(keys), escalated, routed, alone, expensive_correct if expensive_trials else None, expensive_trials,
        confident, caught, unresolved, cheap_cost, expensive_cost,
    )  # fmt: skip


def truth_of(role: str) -> bool | None:
    """The right verdict a certificate role implies: known-good and must-hold pass, must-fail fails.

    Assumes, as the certificate does, that the known-good answers are good. Human labels carry
    their own verdict. Repeats past the first (`reference#1`, ...) return None: the unit of
    evidence is the case, so a repeat is not a second judgment to count.
    """
    if role == 'reference#0' or role.startswith('must_hold:'):
        return True
    if role.startswith('must_fail:'):
        return False
    if role.startswith('human:'):
        return role == 'human:1'
    return None


def keyed(judgments: Sequence[Judgment]) -> dict[MatchKey, Judgment]:
    """Judgments by (case, role, output, occurrence), to match two certificates of the same plan."""
    seen: Counter[tuple[str, str, str]] = Counter()
    out: dict[MatchKey, Judgment] = {}
    for j in judgments:
        base = (j.case, j.role, repr(j.output))
        out[(*base, seen[base])] = j
        seen[base] += 1
    return out


@dataclass
class RoutedCertificate:
    """The certificate of the routing policy as deployed, and what it cost against each judge."""

    certificate: Certificate
    summary: RoutingSummary
    judge: RoutedJudge
    baselines: dict[str, Certificate] = field(default_factory=dict)

    def table(self) -> str:
        return f'{self.certificate.table()}\n{self.summary.table()}'

    def to_dict(self, *, judgments: bool = True) -> dict[str, Any]:
        return {
            'certificate': self.certificate.to_dict(judgments=judgments),
            'routing': self.summary.to_dict(),
            'baselines': {name: cert.to_dict(judgments=False) for name, cert in self.baselines.items()},
        }


def summarize_routing(
    certificate: Certificate,
    *,
    k: int,
    expensive_alone: Certificate | None = None,
    cheap_cost: float = 1.0,
    expensive_cost: float = 1.0,
) -> RoutingSummary:
    """A `RoutingSummary` from a `RoutedJudge`'s certificate, read from each verdict's reason.

    `expensive_alone` is a certificate of the expensive judge on the same plan (same cases,
    controls, labels and seed); without it, the expensive judge's accuracy is measured only on
    the judgments it was escalated, which is not its accuracy alone, so it is left out.
    """
    alone = keyed(expensive_alone.judgments) if expensive_alone is not None else {}
    cheap: dict[MatchKey, Sequence[bool | None]] = {}
    expensive: dict[MatchKey, bool | None] = {}
    truth: dict[MatchKey, bool] = {}
    unrouted = 0
    for key, j in keyed(certificate.judgments).items():
        right = truth_of(j.role)
        if right is None:
            continue
        route = route_of(j.reason or j.error)
        if route is None:  # the routed judge itself errored: the expensive judge failed on an escalation
            unrouted += 1
            continue
        how, samples = route
        cheap[key], truth[key] = samples, right
        if how == 'escalated':
            expensive[key] = j.passed
    summary = replay_routing(cheap, expensive, truth, k=k, cheap_cost=cheap_cost, expensive_cost=expensive_cost)
    if alone:
        both = [key for key in cheap if key in alone and alone[key].passed is not None]
        summary.expensive_trials = len(both)
        summary.expensive_correct = sum(alone[key].passed == truth[key] for key in both)
    else:
        summary.expensive_correct, summary.expensive_trials = None, 0
    if unrouted:
        summary.notes.append(f'{unrouted} judgments errored with no route recorded; left out')
    return summary


async def certify_routing(
    cheap: Evaluator[Any, Any, Any],
    expensive: Evaluator[Any, Any, Any] | None,
    cases: Sequence[JudgeCase],
    *,
    k: int = 2,
    controls: Sequence[Control] = DEFAULT_CONTROLS,
    human_labels: Sequence[HumanLabel] = (),
    repeats: int = 2,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
    assertion: str | None = None,
    max_concurrency: int = 8,
    seed: int = 0,
    baselines: Mapping[str, Certificate] | None = None,
    expensive_baseline: str | None = None,
    cheap_cost: float = 1.0,
    expensive_cost: float = 1.0,
) -> RoutedCertificate:
    """Certify the routing policy as the deployed evaluator, and say what it cost.

    The certificate is `certify_judge` on `RoutedJudge(cheap, expensive, k)`, with the same
    checks and thresholds as any judge: a policy is admissible or not on its own verdicts, not on
    its judges' separate certificates (a policy can escalate exactly the judgments that matter, or
    none of them). `repeats` defaults to 2, not 3: each repeat costs `k` cheap calls, and two are
    enough to measure stability.

    Args:
        baselines: Certificates of other judges on the same plan, reported beside it; pass the
            same `cases`, `controls`, `human_labels` and `seed` so judgments match.
        expensive_baseline: Which of `baselines` is the expensive judge alone, for the accuracy
            comparison in the summary.
        cheap_cost, expensive_cost: Cost of one call of each, in any unit (tokens, seconds).
    """
    judge = RoutedJudge(cheap, expensive, k=k, assertion=assertion)
    certificate = await certify_judge(
        judge,
        cases,
        controls=controls,
        human_labels=human_labels,
        repeats=repeats,
        thresholds=thresholds,
        max_concurrency=max_concurrency,
        seed=seed,
    )
    baselines = dict(baselines or {})
    summary = summarize_routing(
        certificate,
        k=k,
        expensive_alone=baselines.get(expensive_baseline) if expensive_baseline else None,
        cheap_cost=cheap_cost,
        expensive_cost=expensive_cost,
    )
    return RoutedCertificate(certificate, summary, judge, baselines)
