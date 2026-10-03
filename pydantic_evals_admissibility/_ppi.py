"""The agent's real pass rate from a judge on all the traffic and people on a small random part of it.

A judge scores every production output; it is cheap and biased. People label a few; they are
right and expensive. Prediction-powered inference (Angelopoulos et al. 2023, arXiv 2301.09633)
uses both: the judge's pass rate on all N outputs, plus the mean of (human - judge) on the
audited n, the *rectifier*, which measures the judge's bias where it can be seen. On a uniform
audit, plain PPI (`lam = 1`) before clipping is unbiased whatever the judge's mistakes, and its
interval is narrow when the judge is right where people look, because the rectifier then barely
varies. PPI++ (arXiv 2311.01453) weights the judge
by `lam` chosen from the audit: `mean(human) + lam * (judge rate on N - judge rate on the audit)`.
With `lam = 0` it is the human-only rate; with `lam = 1`, plain PPI. A judge that tells nothing
about the labels gets `lam` near 0, so it cannot make the interval much wider than people alone.

**Not exactly unbiased as reported.** The reported `estimate` is clipped to [0, 1], and PPI++'s
`lam` is estimated from the same audit; neither is exactly unbiased. Four outputs that truly pass,
two passed by the judge, every audit of two: plain PPI unclipped averages 1 (`raw_estimate`), but
the audit of the judge's two failures gives 1.5, clipped to 1, and PPI++ sets `lam = 0` on the four
mixed audits; clipped plain PPI and PPI++ both average 11/12. The bias is a small-audit effect:
in simulation, |bias| <= 0.004 from n = 50 (below).

Two adaptations, both stated so they can be checked:

- **The estimand is the pass rate of these N outputs**, not of a process behind them, and the
  judge scores all N, audited ones included. The audit is then a simple random sample without
  replacement from a finite population, and the estimators are survey sampling's difference
  (PPI) and regression (PPI++) estimators: the rectifier's variance carries the finite population
  correction `1 - n/N`, and the best `lam` is `cov(human, judge) / var(judge)`. The paper's
  `(1 + n/N)` factor is for an unlabelled set drawn apart from the labelled one; when n/N is small,
  as in production, the two agree.
- **Two intervals.** `interval` is the paper's: a normal (CLT) interval, valid as n grows, not at
  any n, clipped to [0, 1] and widened, if need be, to hold the estimate. Where it has zero width
  on a partial audit (no disagreement seen, every audited output the same mistake, PPI++ choosing
  `lam = 0` where people all agree, or the whole interval outside [0, 1]), that is not evidence
  that the judge never errs, so `interval` is `exact_interval` instead, with a warning
  (`interval_method` says which). Only a census (n = N) is a point. `exact_interval` is
  finite-sample: the rectifier is +1 on a false fail and -1 on a false pass, so the true rate is
  the judge's rate plus the false-fail rate minus the false-pass rate, and each of those gets a
  Clopper-Pearson interval at `alpha / 2` (Bonferroni). It never covers less than 1 - alpha for
  draws with replacement; it is plain PPI, not tuned, so for an uninformative judge it is wider
  than people alone. Coverage of both is measured in `bench/ppi.py`.

**Only a uniform random audit is valid**, which is why `ppi_pass_rate` takes the `AuditSample`
that `audit_sample` drew and refuses labels for any other set of cases. The rectifier estimates
the judge's bias over all outputs only if the audited outputs are a random draw from them. A
review queue of the judge's failures measures its bias on failures: it contains no false passes
at all, so on a lenient judge it would find the judge too strict. Skipping drawn outputs that are
hard to label does the same quietly, so a drawn case left unlabelled is refused too.

**Against Rogan-Gladen** (`recalibrated_pass_rate` in `_outcomes.py`), with the audit as its
calibration: both are consistent here, with different models of the judge. Rogan-Gladen assumes
the judge's sensitivity and specificity carry over, and divides by `sensitivity + specificity - 1`;
PPI++ before its weight is clipped is `q * PPV + (1 - q) * (1 - NPV)`, with `q` the judge's pass
rate on all N and the predictive values from the audit. Rogan-Gladen is the one to use when
calibration and traffic are different populations (outcomes from last week, traffic today). On a
uniform audit of today's traffic, measured: it is undefined when the audit holds no failure, it is
biased (-0.05) for a judge no better than chance, whose draws it drops, and its bootstrap interval
was the narrowest valid one for a near-perfect judge (0.136 wide at n = 50, against 0.204 exact).

**Measured** (`bench/ppi.py`, `results/ppi.json`; N = 5000, true rate 0.6, 2000 populations): PPI
and PPI++ have |bias| <= 0.004 where the judge alone is off by up to 0.10. `interval` covers
0.93-0.96, and more where it falls back to the exact one: a near-perfect judge's audits saw no
mistake 35% of the time at n = 50 and 13% at n = 100, where the CLT interval alone covered 0.65
and 0.85; `interval` covers 0.99 and 0.98. It is narrower than people alone for any informative
judge (good, n = 200: 0.069 against 0.134; lenient, n = 100: 0.136 against 0.188). The exact
interval always covers (>= 0.997) and is narrower than people alone only for a near-perfect
judge, or a good one from n = 100. For a judge independent of the truth, plain PPI is 36-42% wider
than people alone, and PPI++ about as wide (0.269 against 0.260 at n = 50, 0.133 against 0.134 at
n = 200; covering 0.94). On real replies (80, 90% passing, Codex judges) the judges are so
accurate that audits of 20 or 40 rarely see a mistake, so `interval` is mostly the exact one: it
covered 0.95-1.0; at 40 labels it was 0.11-0.14 wide against Wilson's 0.19 for the two judges
that erred, 0.20 for the one that never did, and about Wilson's width at 20 labels. On 20 replies
audited 10 at a time, plain PPI's CLT interval, where an audit saw a mistake, still covered 0.86.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any

from ._stats import clopper_pearson, wilson


@dataclass(frozen=True)
class AuditSample:
    """The outputs to send to people, drawn uniformly at random by `audit_sample`; keep it with the labels."""

    cases: tuple[str, ...]
    population: int
    seed: int
    digest: str
    """Fingerprint of the case names it was drawn from, so it cannot be applied to other traffic."""


def _digest(names: Iterable[str]) -> str:
    return hashlib.sha256('\n'.join(sorted(names)).encode()).hexdigest()[:16]


def audit_sample(case_names: Iterable[str], n: int, seed: int = 0) -> AuditSample:
    """Draw `n` of the cases uniformly at random, without replacement, for people to label.

    Fix `n` before any label is seen. Stopping when the interval looks narrow enough, then
    reporting it, gives the interval several chances to miss.
    """
    names = sorted(case_names)
    if len(set(names)) != len(names):
        raise ValueError('case names must be unique')
    if not 2 <= n <= len(names):
        raise ValueError(f'the audit needs between 2 and {len(names)} cases, got {n}')
    cases = tuple(random.Random(seed).sample(names, n))
    return AuditSample(cases, len(names), seed, _digest(names))


@dataclass(frozen=True)
class PPIEstimate:
    estimate: float
    """PPI (or PPI++ when tuned) estimate of the true pass rate of all N outputs, clipped to [0, 1]."""
    raw_estimate: float
    """The estimate before clipping; it can leave [0, 1] on a small audit (a warning says so)."""
    interval: tuple[float, float]
    """The normal (CLT) interval from the paper, clipped to [0, 1]; the exact one where the CLT one is degenerate."""
    interval_method: str
    """`'clt'`; `'exact'` when the CLT interval had zero width on a partial audit; `'census'` when n = N."""
    exact_interval: tuple[float, float]
    """Finite-sample interval for plain PPI: Clopper-Pearson on the false-fail and false-pass rates."""
    n_audit: int
    n_total: int
    judge_only: float
    """The judge's pass rate on all N: what you would report without people, biased by the judge's mistakes."""
    human_only: float
    human_only_interval: tuple[float, float]
    """Wilson interval on the audit alone (no finite population correction, so slightly wide when n/N is large)."""
    width_ratio: float | None
    """Width of `interval` over width of `human_only_interval`: below 1, the judge saved labels."""
    lam: float
    """Weight on the judge: 1 for plain PPI, chosen from the audit and clipped to [0, 1] when tuned."""
    tuned: bool
    false_fails: int
    """Audited outputs people passed and the judge failed."""
    false_passes: int
    """Audited outputs people failed and the judge passed."""
    alpha: float
    warnings: tuple[str, ...] = ()

    def summary(self) -> str:
        name = 'PPI++' if self.tuned else 'PPI'
        return (
            f'{name}: {self.estimate:.3f} {_iv(self.interval)} (exact {_iv(self.exact_interval)}) from {self.n_audit} '
            f'labels and {self.n_total} judged outputs; judge alone {self.judge_only:.3f}, people alone '
            f'{self.human_only:.3f} {_iv(self.human_only_interval)}' + ''.join(f'\n- {w}' for w in self.warnings)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            'estimate': self.estimate,
            'raw_estimate': self.raw_estimate,
            'interval': list(self.interval),
            'interval_method': self.interval_method,
            'exact_interval': list(self.exact_interval),
            'n_audit': self.n_audit,
            'n_total': self.n_total,
            'judge_only': self.judge_only,
            'human_only': self.human_only,
            'human_only_interval': list(self.human_only_interval),
            'width_ratio': self.width_ratio,
            'lam': self.lam,
            'tuned': self.tuned,
            'false_fails': self.false_fails,
            'false_passes': self.false_passes,
            'alpha': self.alpha,
            'warnings': list(self.warnings),
        }


def ppi_pass_rate(
    judge_verdicts: Mapping[str, bool],
    audit: Mapping[str, bool],
    *,
    sample: AuditSample,
    alpha: float = 0.05,
    tuned: bool = True,
) -> PPIEstimate:
    """The true pass rate of every judged output, from the judge on all of them and people on `sample`.

    Args:
        judge_verdicts: Case name to the judge's verdict, for every output (True: passed).
        audit: Case name to the person's verdict, for exactly the cases in `sample`.
        sample: What `audit_sample` drew from `judge_verdicts`' case names.
        alpha: One minus the intervals' coverage.
        tuned: PPI++ (weight on the judge chosen from the audit) rather than plain PPI.
    """
    if not 0 < alpha < 1:
        raise ValueError('alpha must be between 0 and 1')
    if sample.digest != _digest(judge_verdicts) or sample.population != len(judge_verdicts):
        raise ValueError('the audit sample was drawn from a different set of outputs than the judge scored')
    if audit_sample(judge_verdicts, len(sample.cases), sample.seed).cases != sample.cases:
        raise ValueError('the audit sample is not the draw audit_sample makes from its seed: it was edited')
    extra, missing = sorted(set(audit) - set(sample.cases)), sorted(set(sample.cases) - set(audit))
    if extra:
        raise ValueError(
            f'labels for {len(extra)} cases that were not drawn (e.g. {extra[:3]}): only a uniform random audit '
            "measures the judge's bias on all outputs; a queue of chosen cases (the judge's failures, say) "
            'measures it on those cases only'
        )
    if missing:
        raise ValueError(
            f'{len(missing)} drawn cases have no label (e.g. {missing[:3]}): leaving out the ones that are hard '
            'to label makes the audit non-uniform; label them, or draw a smaller audit'
        )
    pairs = [(bool(audit[c]), bool(judge_verdicts[c])) for c in sample.cases]
    return _estimate(sum(bool(v) for v in judge_verdicts.values()), len(judge_verdicts), pairs, alpha, tuned)


def _estimate(
    judge_passes: int, total: int, pairs: Sequence[tuple[bool, bool]], alpha: float, tuned: bool
) -> PPIEstimate:
    """The arithmetic, on counts: the judge's passes on all N, and (human, judge) for each audited output.

    Split out so `bench/ppi.py` can run thousands of populations without building their dicts.
    """
    n = len(pairs)
    q = judge_passes / total
    human = [float(h) for h, _ in pairs]
    judge = [float(j) for _, j in pairs]
    mean_h, mean_j = sum(human) / n, sum(judge) / n
    var_j = _cov(judge, judge)
    lam = 1.0
    if tuned and var_j:
        # A judge constant on the audit leaves lam undefined; it then changes no variance, and plain
        # PPI (lam = 1) is the unbiased choice. Falling back to 0 instead discarded a perfect judge.
        lam = min(1.0, max(0.0, _cov(human, judge) / var_j))
    raw = mean_h + lam * (q - mean_j)
    estimate = min(1.0, max(0.0, raw))
    residual = [h - lam * j for h, j in zip(human, judge, strict=True)]
    se = math.sqrt(max(0.0, 1 - n / total) * _cov(residual, residual) / n)
    z = NormalDist().inv_cdf(1 - alpha / 2)

    false_fails = sum(h and not j for h, j in pairs)
    false_passes = sum(j and not h for h, j in pairs)
    warnings: list[str] = []
    if n >= total:
        # A census: q is the judge's rate on the audit itself, so the estimate is people's rate, with no error.
        method, interval = 'census', (estimate, estimate)
        exact = interval
    else:
        ff_low, ff_high = clopper_pearson(false_fails, n, alpha / 2)
        fp_low, fp_high = clopper_pearson(false_passes, n, alpha / 2)
        exact = _ordered(q + ff_low - fp_high, estimate, q + ff_high - fp_low)
        clt = _ordered(raw - z * se, estimate, raw + z * se)
        if clt[1] > clt[0]:
            method, interval = 'clt', clt
        else:
            method, interval = 'exact', exact
            if not false_fails + false_passes:
                warnings.append(
                    f'the judge agreed with people on all {n} audited outputs, so the CLT interval has zero width, '
                    'which is not evidence that it never errs: interval is the finite-sample exact_interval'
                )
            else:
                warnings.append(
                    f'the audit of {n} shows no variation in the rectifier (lam = {lam:.2f}), so the CLT interval '
                    'has zero width: interval is the finite-sample exact_interval'
                )
    if not 0.0 <= raw <= 1.0:
        warnings.append(
            f'the unclipped estimate {raw:.3f} is outside [0, 1] (the audit disagrees with the judge more than its '
            'size can carry): it was clipped, and clipping is not unbiased; audit more outputs'
        )
    if tuned and lam == 0.0 and false_fails + false_passes:
        warnings.append('on this audit the judge carries no information about the labels: the estimate is people alone')

    human_interval = wilson(sum(h for h, _ in pairs), n, z)
    human_width = human_interval[1] - human_interval[0]
    return PPIEstimate(
        estimate=estimate,
        raw_estimate=raw,
        interval=interval,
        interval_method=method,
        exact_interval=exact,
        n_audit=n,
        n_total=total,
        judge_only=q,
        human_only=mean_h,
        human_only_interval=human_interval,
        width_ratio=(interval[1] - interval[0]) / human_width if human_width else None,
        lam=lam,
        tuned=tuned,
        false_fails=false_fails,
        false_passes=false_passes,
        alpha=alpha,
        warnings=tuple(warnings),
    )


def _ordered(low: float, estimate: float, high: float) -> tuple[float, float]:
    """`(low, high)` clipped to [0, 1] and widened, if need be, to hold `estimate` (itself in [0, 1])."""
    return min(estimate, max(0.0, low)), max(estimate, min(1.0, high))


def _cov(a: Sequence[float], b: Sequence[float]) -> float:
    """Sample covariance (n - 1 denominator)."""
    n = len(a)
    if n < 2:
        return 0.0
    ma, mb = sum(a) / n, sum(b) / n
    return sum((x - ma) * (y - mb) for x, y in zip(a, b, strict=True)) / (n - 1)


def _iv(interval: tuple[float, float]) -> str:
    return f'[{interval[0]:.3f}, {interval[1]:.3f}]'
