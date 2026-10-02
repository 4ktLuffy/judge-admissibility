"""Find where competent judges read a rubric differently, and the one sentence that would settle it.

When two strong judges (or two reviewers) give opposite verdicts on the same answer, each
consistently, neither is noisy: they are reading the rubric differently, and no amount of
certifying either judge resolves that. On pydantic-ai's example dataset, judges failed the same
"good" answers on every repeat while others passed them, each side with a reason that holds up
against the rubric's words ("second-person or friendly"). That is a rubric dispute, and the fix is
a clarification of the rubric, not a better judge.

`find_disagreements` works offline on verdicts already paid for, from saved certificates or
reviews:

- every rated output is an item; its disagreement is how evenly the raters split, each rater one
  vote (its majority), so repeats do not outvote a rater asked once;
- the split is attributed: **between raters** (each rater consistent, the raters opposed: a
  dispute about the rubric) or **within a rater** (one rater changes its verdict on the same
  output: noise, or a sensitivity to formatting);
- contested items are clustered by the words of their stated reasons (TF-IDF, cosine, average
  link; no model), and each cluster's competing readings are the words that separate the failing
  side's reasons from the passing side's, with quotes from each.

`clarify_disagreements` then asks an injected `clarify(rubric, cluster)` (a model, in practice) for
one sentence per cluster, for at most `max_calls` clusters. The clustering and the readings are the
evidence; the sentence is a proposal to certify, not a fix.
"""

from __future__ import annotations

import inspect
import math
import re
from collections import Counter, defaultdict
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from ._certify import Certificate, Judgment

_WORD = re.compile(r"[a-z][a-z'\-]*[a-z]")
_STOP = frozenset(
    """
    a about above after again against all also although am an and any are aren't as at be because been before being
    below between both but by can cannot could did didn't do does doesn't doing don't down during each either few for
    from further had has have having he her here hers herself him himself his how however i if in into is isn't it
    it's its itself just let's may me might more most much must my myself no nor not now of off on once only or other
    our ours ourselves out over own same she should so some such than that that's the their theirs them themselves
    then there these they this those through thus to too under until up upon us very via was wasn't we were what when
    where which while who whom why will with within without would yet you're your yours yourself
    output response reply explanation field fields text message provided given one two also clearly appropriate
    """.split()
)  # 'you' is kept on purpose: whether the text says 'you' is what a second-person rubric is about


def tokens(text: str) -> list[str]:
    """Lower-case words of three letters or more, stop words removed, hyphenated words kept whole."""
    text = text.lower().replace('’', "'").replace('‘', "'").replace('‑', '-').replace('‐', '-')
    return [w.strip("'") for w in _WORD.findall(text) if len(w) >= 3 and w not in _STOP]


@dataclass(frozen=True)
class Rating:
    """One verdict by one rater (a judge configuration, a person) on one output."""

    item: str
    """What was rated. Two ratings with the same item are verdicts on the same thing."""
    rater: str
    passed: bool | None
    """None: the rater errored, which is not a verdict."""
    reason: str | None = None
    output: Any = None


def normalized(output: Any) -> str:
    """The output as text with whitespace collapsed: a reformatted copy is the same item."""
    return ' '.join((output if isinstance(output, str) else repr(output)).split())


def ratings_from_certificate(
    certificate: Certificate | Mapping[str, Any],
    *,
    rater: str,
    item: Callable[[Judgment], str | None] | None = None,
) -> list[Rating]:
    """Every judgment in a certificate (or its saved `to_dict()`) as a rating by `rater`.

    `item` names what each judgment rated; None leaves the judgment out. The default is the
    output with whitespace collapsed, so a whitespace control rates the same item as the answer
    it reformats, and another case's borrowed answer rates the donor's item: the right choice for
    a rubric about the output alone (style, tone); for a rubric about correctness, include the
    case name, since the same text can be right for one question and wrong for another.
    """
    cert = certificate if isinstance(certificate, Certificate) else Certificate.from_dict(dict(certificate))
    name = item or (lambda j: normalized(j.output))
    out = []
    for j in cert.judgments:
        key = name(j)
        if key is not None:
            out.append(Rating(key, rater, j.passed, j.reason, j.output))
    return out


@dataclass(frozen=True)
class ContestedItem:
    item: str
    output: Any = field(repr=False)
    by_rater: dict[str, tuple[int, int]]
    """Rater to (passes, verdicts)."""
    pass_reasons: tuple[tuple[str, str], ...] = field(repr=False)
    """(rater, reason) for each passing verdict with a reason."""
    fail_reasons: tuple[tuple[str, str], ...] = field(repr=False)

    @property
    def passes(self) -> int:
        return sum(p for p, _ in self.by_rater.values())

    @property
    def votes(self) -> int:
        return sum(n for _, n in self.by_rater.values())

    @property
    def split(self) -> float:
        """How evenly the raters split, each rater one vote: 1.0 half and half, 0.0 unanimous.

        A rater's vote is its majority verdict (half a pass on a tie), not its verdict count:
        repeats of one rater are evidence about its noise, not more opinions. Counting verdicts
        let three judges asked four times each outvote a dataset author twelve to one.
        """
        votes = [{True: 1.0, False: 0.0, None: 0.5}[self.majority(r)] for r in self.by_rater]
        p = sum(votes) / len(votes)
        return 2 * min(p, 1 - p)

    @property
    def instability(self) -> float:
        """Share of raters that gave both verdicts on this same item."""
        return sum(0 < passes < n for passes, n in self.by_rater.values()) / len(self.by_rater)

    def majority(self, rater: str) -> bool | None:
        passes, n = self.by_rater[rater]
        return None if passes * 2 == n else passes * 2 > n

    @property
    def between(self) -> bool:
        """Two raters' majorities are opposed: a dispute about what the rubric means."""
        return len({self.majority(r) for r in self.by_rater} - {None}) == 2

    @property
    def within(self) -> bool:
        """Some rater gave both verdicts on this same item: noise, or sensitivity to what was not normalized."""
        return any(0 < passes < n for passes, n in self.by_rater.values())

    @property
    def minority(self) -> tuple[str, ...]:
        """Raters whose majority differs from most raters' (none when the raters split evenly)."""
        majorities = {r: self.majority(r) for r in self.by_rater}
        yes = sum(m is True for m in majorities.values())
        no = sum(m is False for m in majorities.values())
        if yes == no:
            return ()
        return tuple(r for r, m in majorities.items() if m is (yes < no))

    @property
    def kind(self) -> str:
        if self.between and self.within:
            return 'between and within raters'
        if self.between:
            return 'between raters'
        return 'within a rater' if self.within else 'split, no rater majority'


@dataclass(frozen=True)
class Reading:
    """One side of a dispute: the words that set its reasons apart, and reasons quoted as given."""

    verdict: str
    terms: tuple[str, ...]
    quotes: tuple[tuple[str, str], ...]
    """(rater, reason)."""


@dataclass(frozen=True)
class Cluster:
    items: tuple[ContestedItem, ...]
    terms: tuple[str, ...]
    """The words that characterize the cluster's reasons (highest summed TF-IDF)."""
    fail: Reading
    passing: Reading
    clarification: str | None = None

    @property
    def between(self) -> int:
        return sum(i.between for i in self.items)


@dataclass(frozen=True)
class DisagreementReport:
    raters: tuple[str, ...]
    items: int
    """Items with at least `min_votes` verdicts."""
    contested: tuple[ContestedItem, ...]
    """Most contested first."""
    clusters: tuple[Cluster, ...]
    """Most contested first: by items disputed between raters, then by size."""

    def outliers(self) -> dict[str, tuple[int, int]]:
        """Per rater: (items disputed between raters where it was in the minority, such items it rated).

        A rubric dispute has competent raters on both sides. When one rater is the minority on
        nearly every disputed item, the dispute is more likely that rater (it cannot see something,
        or reads something no one else does) than the rubric.
        """
        out: dict[str, tuple[int, int]] = {r: (0, 0) for r in self.raters}
        for item in self.contested:
            if not item.between:
                continue
            minority = set(item.minority)
            for rater in item.by_rater:
                k, n = out[rater]
                out[rater] = (k + (rater in minority), n + 1)
        return out

    def table(self, *, quotes: int = 2, width: int = 160) -> str:
        between = sum(i.between for i in self.contested)
        rows = [
            f'{len(self.contested)} of {self.items} items contested ({between} between raters) across '
            f'{len(self.raters)} raters: {", ".join(self.raters)}',
            '',
        ]
        for i, item in enumerate(self.contested):
            votes = ', '.join(f'{r} {p}/{n}' for r, (p, n) in item.by_rater.items())
            rows.append(f'{i + 1:>2}. split {item.split:.2f}  {item.kind:<26} {votes}')
            rows.append(f'    {_clip(item.item, width)}')
        if between:
            rows += ['', 'in the minority on items disputed between raters:']
            rows += [f'  {r}: {k}/{n}' for r, (k, n) in self.outliers().items() if n]
        for c, cluster in enumerate(self.clusters):
            rows += ['', f'cluster {c + 1}: {len(cluster.items)} items ({cluster.between} between raters); '
                     f'about: {", ".join(cluster.terms)}']  # fmt: skip
            for reading in (cluster.fail, cluster.passing):
                rows.append(f'  {reading.verdict} reading: {", ".join(reading.terms) or "-"}')
                rows += [f'    {rater}: "{_clip(reason, width)}"' for rater, reason in reading.quotes[:quotes]]
            if cluster.clarification:
                rows.append(f'  proposed clarification: {cluster.clarification}')
        return '\n'.join(rows)

    def to_dict(self) -> dict[str, Any]:
        def item(i: ContestedItem) -> dict[str, Any]:
            return {
                'item': i.item,
                'split': i.split,
                'instability': i.instability,
                'kind': i.kind,
                'between': i.between,
                'within': i.within,
                'minority': list(i.minority),
                'by_rater': {r: {'passed': p, 'of': n} for r, (p, n) in i.by_rater.items()},
                'pass_reasons': [{'rater': r, 'reason': s} for r, s in i.pass_reasons],
                'fail_reasons': [{'rater': r, 'reason': s} for r, s in i.fail_reasons],
            }

        def reading(r: Reading) -> dict[str, Any]:
            return {'terms': list(r.terms), 'quotes': [{'rater': a, 'reason': s} for a, s in r.quotes]}

        return {
            'raters': list(self.raters),
            'items': self.items,
            'outliers': {r: {'minority': k, 'of': n} for r, (k, n) in self.outliers().items()},
            'contested': [item(i) for i in self.contested],
            'clusters': [
                {
                    'items': [i.item for i in c.items],
                    'between': c.between,
                    'terms': list(c.terms),
                    'fail_reading': reading(c.fail),
                    'pass_reading': reading(c.passing),
                    'clarification': c.clarification,
                }
                for c in self.clusters
            ],
        }


def _clip(text: str, width: int) -> str:
    text = ' '.join(text.split())
    return text if len(text) <= width else text[: width - 3] + '...'


def _vectors(docs: Sequence[list[str]]) -> list[dict[str, float]]:
    """TF-IDF, L2-normalized; idf is smoothed so a word in every document still counts a little."""
    n = len(docs)
    df = Counter(t for doc in docs for t in set(doc))
    out = []
    for doc in docs:
        tf = Counter(doc)
        v = {t: c * (math.log((1 + n) / (1 + df[t])) + 1) for t, c in tf.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        out.append({t: x / norm for t, x in v.items()})
    return out


def _cosine(a: Mapping[str, float], b: Mapping[str, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(x * b.get(t, 0.0) for t, x in a.items())


def _cluster(vectors: Sequence[Mapping[str, float]], threshold: float) -> list[list[int]]:
    """Average-link agglomerative clustering: merge the closest pair while it is at least `threshold` alike."""
    groups = [[i] for i in range(len(vectors))]
    sim = [[_cosine(a, b) for b in vectors] for a in vectors]
    while len(groups) > 1:
        best, pair = -1.0, (0, 0)
        for x in range(len(groups)):
            for y in range(x + 1, len(groups)):
                s = sum(sim[i][j] for i in groups[x] for j in groups[y]) / (len(groups[x]) * len(groups[y]))
                if s > best:
                    best, pair = s, (x, y)
        if best < threshold:
            break
        x, y = pair
        groups[x] = groups[x] + groups[y]
        del groups[y]
    return groups


def _distinctive(side: Sequence[str], other: Sequence[str], k: int) -> tuple[str, ...]:
    """Words more frequent on one side's reasons than the other's (smoothed log ratio), said at least twice."""
    a, b = Counter(t for r in side for t in set(tokens(r))), Counter(t for r in other for t in set(tokens(r)))
    na, nb = max(len(side), 1), max(len(other), 1)
    scored = [
        (math.log((a[t] + 0.5) / (na + 1)) - math.log((b[t] + 0.5) / (nb + 1)), a[t], t)
        for t in a
        if a[t] >= min(2, len(side))
    ]
    return tuple(t for s, _, t in sorted(scored, key=lambda x: (-x[0], -x[1], x[2])) if s > 0)[:k]


def _reading(verdict: str, reasons: Sequence[tuple[str, str]], other: Sequence[tuple[str, str]], k: int) -> Reading:
    terms = _distinctive([r for _, r in reasons], [r for _, r in other], k)
    seen: set[str] = set()
    ranked = sorted(reasons, key=lambda rr: (-len(set(tokens(rr[1])) & set(terms)), len(rr[1])))
    quotes = []
    for rater, reason in ranked:
        key = ' '.join(tokens(reason))
        if key not in seen:
            seen.add(key)
            quotes.append((rater, reason))
    # Different raters first: two phrasings by one rater are one reading.
    order = sorted(range(len(quotes)), key=lambda i: [q[0] for q in quotes[: i + 1]].count(quotes[i][0]))
    return Reading(verdict, terms, tuple(quotes[i] for i in order[:4]))


def find_disagreements(
    ratings: Iterable[Rating],
    *,
    min_votes: int = 2,
    threshold: float = 0.2,
    terms: int = 6,
) -> DisagreementReport:
    """Rank the items whose verdicts split, attribute each split, and cluster them by their reasons.

    Args:
        ratings: Verdicts, from any number of raters, possibly several per rater and item (repeats).
        min_votes: Items with fewer completed verdicts are left out.
        threshold: Cosine similarity of reasons (TF-IDF) at which two clusters merge.
        terms: Words reported per cluster and per reading.
    """
    grouped: dict[str, list[Rating]] = defaultdict(list)
    raters: dict[str, None] = {}
    for r in ratings:
        raters.setdefault(r.rater)
        if r.passed is not None:
            grouped[r.item].append(r)
    items = {k: v for k, v in grouped.items() if len(v) >= min_votes}
    contested: list[ContestedItem] = []
    for key, rs in items.items():
        if len({r.passed for r in rs}) < 2:
            continue
        by_rater: dict[str, tuple[int, int]] = {}
        for r in rs:
            p, n = by_rater.get(r.rater, (0, 0))
            by_rater[r.rater] = (p + bool(r.passed), n + 1)
        contested.append(
            ContestedItem(
                key,
                rs[0].output,
                by_rater,
                tuple((r.rater, r.reason) for r in rs if r.passed and r.reason),
                tuple((r.rater, r.reason) for r in rs if not r.passed and r.reason),
            )
        )
    contested.sort(key=lambda i: (-i.split, -i.instability, -i.votes, i.item))

    docs = [tokens(' '.join(reason for _, reason in i.pass_reasons + i.fail_reasons)) for i in contested]
    vectors = _vectors(docs)
    clusters: list[Cluster] = []
    for group in _cluster(vectors, threshold) if contested else []:
        members = tuple(contested[i] for i in sorted(group))
        weight: Counter[str] = Counter()
        for i in group:
            weight.update(vectors[i])
        fails = [fr for m in members for fr in m.fail_reasons]
        passes = [pr for m in members for pr in m.pass_reasons]
        clusters.append(
            Cluster(
                members,
                tuple(t for t, _ in weight.most_common(terms)),
                _reading('FAIL', fails, passes, terms),
                _reading('PASS', passes, fails, terms),
            )
        )
    clusters.sort(key=lambda c: (-c.between, -len(c.items), -max(i.split for i in c.items)))
    return DisagreementReport(tuple(raters), len(items), tuple(contested), tuple(clusters))


ClarifyResult = str
Clarify = Callable[[str, Cluster], ClarifyResult | Awaitable[ClarifyResult]]
"""`clarify(rubric, cluster)`: one sentence to add to the rubric that settles the cluster's dispute."""


async def clarify_disagreements(
    report: DisagreementReport, rubric: str, clarify: Clarify, *, max_calls: int = 2, between_only: bool = True
) -> DisagreementReport:
    """The report with a proposed clarification on its first `max_calls` clusters.

    With `between_only`, clusters with no item disputed between raters are skipped: a split
    within one rater is noise or a formatting sensitivity, which no rubric sentence settles.
    """
    clusters = list(report.clusters)
    calls = 0
    for i, cluster in enumerate(clusters):
        if calls >= max_calls:
            break
        if between_only and not cluster.between:
            continue
        result = clarify(rubric, cluster)
        if inspect.isawaitable(result):
            result = await result
        clusters[i] = replace(cluster, clarification=' '.join(str(result).split()))
        calls += 1
    return replace(report, clusters=tuple(clusters))
