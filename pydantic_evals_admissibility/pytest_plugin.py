"""A pytest plugin: a test suite uses a judge only once it is certified for what the tests do with it.

    def test_answers_are_correct(certify_judge):
        judge = certify_judge(make_judge(), CASES, decision='gate')
        ...  # use `judge` as usual: it is the same object, now known to be qualified

    async def test_answers_are_correct_async(certify_judge):
        judge = await certify_judge.acertify(make_judge(), CASES, decision='gate')

`certify_judge` certifies the judge once per session, keyed by its `judge_identity` fingerprint
and a hash of the judgments the certificate is made of: every planned (case, role, output the
judge sees, inputs, expected output, metadata), the slice each case falls in, `batch_size`,
thresholds, `assertion` and the plugin's certification version. Fifty tests that use one judge
pay for one certificate; a test that changes what the judge would be shown (a control rewritten,
another slice function) gets a certificate of its own. The judgments are planned once: the key
is the hash of the very plan the judge is then certified on. Two tests asking for the same
certificate at once (tasks of one async test, a thread beside them) share one certification: the
others wait for it. It then asks `qualify` whether that certificate is enough for `decision`
(`report`, `gate`, `promote`, `steer`). A judge that is not qualified fails the test with the
qualification's reasons, the certificate table and the doctor's advice, or skips it
(`--judge-unqualified=skip`).

A judge whose identity is not reliable (`identity_reliable` is False: its configuration cannot be
told apart from another judge's) is certified again every time it is asked for, and its
certificate is never saved: the summary says so.

Sync and async tests: `certify_judge(...)` runs the certification on a fresh event loop, so it is
for sync tests and sync fixtures; called inside a running loop it raises and points to
`await certify_judge.acertify(...)`, which certifies on the test's own loop (the judge may use
the loop's futures, clients bound to it, and the test's context variables). A session-scoped
fixture can use `judge_certificates.certify(...)` (or `await judge_certificates.acertify(...)`
from an async one) with the same arguments, for a judge built once and shared.

    @pytest.mark.judge_certified(decision='promote', context={'slice': 'refunds'})
    def test_optimizer_keeps_the_better_prompt(certify_judge): ...

The marker sets the defaults of `certify_judge` in that test (any `qualify` argument: `decision`,
`context`, `max_age_days`); arguments passed to `certify_judge` win.

Options (each also an ini key without the dashes, e.g. `judge_certificates = certs.json`):

- `--judge-certificates=PATH`: use the certificates saved in PATH instead of calling the judge.
  A judge with no saved certificate for its configuration and cases fails: in CI, a judge that
  changed must be certified again, not trusted on its old evidence. A saved certificate of the
  same name for another configuration is refused by `qualify`'s coverage rule, which names what
  changed.
- `--judge-recertify`: certify every judge now, ignoring saved certificates. With
  `--judge-certificates`, the certificates made in the session are written to PATH at the end
  (atomically: a reader never sees half a file), so
  `pytest --judge-recertify --judge-certificates=certs.json` makes the file CI then reads.
- `--judge-unqualified=fail|skip`: what to do with a test whose judge is not qualified
  (default `fail`).

With pytest-xdist, each worker is its own process with its own cache: a judge is certified once
PER WORKER that uses it, not once per run (save a file with `--judge-recertify` and run CI from
it to avoid paying for that). Workers send what they certified to the controller, which prints
the one summary and writes the file once.

The terminal summary lists every judge the session used: verdict, fingerprint, checks, judge
calls made in this session, and each decision it was asked to qualify for, so a CI log shows
what was trusted.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import dataclasses
import hashlib
import json
import os
import random
import tempfile
import threading
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import pytest

if TYPE_CHECKING:
    from ._cases import HumanLabel, JudgeCase
    from ._certify import Certificate, Thresholds
    from ._controls import Control
    from ._qualify import Decision, Qualification

OnUnqualified = Literal['fail', 'skip']
FILE_VERSION = 2
"""The certificate file's format. Version 1 keys did not cover the materialized judgments, so its certificates are
refused rather than reused."""
CERTIFICATION_VERSION = 3
"""Part of every cache key: bump it when `certify_judge` plans or assesses judgments differently, so a certificate
made by the old algorithm is not reused as if the new one had made it. Version 2: the certificate is made of the
plan its key hashes (version 1 planned again to certify, so its key could describe other judgments). Version 3:
the plan is keyed in the cache's canonical form, which keeps instance state outside fields and private attributes."""
_REGISTRY = pytest.StashKey['JudgeCertificates']()
_WORKER_OUTPUT = 'judge_admissibility_entries'


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup('judge-admissibility', 'certify LLM judges before their verdicts are used')
    group.addoption(
        '--judge-certificates',
        metavar='PATH',
        help='use the judge certificates saved in PATH instead of calling the judges '
        '(with --judge-recertify: write the certificates made in this session to PATH)',
    )
    group.addoption(
        '--judge-recertify', action='store_true', default=None, help='certify every judge again, ignoring saved ones'
    )
    group.addoption(
        '--judge-unqualified',
        choices=('fail', 'skip'),
        help='what to do with a test whose judge is not qualified for its decision (default: fail)',
    )
    parser.addini('judge_certificates', 'saved judge certificates; see --judge-certificates', default='')
    parser.addini('judge_recertify', 'certify every judge again; see --judge-recertify', type='bool', default=False)
    parser.addini('judge_unqualified', 'fail or skip a test whose judge is not qualified', default='fail')


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        'markers',
        'judge_certified(decision=..., context=..., max_age_days=...): '
        'defaults for the `certify_judge` fixture in this test',
    )
    path = config.getoption('judge_certificates') or config.getini('judge_certificates') or None
    recertify = config.getoption('judge_recertify')
    on_unqualified = config.getoption('judge_unqualified') or config.getini('judge_unqualified')
    if on_unqualified not in ('fail', 'skip'):
        raise pytest.UsageError(f'judge_unqualified must be fail or skip, got {on_unqualified!r}')
    config.stash[_REGISTRY] = JudgeCertificates(
        Path(config.rootpath, path) if path else None,
        recertify=bool(config.getini('judge_recertify') if recertify is None else recertify),
        on_unqualified=on_unqualified,
        worker=hasattr(config, 'workerinput'),
    )


@dataclasses.dataclass
class Entry:
    """One judge the session used: its certificate, where it came from, and what it was asked to do."""

    name: str
    key: str
    evidence: str
    certificate: Certificate
    source: Literal['certified', 'saved']
    issued_at: datetime | None
    calls: int
    """Judge calls made in this session to certify it: 0 when the certificate was loaded."""
    uses: dict[str, list[bool]] = dataclasses.field(default_factory=dict)
    """Decision asked for, and whether each use was qualified."""
    warnings: dict[str, None] = dataclasses.field(default_factory=dict)
    """What a qualified use must still disclose (`Qualification.warnings`), once each."""
    reliable: bool = True
    """Whether the judge's identity tells it apart from other judges; if not, the certificate is never reused."""
    certifications: int = 0
    """How many times it was certified in this session: more than 1 under xdist (once per worker), or when its
    identity is not reliable (once per use)."""
    opaque: list[str] = dataclasses.field(default_factory=list)
    """Why its identity is not reliable: what in its configuration the identity could not capture."""

    def to_record(self) -> dict[str, Any]:
        """Plain data, for an xdist worker to send to the controller."""
        return {
            'name': self.name,
            'key': self.key,
            'evidence': self.evidence,
            'certificate': self.certificate.to_dict(),
            'source': self.source,
            'issued_at': self.issued_at.isoformat() if self.issued_at else None,
            'calls': self.calls,
            'uses': self.uses,
            'warnings': list(self.warnings),
            'reliable': self.reliable,
            'certifications': self.certifications,
            'opaque': self.opaque,
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> Entry:
        from ._certify import Certificate

        return cls(
            record['name'],
            record['key'],
            record['evidence'],
            Certificate.from_dict(record['certificate']),
            record['source'],
            datetime.fromisoformat(record['issued_at']) if record['issued_at'] else None,
            record['calls'],
            {decision: list(ok) for decision, ok in record['uses'].items()},
            dict.fromkeys(record['warnings']),
            record['reliable'],
            record['certifications'],
            list(record.get('opaque', [])),
        )


@dataclasses.dataclass
class _Request:
    """One call to `certify`: the judge, what it is certified on, and the key its certificate is cached under."""

    judge: Any
    label: str
    key: str
    fingerprint: str
    evidence: str
    reliable: bool
    opaque: list[str]
    run: Callable[[], Any]
    """Makes the coroutine that certifies the judge on exactly the judgments `evidence` hashes."""


class _Abandoned(Exception):
    """The certification another caller was waiting on was cancelled or interrupted: certify it yourself."""


@dataclasses.dataclass
class _InFlight:
    """A certification under way, shared by every caller that asks for the same key meanwhile."""

    future: concurrent.futures.Future[Entry]
    thread: int
    """The thread certifying: a sync caller on that thread cannot block on it (it would wait on itself)."""


class JudgeCertificates:
    """The session's certificates: each judge is certified once, or loaded from a saved file."""

    def __init__(
        self,
        path: Path | None = None,
        *,
        recertify: bool = False,
        on_unqualified: OnUnqualified = 'fail',
        worker: bool = False,
    ):
        self.path = path
        self.recertify = recertify
        self.on_unqualified: OnUnqualified = on_unqualified
        self.worker = worker
        """Whether this is an xdist worker: it sends its entries to the controller instead of writing the file."""
        self.entries: dict[str, Entry] = {}
        self._saved: list[dict[str, Any]] | None = None
        self.written: Path | None = None
        self._inflight: dict[str, _InFlight] = {}
        self._lock = threading.Lock()

    def certify(
        self,
        judge: Any,
        cases: Sequence[JudgeCase],
        *,
        decision: Decision = 'gate',
        name: str | None = None,
        controls: Sequence[Control] | None = None,
        human_labels: Sequence[HumanLabel] = (),
        repeats: int = 3,
        thresholds: Thresholds | None = None,
        assertion: str | None = None,
        seed: int = 0,
        batch_size: int | None = None,
        slice_by: Callable[[JudgeCase], str] | None = None,
        max_concurrency: int = 8,
        context: Mapping[str, Any] | None = None,
        max_age_days: float | None = None,
        on_unqualified: OnUnqualified | None = None,
    ) -> Any:
        """Return `judge` if it is qualified for `decision`; otherwise fail (or skip) the calling test.

        The certificate arguments are `certify_judge`'s; `decision`, `context` and `max_age_days`
        are `qualify`'s. `name` labels the judge in the summary and in the saved file (by default
        its class and model) and is part of the cache key: give two judges with the same
        configuration but different behaviour (scripted `FunctionModel` judges, say) different names.

        For sync code: the judge is certified on a new event loop. In an async test, use `acertify`.
        """
        request = self._request(
            judge, cases, name, controls, human_labels, repeats, thresholds, assertion, seed, batch_size, slice_by,
            max_concurrency,
        )  # fmt: skip
        entry = self._cached(request)
        if entry is None:
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                entry = self._certify_sync(request)
            else:
                raise RuntimeError(
                    'judge-admissibility: certify_judge(...) was called inside a running event loop, where it '
                    'cannot certify without blocking that loop. In an async test, use '
                    '`await certify_judge.acertify(...)` (or `await judge_certificates.acertify(...)`).'
                )
        return self._qualify(request, entry, decision, context, max_age_days, on_unqualified)

    async def acertify(
        self,
        judge: Any,
        cases: Sequence[JudgeCase],
        *,
        decision: Decision = 'gate',
        name: str | None = None,
        controls: Sequence[Control] | None = None,
        human_labels: Sequence[HumanLabel] = (),
        repeats: int = 3,
        thresholds: Thresholds | None = None,
        assertion: str | None = None,
        seed: int = 0,
        batch_size: int | None = None,
        slice_by: Callable[[JudgeCase], str] | None = None,
        max_concurrency: int = 8,
        context: Mapping[str, Any] | None = None,
        max_age_days: float | None = None,
        on_unqualified: OnUnqualified | None = None,
    ) -> Any:
        """`certify`, for async tests: the judge is certified on the running loop, in the caller's context."""
        request = self._request(
            judge, cases, name, controls, human_labels, repeats, thresholds, assertion, seed, batch_size, slice_by,
            max_concurrency,
        )  # fmt: skip
        entry = self._cached(request)
        if entry is None:
            entry = await self._certify_async(request)
        return self._qualify(request, entry, decision, context, max_age_days, on_unqualified)

    def _claim(self, request: _Request) -> tuple[Entry | None, _InFlight | None, bool]:
        """The cached entry; else the certification under way for this key and whether this caller runs it.

        A judge whose identity is not reliable is never shared: each caller certifies it.
        """
        with self._lock:
            entry = self._cached(request)
            if entry is not None:
                return entry, None, False
            flight = self._inflight.get(request.key) if request.reliable else None
            if flight is not None:
                return None, flight, False
            flight = _InFlight(concurrent.futures.Future(), threading.get_ident())
            if request.reliable:
                self._inflight[request.key] = flight
            return None, flight, True

    def _land(self, request: _Request, flight: _InFlight, certificate: Certificate) -> Entry:
        """Record the certificate, release the key, and hand the entry to the callers waiting on it."""
        with self._lock:
            if self._inflight.get(request.key) is flight:
                del self._inflight[request.key]
            entry = self._certified(request, certificate)
            flight.future.set_result(entry)
            return entry

    def _crash(self, request: _Request, flight: _InFlight, error: BaseException) -> None:
        """Release the key of a certification that raised: a judge that raised raises for every caller waiting on
        it; one cancelled or interrupted leaves each waiter to certify it itself."""
        with self._lock:
            if self._inflight.get(request.key) is flight:
                del self._inflight[request.key]
            flight.future.set_exception(error if isinstance(error, Exception) else _Abandoned())

    def _certify_sync(self, request: _Request) -> Entry:
        while True:
            entry, flight, mine = self._claim(request)
            if entry is not None:
                return entry
            assert flight is not None
            if not mine and flight.thread == threading.get_ident():
                # Under way on a loop of this very thread that is not running now: waiting would never end.
                flight, mine = _InFlight(concurrent.futures.Future(), flight.thread), True
            if not mine:
                try:
                    return flight.future.result()
                except _Abandoned:
                    continue
            try:
                certificate = asyncio.run(request.run())
            except BaseException as error:
                self._crash(request, flight, error)
                raise
            return self._land(request, flight, certificate)

    async def _certify_async(self, request: _Request) -> Entry:
        while True:
            entry, flight, mine = self._claim(request)
            if entry is not None:
                return entry
            assert flight is not None
            if not mine:
                try:
                    # Shielded: a waiter that is cancelled must not cancel the certification it shares.
                    return await asyncio.shield(asyncio.wrap_future(flight.future))
                except _Abandoned:
                    continue
            try:
                certificate = await request.run()
            except BaseException as error:
                self._crash(request, flight, error)
                raise
            return self._land(request, flight, certificate)

    def _request(
        self,
        judge: Any,
        cases: Sequence[JudgeCase],
        name: str | None,
        controls: Sequence[Control] | None,
        human_labels: Sequence[HumanLabel],
        repeats: int,
        thresholds: Thresholds | None,
        assertion: str | None,
        seed: int,
        batch_size: int | None,
        slice_by: Callable[[JudgeCase], str] | None,
        max_concurrency: int,
    ) -> _Request:
        from . import _identity
        from ._certify import DEFAULT_THRESHOLDS, _certify_planned, _plan
        from ._controls import DEFAULT_CONTROLS

        controls = DEFAULT_CONTROLS if controls is None else controls
        thresholds = DEFAULT_THRESHOLDS if thresholds is None else thresholds
        label = name or _describe(judge)
        identity = _identity.judge_identity(judge)
        judge_fingerprint = _identity.fingerprint(identity)
        reliable = getattr(_identity, 'identity_reliable', lambda identity: True)(identity)
        # Planned once: the key hashes these judgments, and the certificate is made of these same ones. A control
        # that makes other outputs when asked again (it holds state, it draws from a clock) is never planned twice.
        rng = random.Random(seed)
        planned = _plan(cases, controls, human_labels, repeats, rng)
        slices = {case.name: slice_by(case) for case in cases} if slice_by is not None else None
        evidence = _evidence_hash(planned, slices, repeats, thresholds, assertion, seed, batch_size)
        started: list[bool] = []

        def run() -> Any:
            if started:  # rng has been drawn from by a sequential certificate: a second run would differ
                raise RuntimeError('judge-admissibility: a planned certification runs once')
            started.append(True)
            return _certify_planned(
                judge,
                cases,
                planned,
                repeats=repeats,
                thresholds=thresholds,
                assertion=assertion,
                max_concurrency=max_concurrency,
                batch_size=batch_size,
                slices=slices,
                rng=rng,
            )

        key = f'{label}|{judge_fingerprint}|{evidence}'
        opaque = [str(o) for o in identity.get('opaque') or ()] if not reliable else []
        return _Request(judge, label, key, judge_fingerprint, evidence, bool(reliable), opaque, run)

    def _cached(self, request: _Request) -> Entry | None:
        """The certificate this session already has for the request, or a saved one; None if it must be certified."""
        if not request.reliable:
            return None  # its identity does not tell it apart from another judge: never reuse a certificate
        entry = self.entries.get(request.key)
        if entry is None and self.path is not None and not self.recertify:
            entry = self.entries[request.key] = self._load(request.label, request.key, request.fingerprint)
        return entry

    def _certified(self, request: _Request, certificate: Certificate) -> Entry:
        now = datetime.now(timezone.utc)
        calls = certificate.calls or 0
        entry = self.entries.get(request.key)
        if entry is None:
            entry = Entry(request.label, request.key, request.evidence, certificate, 'certified', now, 0)
            entry.reliable, entry.opaque = request.reliable, request.opaque
            self.entries[request.key] = entry
        else:  # an unreliable identity, certified again: the latest certificate decides
            entry.certificate, entry.issued_at = certificate, now
        entry.calls += calls
        entry.certifications += 1
        return entry

    def _qualify(
        self,
        request: _Request,
        entry: Entry,
        decision: Decision,
        context: Mapping[str, Any] | None,
        max_age_days: float | None,
        on_unqualified: OnUnqualified | None,
    ) -> Any:
        from ._qualify import qualify

        q = qualify(
            request.judge,
            entry.certificate,
            decision=decision,
            context=context,
            max_age_days=max_age_days,
            issued_at=entry.issued_at,
        )
        entry.uses.setdefault(decision, []).append(q.qualified)
        entry.warnings.update(dict.fromkeys(f'{decision}: {w}' for w in q.warnings))
        if not q.qualified:
            message = _refusal(request.label, q, entry, request.judge)
            if (on_unqualified or self.on_unqualified) == 'skip':
                pytest.skip(message)
            pytest.fail(message, pytrace=False)
        return request.judge

    def _load(self, label: str, key: str, judge_fingerprint: str) -> Entry:
        """The saved certificate for this key; failing that, one saved under the same name, for `qualify` to refuse."""
        from ._certify import Certificate

        assert self.path is not None
        if self._saved is None:
            if not self.path.exists():
                pytest.fail(
                    f'judge-admissibility: no certificate file at {self.path}; make it with --judge-recertify',
                    pytrace=False,
                )
            data = json.loads(self.path.read_text())
            if data.get('version') != FILE_VERSION:
                pytest.fail(
                    f'judge-admissibility: {self.path} was written by another version of the plugin '
                    f'(file version {data.get("version")}, this one reads {FILE_VERSION}). '
                    'Certify again with --judge-recertify.',
                    pytrace=False,
                )
            self._saved = list(data['certificates'])
        same = [s for s in self._saved if s['key'] == key] or [s for s in self._saved if s['name'] == label]
        if not same:
            known = ', '.join(sorted({s['name'] for s in self._saved})) or 'none'
            pytest.fail(
                f'judge-admissibility: {self.path} has no certificate for {label} (saved: {known}). '
                'Certify it with --judge-recertify.',
                pytrace=False,
            )
        saved = same[0]
        if saved['key'] != key and saved['fingerprint'] == judge_fingerprint:
            pytest.fail(
                f'judge-admissibility: the certificate for {label} in {self.path} was made on other judgments than '
                'this test asks for (cases, controls, human labels, slices, batch size, thresholds or the '
                'certification version). Certify it again with --judge-recertify.',
                pytrace=False,
            )
        when = saved.get('issued_at')
        return Entry(
            label,
            key,
            saved['evidence'],
            Certificate.from_dict(saved['certificate']),
            'saved',
            datetime.fromisoformat(when) if when else None,
            0,
        )

    def merge(self, records: Sequence[dict[str, Any]]) -> None:
        """Add the entries an xdist worker sent: the same judge from two workers is one entry, certified twice."""
        for record in records:
            new = Entry.from_record(record)
            old = self.entries.get(new.key)
            if old is None:
                self.entries[new.key] = new
                continue
            if new.source == 'certified' and old.source == 'saved':
                old.certificate, old.source, old.issued_at = new.certificate, 'certified', new.issued_at
            old.calls += new.calls
            old.certifications += new.certifications
            old.reliable = old.reliable and new.reliable
            old.opaque += [o for o in new.opaque if o not in old.opaque]
            for decision, ok in new.uses.items():
                old.uses.setdefault(decision, []).extend(ok)
            old.warnings.update(new.warnings)

    def save(self) -> Path | None:
        """Write the certificates made in this session to `path` (with `recertify`), keeping the others.

        The file is replaced atomically: a reader sees the old file or the new one, never half of one. An
        xdist worker never writes it: the controller does, once, with every worker's certificates.
        A certificate of a judge whose identity is not reliable is not written: it could never be reused.
        """
        made = [e for e in self.entries.values() if e.source == 'certified' and e.reliable]
        if self.worker or self.path is None or not self.recertify or not made:
            return None
        kept: list[dict[str, Any]] = []
        if self.path.exists():
            data = json.loads(self.path.read_text())
            if data.get('version') == FILE_VERSION:
                keys = {e.key for e in made}
                kept = [s for s in data['certificates'] if s['key'] not in keys]
        records = kept + [
            {
                'name': e.name,
                'key': e.key,
                'fingerprint': e.certificate.fingerprint,
                'evidence': e.evidence,
                'issued_at': e.issued_at.isoformat() if e.issued_at else None,
                'certificate': e.certificate.to_dict(),
            }
            for e in made
        ]
        _write_atomically(self.path, json.dumps({'version': FILE_VERSION, 'certificates': records}, indent=2))
        return self.path

    def summary(self) -> list[str]:
        """One block per judge: verdict, fingerprint, checks, calls, and each decision it was used for."""
        lines: list[str] = []
        for e in self.entries.values():
            c = e.certificate
            lines.append(f'{e.name}: {c.verdict}  fingerprint {c.fingerprint or "-"}')
            lines.append('  checks: ' + ', '.join(f'{check.name} {check.status}' for check in c.checks))
            if not e.reliable:
                lines.append(
                    f'  calls: {e.calls} (certified {_times(e.certifications)} in this session: its identity is not '
                    'reliable, so its certificate is never reused or saved)'
                )
                lines += [f'  not identified: {o}' for o in e.opaque]
            elif e.source == 'certified' and e.certifications > 1:
                lines.append(
                    f'  calls: {e.calls} (certified {_times(e.certifications)} in this session, once per xdist worker)'
                )
            elif e.source == 'certified':
                lines.append(f'  calls: {e.calls} (certified in this session)')
            else:
                lines.append(f'  calls: 0 (saved certificate, {c.calls} calls when made {_date(e.issued_at)})')
            used = [
                f'{decision} {"qualified" if all(ok) else "NOT qualified"} (asked {_times(len(ok))})'
                for decision, ok in e.uses.items()
            ]
            lines.append('  used for: ' + ('; '.join(used) or 'nothing'))
            lines += [f'  warning: {w}' for w in e.warnings]
        return lines


class CertifyJudge:
    """The `certify_judge` fixture: call it from a sync test, `await certify_judge.acertify(...)` from an async one."""

    def __init__(self, registry: JudgeCertificates, defaults: Mapping[str, Any]):
        self.registry = registry
        self.defaults = dict(defaults)

    def __call__(self, judge: Any, cases: Sequence[JudgeCase], **kwargs: Any) -> Any:
        """`JudgeCertificates.certify` with the test marker's defaults: certifies on a new event loop."""
        return self.registry.certify(judge, cases, **{**self.defaults, **kwargs})

    async def acertify(self, judge: Any, cases: Sequence[JudgeCase], **kwargs: Any) -> Any:
        """`JudgeCertificates.acertify` with the test marker's defaults: certifies on the test's own loop."""
        return await self.registry.acertify(judge, cases, **{**self.defaults, **kwargs})


@pytest.fixture(scope='session')
def judge_certificates(pytestconfig: pytest.Config) -> JudgeCertificates:
    """The session's judge certificates; `judge_certificates.certify(judge, cases, ...)` for a shared fixture."""
    return pytestconfig.stash[_REGISTRY]


@pytest.fixture
def certify_judge(request: pytest.FixtureRequest, judge_certificates: JudgeCertificates) -> CertifyJudge:
    """`certify_judge(judge, cases, decision='gate', ...)` returns the judge once it is qualified, or fails the test.

    In an async test: `await certify_judge.acertify(judge, cases, ...)`. Defaults come from the
    test's `@pytest.mark.judge_certified(...)` marker, if any.
    """
    marker = request.node.get_closest_marker('judge_certified')
    defaults = dict(marker.kwargs) if marker else {}
    if marker and marker.args:
        defaults.setdefault('decision', marker.args[0])
    return CertifyJudge(judge_certificates, defaults)


def pytest_terminal_summary(terminalreporter: Any, config: pytest.Config) -> None:
    registry = config.stash.get(_REGISTRY, None)
    if registry is None or registry.worker or not registry.entries:
        return
    terminalreporter.write_sep('=', 'judge certificates')
    for line in registry.summary():
        terminalreporter.write_line(line)
    if registry.written is not None:
        terminalreporter.write_line(f'certificates written to {registry.written}')


def pytest_sessionfinish(session: pytest.Session) -> None:
    registry = session.config.stash.get(_REGISTRY, None)
    if registry is None:
        return
    if registry.worker:
        workeroutput = getattr(session.config, 'workeroutput', None)
        if workeroutput is not None:
            workeroutput[_WORKER_OUTPUT] = json.dumps([e.to_record() for e in registry.entries.values()])
        return
    registry.written = registry.save()


@pytest.hookimpl(optionalhook=True)
def pytest_testnodedown(node: Any, error: Any) -> None:
    """On the xdist controller: collect what a finished worker certified (`pytest_sessionfinish` sent it)."""
    records = getattr(node, 'workeroutput', {}).get(_WORKER_OUTPUT)
    registry = node.config.stash.get(_REGISTRY, None)
    if records and registry is not None:
        registry.merge(json.loads(records))


def _write_atomically(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f'.{path.name}.', suffix='.tmp')
    try:
        with os.fdopen(fd, 'w') as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        os.unlink(tmp)
        raise


def _refusal(label: str, q: Qualification, entry: Entry, judge: Any) -> str:
    from ._diagnose import diagnose

    certificate = entry.certificate
    origin = 'certified in this session' if entry.source == 'certified' else 'saved certificate'
    lines = [f'judge-admissibility: {label} is not qualified to {q.decision} ({origin}, fingerprint {q.fingerprint})']
    lines += [f'- {reason}' for reason in q.reasons]
    lines += ['', certificate.table()]
    advice = diagnose(certificate, judge) if certificate.covers(judge) else []
    if advice:
        lines += ['', 'what to try:'] + [f'- {a}' for a in advice]
    return '\n'.join(lines)


def _describe(judge: Any) -> str:
    from ._certify import _describe

    return _describe(judge)


def _evidence_hash(
    planned: Sequence[tuple[JudgeCase, Any, str]],
    slices: Mapping[str, str] | None,
    repeats: int,
    thresholds: Thresholds,
    assertion: str | None,
    seed: int,
    batch_size: int | None,
) -> str:
    """What the certificate is made of, so it is not reused for other judgments, slices or rules.

    Hashes the materialized plan, not a description of it: every judgment `certify_judge` will ask
    for, with the output the judge is actually shown and the inputs, expected output and metadata
    of the case as the judge sees them. A control's name says nothing about what it does; the
    outputs it makes do. `planned` is the very list the certificate is then made of: hashing one
    plan and certifying on another would let a control that changes between calls slip through.
    """
    from ._cache import _canonical, repr_fallback  # pyright: ignore[reportPrivateUsage]

    data = {
        'certification': CERTIFICATION_VERSION,
        'judgments': [
            [case.name, role, output, case.inputs, case.expected_output, case.metadata]
            for case, output, role in planned
        ],
        'slices': [[name, kind] for name, kind in slices.items()] if slices is not None else None,
        'repeats': repeats,
        'seed': seed,  # the order a sequential certificate judges its batches in
        'batch_size': batch_size,
        'thresholds': thresholds,
        'assertion': assertion,
    }
    # The cache's canonical form: types, order and container-subclass state kept; `repr` only for
    # what has no faithful form, so state a judge can read is never dropped from the key.
    text = json.dumps(_canonical(data, repr_fallback, 'plan'), separators=(',', ':'), sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _times(n: int) -> str:
    return f'{n} time{"s" * (n != 1)}'


def _date(when: datetime | None) -> str:
    return when.date().isoformat() if when else 'at an unknown time'
