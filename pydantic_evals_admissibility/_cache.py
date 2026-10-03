"""Re-certify after a dataset edit, paying only for the judgments the edit changed.

Certifying is the expensive part: every case is judged several times, and again under every
control. Most edits to a dataset touch a few cases, yet `certify_judge` asks the judge about all
of them again. `certify_judge_cached` keeps each verdict under a key made of everything the
verdict depends on, and asks the judge only about keys it has not seen.

The key is content-addressed, so a stored verdict is reused only for exactly what was judged:

- the judge: a hash of its full `judge_identity` (rubric, model, what it is shown, settings, the
  pydantic-evals version; a function-backed model by the `model_name` its owner gave it). Change
  any of them and every key changes; a verdict is never carried from one judge to another. A judge
  whose identity is not reliable (`identity_reliable`: a function-backed model without a name of
  its own, closed-over mutable state) is never cached at all.
- what the judge is shown: the case name, inputs, expected output and metadata as the
  `EvaluatorContext` carries them, and the output actually judged, each with its exact types and
  its order (`LLMJudge` writes a dict's items into its prompt in insertion order, and a custom
  evaluator can tell a tuple from a list). A value with no faithful form raises `TypeError`
  unless a `serializer` is given. For a control the output is the control's, not the case's: a
  `MismatchedOutput` borrows another case's answer through a seeded random choice, so editing the
  donor, or changing the seed, changes the key.
- the role (`reference#2`, `must_fail:empty_output`, `human:1`, ...) and how many identical items
  came before it in the plan. Repeats exist to measure the judge's noise, so `reference#1` must
  never be served the verdict of `reference#0`; two identical human labels get two judgments.

Errored judgments are not stored: an error is not a verdict, so it is asked again next time.

The certificate is built by the same planning and assessment as `certify_judge`, from the same
seed, so given the same verdicts it is the same certificate, `calls` included: `calls` counts the
judgments the certificate rests on, as in an uncached run. The calls the cache saved are in
`CacheStats`.
"""

from __future__ import annotations

import asyncio
import dataclasses
import datetime
import decimal
import enum
import hashlib
import json
import random
import re
import sqlite3
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import Any

import pydantic
from pydantic_evals.evaluators import Evaluator

from ._cases import HumanLabel, JudgeCase
from ._certify import (
    DEFAULT_THRESHOLDS,
    Certificate,
    Judgment,
    Thresholds,
    Verdict,
    _assess,  # pyright: ignore[reportPrivateUsage]
    _describe,  # pyright: ignore[reportPrivateUsage]
    _judge,  # pyright: ignore[reportPrivateUsage]
    _plan,  # pyright: ignore[reportPrivateUsage]
)
from ._controls import DEFAULT_CONTROLS, Control
from ._identity import (
    _class_digest,  # pyright: ignore[reportPrivateUsage]
    _function_digest,  # pyright: ignore[reportPrivateUsage]
    _instance_state,  # pyright: ignore[reportPrivateUsage]
    fingerprint,
    identity_reliable,
    judge_identity,
)

CACHE_FORMAT = 2
"""Part of every key: a change to how keys are built must not hit entries built the old way."""


@dataclass(frozen=True)
class CacheStats:
    """What the cache did on one certification."""

    hits: int
    """Judgments served from the cache: judge calls not made."""
    misses: int
    """Judgments the judge was asked for: the calls this run made."""
    errors: int = 0
    """Misses whose judgment errored. Not stored, so they are asked again next time."""
    missed: tuple[tuple[str, str], ...] = ()
    """(case, role) of every miss, to see what an edit cost."""
    unreliable_identity: bool = False
    """The judge's identity is not complete (`identity_reliable`): nothing was read from or written
    to the cache, since a stored verdict could be another judge's. Every judgment was a miss."""

    @property
    def calls_saved(self) -> int:
        return self.hits


class JudgmentCache:
    """Verdicts by content key, in a SQLite file (or in memory with the default `':memory:'`).

    SQLite rather than a JSON file because each verdict is committed as soon as it arrives: a run
    that stops part way (a timeout, credits running out) keeps what it paid for, and the next run
    picks up from there. Rows carry the judge's fingerprint, case and role for auditing; lookups
    use only the key.
    """

    def __init__(self, path: str | Path = ':memory:') -> None:
        self.path = str(path)
        self._db = sqlite3.connect(self.path)
        self._db.execute(
            'CREATE TABLE IF NOT EXISTS judgments ('
            'key TEXT PRIMARY KEY, passed INTEGER NOT NULL, reason TEXT, '
            'fingerprint TEXT NOT NULL, case_name TEXT NOT NULL, role TEXT NOT NULL)'
        )
        self._db.commit()

    def get(self, key: str) -> tuple[bool, str | None] | None:
        row = self._db.execute('SELECT passed, reason FROM judgments WHERE key = ?', (key,)).fetchone()
        return None if row is None else (bool(row[0]), row[1])

    def put(self, key: str, passed: bool, reason: str | None, *, judge: str, case: str, role: str) -> None:
        self._db.execute(
            'INSERT OR REPLACE INTO judgments VALUES (?, ?, ?, ?, ?, ?)', (key, int(passed), reason, judge, case, role)
        )
        self._db.commit()

    def __len__(self) -> int:
        return self._db.execute('SELECT COUNT(*) FROM judgments').fetchone()[0]

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> JudgmentCache:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


Serializer = Callable[[Any], Any]
"""Turns a value the cache cannot key faithfully into one it can (plain data, dataclasses, models)."""

_ADDRESS = re.compile(r' at 0x[0-9a-fA-F]+')

_AS_TEXT: dict[type, Callable[[Any], str]] = {
    # with `fold` and the tzinfo object itself: the ISO text holds only the offset
    datetime.datetime: lambda v: f'{v.isoformat()} fold={v.fold} tz={_ADDRESS.sub("", repr(v.tzinfo))}',
    datetime.date: datetime.date.isoformat,
    datetime.time: lambda v: f'{v.isoformat()} fold={v.fold} tz={_ADDRESS.sub("", repr(v.tzinfo))}',
    datetime.timedelta: repr,
    decimal.Decimal: str,
    uuid.UUID: str,
}
"""Types whose text form is the whole value, so it can stand for it in a key."""


def _canonical(value: Any, serializer: Serializer | None = None, where: str = 'value') -> Any:
    """A JSON form of `value` that keeps everything a judge, or its prompt, could tell apart.

    `serializer`, when given, is asked first about every value, at every depth: what it returns in
    place of the value is keyed instead (tagged with the original's class), and a value it returns
    unchanged (the same object) is keyed as below. So it can take over any type, a dict subclass
    or a dataclass as much as an object the cache has no form for.

    Every value is tagged with its exact type: a tuple is not a list, a frozenset is not a set,
    `1` is not `True` or `1.0`, an `OrderedDict` is not a dict, a model or dataclass is tagged with
    its class. A subclass of dict, list, tuple, set or frozenset is keyed with its class and its
    instance state (its `__dict__` and `__slots__`; a defaultdict's factory), since a judge can read
    that state as well as the items. Order is kept where the value has one: a dict's items and a
    model's or dataclass's fields in their own order, as `LLMJudge` writes them into its prompt.
    Only a set's members are sorted, by their canonical form, since a set has no order to keep.

    A value with no faithful form here raises `TypeError` rather than falling back to `repr`, which
    can hide what the judge sees; `serializer` turns such values into ones that have one.
    """
    kind = type(value)
    name = f'{kind.__module__}.{kind.__qualname__}'
    if serializer is not None:
        converted = serializer(value)
        if converted is not value:
            return {'serialized': name, 'value': _canonical(converted, None, where)}
    if value is None or kind in (bool, int, str):
        return value
    if kind is float:
        return {'float': repr(value)}  # nan, inf and -0.0 kept apart, and 1.0 never equal to 1
    if kind in (list, tuple):
        return {kind.__name__: [_canonical(v, serializer, f'{where}[{i}]') for i, v in enumerate(value)]}
    if kind in (set, frozenset):
        members = (_canonical(v, serializer, f'{where}{{}}') for v in value)
        return {kind.__name__: sorted(members, key=lambda m: json.dumps(m, sort_keys=True))}
    if kind in (bytes, bytearray):
        return {kind.__name__: value.hex()}
    if isinstance(value, dict):  # dict and its subclasses, tagged with the class when not a plain dict
        items: dict[Any, Any] = value
        pairs = [[_canonical(k, serializer, f'{where} key'), _canonical(v, serializer, f'{where}[{k!r}]')]
                 for k, v in items.items()]  # fmt: skip
        if kind is dict:
            return {'dict': pairs}
        return {'mapping': name, 'items': pairs, 'state': _state(value, serializer, where), 'code': _code_of(kind)}
    if isinstance(value, list | tuple | set | frozenset) and not isinstance(value, enum.Enum):
        container: Any = value
        members = [_canonical(v, serializer, f'{where}[{i}]') for i, v in enumerate(container)]
        if isinstance(value, set | frozenset):
            members.sort(key=lambda m: json.dumps(m, sort_keys=True))
        return {'collection': name, 'items': members, 'state': _state(value, serializer, where), 'code': _code_of(kind)}
    if isinstance(value, enum.Enum):
        return {'enum': name, 'value': _canonical(value.value, serializer, where)}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        fields = [[f.name, _canonical(getattr(value, f.name), serializer, f'{where}.{f.name}')]
                  for f in dataclasses.fields(value)]  # fmt: skip
        names = {f.name for f in dataclasses.fields(value)}  # and what `__post_init__` stored besides
        extra = [[k, _canonical(v, serializer, f'{where}.{k}')] for k, v in sorted(
            ((k, v) for k, v in _instance_state(value) if k not in names), key=lambda kv: kv[0])]  # fmt: skip
        return {'dataclass': name, 'fields': fields, 'state': extra, 'code': _code_of(kind)}
    if isinstance(value, pydantic.BaseModel):
        # Field by field in the model's order, values as they are (a nested model keeps its class);
        # extra fields after, in the order they came.
        names = [*type(value).model_fields, *(value.model_extra or {})]
        fields = [[n, _canonical(getattr(value, n), serializer, f'{where}.{n}')] for n in names]
        private = sorted((getattr(value, '__pydantic_private__', None) or {}).items())
        fields += [[f'private:{k}', _canonical(v, serializer, f'{where}.{k}')] for k, v in private]
        return {'model': name, 'fields': fields, 'fields_set': sorted(value.model_fields_set), 'code': _code_of(kind)}
    if kind in _AS_TEXT:
        return {'text': name, 'value': _AS_TEXT[kind](value)}
    if isinstance(value, PurePath):
        return {'text': name, 'value': str(value)}
    if serializer is not None:
        raise TypeError(f'serializer returned {where} ({name}) unchanged; it must return data the cache can key')
    raise TypeError(
        f'cannot key {where}: a {name} has no faithful form here, and its repr may hide what the judge '
        'sees. Pass serializer= to certify_judge_cached (a function returning plain data, a dataclass or '
        'a pydantic model for it).'
    )


def _faithful(value: Any) -> bool:
    """Whether `_canonical` has a faithful form of its own for `value` (not counting what it holds)."""
    return (
        value is None
        or type(value) in (bool, int, float, str, bytes, bytearray)
        or type(value) in _AS_TEXT
        or isinstance(value, dict | list | tuple | set | frozenset | enum.Enum | pydantic.BaseModel | PurePath)
        or (dataclasses.is_dataclass(value) and not isinstance(value, type))
    )


def repr_fallback(value: Any) -> Any:
    """A serializer for `_canonical` that keys only what has no faithful form by its `repr` (memory
    addresses removed, tagged with the class); everything else, at every depth, is keyed as usual."""
    return value if _faithful(value) else {'repr': _ADDRESS.sub('', repr(value))}


def _callable_key(fn: Any) -> str:
    """A function a value holds (a defaultdict's factory), by `_function_digest`; one that cannot be
    pinned down gets a key no later run can match."""
    if isinstance(fn, type):
        return f'{fn.__module__}.{fn.__qualname__} {_code_of(fn)}'
    opaque: list[str] = []
    digest = _function_digest(fn, opaque)
    return f'{digest} unidentified {uuid.uuid4()}' if opaque else digest


def _code_of(kind: type) -> str:
    """The class's own code (`_class_digest`): two classes of one name can behave differently (a
    property, `__getitem__`), and the judge sees that behaviour, not the name."""
    opaque: list[str] = []
    digest = _class_digest(kind, opaque)
    # What the digest could not pin down (a method closing over a list) can change unseen: such a
    # value gets a key no later run can match, so its verdicts are never reused.
    return f'{digest} unidentified {uuid.uuid4()}' if opaque else digest


def _state(value: Any, serializer: Serializer | None, where: str) -> list[list[Any]]:
    """A container subclass's instance state, by attribute name. A defaultdict's factory by its name."""
    out: list[list[Any]] = []
    for attr, held in _instance_state(value):
        if attr == 'default_factory' and (held is None or callable(held)):
            out.append([attr, {'factory': None if held is None else _callable_key(held)}])
        else:
            out.append([attr, _canonical(held, serializer, f'{where}.{attr}')])
    return out


def _digest(material: Any) -> str:
    return hashlib.sha256(json.dumps(material, sort_keys=True, allow_nan=True).encode()).hexdigest()


def judgment_key(
    judge_digest: str,
    case: JudgeCase,
    output: Any,
    role: str,
    *,
    assertion: str | None,
    occurrence: int,
    salt: str,
    serializer: Serializer | None = None,
) -> str:
    """The cache key of one judgment: the judge, everything it is shown, the role, the occurrence."""
    return _digest(
        {
            'format': CACHE_FORMAT,
            'judge': judge_digest,
            'salt': salt,
            'assertion': assertion,
            'name': case.name,
            'inputs': _canonical(case.inputs, serializer, 'inputs'),
            'expected_output': _canonical(case.expected_output, serializer, 'expected_output'),
            'metadata': _canonical(case.metadata, serializer, 'metadata'),
            'output': _canonical(output, serializer, 'output'),
            'role': role,
            'occurrence': occurrence,
        }
    )


def _keyed_plan(
    judge: Evaluator[Any, Any, Any],
    cases: Sequence[JudgeCase],
    *,
    controls: Sequence[Control],
    human_labels: Sequence[HumanLabel],
    repeats: int,
    seed: int,
    assertion: str | None,
    salt: str,
    serializer: Serializer | None = None,
) -> tuple[random.Random, list[tuple[JudgeCase, Any, str]], list[str | None]]:
    """`certify_judge`'s plan, from the same seed, with a key for every planned judgment.

    The rng is returned in the state `certify_judge` leaves it in, since a sequential run goes on
    to shuffle the cases with it. A judge whose identity is not reliable gets no keys (all None):
    nothing may be looked up or stored for it.
    """
    if repeats < 1:
        raise ValueError(f'repeats must be >= 1, got {repeats}')
    if not cases:
        raise ValueError('no cases to certify the judge on')
    names = [case.name for case in cases]
    if len(set(names)) != len(names):
        raise ValueError('case names must be unique: human labels are matched to cases by name')
    rng = random.Random(seed)
    planned = _plan(cases, controls, human_labels, repeats, rng)
    identity = judge_identity(judge)
    if not identity_reliable(identity):
        return rng, planned, [None] * len(planned)
    judge_digest = _digest(identity)
    seen: dict[str, int] = {}
    keys: list[str | None] = []
    for case, output, role in planned:
        base = judgment_key(
            judge_digest, case, output, role, assertion=assertion, occurrence=0, salt=salt, serializer=serializer
        )
        occurrence = seen.get(base, 0)
        seen[base] = occurrence + 1
        if occurrence:
            base = judgment_key(
                judge_digest, case, output, role, assertion=assertion, occurrence=occurrence, salt=salt,
                serializer=serializer,
            )  # fmt: skip
        keys.append(base)
    return rng, planned, keys


def uncached_judgments(
    judge: Evaluator[Any, Any, Any],
    cases: Sequence[JudgeCase],
    *,
    cache: JudgmentCache,
    controls: Sequence[Control] = DEFAULT_CONTROLS,
    human_labels: Sequence[HumanLabel] = (),
    repeats: int = 3,
    assertion: str | None = None,
    seed: int = 0,
    salt: str = '',
    serializer: Serializer | None = None,
) -> list[tuple[str, str]]:
    """(case, role) of every judgment a cached certification would ask the judge for, without asking.

    What re-certifying will cost before it is paid. A sequential run (`batch_size`) that stops
    early asks for fewer. For a judge whose identity is not reliable, every planned judgment.
    """
    _, planned, keys = _keyed_plan(
        judge, cases, controls=controls, human_labels=human_labels, repeats=repeats, seed=seed,
        assertion=assertion, salt=salt, serializer=serializer,
    )  # fmt: skip
    return [
        (case.name, role)
        for (case, _, role), key in zip(planned, keys, strict=True)
        if key is None or cache.get(key) is None
    ]


async def certify_judge_cached(
    judge: Evaluator[Any, Any, Any],
    cases: Sequence[JudgeCase],
    *,
    cache: JudgmentCache,
    controls: Sequence[Control] = DEFAULT_CONTROLS,
    human_labels: Sequence[HumanLabel] = (),
    repeats: int = 3,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
    assertion: str | None = None,
    max_concurrency: int = 8,
    seed: int = 0,
    batch_size: int | None = None,
    slice_by: Callable[[JudgeCase], str] | None = None,
    salt: str = '',
    serializer: Serializer | None = None,
) -> tuple[Certificate, CacheStats]:
    """`certify_judge`, asking the judge only for judgments not already in `cache`.

    Takes `certify_judge`'s arguments, plus:

    Args:
        cache: Where verdicts are kept between runs.
        salt: Added to every key. `judge_identity` does not follow what a judge's code reaches
            through module globals (a helper it calls, a prompt file it reads), and identifies a
            named `FunctionModel` by its name. Give such a judge a salt that changes when it does
            (a version, a hash of its source).
        serializer: Asked first about every value in a case or output, at every depth: return
            a replacement (plain data, a dataclass, a pydantic model) to have it keyed instead, or
            the value itself (the same object) to leave it to the cache. Needed for a value the
            cache cannot key faithfully (an object that is not plain data, a dataclass or a
            pydantic model): without it such a value raises `TypeError` rather than being keyed by
            its `repr`. A function-backed judge is cached only under a `model_name` of its own (see
            `judge_identity`); bump that name, or the salt, when its code changes.

    A judge whose identity is not reliable (`identity_reliable`: it closes over mutable state or
    objects that cannot be identified by value) is never served from the cache, and nothing is
    stored for it: `CacheStats.unreliable_identity` says so, and every judgment is a miss.
    """
    if max_concurrency < 1 or (batch_size is not None and batch_size < 1):
        raise ValueError('max_concurrency and batch_size must be >= 1')
    rng, planned, keys = _keyed_plan(
        judge, cases, controls=controls, human_labels=human_labels, repeats=repeats, seed=seed,
        assertion=assertion, salt=salt, serializer=serializer,
    )  # fmt: skip
    limit = asyncio.Semaphore(max_concurrency)
    slices = {case.name: slice_by(case) for case in cases} if slice_by else None
    identity = judge_identity(judge)
    short = fingerprint(identity)
    hits: list[int] = []
    missed: list[tuple[str, str]] = []
    errors: list[int] = []

    async def one(i: int) -> Judgment:
        case, output, role = planned[i]
        key = keys[i]
        stored = None if key is None else cache.get(key)
        if stored is not None:
            hits.append(i)
            return Judgment(case.name, role, output, stored[0], stored[1])
        missed.append((case.name, role))
        judgment = await _judge(judge, case, output, role, assertion, limit)
        if judgment.passed is None:
            errors.append(i)
        elif key is not None:
            cache.put(key, judgment.passed, judgment.reason, judge=short, case=case.name, role=role)
        return judgment

    looks = 1
    if batch_size is None:
        judgments = list(await asyncio.gather(*(one(i) for i in range(len(planned)))))
        verdict, checks = _assess(judgments, repeats, thresholds, slices=slices)
    else:
        # `_certify_sequentially`, step for step: the same shuffle of the same rng, the same looks.
        order = [case.name for case in cases]
        rng.shuffle(order)
        batches = [order[i : i + batch_size] for i in range(0, len(order), batch_size)]
        looks = len(batches)
        judgments = []
        verdict: Verdict = 'UNVALIDATED'
        checks = ()
        for batch in batches:
            names = set(batch)
            todo = [i for i, (c, _, _) in enumerate(planned) if c.name in names]
            judgments += await asyncio.gather(*(one(i) for i in todo))
            verdict, checks = _assess(judgments, repeats, thresholds, looks=looks, slices=slices)
            if verdict == 'INADMISSIBLE':
                break
    certificate = Certificate(
        verdict,
        checks,
        tuple(judgments),
        judge=_describe(judge),
        calls=len(judgments),
        planned=len(planned),
        looks=looks,
        identity=identity,
    )
    # Every lookup happens before the coroutine's first await, so `missed` is in plan order.
    unreliable = any(key is None for key in keys)
    return certificate, CacheStats(len(hits), len(missed), len(errors), tuple(missed), unreliable)
