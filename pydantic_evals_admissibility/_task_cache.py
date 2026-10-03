"""Re-score without re-running the task: keep each task output, and run only the evaluators again.

pydantic-evals runs the task on every case each time a dataset is evaluated, so trying another
evaluator, or a reworded rubric, pays for every model call again (pydantic-ai issues #1350 and
#3314). `evaluate_cached` runs the task once per case and keeps the output in a `TaskCache`; later
evaluations serve the stored output and run only the evaluators. It is `Dataset.evaluate` itself,
so cases still stream through task and evaluators together under `max_concurrency`, and the result
is an ordinary `EvaluationReport`.

An output is reused only for exactly what produced it. Its key is made of:

- the task's identity (`task_identity`): the code of the task function, its default arguments and
  the values it closes over (by `_identity._function_digest`), and what it reaches through its
  module's globals: plain immutable data by value, functions and classes of the same module by
  their code (followed in turn), modules and functions or classes of other modules by name. Edit
  the task, a constant it reads or a helper next to it, and every key changes.
- the inputs as the task sees them, with their exact types and order (`_cache._canonical`): the
  task is called with `case.inputs` and nothing else, so a case's expected output and metadata
  are not part of the key and editing them does not rerun it. With a `lifecycle`, whose `setup`
  sees the whole case and may prepare what the task reads, the case's name, expected output and
  metadata are in the key too, and so is the lifecycle's code.
- the module globals read by the methods of the classes the inputs hold (the task calls them):
  edit a constant an input's `render()` reads and the key changes; a class whose methods read a
  mutable global or an object gets a key no later run matches.
- the occurrence: the k-th case (or `repeat` run) with the same task and inputs gets the k-th
  output, so two identical cases, or the runs of `repeat=3`, stay independent samples.

A task whose identity is not reliable is never served from the cache and nothing is stored for it
(the package's rule for judges). Unreliable means something the task reaches could change without
its key changing: a mutable value it closes over or reaches as a global (a counter, a config dict),
an object (a pydantic-ai `Agent`, a client), a callable object. `version=` is the user's claim
that the task is the same task, as `FunctionModel(model_name=...)` is for a judge's model: with
it, such a task is cached under that version (and its code, which still counts), and bumping the
version is how the user says it changed. What reaches beyond the module (a helper in another
module, a prompt file, the library versions) is not followed, with or without a version.

Outputs are stored faithfully or not at all: plain data, tuples, sets, bytes, enums, datetimes,
dataclasses and pydantic models (fields, extra and private attributes, which fields were set) are
stored with their classes and read back with the class's code checked; a stored output whose class
has changed is a miss. An output that cannot be stored faithfully is used for the report and not
stored (`codec=` gives a dump/load pair for it). Errors are not stored.

A served output is marked: the case's `task_cache` attribute says it came from the cache, when it
was stored, and the original run's duration and metrics. The report's `task_duration` and
`metrics` are this run's (no model was called), never the original's; attributes the task set are
replayed, since evaluators read them as part of the output; and the span tree is unavailable
(`ctx.span_tree` raises), since no task spans were recorded.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextvars
import dataclasses
import datetime
import decimal
import enum
import functools
import importlib
import inspect
import json
import sqlite3
import sys
import threading
import time
import types
import uuid
from collections.abc import Awaitable, Callable, Collection, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePath
from typing import Any

import anyio.to_thread
import pydantic
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators.context import EvaluatorContext
from pydantic_evals.lifecycle import CaseLifecycle
from pydantic_evals.otel._errors import SpanTreeRecordingError
from pydantic_evals.reporting import EvaluationReport, ReportCase, ReportCaseFailure

from ._cache import (
    Serializer,
    _canonical,  # pyright: ignore[reportPrivateUsage]
    _digest,  # pyright: ignore[reportPrivateUsage]
)
from ._identity import (
    _LIBRARY_ROOTS,  # pyright: ignore[reportPrivateUsage]
    _MAX_DEPTH,  # pyright: ignore[reportPrivateUsage]
    _class_digest,  # pyright: ignore[reportPrivateUsage]
    _data,  # pyright: ignore[reportPrivateUsage]
    _function_digest,  # pyright: ignore[reportPrivateUsage]
    _generated,  # pyright: ignore[reportPrivateUsage]
    _instance_state,  # pyright: ignore[reportPrivateUsage]
    _name,  # pyright: ignore[reportPrivateUsage]
    _NotData,  # pyright: ignore[reportPrivateUsage]
    _value,  # pyright: ignore[reportPrivateUsage]
    fingerprint,
)

TASK_CACHE_FORMAT = 1
"""Part of every key: a change to how keys or stored outputs are built must not hit old entries."""

TASK_CACHE_ATTRIBUTE = 'task_cache'
"""The case attribute that says where a case's output came from."""


# --- identity ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class OutputCodec:
    """How to store an output the cache has no faithful form for: `dump` turns it into data the
    cache can store (plain data, dataclasses, pydantic models), `load` turns that back into it."""

    dump: Callable[[Any], Any]
    load: Callable[[Any], Any]


def _code_names(code: types.CodeType) -> set[str]:
    """The names a function's code looks up, nested functions' and comprehensions' included."""
    names = set(code.co_names)
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            names |= _code_names(const)
    return names


def _library(obj: Any) -> bool:
    return (getattr(obj, '__module__', None) or '').split('.')[0] in _LIBRARY_ROOTS


def _global(value: Any, owner: types.FunctionType, opaque: list[str], where: str, follow: list[Any]) -> Any:
    """A global a function reads: data by value, its own module's code by digest, the rest by name."""
    try:
        return ['data', _data(value)]
    except _NotData:
        pass
    same_module = getattr(value, '__module__', None) == owner.__module__
    if isinstance(value, types.ModuleType):
        return ['module', value.__name__]
    if isinstance(value, type):
        if same_module:
            follow.append(value)
            return ['class', _name(value), _class_digest(value, opaque)]
        return ['class', _name(value)]
    if isinstance(value, enum.Enum):
        return _value(value, opaque, where, 0, ())
    if isinstance(value, types.FunctionType):
        if same_module:
            follow.append(value)
            return ['function', _name(value), _function_digest(value, opaque, where)]
        return ['function', _name(value)]
    if isinstance(value, types.BuiltinFunctionType) and isinstance(value.__self__, types.ModuleType | None):
        return ['builtin', _name(value)]
    if isinstance(value, types.GenericAlias | types.UnionType) or (type(value).__module__ or '') == 'typing':
        return ['typing', repr(value)]  # `Optional[int]`, `Any`: what the code names, not state
    # A list or dict may be a counter or a config that changes; an object (an Agent, a client)
    # has no value we can trust to mean its behaviour.
    opaque.append(f'{where}: {type(value).__qualname__}')
    return ['opaque', _name(type(value))]


def _reached_globals(root: Any, opaque: list[str], where: str) -> dict[str, Any]:
    """Every module global the code reachable from `root` reads, by `module.name`."""
    out: dict[str, Any] = {}
    seen: set[int] = set()
    stack: list[tuple[Any, int]] = [(root, 0)]
    while stack:
        obj, depth = stack.pop()
        if id(obj) in seen or obj is None:
            continue
        seen.add(id(obj))
        if depth > _MAX_DEPTH:
            opaque.append(f'{where}: code nested deeper than {_MAX_DEPTH} levels')
            continue
        follow: list[Any] = []
        if isinstance(obj, functools.partial):
            follow += [obj.func, *obj.args, *obj.keywords.values()]
        elif isinstance(obj, types.MethodType):
            follow += [obj.__func__, obj.__self__]
        elif isinstance(obj, type):
            if not _library(obj):
                for klass in obj.__mro__:
                    if klass is object or _library(klass):
                        continue
                    for name, attr in vars(klass).items():
                        held = attr.__func__ if isinstance(attr, staticmethod | classmethod) else attr
                        if isinstance(held, property):
                            follow += [held.fget, held.fset, held.fdel]
                        elif (
                            isinstance(held, types.FunctionType)
                            and held.__module__ == klass.__module__  # written with the class, not made by a library
                            and not _generated(klass, name, held)
                        ):
                            follow.append(held)
        elif isinstance(obj, types.FunctionType):
            for name in sorted(_code_names(obj.__code__)):
                if name not in obj.__globals__:
                    continue  # a builtin, or an attribute name
                label = f'{obj.__module__}.{name}'
                if label not in out:
                    out[label] = _global(obj.__globals__[name], obj, opaque, f'{where} global {label}', follow)
            for cell in obj.__closure__ or ():
                try:
                    follow.append(cell.cell_contents)
                except ValueError:
                    pass
            follow += [*(obj.__defaults__ or ()), *(obj.__kwdefaults__ or {}).values()]
        elif not isinstance(obj, types.BuiltinFunctionType) and callable(obj):
            follow.append(type(obj))  # a callable object: its class's methods (its state is opaque already)
        stack += [(f, depth + 1) for f in follow if callable(f)]
    return dict(sorted(out.items()))


def _part(obj: Any, opaque: list[str], where: str) -> dict[str, Any]:
    if isinstance(obj, type):
        code = _class_digest(obj, opaque)
    else:
        code = _function_digest(obj, opaque, where)
    return {'name': _name(obj), 'code': code, 'globals': _reached_globals(obj, opaque, where)}


def task_identity(
    task: Callable[..., Any],
    *,
    version: str | None = None,
    lifecycle: Any = None,
    codec: OutputCodec | None = None,
    serializer: Serializer | None = None,
) -> dict[str, Any]:
    """What makes a task the task it is, as plain data; what could not be pinned down under `opaque`.

    The task's code, defaults and closed-over values (`_identity._function_digest`), the module
    globals its code reads, the user's `version`, and the code of the `lifecycle`, `codec` and
    `serializer` an evaluation uses with it, since each decides what is run or stored.
    """
    opaque: list[str] = []
    identity: dict[str, Any] = {'format': TASK_CACHE_FORMAT, 'task': _part(task, opaque, 'task'), 'version': version}
    if lifecycle is not None:
        identity['lifecycle'] = _part(lifecycle, opaque, 'lifecycle')
    if codec is not None:
        identity['codec'] = [_part(codec.dump, opaque, 'codec.dump'), _part(codec.load, opaque, 'codec.load')]
    if serializer is not None:
        identity['serializer'] = _part(serializer, opaque, 'serializer')
    if opaque:
        identity['opaque'] = sorted(set(opaque))
    return identity


def task_identity_reliable(identity: dict[str, Any]) -> bool:
    """Whether outputs may be reused on this identity: nothing opaque, or a `version` that claims it."""
    return not identity.get('opaque') or identity.get('version') is not None


# --- faithful storage -------------------------------------------------------------------------


class _Unstorable(TypeError):
    """An output with no faithful stored form."""


class _Stale(Exception):
    """A stored output whose class is gone or has changed since it was stored."""


def _class_ref(kind: type, where: str) -> dict[str, Any]:
    """A class by import path and code (`_class_digest`), checked again when the output is read."""
    module, qualname = kind.__module__, kind.__qualname__
    try:
        found = _import(module, qualname)
    except Exception:
        found = None
    if found is not kind:
        raise _Unstorable(f'{where}: class {module}.{qualname} cannot be imported by name (defined in a function?)')
    return {'module': module, 'qualname': qualname, 'code': _class_digest(kind, [])}


def _import(module: str, qualname: str) -> Any:
    obj: Any = importlib.import_module(module)
    for part in qualname.split('.'):
        obj = getattr(obj, part)
    return obj


def _resolve(ref: dict[str, Any]) -> Any:
    try:
        kind = _import(ref['module'], ref['qualname'])
    except Exception as exc:
        raise _Stale(f'class {ref["module"]}.{ref["qualname"]} is gone') from exc
    if not isinstance(kind, type) or _class_digest(kind, []) != ref['code']:
        raise _Stale(f'class {ref["module"]}.{ref["qualname"]} has changed since the output was stored')
    return kind


def _tz(tz: datetime.tzinfo | None, where: str) -> Any:
    if tz is None:
        return None
    if type(tz) is datetime.timezone:
        offset = tz.utcoffset(None)
        named = repr(datetime.timezone(offset)) != repr(tz)
        return {'offset': _encode(offset, where), 'name': tz.tzname(None) if named else None}
    key = getattr(tz, 'key', None)
    if type(tz).__module__ == 'zoneinfo' and isinstance(key, str):
        return {'zone': key}
    raise _Unstorable(f'{where}: time zone {type(tz).__qualname__} has no faithful stored form')


def _untz(data: Any) -> datetime.tzinfo | None:
    if data is None:
        return None
    if 'zone' in data:
        import zoneinfo

        return zoneinfo.ZoneInfo(data['zone'])
    offset = _decode(data['offset'])
    return datetime.timezone(offset) if data['name'] is None else datetime.timezone(offset, data['name'])


def _encode(value: Any, where: str = 'output', depth: int = 0) -> Any:
    """A JSON form `_decode` turns back into an equal value of the same types, or `_Unstorable`."""
    if depth > 100:
        raise _Unstorable(f'{where}: nested too deeply (a cycle?)')
    kind = type(value)
    deeper = depth + 1
    if value is None or kind in (bool, int, str):
        return value
    if kind is float:
        return {'t': 'float', 'v': repr(value)}
    if kind in (list, tuple, set, frozenset):
        items = [_encode(v, f'{where}[{i}]', deeper) for i, v in enumerate(value)]
        if kind in (set, frozenset):  # no order of its own; iteration order follows the hash seed
            items.sort(key=lambda m: json.dumps(m, sort_keys=True))
        return {'t': kind.__name__, 'v': items}
    if kind is dict:
        pairs = [[_encode(k, f'{where} key', deeper), _encode(v, f'{where}[{k!r}]', deeper)] for k, v in value.items()]
        return {'t': 'dict', 'v': pairs}
    if kind in (bytes, bytearray):
        return {'t': kind.__name__, 'v': value.hex()}
    if kind is datetime.datetime:
        return {'t': 'datetime', 'v': value.replace(tzinfo=None).isoformat(), 'fold': value.fold,
                'tz': _tz(value.tzinfo, where)}  # fmt: skip
    if kind is datetime.date:
        return {'t': 'date', 'v': value.isoformat()}
    if kind is datetime.time:
        return {'t': 'time', 'v': value.replace(tzinfo=None).isoformat(), 'fold': value.fold,
                'tz': _tz(value.tzinfo, where)}  # fmt: skip
    if kind is datetime.timedelta:
        return {'t': 'timedelta', 'v': [value.days, value.seconds, value.microseconds]}
    if kind is decimal.Decimal:
        return {'t': 'decimal', 'v': str(value)}
    if kind is uuid.UUID:
        return {'t': 'uuid', 'v': str(value)}
    if isinstance(value, PurePath) and kind.__module__ == 'pathlib':
        return {'t': 'path', 'class': _class_ref(kind, where), 'v': str(value)}
    if isinstance(value, enum.Enum):
        ref = _class_ref(kind, where)
        name = value.name
        if name is not None and kind.__members__.get(name) is value:
            return {'t': 'enum', 'class': ref, 'name': name}
        return {'t': 'enum', 'class': ref, 'value': _encode(value.value, where, deeper)}
    if isinstance(value, tuple) and hasattr(kind, '_fields') and hasattr(kind, '_make'):
        items = [_encode(v, f'{where}[{i}]', deeper) for i, v in enumerate(value)]  # a namedtuple
        return {'t': 'namedtuple', 'class': _class_ref(kind, where), 'v': items}
    if isinstance(value, pydantic.BaseModel):
        # Its whole `__dict__` (fields, and what a `cached_property` or a method stored there) and
        # any slots of its own, not only the fields: all of it can be read from the output.
        fields = [[n, _encode(v, f'{where}.{n}', deeper)] for n, v in value.__dict__.items()]
        slots = [[k, _encode(v, f'{where}.{k}', deeper)] for k, v in _instance_state(value)
                 if k not in value.__dict__ and not k.startswith('__pydantic_')]  # fmt: skip
        extra = value.__pydantic_extra__
        private = value.__pydantic_private__
        return {
            't': 'model',
            'class': _class_ref(kind, where),
            'fields': fields,
            'slots': slots,
            'fields_set': sorted(value.model_fields_set),
            'extra': None if extra is None else [[k, _encode(v, f'{where}.{k}', deeper)] for k, v in extra.items()],
            'private': None
            if private is None
            else [[k, _encode(v, f'{where}.{k}', deeper)] for k, v in private.items()],
        }
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        state = [[k, _encode(v, f'{where}.{k}', deeper)] for k, v in _instance_state(value)]
        return {'t': 'dataclass', 'class': _class_ref(kind, where), 'state': state}
    raise _Unstorable(
        f'{where}: a {kind.__module__}.{kind.__qualname__} has no faithful stored form here; pass '
        'codec=OutputCodec(dump, load) to store it'
    )


def _decode(data: Any) -> Any:
    if data is None or type(data) in (bool, int, str):
        return data
    tag = data['t']
    if tag == 'float':
        return float(data['v'])
    if tag in ('list', 'tuple', 'set', 'frozenset'):
        kinds: dict[str, Any] = {'list': list, 'tuple': tuple, 'set': set, 'frozenset': frozenset}
        return kinds[tag](_decode(v) for v in data['v'])
    if tag == 'dict':
        return {_decode(k): _decode(v) for k, v in data['v']}
    if tag == 'bytes':
        return bytes.fromhex(data['v'])
    if tag == 'bytearray':
        return bytearray.fromhex(data['v'])
    if tag == 'datetime':
        moment = datetime.datetime.fromisoformat(data['v'])
        return moment.replace(tzinfo=_untz(data['tz']), fold=data['fold'])
    if tag == 'date':
        return datetime.date.fromisoformat(data['v'])
    if tag == 'time':
        return datetime.time.fromisoformat(data['v']).replace(tzinfo=_untz(data['tz']), fold=data['fold'])
    if tag == 'timedelta':
        days, seconds, micro = data['v']
        return datetime.timedelta(days=days, seconds=seconds, microseconds=micro)
    if tag == 'decimal':
        return decimal.Decimal(data['v'])
    if tag == 'uuid':
        return uuid.UUID(data['v'])
    kind = _resolve(data['class'])
    if tag == 'path':
        return kind(data['v'])
    if tag == 'enum':
        return kind[data['name']] if 'name' in data else kind(_decode(data['value']))
    if tag == 'namedtuple':
        return kind._make(_decode(v) for v in data['v'])
    if tag == 'model':
        model = kind.__new__(kind)
        object.__setattr__(model, '__dict__', {n: _decode(v) for n, v in data['fields']})
        object.__setattr__(model, '__pydantic_fields_set__', set(data['fields_set']))
        extra = data['extra']
        object.__setattr__(model, '__pydantic_extra__', None if extra is None else {k: _decode(v) for k, v in extra})
        private = data['private']
        object.__setattr__(
            model, '__pydantic_private__', None if private is None else {k: _decode(v) for k, v in private}
        )
        for name, held in data['slots']:
            object.__setattr__(model, name, _decode(held))
        return model
    if tag == 'dataclass':
        instance = kind.__new__(kind)
        for name, held in data['state']:
            object.__setattr__(instance, name, _decode(held))
        return instance
    raise _Stale(f'unknown stored form {tag!r}')  # pragma: no cover


def _store_form(output: Any, codec: OutputCodec | None) -> Any:
    """The stored form of an output, checked by reading it back: `_Unstorable` if it does not come
    back the same, with the same types (by `_cache._canonical`, which keeps every type and order)."""
    dumped = codec.dump(output) if codec is not None else output
    stored = _encode(dumped)
    back = _decode(json.loads(json.dumps(stored)))
    again = codec.dump(codec.load(back)) if codec is not None else back
    # `_encode` is exact: every type (a tuple is not a list, 1 is not 1.0 or True), every order,
    # each class by its code, a dataclass's whole instance state, a model's fields, extra and
    # private attributes and which fields were set. What reads back with another form is not stored.
    if json.dumps(_encode(again)) != json.dumps(stored):
        raise _Unstorable('output: does not read back the same after storing; pass codec=OutputCodec(dump, load)')
    return stored


_STDLIB_ROOTS = frozenset(sys.stdlib_module_names)


def _value_classes(value: Any, found: dict[str, type], seen: set[int], depth: int = 0) -> None:
    """Every class of user code an input holds, at any depth: its methods run when the task uses it."""
    if id(value) in seen or depth > 100:
        return
    seen.add(id(value))
    kind = type(value)
    root = (kind.__module__ or '').split('.')[0]
    if root not in _STDLIB_ROOTS and not _library(kind):
        found.setdefault(_name(kind), kind)
    held: list[Any] = []
    if isinstance(value, dict):
        mapping: dict[Any, Any] = value
        held += [*mapping.keys(), *mapping.values()]
    elif isinstance(value, list | tuple | set | frozenset):
        held += list(value)
    if isinstance(value, pydantic.BaseModel):
        held += [*value.__dict__.values(), *(value.__pydantic_extra__ or {}).values(),
                 *(value.__pydantic_private__ or {}).values()]  # fmt: skip
    elif not isinstance(value, type):
        held += [v for _, v in _instance_state(value)]
    for item in held:
        _value_classes(item, found, seen, depth + 1)


def _key_form(value: Any, serializer: Serializer | None, where: str) -> Any:
    """What a key holds of a value: `_cache._canonical`'s form (exact types, order, instance state,
    each class by its code), and the module globals its classes' methods read, which the task reaches
    when it calls them. A class whose methods read what cannot be pinned down (a mutable global, an
    object) gets a key no later run matches, as `_cache._code_of` does for unidentified code."""
    classes: dict[str, type] = {}
    _value_classes(value, classes, set())
    reached: dict[str, Any] = {}
    for label, kind in sorted(classes.items()):
        opaque: list[str] = []
        reached[label] = _reached_globals(kind, opaque, f'{where} {label}')
        if opaque:
            reached[label] = f'unidentified {uuid.uuid4()}: ' + '; '.join(sorted(set(opaque)))
    return {'canonical': _canonical(value, serializer, where), 'globals': reached}


# --- the cache --------------------------------------------------------------------------------


@dataclass(frozen=True)
class StoredOutput:
    """One stored task run."""

    output: Any
    attributes: dict[str, Any]
    duration: float | None
    metrics: dict[str, Any]
    stored_at: str


class TaskCache:
    """Task outputs by content key, in a SQLite file (or in memory with the default `':memory:'`).

    Each output is committed as soon as its case has run, so an evaluation that stops part way keeps
    what it paid for. Within a process, two evaluations sharing one cache never run the same key
    twice at once: the second waits for the first's output (across threads and event loops too).
    Two processes on one file are not coordinated: both may run a case, and the last write wins.
    """

    def __init__(self, path: str | Path = ':memory:') -> None:
        self.path = str(path)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.execute(
            'CREATE TABLE IF NOT EXISTS task_outputs ('
            'key TEXT PRIMARY KEY, output TEXT NOT NULL, attributes TEXT NOT NULL, duration REAL, '
            'metrics TEXT NOT NULL, stored_at TEXT NOT NULL, task TEXT NOT NULL, case_name TEXT NOT NULL)'
        )
        self._db.commit()
        self._running: dict[str, concurrent.futures.Future[bool]] = {}
        self.runs = 0
        """Task runs made through this cache object (in this process)."""
        self.hits = 0
        """Outputs served instead of running the task (stored, or shared from a concurrent run)."""

    def _row(self, key: str) -> tuple[str, str, float | None, str, str] | None:
        with self._lock:
            return self._db.execute(
                'SELECT output, attributes, duration, metrics, stored_at FROM task_outputs WHERE key = ?', (key,)
            ).fetchone()

    def __contains__(self, key: str) -> bool:
        return self._row(key) is not None

    def get(self, key: str, codec: OutputCodec | None = None) -> StoredOutput | None:
        """The stored run for `key`, read back; None if there is none or its classes have changed."""
        row = self._row(key)
        if row is None:
            return None
        try:
            output = _decode(json.loads(row[0]))
            if codec is not None:
                output = codec.load(output)
            attributes = _decode(json.loads(row[1]))
        except _Stale:
            return None
        return StoredOutput(output, attributes, row[2], json.loads(row[3]), row[4])

    def _put(self, key: str, output: str, attributes: str, duration: float | None, metrics: dict[str, Any],
             *, task: str, case: str) -> None:  # fmt: skip
        stored_at = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')
        with self._lock:
            self._db.execute(
                'INSERT OR REPLACE INTO task_outputs VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                (key, output, attributes, duration, json.dumps(metrics), stored_at, task, case),
            )
            self._db.commit()

    def delete(self, keys: Collection[str]) -> int:
        """Forget these keys; how many were stored."""
        with self._lock:
            removed = 0
            for key in keys:
                removed += self._db.execute('DELETE FROM task_outputs WHERE key = ?', (key,)).rowcount
            self._db.commit()
        return removed

    def _claim(self, key: str) -> tuple[concurrent.futures.Future[bool], bool]:
        """The in-flight run of `key`, and whether the caller now owns it (must run and release it)."""
        with self._lock:
            running = self._running.get(key)
            if running is not None:
                return running, False
            running = concurrent.futures.Future()
            self._running[key] = running
            return running, True

    def _release(self, key: str, stored: bool) -> None:
        with self._lock:
            running = self._running.pop(key, None)
        if running is not None and not running.done():
            running.set_result(stored)

    def _count(self, *, runs: int = 0, hits: int = 0) -> None:
        with self._lock:
            self.runs += runs
            self.hits += hits

    def __len__(self) -> int:
        with self._lock:
            return self._db.execute('SELECT COUNT(*) FROM task_outputs').fetchone()[0]

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def __enter__(self) -> TaskCache:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


# --- evaluation -------------------------------------------------------------------------------


@dataclass(frozen=True)
class TaskCacheStats:
    """What the cache did on one evaluation; also in `report.experiment_metadata['task_cache']`."""

    runs: int
    """Task runs this evaluation made."""
    hits: int
    """Outputs served instead of a run: from the cache, or shared from a concurrent evaluation's run."""
    refreshed: int = 0
    """Stored outputs dropped by `refresh` before the evaluation."""
    unreliable_identity: bool = False
    """The task's identity is not reliable: nothing was read or stored, every case ran."""
    ran: tuple[str, ...] = ()
    """The cases the task ran on (a repeat run as `name#k`)."""
    unstored: tuple[tuple[str, str], ...] = ()
    """(case, reason) of runs whose output could not be stored faithfully, so will run again."""
    task: str = ''
    """The task identity's fingerprint."""

    def to_dict(self) -> dict[str, Any]:
        return {**dataclasses.asdict(self), 'ran': list(self.ran), 'unstored': [list(u) for u in self.unstored]}

    @classmethod
    def from_report(cls, report: EvaluationReport[Any, Any, Any]) -> TaskCacheStats:
        data = dict((report.experiment_metadata or {})[TASK_CACHE_ATTRIBUTE])
        data['ran'] = tuple(data['ran'])
        data['unstored'] = tuple(tuple(u) for u in data['unstored'])
        return cls(**data)


@dataclass
class _Slot:
    """One case run's dealings with the cache, shared by its lifecycle and the task wrapper."""

    label: str
    key: str | None
    owner: bool = False
    ran: bool = False
    seconds: float | None = None
    served: StoredOutput | None = None
    shared: bool = False
    error: str | None = None
    note: dict[str, Any] = field(default_factory=dict[str, Any])


_SLOT = contextvars.ContextVar['_Slot | None']('_TASK_CACHE_SLOT', default=None)
"""The current case run's `_Slot`: set by the lifecycle's `setup`, read by the task wrapper."""


def _case_names(cases: Sequence[Case[Any, Any, Any]]) -> list[str]:
    return [case.name or f'Case {i}' for i, case in enumerate(cases, 1)]


def _plan(
    dataset: Dataset[Any, Any, Any],
    identity: dict[str, Any],
    *,
    repeat: int,
    whole_case: bool,
    serializer: Serializer | None,
) -> list[tuple[str, list[str | None]]]:
    """(case name, a key per repeat run) for every case, in dataset order; all None when unreliable."""
    if repeat < 1:
        raise ValueError(f'repeat must be >= 1, got {repeat}')
    names = _case_names(dataset.cases)
    if not task_identity_reliable(identity):
        return [(name, [None] * repeat) for name in names]
    task = _digest(identity)
    seen: dict[str, int] = {}
    plan: list[tuple[str, list[str | None]]] = []
    for name, case in zip(names, dataset.cases, strict=True):
        material: dict[str, Any] = {
            'format': TASK_CACHE_FORMAT,
            'task': task,
            'inputs': _key_form(case.inputs, serializer, f'{name}: inputs'),
        }
        if whole_case:  # a lifecycle's setup sees the whole case and may prepare what the task reads
            material['case'] = [
                name,
                _key_form(case.expected_output, serializer, f'{name}: expected_output'),
                _key_form(case.metadata, serializer, f'{name}: metadata'),
            ]
        base = _digest(material)
        keys: list[str | None] = []
        for _ in range(repeat):
            occurrence = seen.get(base, 0)
            seen[base] = occurrence + 1
            keys.append(base if occurrence == 0 else _digest({'base': base, 'occurrence': occurrence}))
        plan.append((name, keys))
    return plan


def _task_label(task: Callable[..., Any]) -> str:
    inner: Any = task
    while isinstance(inner, functools.partial):
        inner = inner.func
    inner = inspect.unwrap(inner)
    return getattr(inner, '__name__', type(inner).__name__)


def _is_async(task: Any) -> bool:
    inner = task
    while isinstance(inner, functools.partial):
        inner = inner.func
    return inspect.iscoroutinefunction(inner) or inspect.iscoroutinefunction(type(inner).__call__)


def _jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, default=repr))


async def evaluate_cached(
    dataset: Dataset[Any, Any, Any],
    task: Callable[[Any], Awaitable[Any]] | Callable[[Any], Any],
    *,
    cache: TaskCache,
    version: str | None = None,
    refresh: bool | Collection[str] = False,
    codec: OutputCodec | None = None,
    serializer: Serializer | None = None,
    name: str | None = None,
    max_concurrency: int | None = None,
    progress: bool = True,
    retry_task: Any = None,
    retry_evaluators: Any = None,
    task_name: str | None = None,
    metadata: dict[str, Any] | None = None,
    repeat: int = 1,
    lifecycle: Any = None,
) -> EvaluationReport[Any, Any, Any]:
    """`dataset.evaluate(task, ...)`, running the task only on cases whose output is not in `cache`.

    Takes `Dataset.evaluate`'s arguments, plus:

    Args:
        cache: Where outputs are kept between evaluations.
        version: The user's claim that the task is the same task. Needed to cache a task whose
            identity is not reliable (it reads an Agent or a mutable value through a global or a
            closure); bump it whenever anything the task reaches changes. Part of every key.
        refresh: `True` to drop every stored output this evaluation would use and run the task on
            every case; a collection of case names to rerun only those.
        codec: A dump/load pair to store outputs the cache has no faithful form for.
        serializer: For inputs the key cannot hold faithfully, as in `certify_judge_cached`.

    Returns the report `Dataset.evaluate` returns, with `experiment_metadata['task_cache']` holding
    `TaskCacheStats` and each case's `task_cache` attribute saying where its output came from.
    """
    identity = task_identity(task, version=version, lifecycle=lifecycle, codec=codec, serializer=serializer)
    reliable = task_identity_reliable(identity)
    task_fp = fingerprint(identity)
    plan = _plan(dataset, identity, repeat=repeat, whole_case=lifecycle is not None, serializer=serializer)

    refreshed = 0
    if refresh is not False and refresh is not None:
        names = {n for n, _ in plan}
        wanted: set[str] = names if isinstance(refresh, bool) else set(refresh)
        if unknown := wanted - names:
            raise ValueError(f'refresh names cases not in the dataset: {sorted(unknown)}')
        refreshed = cache.delete([k for n, keys in plan if n in wanted for k in keys if k is not None])

    queues: dict[int, list[tuple[str, str | None]]] = {}
    for case, (case_name, keys) in zip(dataset.cases, plan, strict=True):
        labels = [case_name if repeat == 1 else f'{case_name}#{i}' for i in range(len(keys))]
        queues.setdefault(id(case), []).extend(zip(labels, keys, strict=True))
    slots: list[_Slot] = []
    unreliable_reason = 'identity not reliable: ' + '; '.join(identity.get('opaque', []))
    var = _SLOT
    run_async = _is_async(task)

    async def call_task(inputs: Any) -> Any:
        if run_async:
            result = task(inputs)
        else:
            result = await anyio.to_thread.run_sync(task, inputs)
        return await result if inspect.isawaitable(result) else result

    async def cached_task(inputs: Any) -> Any:
        slot: _Slot | None = var.get()
        if slot is None or slot.key is None:  # not reliable: run, and keep nothing
            if slot is not None:
                slot.ran = True
            cache._count(runs=1)  # pyright: ignore[reportPrivateUsage]
            return await call_task(inputs)
        key = slot.key
        shared = False
        while True:
            stored = cache.get(key, codec)
            if stored is not None:
                slot.served, slot.shared = stored, shared
                cache._count(hits=1)  # pyright: ignore[reportPrivateUsage]
                return stored.output
            running, owner = cache._claim(key)  # pyright: ignore[reportPrivateUsage]
            if owner:
                break
            # Another evaluation is running this key: wait for its output rather than pay again.
            if await asyncio.wrap_future(running):
                shared = True
        slot.owner, slot.ran, slot.served = True, True, None
        started = time.perf_counter()
        try:
            output = await call_task(inputs)
            slot.seconds = time.perf_counter() - started
        except BaseException:
            slot.owner = False
            cache._release(key, False)  # pyright: ignore[reportPrivateUsage]
            raise
        cache._count(runs=1)  # pyright: ignore[reportPrivateUsage]
        return output  # stored by the lifecycle's prepare_context, once the run's metrics are known

    class _Lifecycle(CaseLifecycle[Any, Any, Any]):
        def __init__(self, case: Case[Any, Any, Any]) -> None:
            super().__init__(case)
            label, key = queues[id(case)].pop(0)
            self.slot = _Slot(label, key)
            slots.append(self.slot)
            self.inner: CaseLifecycle[Any, Any, Any] | None = lifecycle(case) if lifecycle is not None else None
            self.token: Any = None

        async def setup(self) -> None:
            if self.inner is not None:
                await self.inner.setup()
            self.token = var.set(self.slot)

        async def prepare_context(self, ctx: EvaluatorContext[Any, Any, Any]) -> EvaluatorContext[Any, Any, Any]:
            slot = self.slot
            if slot.served is not None:
                served = slot.served
                ctx.attributes.update(served.attributes)
                slot.note = {
                    'cached': True,
                    'source': 'concurrent run' if slot.shared else 'cache',
                    'stored_at': served.stored_at,
                    'original_task_duration': served.duration,
                    'original_metrics': served.metrics,
                }
                ctx = dataclasses.replace(
                    ctx,
                    _span_tree=SpanTreeRecordingError(
                        'this output was served from the task cache: the task did not run, so it recorded no spans'
                    ),
                )
            elif slot.owner and slot.key is not None:
                stored = False
                try:
                    form = json.dumps(_store_form(ctx.output, codec))
                    attributes = json.dumps(_encode(dict(ctx.attributes), 'attributes'))
                except Exception as exc:  # never fail a case whose run is paid for: keep the output, store nothing
                    slot.error = str(exc) if isinstance(exc, _Unstorable) else f'{type(exc).__name__}: {exc}'
                else:
                    cache._put(  # pyright: ignore[reportPrivateUsage]
                        slot.key, form, attributes, slot.seconds, _jsonable(dict(ctx.metrics)),
                        task=task_fp, case=slot.label,
                    )  # fmt: skip
                    stored = True
                finally:
                    slot.owner = False
                    cache._release(slot.key, stored)  # pyright: ignore[reportPrivateUsage]
                slot.note = {'cached': False, 'stored': stored}
                if slot.error:
                    slot.note['reason'] = slot.error
            else:
                slot.note = {'cached': False, 'stored': False, 'reason': unreliable_reason}
            ctx.attributes[TASK_CACHE_ATTRIBUTE] = {**slot.note, 'task': task_fp}
            if self.inner is not None:
                ctx = await self.inner.prepare_context(ctx)
            return ctx

        async def teardown(self, result: ReportCase[Any, Any, Any] | ReportCaseFailure[Any, Any, Any] | None) -> None:
            if self.slot.owner and self.slot.key is not None:  # ran but never reached prepare_context
                self.slot.owner = False
                cache._release(self.slot.key, False)  # pyright: ignore[reportPrivateUsage]
            if self.token is not None:
                try:
                    var.reset(self.token)
                except ValueError:  # set in another context: nothing of ours is left in this one
                    pass
            if self.inner is not None:
                await self.inner.teardown(result)

    report = await dataset.evaluate(
        cached_task,
        name=name,
        max_concurrency=max_concurrency,
        progress=progress,
        retry_task=retry_task,
        retry_evaluators=retry_evaluators,
        task_name=task_name or _task_label(task),
        metadata=metadata,
        repeat=repeat,
        lifecycle=_Lifecycle,
    )
    stats = TaskCacheStats(
        runs=sum(1 for s in slots if s.ran),
        hits=sum(1 for s in slots if s.served is not None),
        refreshed=refreshed,
        unreliable_identity=not reliable,
        ran=tuple(sorted(s.label for s in slots if s.ran)),
        unstored=tuple(sorted((s.label, s.error) for s in slots if s.error)),
        task=task_fp,
    )
    report.experiment_metadata = {**(metadata or {}), TASK_CACHE_ATTRIBUTE: stats.to_dict()}
    return report


def evaluate_cached_sync(
    dataset: Dataset[Any, Any, Any],
    task: Callable[[Any], Awaitable[Any]] | Callable[[Any], Any],
    **kwargs: Any,
) -> EvaluationReport[Any, Any, Any]:
    """`evaluate_cached` for code that is not async (as `Dataset.evaluate_sync` is)."""
    return asyncio.run(evaluate_cached(dataset, task, **kwargs))


def uncached_cases(
    dataset: Dataset[Any, Any, Any],
    task: Callable[..., Any],
    *,
    cache: TaskCache,
    version: str | None = None,
    codec: OutputCodec | None = None,
    serializer: Serializer | None = None,
    repeat: int = 1,
    lifecycle: Any = None,
) -> list[str]:
    """The cases `evaluate_cached` would run the task on, without running anything: what an
    evaluation will cost before it is paid. Every case, for a task whose identity is not reliable."""
    identity = task_identity(task, version=version, lifecycle=lifecycle, codec=codec, serializer=serializer)
    plan = _plan(dataset, identity, repeat=repeat, whole_case=lifecycle is not None, serializer=serializer)
    return [
        name if repeat == 1 else f'{name}#{i}'
        for name, keys in plan
        for i, key in enumerate(keys)
        if key is None or key not in cache
    ]
