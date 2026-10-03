"""What a certificate certified: the judge's configuration, so it cannot vouch for another one.

A certificate covers one judge configuration. Change the rubric, the model, what the judge sees,
its settings, or the pydantic-evals version that builds its prompt, and the evidence is about a
different judge. `judge_identity` records those settings when a judge is certified, and
`Certificate.covers(judge)` compares them before a certificate is used to trust its scores.

What the identity captures:

- An `LLMJudge`'s rubric, model name, `include_input`, `include_expected_output`, model settings,
  score and assertion configuration; any other dataclass evaluator's class, module, code (its
  methods' `_function_digest` and its other class attributes), fields and state stored outside
  them; the pydantic-evals version.
- A model by its name (with its provider's system): `openai:gpt-5`, or
  `FunctionModel(fn, model_name='my-judge-v2')`. The name is the claim that it is that model, and
  the user owns it: change what the model does without changing its name, and nothing here can
  tell. Name a function-backed judge after a version you bump when its code changes.
- Settings and a judge's own fields by value, with their types: a tuple is not a list, a set is
  not a frozenset, a dict subclass (with its instance state) is not a dict, a dataclass or model
  carries its class. Mutable containers held as a judge's field are configuration, so they are
  recorded by value like the rest.
- Judges held by a judge, at any depth of dicts, lists, tuples, sets and dataclasses (a router's
  judges, a jury's members), by their own identity.

What is recorded only on a best-effort basis, and so marks the identity as not reliable:

- A function-backed model without a name of its own (pydantic-ai names it `function:<fn>:`).
  Arbitrary code cannot be identified soundly: what a function reaches through globals, the
  helpers it calls, the objects it holds are not followed. Its label carries a digest of its code
  (bytecode, constants including nested functions' code, the names it uses, its parameters by
  name, order and kind, its default arguments by value, and what it closes over), which tells
  judges from one factory apart within a process for `covers` and messages, but the identity
  lists the model under `opaque`, so `identity_reliable` is False and nothing that reuses work
  (the cache, saved and session certificates, jury reuse) trusts it.
- Mutable or unidentifiable state closed over by a function (a list, dict or set, which may be a
  counter that grows as the judge runs; any other object, or a method bound to one): recorded by
  its type only, and named in `opaque`.
- An evaluator that is not a dataclass, recorded by its `repr`; any other setting that is not
  data, recorded by its text.

`identity_reliable` is False whenever `opaque` is not empty: two judges that differ only there
share the identity, so a cache or a saved certificate must not be reused on it alone.

Functions held directly as a judge's setting (not as its model) are recorded by the same digest
and are not marked opaque unless what they close over is; what they reach through globals is not
followed. Bytecode differs between Python versions, so a digest is stable across processes on one
Python version, not across versions. Nothing holds a memory address.
"""

from __future__ import annotations

import collections
import contextvars
import dataclasses
import enum
import functools
import hashlib
import inspect
import json
import os
import re
import types
from typing import Any

import pydantic

_MAX_DEPTH = 6
"""How deep functions are followed through closures before the rest is called opaque."""


class _NotData(Exception):
    """A value that is not plain immutable data."""


def _data(value: Any) -> Any:
    """Plain immutable data by value, with type tags so `1`, `1.0`, `True` and `'1'` differ."""
    kind = type(value)
    if value is None:
        return ['none']
    if value is Ellipsis:
        return ['ellipsis']
    if kind in (bool, int, str):
        return [kind.__name__, value]
    if kind in (float, complex):
        return [kind.__name__, repr(value)]  # repr keeps nan, inf and -0.0 apart
    if kind is bytes:
        return ['bytes', value.hex()]
    if kind is tuple:
        return ['tuple', [_data(v) for v in value]]
    if kind is frozenset:  # sorted by a stable form, not iteration order (which follows the hash seed)
        return ['frozenset', sorted((_data(v) for v in value), key=json.dumps)]
    if isinstance(value, types.CodeType):
        return ['code', _code_digest(value)]
    raise _NotData


def _code_digest(code: types.CodeType) -> str:
    """The code itself: bytecode, constants (nested functions' code included), the names it uses, and
    its parameters by name, order and kind, so `f(output, expected)` and `f(expected, output)`, which
    differ when called by keyword, differ here too."""
    consts = []
    for const in code.co_consts:
        try:
            consts.append(_data(const))
        except _NotData:  # pragma: no cover  # the compiler only emits plain constants
            consts.append(['type', _name(type(const))])
    flags = code.co_flags & (inspect.CO_VARARGS | inspect.CO_VARKEYWORDS)
    count = code.co_argcount + code.co_kwonlyargcount + bin(flags).count('1')  # with *args and **kwargs
    params = [list(code.co_varnames[:count]), code.co_argcount, code.co_posonlyargcount, code.co_kwonlyargcount, flags]
    material = [code.co_code.hex(), consts, list(code.co_names), params]
    return hashlib.sha256(json.dumps(material).encode()).hexdigest()[:16]


def _name(obj: Any) -> str:
    return f'{getattr(obj, "__module__", "")}.{getattr(obj, "__qualname__", type(obj).__qualname__)}'


def _value(value: Any, opaque: list[str], where: str, depth: int, path: tuple[int, ...]) -> Any:
    """A value a function closes over (or a default argument): by value where that is its meaning."""
    try:
        return _data(value)
    except _NotData:
        pass
    if _is_evaluator(value):
        identity = judge_identity(value)
        opaque.extend(f'{where}.{item}' for item in identity.get('opaque', ()))
        return ['judge', identity]
    if isinstance(value, type):
        return ['class', _name(value), _class_digest(value, opaque)]
    if isinstance(value, enum.Enum):
        return ['enum', _name(type(value)), value.name, _value(value.value, opaque, where, depth, path),
                _class_digest(type(value), opaque)]  # fmt: skip
    if callable(value):
        return ['function', _function_digest(value, opaque, where, depth + 1, path)]
    # A list, dict or set may be a counter or a log that changes as the judge runs, and an object
    # has no value we can trust to mean its behaviour: its type, and the identity says so.
    opaque.append(f'{where}: {type(value).__qualname__}')
    return ['opaque', _name(type(value))]


def _function_digest(
    fn: Any, opaque: list[str] | None = None, where: str = 'function', depth: int = 0, path: tuple[int, ...] = ()
) -> str:
    """A stable digest of a callable: its name, its code, its defaults, and what it closes over.

    What cannot be identified by value is appended to `opaque`, described by where it was found.
    """
    opaque = [] if opaque is None else opaque
    parts: list[Any] = [_name(fn)]
    if id(fn) in path:  # a function that closes over itself (recursion): already being described
        return hashlib.sha256(json.dumps(['recursive', _name(fn)]).encode()).hexdigest()[:12]
    path = (*path, id(fn))
    if depth > _MAX_DEPTH:
        opaque.append(f'{where}: nested deeper than {_MAX_DEPTH} functions')
        parts.append('too deep')
    elif isinstance(fn, functools.partial):
        parts.append(['partial', _function_digest(fn.func, opaque, f'{where}.func', depth + 1, path)])
        parts.append([_value(a, opaque, f'{where}.args[{i}]', depth, path) for i, a in enumerate(fn.args)])
        parts.append({k: _value(v, opaque, f'{where}.{k}', depth, path) for k, v in sorted(fn.keywords.items())})
    elif isinstance(fn, types.MethodType):
        parts.append(_function_digest(fn.__func__, opaque, where, depth + 1, path))
        parts.append(_value(fn.__self__, opaque, f'{where}.__self__', depth, path))
    elif isinstance(fn, types.FunctionType):
        code = fn.__code__
        parts.append(_code_digest(code))
        parts.append([_value(d, opaque, f'{where}.defaults[{i}]', depth, path)
                      for i, d in enumerate(fn.__defaults__ or ())])  # fmt: skip
        kwdefaults = fn.__kwdefaults__ or {}
        parts.append({k: _value(v, opaque, f'{where}.{k}', depth, path) for k, v in sorted(kwdefaults.items())})
        for name, cell in zip(code.co_freevars, fn.__closure__ or (), strict=True):
            try:
                value = cell.cell_contents
            except ValueError:  # an empty cell: a name assigned later in the enclosing function
                parts.append([name, 'empty'])
                continue
            parts.append([name, _value(value, opaque, f'{where}.{name}', depth, path)])
    elif isinstance(fn, types.BuiltinFunctionType):
        owner = fn.__self__
        if owner is not None and not isinstance(owner, types.ModuleType):  # a method of some object
            parts.append(_value(owner, opaque, f'{where}.__self__', depth, path))
    else:  # a callable object: its state is not something we can read faithfully
        opaque.append(f'{where}: {type(fn).__qualname__}')
    return hashlib.sha256(json.dumps(parts).encode()).hexdigest()[:12]


def _function_backed(model: Any) -> Any:
    """The model a `FunctionModel`-like model is, through wrappers (`.wrapped`), or None."""
    for _ in range(_MAX_DEPTH):
        if model is None:
            return None
        if callable(getattr(model, 'function', None)) or callable(getattr(model, 'stream_function', None)):
            return model
        model = getattr(model, 'wrapped', None)
    return None  # pragma: no cover


def _default_function_name(model: Any) -> str:
    """The name pydantic-ai gives a `FunctionModel` that was not given one: `function:<fn>:<stream fn>`."""
    names = [getattr(fn, '__name__', type(fn).__name__) if fn is not None else ''
             for fn in (getattr(model, 'function', None), getattr(model, 'stream_function', None))]  # fmt: skip
    return f'function:{names[0]}:{names[1]}'


def _model_name(model: Any, opaque: list[str] | None = None, where: str = 'model') -> str | None:
    """A model by its name, which the user owns: `FunctionModel(fn, model_name='my-judge-v2')` is
    identified as 'my-judge-v2' (with its system), whatever `fn` does, as `openai:gpt-5` is.

    A function-backed model without a name of its own (pydantic-ai's `function:<fn>:<stream fn>`)
    is named only after Python code, and code cannot be identified soundly (what it reaches through
    globals, what it calls, objects it holds): it is listed as opaque, so its identity is never
    reliable. The label still carries a best-effort digest of the code, which tells judges from one
    factory apart within a process (`covers`, messages), but nothing is reused on the strength of it.
    """
    opaque = [] if opaque is None else opaque
    if model is None or isinstance(model, str):
        return model
    system, name = getattr(model, 'system', None), getattr(model, 'model_name', None)
    if name is None:
        opaque.append(f'{where}: {type(model).__qualname__} has no model_name')
        return type(model).__qualname__
    label = f'{system}:{name}' if system else str(name)
    backed = _function_backed(model)
    if backed is not None and str(name) == _default_function_name(backed):
        opaque.append(f'{where}: unnamed function-backed model ({name}); give it a model_name to have it trusted')
        digests = [_function_digest(fn, opaque, f'{where}.{attr}')
                   for attr in ('function', 'stream_function')
                   if (fn := getattr(backed, attr, None)) is not None]  # fmt: skip
        label += '#' + '+'.join(digests)
    return label


def _is_evaluator(value: Any) -> bool:
    return not isinstance(value, type) and callable(getattr(value, 'evaluate', None))


_TAG = '!type'
"""The key that marks a tagged value in `_plain`'s output. A plain dict holding this key is tagged
itself, so no setting can pass for a tagged one."""


def _instance_state(value: Any) -> list[tuple[str, Any]]:
    """What an instance of a builtin container's subclass holds besides its items: its `__dict__`
    and its `__slots__`, by attribute name (a defaultdict's `default_factory` too)."""
    state: dict[str, Any] = dict(getattr(value, '__dict__', None) or {})
    for cls in type(value).__mro__:
        slots = cls.__dict__.get('__slots__', ())
        for slot in (slots,) if isinstance(slots, str) else slots:
            if slot in ('__dict__', '__weakref__'):
                continue
            attr = f'_{cls.__name__.lstrip("_")}{slot}' if slot.startswith('__') and not slot.endswith('__') else slot
            missing = object()
            held = getattr(value, attr, missing)
            if held is not missing:
                state[attr] = held
    if isinstance(value, collections.defaultdict):
        state['default_factory'] = value.default_factory  # pyright: ignore[reportUnknownMemberType]
    return sorted(state.items())


def _is_model(value: Any) -> bool:
    """A pydantic-ai model (a `FunctionModel`, a provider's model, a wrapper of one)."""
    try:
        from pydantic_ai.models import Model
    except ImportError:  # pragma: no cover
        return False
    return isinstance(value, Model)


def _plain(value: Any, opaque: list[str] | None = None, where: str = 'setting') -> Any:
    """A JSON-stable form of a setting: judges by their identity, everything else by value and type.

    A judge inside a judge (a router's cheap and expensive judge, a jury's members, judges held in
    a dict) is recorded by its own identity, never its `repr`, which can hold a memory address and
    so differ every run. A model is recorded as `_model_name` records it. Functions are recorded by
    name and digest (see `_function_digest`).

    Containers in a judge's own fields are configuration and are recorded by value, with their type:
    a plain list and a plain dict with string keys as themselves, and everything else (a tuple, a
    set, a frozenset, a dict with other keys, any subclass of these with its instance state, a
    dataclass or pydantic model with its class, an enum, a class, a function) as a dict tagged with
    `_TAG`, so `()` and `[]`, `{1}` and `frozenset({1})`, `{'a': 1}` and `OrderedDict(a=1)` differ.
    """
    opaque = [] if opaque is None else opaque
    kind = type(value)
    if value is None or kind in (bool, int, float, str):
        return value
    if _is_evaluator(value):
        identity = judge_identity(value)
        opaque.extend(f'{where}.{item}' for item in identity.get('opaque', ()))
        return identity
    if _is_model(value):
        return {_TAG: 'model', 'name': _model_name(value, opaque, where)}
    if kind is dict and _TAG not in value and all(type(k) is str for k in value):  # pyright: ignore[reportUnknownVariableType]
        items: dict[str, Any] = value
        if list(items) == sorted(items):
            return {k: _plain(v, opaque, f'{where}.{k}') for k, v in items.items()}
        # Out of sorted order: kept in its order, since a judge can read it in that order.
        return {_TAG: 'dict', 'items': [[k, _plain(v, opaque, f'{where}.{k}')] for k, v in items.items()]}
    if kind is list:
        return [_plain(v, opaque, f'{where}[{i}]') for i, v in enumerate(value)]  # pyright: ignore[reportUnknownArgumentType,reportUnknownVariableType]
    if isinstance(value, pydantic.BaseModel):
        fields = {n: _plain(getattr(value, n), opaque, f'{where}.{n}')
                  for n in [*type(value).model_fields, *(value.model_extra or {})]}  # fmt: skip
        out = {_TAG: f'pydantic:{_name(kind)}', 'fields': fields, 'code': _class_digest(kind, opaque),
               'fields_set': sorted(value.model_fields_set)}  # fmt: skip
        if private := getattr(value, '__pydantic_private__', None):  # `PrivateAttr`s a judge can read too
            out['private'] = {k: _plain(v, opaque, f'{where}.{k}') for k, v in sorted(private.items())}
        return out
    if (
        not isinstance(value, type)
        and callable(getattr(value, 'model_dump', None))
        and (kind.__module__ or '').split('.')[0] == __name__.split('.')[0]
    ):
        dumpable: Any = value  # one of this package's judges that says how it is recorded (a jury's members)
        return _plain(dumpable.model_dump(), opaque, where)
    if isinstance(value, dict | list | tuple | set | frozenset):
        container: Any = value
        tag = kind.__name__ if kind in (dict, tuple, set, frozenset) else f'{_name(kind)}'
        if isinstance(value, dict):
            members = [[_plain(k, opaque, f'{where} key'), _plain(v, opaque, f'{where}.{k}')]
                       for k, v in container.items()]  # fmt: skip
        elif isinstance(value, set | frozenset):
            members = sorted((_plain(v, opaque, f'{where}{{}}') for v in container),
                             key=lambda v: json.dumps(v, sort_keys=True))  # fmt: skip
        else:
            members = [_plain(v, opaque, f'{where}[{i}]') for i, v in enumerate(container)]
        out: dict[str, Any] = {_TAG: tag, 'items': members}
        if kind not in (dict, list, tuple, set, frozenset):  # a subclass: its state and its methods
            if state := _instance_state(value):
                out['state'] = {k: _plain(v, opaque, f'{where}.{k}') for k, v in state}
            out['code'] = _class_digest(kind, opaque)
        return out
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        fields = {
            f.name: _plain(getattr(value, f.name), opaque, f'{where}.{f.name}') for f in dataclasses.fields(value)
        }
        out = {_TAG: f'dataclass:{_name(kind)}', 'fields': fields, 'code': _class_digest(kind, opaque)}
        if extra := _extra_state(value):
            out['state'] = {k: _plain(v, opaque, f'{where}.{k}') for k, v in extra}
        return out
    if isinstance(value, enum.Enum):  # its value and its class's code: a judge can read either
        return {_TAG: f'enum:{_name(kind)}', 'name': value.name, 'value': _plain(value.value, opaque, where),
                'code': _class_digest(kind, opaque)}  # fmt: skip
    if isinstance(value, type):  # a class held as configuration: its behaviour is its code
        return {_TAG: 'class', 'name': _name(value), 'code': _class_digest(value, opaque)}
    if callable(value):
        # Code can be fingerprinted only so far (not what it reaches through globals), so a function
        # held as a setting is recorded, best effort, and marked opaque: nothing reuses on its strength.
        opaque.append(f'{where}: function {_name(value)} (its code is fingerprinted best effort only)')
        return {_TAG: 'function', 'name': _name(value), 'digest': _function_digest(value, opaque, where)}
    opaque.append(f'{where}: {type(value).__qualname__}')
    return {_TAG: 'opaque', 'type': _name(kind), 'text': re.sub(r' at 0x[0-9a-f]+', '', str(value))}


_LIBRARY_ROOTS = frozenset(
    {'builtins', 'typing', 'abc', 'enum', 'collections', 'pydantic', 'pydantic_core', 'pydantic_evals'}
)
"""Packages whose classes are identified by the installed version (pydantic-evals is recorded), not by digest."""

_STDLIB = os.path.dirname(dataclasses.__file__)
_ANNOTATION_MACHINERY = frozenset({'__annotate__', '__annotate_func__', '__annotations__', '__annotations_cache__'})
"""What Python 3.14 keeps on every class for its annotations: `__annotate_func__` closes over the class
namespace, so digesting it marked every class written without `from __future__ import annotations`
opaque. A class's annotated fields are recorded by value already."""
_DATACLASS_METHODS = frozenset({'__init__', '__repr__', '__eq__', '__hash__', '__setattr__', '__delattr__',
                                '__lt__', '__le__', '__gt__', '__ge__', '__getstate__', '__setstate__'})  # fmt: skip
_DIGESTING: contextvars.ContextVar[frozenset[int]] = contextvars.ContextVar('_DIGESTING', default=frozenset())


def _sunder(name: str) -> bool:
    """`_name_`: what `enum` keeps on every enum class (its members by name and value, `_new_member_`)."""
    return len(name) > 2 and name[0] == name[-1] == '_' and name[1] != '_' and name[-2] != '_'


def _generated(klass: type, name: str, attr: Any) -> bool:
    """A method Python wrote, not the class's author: one from the standard library (a dataclass's
    `__replace__`, `reprlib`'s wrapper of `__repr__`), or one `dataclasses` built from the fields,
    which the identity records already."""
    if not isinstance(attr, types.FunctionType):
        return False
    filename = attr.__code__.co_filename
    if os.path.dirname(filename) == _STDLIB:
        return True
    return filename == '<string>' and name in _DATACLASS_METHODS and dataclasses.is_dataclass(klass)


def _class_digest(cls: type, opaque: list[str]) -> str:
    """The code of an evaluator class and of its bases outside pydantic-evals: every method's
    `_function_digest` (code, defaults, what it closes over; properties, static and class methods
    unwrapped) and every other class attribute by value, so editing `evaluate`, or a class-level
    setting it reads, makes a different judge. Like a named model's name, it cannot follow what
    the methods reach through globals (a helper in another module, a prompt file)."""
    digesting = _DIGESTING.get()
    if id(cls) in digesting:  # a method that refers to its own class (`super()`, a classmethod's `cls`)
        return f'recursive:{_name(cls)}'
    token = _DIGESTING.set(digesting | {id(cls)})
    try:
        return _class_material(cls, opaque)
    finally:
        _DIGESTING.reset(token)


def _class_material(cls: type, opaque: list[str]) -> str:
    material: list[Any] = []
    meta = type(cls)  # a metaclass's methods are the class's too (`Config.permissive()`)
    if meta is not type and (meta.__module__ or '').split('.')[0] not in _LIBRARY_ROOTS:
        material.append(['metaclass', _name(meta), _class_digest(meta, opaque)])
    for klass in cls.__mro__:
        module = klass.__module__ or ''
        if klass is object or module.split('.')[0] in _LIBRARY_ROOTS:
            continue
        for name, attr in sorted(vars(klass).items()):
            held = attr.__func__ if isinstance(attr, staticmethod | classmethod) else attr
            dunder = name.startswith('__') and name.endswith('__')
            if (
                (dunder and not isinstance(held, types.FunctionType | property))
                or name.startswith('_abc_')
                or name in _ANNOTATION_MACHINERY
                or type(attr).__name__ == '_tuplegetter'  # a namedtuple's field accessors, built by `collections`
                or _generated(klass, name, held)
                or (issubclass(klass, enum.Enum) and (_sunder(name) or isinstance(attr, klass)))
            ):
                continue  # what Python, abc and dataclasses put on every class (__module__, a generated __eq__)
            where = f'{klass.__qualname__}.{name}'
            if isinstance(held, property):
                parts = [_function_digest(f, opaque, where) for f in (held.fget, held.fset, held.fdel) if f]
                material.append([_name(klass), name, 'property', parts])
            elif isinstance(held, types.FunctionType):
                material.append([_name(klass), name, _function_digest(held, opaque, where)])
            else:
                material.append([_name(klass), name, _plain(held, opaque, where)])
    return hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()[:16]


_RESERVED = frozenset({'evaluator', 'pydantic_evals', 'opaque'})
"""Identity keys a dataclass field could share; the identity's own keys that are not identifiers start with `!`."""


def _extra_state(value: Any) -> list[tuple[str, Any]]:
    """A dataclass instance's attributes that are not its fields, by name, in sorted order."""
    names = {f.name for f in dataclasses.fields(value)}
    return sorted(((k, v) for k, v in _instance_state(value) if k not in names), key=lambda kv: kv[0])


def judge_identity(judge: Any) -> dict[str, Any]:
    """The settings that make a judge the judge it is, as plain data.

    For an `LLMJudge`: its rubric, model, what it is shown, model settings, score and assertion
    configuration, and the pydantic-evals version that writes its prompt. For any other
    evaluator: its class and its dataclass fields, or its `repr`. What could not be identified is
    listed under `opaque` (absent when everything was); see the module docstring.
    """
    from importlib.metadata import PackageNotFoundError, version

    try:
        evals_version = version('pydantic-evals')
    except PackageNotFoundError:  # pragma: no cover
        evals_version = None
    kind = type(judge).__qualname__
    opaque: list[str] = []
    upstream = kind == 'LLMJudge' and type(judge).__module__.startswith('pydantic_evals.')
    if upstream:
        fields = {
            'rubric': judge.rubric,
            'model': _model_name(judge.model, opaque),
            'include_input': judge.include_input,
            'include_expected_output': judge.include_expected_output,
            'model_settings': _plain(judge.model_settings, opaque, 'model_settings'),
            'score': _plain(judge.score, opaque, 'score'),
            'assertion': _plain(judge.assertion, opaque, 'assertion'),
        }
    elif dataclasses.is_dataclass(judge) and not isinstance(judge, type):
        values = {f.name: getattr(judge, f.name) for f in dataclasses.fields(judge)}
        fields = {k: _model_name(v, opaque) if k == 'model' else _plain(v, opaque, k) for k, v in values.items()}
        # What `__post_init__` (an `InitVar`, a cache) stores outside the fields can decide verdicts too.
        if extra := _extra_state(judge):
            fields['!state'] = {k: _plain(v, opaque, k) for k, v in extra}
        for clash in fields.keys() & _RESERVED:
            opaque.append(f"{clash}: a field named like the identity's own keys")
    else:
        fields = {'repr': re.sub(r' at 0x[0-9a-f]+', '', repr(judge))}
        opaque.append(f'repr: {kind} is not a dataclass')
    _lift(fields, opaque, '')
    identity = {**fields, 'evaluator': kind, 'pydantic_evals': evals_version}
    if not upstream:  # two `Judge` classes in two modules, or one edited, are two judges
        identity['!module'] = type(judge).__module__
        identity['!code'] = _class_digest(type(judge), opaque)
    if opaque:
        identity['opaque'] = sorted(set(opaque))
    return dict(sorted(identity.items()))  # keys sorted, so an identity held inside another reads as plain data


def _lift(value: Any, opaque: list[str], where: str) -> None:
    """Collect `opaque` from identities nested where `_plain` did not build them (a jury's `model_dump`)."""
    if isinstance(value, dict):
        mapping: dict[str, Any] = value
        if 'evaluator' in mapping and mapping.get('opaque'):
            opaque.extend(f'{where}.{item}'.lstrip('.') for item in mapping['opaque'])
        for k, v in mapping.items():
            if k != 'opaque':
                _lift(v, opaque, f'{where}.{k}')
    elif isinstance(value, list):
        for i, v in enumerate(value):  # pyright: ignore[reportUnknownArgumentType,reportUnknownVariableType]
            _lift(v, opaque, f'{where}[{i}]')


def identity_reliable(identity: dict[str, Any]) -> bool:
    """Whether the identity holds everything that makes the judge what it is: nothing opaque.

    When False, two judges that differ only in what is listed under `opaque` share this identity;
    reusing a stored verdict or certificate on the strength of it alone is not safe.
    """
    found: list[str] = []
    _lift(identity, found, '')
    return not found


def fingerprint(identity: dict[str, Any]) -> str:
    """A short, stable hash of an identity, for logs and certificate files."""
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]


def differences(certified: dict[str, Any], judge: dict[str, Any]) -> list[str]:
    """The settings that differ, by name: what changed since the certificate was issued.

    Compared exactly, as the fingerprint is: `True`, `1` and `1.0` are different settings.
    """

    def exact(value: Any) -> str:
        return json.dumps(value, sort_keys=True)

    return sorted(k for k in certified.keys() | judge.keys() if exact(certified.get(k)) != exact(judge.get(k)))
