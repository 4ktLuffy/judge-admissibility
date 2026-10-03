"""Re-score without re-running the task: one run per case, reused only for exactly what produced it."""

from __future__ import annotations

import asyncio
import dataclasses
import datetime
import decimal
import enum
import functools
import json
import threading
import uuid
from collections import namedtuple
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict, PrivateAttr
from pydantic_evals import Case, Dataset
from pydantic_evals.dataset import increment_eval_metric, set_eval_attribute
from pydantic_evals.evaluators import Evaluator, EvaluatorContext
from pydantic_evals.lifecycle import CaseLifecycle
from pydantic_evals.otel._errors import SpanTreeRecordingError

from pydantic_evals_admissibility._task_cache import (
    OutputCodec,
    TaskCache,
    TaskCacheStats,
    _encode,  # pyright: ignore[reportPrivateUsage]
    evaluate_cached,
    evaluate_cached_sync,
    task_identity,
    task_identity_reliable,
    uncached_cases,
)

CALLS: list[Any] = []  # module state: a task reading it is not fully identified, so these tasks carry a version
PREFIX = 'answer: '


@pytest.fixture(autouse=True)
def _reset() -> None:
    CALLS.clear()


@dataclasses.dataclass
class StartsWith(Evaluator[Any, Any, Any]):
    prefix: str = 'answer'

    def evaluate(self, ctx: EvaluatorContext[Any, Any, Any]) -> bool:
        return str(ctx.output).startswith(self.prefix)


@dataclasses.dataclass
class Longer(Evaluator[Any, Any, Any]):
    than: int = 5

    def evaluate(self, ctx: EvaluatorContext[Any, Any, Any]) -> bool:
        return len(str(ctx.output)) > self.than


async def counted(inputs: str) -> str:
    CALLS.append(inputs)
    await asyncio.sleep(0.01)
    return PREFIX + inputs.upper()


def dataset(*inputs: str, evaluators: tuple[Evaluator[Any, Any, Any], ...] = (StartsWith(),)) -> Dataset[Any, Any, Any]:
    return Dataset(
        name='d', cases=[Case(name=f'c{i}', inputs=x) for i, x in enumerate(inputs)], evaluators=list(evaluators)
    )


def stats(report: Any) -> TaskCacheStats:
    return TaskCacheStats.from_report(report)


async def run(ds: Dataset[Any, Any, Any], task: Any, cache: TaskCache, **kw: Any) -> Any:
    return await evaluate_cached(ds, task, cache=cache, progress=False, **kw)


async def test_the_task_runs_once_per_case_across_rescorings() -> None:
    cache = TaskCache()
    first = await run(dataset('a', 'b', 'c'), counted, cache, version='v1')
    assert sorted(CALLS) == ['a', 'b', 'c'] and (stats(first).runs, stats(first).hits) == (3, 0)
    # Re-scored twice with other evaluators: the evaluators run, the task does not.
    for evaluators in [(StartsWith(), Longer()), (Longer(than=100),)]:
        again = await run(dataset('a', 'b', 'c', evaluators=evaluators), counted, cache, version='v1')
        assert len(CALLS) == 3 and (stats(again).runs, stats(again).hits) == (0, 3)
        assert [c.output for c in again.cases] == [c.output for c in first.cases]
        assert {n for c in again.cases for n in c.assertions} == {type(e).__name__ for e in evaluators}
    assert (cache.runs, cache.hits) == (3, 6)
    longer = {c.name: c.assertions['Longer'].value for c in again.cases}
    assert longer == {'c0': False, 'c1': False, 'c2': False}


async def test_a_served_output_is_marked_and_keeps_this_runs_duration_and_metrics() -> None:
    async def task(inputs: str) -> str:
        CALLS.append(inputs)
        set_eval_attribute('retrieved', ['doc-1', inputs])
        increment_eval_metric('tokens', 7)
        await asyncio.sleep(0.05)
        return inputs

    cache = TaskCache()
    fresh = (await run(dataset('a'), task, cache, version='v1')).cases[0]
    served = (await run(dataset('a'), task, cache, version='v1')).cases[0]
    assert fresh.attributes['task_cache']['cached'] is False and fresh.attributes['task_cache']['stored'] is True
    note = served.attributes['task_cache']
    assert note['cached'] is True and note['source'] == 'cache'
    assert 0.05 <= note['original_task_duration'] <= fresh.task_duration
    assert served.task_duration < 0.05  # this run's duration, not the original's
    assert fresh.metrics == {'tokens': 7} and served.metrics == {} and note['original_metrics'] == {'tokens': 7}
    assert served.attributes['retrieved'] == ['doc-1', 'a']  # what the task set is part of its output


async def test_a_served_output_has_no_span_tree() -> None:
    seen: list[str] = []

    @dataclasses.dataclass
    class Spans(Evaluator[Any, Any, Any]):
        def evaluate(self, ctx: EvaluatorContext[Any, Any, Any]) -> bool:
            try:
                _ = ctx.span_tree
            except SpanTreeRecordingError as exc:
                seen.append(str(exc))
            return True

    cache = TaskCache()
    await run(dataset('a', evaluators=(Spans(),)), counted, cache, version='v1')
    seen.clear()
    await run(dataset('a', evaluators=(Spans(),)), counted, cache, version='v1')
    assert seen and 'served from the task cache' in seen[0]


async def test_an_edited_case_reruns_and_nothing_else() -> None:
    cache = TaskCache()
    await run(dataset('a', 'b', 'c'), counted, cache, version='v1')
    CALLS.clear()
    report = await run(dataset('a', 'B', 'c'), counted, cache, version='v1')
    assert CALLS == ['B'] and stats(report).ran == ('c1',)
    assert uncached_cases(dataset('a', 'B', 'c'), counted, cache=cache, version='v1') == []
    assert uncached_cases(dataset('x', 'B'), counted, cache=cache, version='v1') == ['c0']


async def test_inputs_are_keyed_with_their_types_and_order() -> None:
    async def echo(inputs: Any) -> str:
        CALLS.append(inputs)
        return repr(inputs)

    variants: list[Any] = [1, 1.0, True, '1', ('a', 'b'), ['a', 'b'], ['b', 'a'], {'x': 1, 'y': 2}, {'y': 2, 'x': 1}]
    cache = TaskCache()
    for v in variants:
        ds = Dataset(name='d', cases=[Case(name='only', inputs=v)])
        await run(ds, echo, cache, version='v1')
    assert len(CALLS) == len(variants) and len(cache) == len(variants)
    report = await run(Dataset(name='d', cases=[Case(name='only', inputs={'y': 2, 'x': 1})]), echo, cache, version='v1')
    assert report.cases[0].output == "{'y': 2, 'x': 1}" and len(CALLS) == len(variants)


def _make(source: str) -> Any:
    """A module-level `task` from source, so two versions share a name and differ only in code."""
    namespace = dict(globals())
    exec(compile(source, __file__, 'exec'), namespace)
    return namespace['task']


async def test_edited_task_code_reruns_every_case() -> None:
    v1 = _make('async def task(inputs):\n    CALLS.append(inputs)\n    return inputs.upper()\n')
    v2 = _make('async def task(inputs):\n    CALLS.append(inputs)\n    return inputs.lower()\n')
    v1_again = _make('async def task(inputs):\n    CALLS.append(inputs)\n    return inputs.upper()\n')
    cache = TaskCache()
    await run(dataset('a', 'b'), v1, cache, version='v1')
    await run(dataset('a', 'b'), v2, cache, version='v1')
    assert len(CALLS) == 4
    report = await run(dataset('a', 'b'), v1_again, cache, version='v1')  # the same code: served
    assert len(CALLS) == 4 and stats(report).hits == 2


async def test_a_bumped_version_reruns() -> None:
    cache = TaskCache()
    await run(dataset('a'), counted, cache, version='v1')
    await run(dataset('a'), counted, cache, version='v2')
    assert len(CALLS) == 2


def make_task(suffix: str, n: int = 1) -> Any:
    async def task(inputs: str) -> str:
        return inputs + suffix * n

    return task


async def test_closed_over_values_and_defaults_are_part_of_the_key() -> None:
    cache = TaskCache()
    first = await run(dataset('a', 'b'), make_task('!'), cache)
    assert task_identity_reliable(task_identity(make_task('!')))
    assert not stats(first).unreliable_identity and stats(first).runs == 2
    assert stats(await run(dataset('a', 'b'), make_task('!'), cache)).hits == 2  # a new closure, same values
    assert stats(await run(dataset('a', 'b'), make_task('?'), cache)).runs == 2
    assert stats(await run(dataset('a', 'b'), make_task('!', 2), cache)).runs == 2
    assert stats(await run(dataset('a', 'b'), make_task('!', True), cache)).runs == 2  # True is not 1


async def prefixed(inputs: str) -> str:
    return PREFIX + inputs


async def test_a_module_constant_the_task_reads_is_part_of_the_key(monkeypatch: pytest.MonkeyPatch) -> None:
    cache = TaskCache()
    assert stats(await run(dataset('a'), prefixed, cache)).runs == 1
    assert stats(await run(dataset('a'), prefixed, cache)).hits == 1
    monkeypatch.setitem(globals(), 'PREFIX', 'reply: ')
    report = await run(dataset('a'), prefixed, cache)
    assert stats(report).runs == 1 and report.cases[0].output == 'reply: a'


def helper(text: str) -> str:
    return text.title()


async def uses_helper(inputs: str) -> str:
    return helper(inputs)


async def test_a_helper_in_the_same_module_is_followed(monkeypatch: pytest.MonkeyPatch) -> None:
    cache = TaskCache()
    await run(dataset('a b'), uses_helper, cache)
    assert stats(await run(dataset('a b'), uses_helper, cache)).hits == 1

    def helper(text: str) -> str:  # noqa: F811  # an edit of the helper, not of the task
        return text.upper()

    helper.__module__ = __name__
    monkeypatch.setitem(globals(), 'helper', helper)
    report = await run(dataset('a b'), uses_helper, cache)
    assert stats(report).runs == 1 and report.cases[0].output == 'A B'


async def test_an_unreliable_identity_always_reruns_and_stores_nothing() -> None:
    log: list[str] = []

    async def closes_over_a_list(inputs: str) -> str:
        log.append(inputs)
        return inputs

    identity = task_identity(counted)  # reads the mutable global CALLS
    assert not task_identity_reliable(identity) and any('CALLS' in o for o in identity['opaque'])
    assert not task_identity_reliable(task_identity(closes_over_a_list))
    cache = TaskCache()
    for _ in range(3):
        report = await run(dataset('a', 'b'), counted, cache)
        assert stats(report).unreliable_identity and stats(report).runs == 2
        assert 'identity not reliable' in report.cases[0].attributes['task_cache']['reason']
        await run(dataset('a', 'b'), closes_over_a_list, cache)
    assert len(CALLS) == 6 and len(log) == 6 and len(cache) == 0
    assert uncached_cases(dataset('a', 'b'), counted, cache=cache) == ['c0', 'c1']
    # A version is the user's claim that it is the same task: then it is cached.
    await run(dataset('a', 'b'), counted, cache, version='v1')
    await run(dataset('a', 'b'), counted, cache, version='v1')
    assert len(CALLS) == 8


class Colour(enum.Enum):
    RED = 'red'
    CRIMSON = 'red'  # an alias: the member read back must be the one stored
    BLUE = 'blue'


@dataclasses.dataclass(frozen=True)
class Span:
    start: int
    end: int
    label: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, 'width', self.end - self.start)


Point = namedtuple('Point', ['x', 'y'])


class Inner(BaseModel):
    score: float
    when: datetime.datetime


class Answer(BaseModel):
    model_config = ConfigDict(extra='allow')
    text: str
    spans: list[Span]
    inner: Inner | None = None
    tags: frozenset[str] = frozenset()
    _secret: str = PrivateAttr(default='')


def rich_output(inputs: str) -> Any:
    answer = Answer(
        text=inputs,
        spans=[Span(0, 3, ('a', 'b'))],
        inner=Inner(score=-0.0, when=datetime.datetime(2026, 1, 2, 3, 4, tzinfo=datetime.timezone.utc)),
        tags=frozenset({'x', 'y'}),
        note='extra',  # pyright: ignore[reportCallIssue]
    )
    answer._secret = 'kept'
    return {
        'answer': answer,
        1: (1, 1.0, True, None, float('nan')),
        'set': {3, 1, 2},
        'bytes': b'\x00\xff',
        'decimal': decimal.Decimal('1.10'),
        'id': uuid.UUID(int=7),
        'colour': Colour.RED,
        'point': Point(1, 2),
        'delta': datetime.timedelta(days=1, microseconds=5),
        'date': datetime.date(2026, 1, 1),
        'path': Path('a/b'),
        'tz': datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone(datetime.timedelta(hours=3), 'EAT')),
    }


def rich(inputs: str) -> Any:  # a sync task: run in a thread, as pydantic-evals does
    return rich_output(inputs)


async def test_outputs_round_trip_with_their_types(tmp_path: Path) -> None:
    path = tmp_path / 'tasks.sqlite'
    with TaskCache(path) as cache:
        first = await run(dataset('a'), rich, cache)
        assert stats(first).unstored == () and stats(first).runs == 1
    with TaskCache(path) as reopened:  # from the file, in a new cache object
        report = await run(dataset('a'), rich, reopened)
    assert stats(report).hits == 1 and stats(report).runs == 0
    out, original = report.cases[0].output, rich_output('a')
    assert json.dumps(_encode(out)) == json.dumps(_encode(original))
    answer = out['answer']
    assert type(answer) is Answer and type(answer.spans[0]) is Span and type(answer.inner) is Inner
    assert answer.spans[0].label == ('a', 'b') and vars(answer.spans[0])['width'] == 3  # state set in __post_init__
    assert answer._secret == 'kept' and answer.model_extra == {'note': 'extra'}
    assert answer.model_fields_set == original['answer'].model_fields_set
    assert answer.tags == frozenset({'x', 'y'}) and type(answer.tags) is frozenset
    assert str(answer.inner.score) == '-0.0' and answer.inner.when.tzinfo == datetime.timezone.utc
    assert [type(v) for v in out[1][:4]] == [int, float, bool, type(None)] and out[1][4] != out[1][4]
    assert out['colour'] is Colour.RED and type(out['point']) is Point and out['point'].y == 2
    assert out['tz'].tzname() == 'EAT' and out['decimal'].as_tuple() == decimal.Decimal('1.10').as_tuple()
    assert answer is not first.cases[0].output['answer']  # each case is given its own copy


class Opaque:
    def __init__(self, value: str) -> None:
        self.value = value


def opaque_task(inputs: str) -> Opaque:
    return Opaque(inputs)


async def test_an_output_that_cannot_be_stored_faithfully_is_used_and_not_stored() -> None:
    cache = TaskCache()
    for _ in range(2):
        report = await run(dataset('a'), opaque_task, cache)
        assert stats(report).runs == 1 and 'codec' in stats(report).unstored[0][1]
        assert report.cases[0].output.value == 'a' and report.cases[0].attributes['task_cache']['stored'] is False
    assert len(cache) == 0
    codec = OutputCodec(dump=lambda o: {'value': o.value}, load=lambda d: Opaque(d['value']))
    await run(dataset('a'), opaque_task, cache, codec=codec)
    report = await run(dataset('a'), opaque_task, cache, codec=codec)
    assert stats(report).hits == 1 and type(report.cases[0].output) is Opaque and report.cases[0].output.value == 'a'


async def test_a_lossy_codec_is_refused() -> None:
    codec = OutputCodec(dump=lambda o: {'value': o.value}, load=lambda d: Opaque(d['value'].upper()))
    report = await run(dataset('a'), opaque_task, TaskCache(), codec=codec)
    assert stats(report).unstored and 'read back the same' in stats(report).unstored[0][1]


async def test_concurrent_evaluations_do_not_run_a_case_twice() -> None:
    cache = TaskCache()
    ds = dataset('a', 'b', 'c', 'd')
    first, second = await asyncio.gather(run(ds, counted, cache, version='v1'), run(ds, counted, cache, version='v1'))
    assert sorted(CALLS) == ['a', 'b', 'c', 'd']
    assert stats(first).runs + stats(second).runs == 4 and stats(first).hits + stats(second).hits == 4
    sources = {c.attributes['task_cache'].get('source') for r in (first, second) for c in r.cases}
    assert 'concurrent run' in sources


def test_concurrent_evaluations_in_threads_do_not_run_a_case_twice() -> None:
    cache, reports = TaskCache(), []
    ds = dataset('a', 'b', 'c')

    def evaluate() -> None:
        reports.append(evaluate_cached_sync(ds, counted, cache=cache, version='v1', progress=False))

    threads = [threading.Thread(target=evaluate) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(CALLS) == ['a', 'b', 'c'] and sum(stats(r).runs for r in reports) == 3


async def test_refresh_reruns_all_or_only_the_named_cases() -> None:
    cache = TaskCache()
    await run(dataset('a', 'b', 'c'), counted, cache, version='v1')
    CALLS.clear()
    report = await run(dataset('a', 'b', 'c'), counted, cache, version='v1', refresh=['c1'])
    assert CALLS == ['b'] and stats(report).refreshed == 1 and stats(report).hits == 2
    report = await run(dataset('a', 'b', 'c'), counted, cache, version='v1', refresh=True)
    assert len(CALLS) == 4 and stats(report).refreshed == 3
    with pytest.raises(ValueError, match='not in the dataset'):
        await run(dataset('a'), counted, cache, version='v1', refresh=['nope'])


async def test_identical_cases_and_repeats_stay_independent_samples() -> None:
    cache = TaskCache()
    ds = Dataset(name='d', cases=[Case(name='x', inputs='a'), Case(name='y', inputs='a')])
    await run(ds, counted, cache, version='v1')
    assert len(CALLS) == 2  # two cases with the same inputs: two runs, not one
    report = await run(ds, counted, cache, version='v1', repeat=3)
    assert len(CALLS) == 6 and stats(report).hits == 2 and stats(report).runs == 4
    report = await run(ds, counted, cache, version='v1', repeat=3)
    assert len(CALLS) == 6 and stats(report).hits == 6


async def test_errors_are_not_stored() -> None:
    failed: set[str] = set()

    async def flaky(inputs: str) -> str:
        CALLS.append(inputs)
        if inputs == 'b' and 'b' not in failed:
            failed.add('b')
            raise RuntimeError('rate limited')
        return inputs

    cache = TaskCache()
    first = await run(dataset('a', 'b'), flaky, cache, version='v1')
    assert len(first.failures) == 1 and len(cache) == 1
    second = await run(dataset('a', 'b'), flaky, cache, version='v1')
    assert not second.failures and stats(second).ran == ('c1',) and len(CALLS) == 3


async def test_a_stored_output_whose_class_changed_is_a_miss() -> None:
    cache = TaskCache()
    await run(dataset('a'), rich, cache)
    db = cache._db  # pyright: ignore[reportPrivateUsage]
    (key, output) = db.execute('SELECT key, output FROM task_outputs').fetchone()
    db.execute('UPDATE task_outputs SET output = ? WHERE key = ?', (output.replace('"code": "', '"code": "x'), key))
    report = await run(dataset('a'), rich, cache)
    assert stats(report).runs == 1 and type(report.cases[0].output['answer']) is Answer


async def test_metadata_is_not_keyed_unless_a_lifecycle_can_show_it_to_the_task() -> None:
    def ds(meta: str) -> Dataset[Any, Any, Any]:
        return Dataset(name='d', cases=[Case(name='c', inputs='a', metadata={'m': meta})])

    cache = TaskCache()
    await run(ds('one'), counted, cache, version='v1')
    await run(ds('two'), counted, cache, version='v1')  # the task is called with the inputs only
    assert len(CALLS) == 1

    class Fixture(CaseLifecycle[Any, Any, Any]):
        async def setup(self) -> None:
            CALLS.append(('setup', self.case.metadata))

    CALLS.clear()
    await run(ds('one'), counted, cache, version='v1', lifecycle=Fixture)
    await run(ds('one'), counted, cache, version='v1', lifecycle=Fixture)
    await run(ds('two'), counted, cache, version='v1', lifecycle=Fixture)
    assert [c for c in CALLS if c == 'a'] == ['a', 'a']  # a new key with a lifecycle, then the metadata edit
    assert CALLS.count(('setup', {'m': 'one'})) == 2  # the user's lifecycle still runs on every case


async def test_unkeyable_inputs_need_a_serializer() -> None:
    ds = Dataset(name='d', cases=[Case(name='c', inputs=Opaque('a'))])

    async def task(inputs: Opaque) -> str:
        return inputs.value

    with pytest.raises(TypeError, match='serializer'):
        await run(ds, task, TaskCache())

    def serializer(value: Any) -> Any:
        return {'value': value.value} if isinstance(value, Opaque) else value

    cache = TaskCache()
    await run(ds, task, cache, serializer=serializer)
    assert stats(await run(ds, task, cache, serializer=serializer)).hits == 1


INPUT_PREFIX = 'old:'


@dataclasses.dataclass
class Prompted:
    value: str

    def render(self) -> str:
        return INPUT_PREFIX + self.value


def renders(inputs: Prompted) -> str:
    return inputs.render()


async def test_a_global_an_input_method_reads_is_part_of_the_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Review C1: the task calls a method of its input, and that method reads a module constant."""
    ds = Dataset(name='d', cases=[Case(name='c', inputs=Prompted('hello'))])
    cache = TaskCache()
    assert (await run(ds, renders, cache)).cases[0].output == 'old:hello'
    assert stats(await run(ds, renders, cache)).hits == 1
    monkeypatch.setitem(globals(), 'INPUT_PREFIX', 'new:')
    report = await run(ds, renders, cache)
    assert not stats(report).unreliable_identity
    assert stats(report).runs == 1 and report.cases[0].output == 'new:hello'


@dataclasses.dataclass
class Logged:
    value: str

    def render(self) -> str:
        CALLS.append(self.value)  # a mutable global: what it holds cannot be pinned down
        return self.value


def renders_logged(inputs: Logged) -> str:
    return inputs.render()


async def test_an_input_whose_methods_read_mutable_state_is_never_served() -> None:
    ds = Dataset(name='d', cases=[Case(name='c', inputs=Logged('a'))])
    cache = TaskCache()
    for _ in range(2):
        assert stats(await run(ds, renders_logged, cache, version='v1')).runs == 1
    assert CALLS == ['a', 'a']


class WithCachedProperty(BaseModel):
    value: int

    @functools.cached_property
    def computed(self) -> int:
        return self.value * 2


def sets_cached_property(inputs: int) -> WithCachedProperty:
    out = WithCachedProperty(value=inputs)
    out.computed = 99  # pyright: ignore[reportAttributeAccessIssue]
    return out


async def test_a_models_cached_property_state_is_stored() -> None:
    """Review C2: state a model keeps in its `__dict__` besides its fields comes back as it was."""
    ds = Dataset(name='d', cases=[Case(name='c', inputs=1)])
    cache = TaskCache()
    fresh = (await run(ds, sets_cached_property, cache, version='v1')).cases[0]
    served = (await run(ds, sets_cached_property, cache, version='v1')).cases[0]
    assert served.attributes['task_cache']['cached'] is True
    assert fresh.output.computed == served.output.computed == 99
