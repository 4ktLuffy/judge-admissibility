"""Incremental re-certification: reuse a verdict only for exactly what was judged, by exactly that judge."""

from __future__ import annotations

import dataclasses
import json
import random
from collections import OrderedDict, defaultdict
from typing import Any

import pytest
from judges import judge, oracle, yes_man
from pydantic import BaseModel
from pydantic_evals.evaluators import Evaluator, EvaluatorContext
from test_certify import CASES

from pydantic_evals_admissibility import Certificate, HumanLabel, JudgeCase, MismatchedOutput, certify_judge
from pydantic_evals_admissibility._cache import (
    JudgmentCache,
    _keyed_plan,  # pyright: ignore[reportPrivateUsage]
    certify_judge_cached,
    judgment_key,
    uncached_judgments,
)
from pydantic_evals_admissibility._controls import DEFAULT_CONTROLS

CALLS: list[int] = []  # module state, not a closure: a judge closing over a counter is not fully identified


def counted_oracle(output: str, expected: str) -> bool:
    CALLS.append(1)
    return oracle(output, expected)


def edit(cases: list[JudgeCase], names: set[str]) -> list[JudgeCase]:
    """The same answers, reworded for `names`: still right, but no longer the text that was judged."""
    return [dataclasses.replace(c, output=f'It is {c.output}.') if c.name in names else c for c in cases]


async def test_a_cold_cache_gives_the_uncached_certificate() -> None:
    sound = judge(oracle)
    for batch_size in (None, 7):
        plain = await certify_judge(sound, CASES, batch_size=batch_size)
        cached, stats = await certify_judge_cached(sound, CASES, cache=JudgmentCache(), batch_size=batch_size)
        assert cached == plain and cached.to_dict() == plain.to_dict()
        assert (stats.hits, stats.misses) == (0, plain.calls)


async def test_a_warm_cache_asks_the_judge_nothing_and_changes_nothing() -> None:
    sound, cache = judge(counted_oracle), JudgmentCache()
    first, _ = await certify_judge_cached(sound, CASES, cache=cache)
    made = len(CALLS)
    again, stats = await certify_judge_cached(sound, CASES, cache=cache)
    assert len(CALLS) == made and stats.misses == 0 and stats.calls_saved == first.calls
    assert again == first


async def test_an_edit_costs_only_the_judgments_it_changed() -> None:
    sound, cache = judge(oracle), JudgmentCache()
    first, _ = await certify_judge_cached(sound, CASES, cache=cache)
    owner = {c.output: c.name for c in CASES}
    lent = sorted(owner[j.output] for j in first.judgments if j.role.endswith('mismatched_output'))
    changed = {lent[0], lent[-1]}  # answers some other case borrowed, so the borrowing control changes too
    edited = edit(CASES, changed)
    old = {c.output for c in CASES if c.name in changed}
    # Changed: each edited case's repeats and its reformatted answer, and every borrowed answer
    # that was one of the edited answers. Not changed: the empty answer (the judge never sees the
    # original), and the edited cases' own borrowed answers (same question, same donor).
    expected = {(n, f'reference#{i}') for n in changed for i in range(3)} | {
        (n, 'must_hold:whitespace_reformat') for n in changed
    }
    borrowed = {(j.case, j.role) for j in first.judgments if j.role.endswith('mismatched_output') and j.output in old}
    assert borrowed
    expected |= borrowed
    assert set(uncached_judgments(sound, edited, cache=cache)) == expected

    again, stats = await certify_judge_cached(sound, edited, cache=cache)
    assert set(stats.missed) == expected and stats.misses == len(expected)
    assert stats.hits + stats.misses == again.planned
    assert again == await certify_judge(sound, edited)


async def test_any_change_to_the_judge_misses_every_verdict() -> None:
    sound, cache = judge(oracle), JudgmentCache()
    await certify_judge_cached(sound, CASES, cache=cache)
    changes = [
        dataclasses.replace(sound, rubric='The answer is polite.'),
        dataclasses.replace(sound, include_input=True),
        dataclasses.replace(sound, include_expected_output=False),
        dataclasses.replace(sound, model='openai:gpt-5'),
        dataclasses.replace(sound, model_settings={'temperature': 0.7}),
    ]
    planned = len(_keyed_plan(sound, CASES, controls=DEFAULT_CONTROLS, human_labels=(), repeats=3, seed=0,
                              assertion=None, salt='')[1])  # fmt: skip
    assert len(uncached_judgments(sound, CASES, cache=cache)) == 0
    for changed in changes:
        assert len(uncached_judgments(changed, CASES, cache=cache)) == planned
    # What identity cannot see, a salt can: the same settings, a different function behind them.
    assert len(uncached_judgments(sound, CASES, cache=cache, salt='v2')) == planned
    assert len(uncached_judgments(sound, CASES, cache=cache, assertion='pass')) == planned


NOISE = random.Random(3)  # module state: `coin(3)` closes over its rng, so it is not fully identified


def noisy_decide(output: str, expected: str) -> bool:
    return NOISE.random() < 0.5


async def test_repeats_are_kept_apart() -> None:
    noisy, cache = judge(noisy_decide), JudgmentCache()
    first, _ = await certify_judge_cached(noisy, CASES, cache=cache, repeats=3)
    by_case: dict[str, set[bool | None]] = {}
    for j in first.judgments:
        if j.role.startswith('reference#'):
            by_case.setdefault(j.case, set()).add(j.passed)
    assert any(len(v) > 1 for v in by_case.values())  # the repeats disagree somewhere, so a shared key would show
    again, stats = await certify_judge_cached(noisy, CASES, cache=cache, repeats=3)
    assert stats.misses == 0 and again == first

    # One more repeat costs one judgment per case, and nothing else.
    _, more = await certify_judge_cached(noisy, CASES, cache=cache, repeats=4)
    assert set(more.missed) == {(c.name, 'reference#3') for c in CASES}


async def test_borrowed_answers_are_keyed_on_the_answer_borrowed() -> None:
    sound, cache = judge(oracle), JudgmentCache()
    controls = (MismatchedOutput(),)
    first, _ = await certify_judge_cached(sound, CASES, cache=cache, controls=controls, seed=0)
    second, stats = await certify_judge_cached(sound, CASES, cache=cache, controls=controls, seed=1)
    donor = {j.case: j.output for j in first.judgments if j.role.startswith('must_fail')}
    redrawn = {j.case for j in second.judgments if j.role.startswith('must_fail') and j.output != donor[j.case]}
    assert redrawn  # a new seed draws different donors for some cases
    assert set(stats.missed) == {(n, 'must_fail:mismatched_output') for n in redrawn}


FAILING = {'on': True}


def flaky(output: str, expected: str) -> bool:
    if FAILING['on'] and output == 'Paris':
        raise RuntimeError('timeout')
    return oracle(output, expected)


async def test_errors_are_asked_again() -> None:
    FAILING['on'] = True
    sound, cache = judge(flaky), JudgmentCache()
    _, stats = await certify_judge_cached(sound, CASES, cache=cache)
    assert stats.errors == 3  # three repeats of France
    FAILING['on'] = False
    certificate, stats = await certify_judge_cached(sound, CASES, cache=cache)
    assert set(stats.missed) == {('France', f'reference#{i}') for i in range(3)}
    assert certificate == await certify_judge(sound, CASES)


async def test_identical_labels_get_their_own_judgments(tmp_path) -> None:  # type: ignore[no-untyped-def]
    labels = [HumanLabel('Peru', 'Lima', True), HumanLabel('Peru', 'Lima', True)]
    path = tmp_path / 'verdicts.sqlite'
    with JudgmentCache(path) as cache:
        first, stats = await certify_judge_cached(judge(oracle), CASES, cache=cache, human_labels=labels)
        assert len(cache) == stats.misses == first.planned  # no two planned judgments share a key
    with JudgmentCache(path) as reopened:  # and they survive the process
        again, stats = await certify_judge_cached(judge(oracle), CASES, cache=reopened, human_labels=labels)
    assert stats.misses == 0 and again == first
    assert Certificate.from_dict(json.loads(json.dumps(again.to_dict()))).verdict == first.verdict


def _key(inputs: Any, output: Any = 'Paris', **kwargs: Any) -> str:
    return judgment_key('j', JudgeCase('c', inputs, output), output, 'reference#0', assertion=None, occurrence=0,
                        salt='', **kwargs)  # fmt: skip


async def test_dict_order_is_part_of_the_key() -> None:
    """Found in review: keys sorted dict keys, but `LLMJudge` writes inputs in insertion order, so a
    reordered input is a different prompt. Reversing the keys returned 20 cached passes, no calls;
    judged afresh, the order-sensitive judge passed none."""

    forward = [JudgeCase(c.name, {'country': c.inputs, 'ask': 'capital'}, c.output, c.expected_output) for c in CASES]
    backward = [dataclasses.replace(c, inputs=dict(reversed(c.inputs.items()))) for c in forward]
    assert _key(forward[0].inputs) != _key(backward[0].inputs)
    cache = JudgmentCache()
    sound = judge(oracle)
    await certify_judge_cached(sound, forward, cache=cache)
    _, stats = await certify_judge_cached(sound, backward, cache=cache)
    assert stats.hits == 0  # every judgment saw a different prompt


def test_types_are_part_of_the_key() -> None:
    class Ask(BaseModel):
        country: str
        n: int = 1

    class Other(BaseModel):
        country: str
        n: int = 1

    distinct = [
        {1, 2}, frozenset({1, 2}), [1, 2], (1, 2), {'country': 'Peru', 'n': 1}, Ask(country='Peru'),
        Other(country='Peru'), OrderedDict(country='Peru', n=1), 1, 1.0, True, '1', b'1',
    ]  # fmt: skip
    keys = [_key(v) for v in distinct]
    assert len(set(keys)) == len(keys)
    assert _key({3, 1, 2}) == _key({2, 3, 1})  # a set has no order to keep
    assert _key([Ask(country='Peru')]) != _key([Other(country='Peru')])  # nested models keep their class


class Opaque:
    def __init__(self, text: str) -> None:
        self.text = text

    def __repr__(self) -> str:
        return 'Opaque()'  # hides what the judge would see


def test_values_without_a_faithful_form_are_refused() -> None:
    with pytest.raises(TypeError, match=r'Opaque.*serializer='):
        _key({'q': Opaque('a')})
    by_text = lambda v: v.text if isinstance(v, Opaque) else v  # noqa: E731
    assert _key({'q': Opaque('a')}, serializer=by_text) != _key({'q': Opaque('b')}, serializer=by_text)
    with pytest.raises(TypeError, match='serializer returned'):
        _key({'q': Opaque('a')}, serializer=lambda v: v)  # a serializer that leaves it is no help


class Flagged(dict[str, Any]):
    """A dict that carries state the judge reads besides its items."""

    def __init__(self, items: dict[str, Any], flag: bool) -> None:
        super().__init__(items)
        self.flag = flag


class Slotted(list[Any]):
    __slots__ = ('strict',)

    def __init__(self, items: list[Any], strict: bool) -> None:
        super().__init__(items)
        self.strict = strict


@dataclasses.dataclass
class FlagReader(Evaluator[Any, Any, Any]):
    """Judges by the flag its inputs carry: the items are the same, the verdicts are not."""

    async def evaluate(self, ctx: EvaluatorContext[Any, Any, Any]) -> bool:
        return oracle(ctx.output, ctx.expected_output or '') and not getattr(ctx.inputs, 'flag', False)


async def test_container_subclass_state_is_part_of_the_key() -> None:
    """Found in review: a dict subclass was keyed by its class and items only, so the same items
    with `.flag` True then False got 180 cached judgments (ADMISSIBLE), where judging afresh gave
    INADMISSIBLE."""
    assert _key(Flagged({'q': 'Peru'}, True)) != _key(Flagged({'q': 'Peru'}, False))
    assert _key(Flagged({'q': 'Peru'}, True)) == _key(Flagged({'q': 'Peru'}, True))
    assert _key(Slotted([1], True)) != _key(Slotted([1], False)) != _key([1])
    assert _key(defaultdict(int, q=1)) != _key(defaultdict(list, q=1))

    off = [dataclasses.replace(c, inputs=Flagged({'country': c.inputs}, False)) for c in CASES]
    on = [dataclasses.replace(c, inputs=Flagged({'country': c.inputs}, True)) for c in CASES]
    cache, reader = JudgmentCache(), FlagReader()
    first, _ = await certify_judge_cached(reader, off, cache=cache)
    second, stats = await certify_judge_cached(reader, on, cache=cache)
    assert first.admissible and stats.hits == 0
    assert second.verdict == (await certify_judge(reader, on)).verdict != 'ADMISSIBLE'


def test_the_serializer_is_asked_first() -> None:
    """Found in review: the serializer was asked only about values the cache had no form for, so it
    could not take over a dict subclass (or anything else the cache keys by itself)."""
    asked: list[type] = []

    def by_flag(v: Any) -> Any:
        asked.append(type(v))
        return {'items': dict(v), 'flag': v.flag} if isinstance(v, Flagged) else v

    assert _key(Flagged({'q': 'Peru'}, True), serializer=by_flag) != _key(
        Flagged({'q': 'Peru'}, False), serializer=by_flag
    )
    assert Flagged in asked and str in asked  # every value, nested and plain alike
    upper = lambda v: v.upper() if isinstance(v, str) else v  # noqa: E731
    assert _key({'q': 'peru'}, serializer=upper) == _key({'q': 'PERU'}, serializer=upper)  # it decides


async def test_a_judge_not_fully_identified_is_never_served_from_the_cache() -> None:
    state: list[int] = []

    def counted(output: str, expected: str) -> bool:
        state.append(1)
        return yes_man(output, expected)

    lax, cache = judge(counted), JudgmentCache()
    first, cold = await certify_judge_cached(lax, CASES, cache=cache)
    again, warm = await certify_judge_cached(lax, CASES, cache=cache)
    assert cold.unreliable_identity and warm.unreliable_identity
    assert warm.hits == 0 and warm.misses == first.planned and len(cache) == 0
    assert len(uncached_judgments(lax, CASES, cache=cache)) == first.planned
    assert again == first


async def test_an_unnamed_function_model_is_never_served_from_the_cache() -> None:
    """Arbitrary code cannot be fingerprinted soundly: a `FunctionModel` with pydantic-ai's default
    name is not reused, however its code digest looks; named, its owner vouches for it and it is."""
    unnamed, cache = judge(oracle, named=False), JudgmentCache()
    _, cold = await certify_judge_cached(unnamed, CASES, cache=cache)
    _, warm = await certify_judge_cached(unnamed, CASES, cache=cache)
    assert cold.unreliable_identity and warm.hits == 0 and len(cache) == 0
    named = judge(oracle)
    first, _ = await certify_judge_cached(named, CASES, cache=cache)
    _, again = await certify_judge_cached(named, CASES, cache=cache)
    assert not again.unreliable_identity and again.misses == 0 and again.hits == first.planned
