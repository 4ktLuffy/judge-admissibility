"""A certificate covers the judge it certified, as configured then, and nothing else."""

from __future__ import annotations

import dataclasses
import json
import os
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

import pydantic
import pytest
from judges import judge, oracle, yes_man
from pydantic_ai.models.function import FunctionModel
from pydantic_evals.evaluators import Evaluator, EvaluatorContext, LLMJudge
from test_certify import CASES

from pydantic_evals_admissibility import Certificate, InadmissibleJudge, certify_judge, decide, judge_identity
from pydantic_evals_admissibility._identity import fingerprint, identity_reliable


async def test_a_certificate_covers_only_the_configuration_it_certified() -> None:
    sound = judge(oracle)
    certificate = await certify_judge(sound, CASES)
    assert certificate.admissible and certificate.covers(sound)

    changes = {
        'rubric': dataclasses.replace(sound, rubric='The answer is polite.'),
        'include_input': dataclasses.replace(sound, include_input=True),
        'include_expected_output': dataclasses.replace(sound, include_expected_output=False),
        'model': dataclasses.replace(sound, model='openai:gpt-5'),
        'model_settings': dataclasses.replace(sound, model_settings={'temperature': 0.7}),
    }
    for setting, changed in changes.items():
        assert certificate.differences(changed) == [setting], setting
        assert not certificate.covers(changed)


async def test_the_gate_refuses_a_certificate_for_another_judge() -> None:
    sound = judge(oracle)
    certificate = await certify_judge(sound, CASES)
    better = {c.name: [True] for c in CASES}
    worse = {c.name: [False] for c in CASES}
    assert decide(worse, better, certificate=certificate, judge=sound).decision == 'PROMOTE'
    other = dataclasses.replace(sound, include_input=True)
    refused = decide(worse, better, certificate=certificate, judge=other)
    assert refused.decision == 'REFUSED' and 'include_input' in refused.reason

    with pytest.raises(InadmissibleJudge, match='include_input changed'):
        certificate.raise_unless_admissible(other)


async def test_identity_survives_saving_and_names_no_object_addresses() -> None:
    certificate = await certify_judge(judge(oracle), CASES)
    again = Certificate.from_dict(json.loads(json.dumps(certificate.to_dict())))
    assert again.identity == certificate.identity and again.fingerprint == certificate.fingerprint
    assert ' at 0x' not in json.dumps(judge_identity(judge(oracle)))  # stable across processes


def test_two_scripted_judges_from_one_factory_have_different_identities() -> None:
    """Found by the pytest plugin's tests: `judge(oracle)` and `judge(yes_man)` shared a fingerprint,
    so a sound judge's certificate covered a judge that passes everything."""
    assert judge_identity(judge(oracle))['model'] != judge_identity(judge(yes_man))['model']
    assert judge_identity(judge(oracle)) == judge_identity(judge(oracle))  # still stable for the same judge


def test_a_judge_made_of_judges_has_a_stable_identity() -> None:
    """Found while building juries: a router's identity held its judges' memory addresses, so the
    same configuration had a new fingerprint in every process."""
    import subprocess
    import sys

    code = (
        'from judges import judge, oracle, yes_man\n'
        'from pydantic_evals_admissibility import RoutedJudge, judge_identity\n'
        'from pydantic_evals_admissibility._identity import fingerprint\n'
        'print(fingerprint(judge_identity(RoutedJudge(judge(oracle), judge(yes_man)))))\n'
    )
    here = Path(__file__).parent
    env = {**os.environ, 'PYTHONPATH': f'{here.parent}{os.pathsep}{here}', 'PYDANTIC_AI_NO_BANNER': '1'}
    runs = {subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, env=env, check=True).stdout
            for _ in range(2)}  # fmt: skip
    assert len(runs) == 1, runs


def _always(always: bool) -> Callable[[str, str], bool]:
    def decide(output: str, expected: str) -> bool:
        return always or oracle(output, expected)

    return decide


def test_closure_values_are_part_of_the_identity() -> None:
    """Found in review: identity hashed the TYPES a function closes over, so a factory's
    `always=True` judge (passes everything) shared the `always=False` judge's identity and
    reused its ADMISSIBLE certificate. The digest still tells them apart (for `covers` and
    messages), but an unnamed function-backed model is never reliable: nothing is reused on it."""
    strict, lenient = judge(_always(False), named=False), judge(_always(True), named=False)
    assert judge_identity(strict) != judge_identity(lenient)
    assert judge_identity(strict) == judge_identity(judge(_always(False), named=False))
    assert not identity_reliable(judge_identity(strict)) and not identity_reliable(judge_identity(lenient))


def test_function_code_is_part_of_the_identity() -> None:
    def decide(output: str, expected: str) -> bool:
        return oracle(output, expected)

    def impostor(output: str, expected: str) -> bool:
        return True

    impostor.__qualname__, impostor.__name__ = decide.__qualname__, decide.__name__  # same name, other code
    assert judge_identity(judge(decide, named=False)) != judge_identity(judge(impostor, named=False))


def test_parameter_names_are_part_of_the_code_digest() -> None:
    """Found in review: `f(output, expected)` and `f(expected, output)` compile to the same bytecode
    (locals are read by index), so two judges that call them by keyword, and so decide oppositely,
    shared a reliable identity."""

    def forward(output: str, expected: str) -> bool:
        return output.startswith(expected)

    def swapped(expected: str, output: str) -> bool:
        return expected.startswith(output)

    swapped.__qualname__, swapped.__name__ = forward.__qualname__, forward.__name__
    assert forward.__code__.co_code == swapped.__code__.co_code
    assert forward(output='Lima, Peru', expected='Lima') != swapped(output='Lima, Peru', expected='Lima')
    a, b = _KeywordJudge(forward), _KeywordJudge(swapped)
    assert judge_identity(a) != judge_identity(b)
    assert judge_identity(judge(forward, named=False)) != judge_identity(judge(swapped, named=False))


@dataclasses.dataclass
class _KeywordJudge(Evaluator[Any, Any, Any]):
    decide: Callable[..., bool]

    async def evaluate(self, ctx: EvaluatorContext[Any, Any, Any]) -> bool:  # pragma: no cover
        return self.decide(output=ctx.output, expected=ctx.expected_output)


def test_an_unnamed_function_model_is_not_reliable_and_a_named_one_is() -> None:
    """Arbitrary code cannot be fingerprinted soundly, so a judge on a `FunctionModel` with
    pydantic-ai's default name is never trusted for reuse; a name of its own is its owner's claim."""
    unnamed = judge_identity(judge(oracle, named=False))
    assert not identity_reliable(unnamed)
    assert any('unnamed function-backed model' in item for item in unnamed['opaque'])
    assert unnamed['model'].startswith('function:function:model:#')  # still a best-effort digest in the label

    def model(messages: Any, info: Any) -> Any:  # pragma: no cover
        raise NotImplementedError

    named = judge_identity(LLMJudge(rubric='r', model=FunctionModel(model, model_name='my-judge-v2')))
    assert identity_reliable(named) and named['model'] == 'function:my-judge-v2'
    renamed = judge_identity(LLMJudge(rubric='r', model=FunctionModel(model, model_name='my-judge-v3')))
    assert fingerprint(renamed) != fingerprint(named)
    assert identity_reliable(judge_identity(LLMJudge(rubric='r', model='openai:gpt-5')))
    # The test suite's scripted judges are named after their decide function, so they are reliable.
    assert identity_reliable(judge_identity(judge(oracle)))
    assert judge_identity(judge(oracle)) != judge_identity(judge(yes_man))
    # Inside another field, a model is identified the same way.
    assert not identity_reliable(judge_identity(_Holder(FunctionModel(model))))
    assert identity_reliable(judge_identity(_Holder(FunctionModel(model, model_name='named'))))


@dataclasses.dataclass
class _Holder(Evaluator[Any, Any, Any]):
    backup: Any

    async def evaluate(self, ctx: EvaluatorContext[Any, Any, Any]) -> bool:  # pragma: no cover
        return True


@dataclasses.dataclass
class _Configured(Evaluator[Any, Any, Any]):
    config: Any = ()

    async def evaluate(self, ctx: EvaluatorContext[Any, Any, Any]) -> bool:  # pragma: no cover
        return isinstance(self.config, tuple)


class _Tagged(dict[str, Any]):
    pass


def test_container_types_in_a_judges_fields_are_part_of_the_identity() -> None:
    """Found in review: a dataclass judge with `config=()` and one with `config=[]` had the same
    reliable identity; 180 cached judgments were reused (ADMISSIBLE) where judging afresh gave
    INADMISSIBLE. Containers are configuration: recorded by value, with their type."""
    distinct = [(), [], set(), frozenset(), {}, _Tagged(), {1: 'a'}, {'1': 'a'}, (1,), [1], {1}, frozenset({1}),
                OrderedDict(a=1), {'a': 1}, {'!type': 'tuple', 'items': []}]  # fmt: skip
    identities = [judge_identity(_Configured(v)) for v in distinct]
    assert len({fingerprint(i) for i in identities}) == len(distinct)
    assert all(identity_reliable(i) for i in identities)  # a judge's own mutable fields are configuration
    assert judge_identity(_Configured([1, 2])) == judge_identity(_Configured([1, 2]))
    assert judge_identity(_Configured({3, 1, 2})) == judge_identity(_Configured({2, 3, 1}))
    flagged = _Tagged(a=1)
    flagged.flag = True  # pyright: ignore[reportAttributeAccessIssue]
    assert judge_identity(_Configured(flagged)) != judge_identity(_Configured(_Tagged(a=1)))


async def test_tuple_and_list_config_do_not_share_a_cache() -> None:
    from pydantic_evals_admissibility._cache import JudgmentCache, certify_judge_cached

    cache = JudgmentCache()
    _, cold = await certify_judge_cached(_TupleOnly(()), CASES, cache=cache, controls=())
    _, other = await certify_judge_cached(_TupleOnly([]), CASES, cache=cache, controls=())
    assert not cold.unreliable_identity and other.hits == 0


@dataclasses.dataclass
class _TupleOnly(Evaluator[Any, Any, Any]):
    config: Any

    async def evaluate(self, ctx: EvaluatorContext[Any, Any, Any]) -> bool:
        return isinstance(self.config, tuple) and bool(ctx.output)


@dataclasses.dataclass
class _Router(Evaluator[Any, Any, Any]):
    judges: dict[str, Any]
    order: tuple[Any, ...] = ()

    async def evaluate(self, ctx: EvaluatorContext[Any, Any, Any]) -> bool:  # pragma: no cover
        return True


def test_judges_inside_dicts_and_nested_containers_are_identified() -> None:
    """Found in review: `_plain` only recognised a flat list of judges, so a router holding the
    oracle and the yes-man in a dict had the same identity with the two swapped."""
    sound, lax = judge(oracle), judge(yes_man)
    assert judge_identity(_Router({'a': sound, 'b': lax})) != judge_identity(_Router({'a': lax, 'b': sound}))
    nested_a = _Router({}, order=((sound, 1), [lax]))
    nested_b = _Router({}, order=((lax, 1), [sound]))
    assert judge_identity(nested_a) != judge_identity(nested_b)
    assert ' at 0x' not in json.dumps(judge_identity(nested_a))


def test_mutable_state_is_opaque_and_does_not_change_the_identity() -> None:
    calls: list[int] = []

    def counted(output: str, expected: str) -> bool:
        calls.append(1)
        return oracle(output, expected)

    sound = judge(counted)  # unnamed: its digest is not complete
    before = judge_identity(sound)
    calls.extend([1, 2, 3])
    assert judge_identity(sound) == before  # a counter growing is not a new judge
    assert not identity_reliable(before) and before['opaque']  # but it is not fully identified either
    assert identity_reliable(judge_identity(judge(oracle)))
    assert not identity_reliable(judge_identity(_Router({'a': sound})))  # opacity of a member is the router's


def test_identity_with_closure_values_is_stable_across_processes() -> None:
    import subprocess
    import sys

    code = (
        'from judges import judge, oracle, yes_man\n'
        'from pydantic_evals_admissibility import judge_identity\n'
        'from pydantic_evals_admissibility._identity import fingerprint\n'
        'def make(flags, names, table):\n'
        '    def decide(o, e):\n'
        '        return flags[0] and o in names and table is not None and oracle(o, e)\n'
        '    return decide\n'
        "j = judge(make((True, 1.5, b'x', None), frozenset({'Paris', 'Lima', 'Rome', 'Oslo'}), {'k': (1, 2)}))\n"
        'print(fingerprint(judge_identity(j)))\n'
    )
    here = Path(__file__).parent
    runs = set()
    for seed in ('1', '2', '3'):  # set order follows the string hash seed
        env = {**os.environ, 'PYTHONPATH': f'{here.parent}{os.pathsep}{here}', 'PYDANTIC_AI_NO_BANNER': '1',
               'PYTHONHASHSEED': seed}  # fmt: skip
        runs.add(subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, env=env,
                                check=True).stdout)  # fmt: skip
    assert len(runs) == 1, runs


def test_settings_are_compared_exactly_and_functions_are_never_trusted() -> None:
    """`True`, `1` and `1.0` are different settings; a function held as a setting is never reliable."""
    from pydantic_evals_admissibility import identity_reliable
    from pydantic_evals_admissibility._identity import differences

    assert differences({'threshold': True}, {'threshold': 1}) == ['threshold']
    assert differences({'threshold': 1}, {'threshold': 1.0}) == ['threshold']
    assert differences({'threshold': 1}, {'threshold': 1}) == []

    @dataclasses.dataclass
    class WithCheck:
        check: Any

        async def evaluate(self, ctx: Any) -> bool:
            return bool(self.check(ctx))

    assert not identity_reliable(judge_identity(WithCheck(len)))


@dataclasses.dataclass
class _FirstKey(Evaluator[Any, Any, Any]):
    config: Any

    async def evaluate(self, ctx: EvaluatorContext[Any, Any, Any]) -> bool:  # pragma: no cover
        return next(iter(self.config)) == 'lax'


@dataclasses.dataclass
class _InitState(Evaluator[Any, Any, Any]):
    always: dataclasses.InitVar[bool]

    def __post_init__(self, always: bool) -> None:
        self._always = always

    async def evaluate(self, ctx: EvaluatorContext[Any, Any, Any]) -> bool:  # pragma: no cover
        return self._always


def _judge_class_in(module: str) -> type:
    @dataclasses.dataclass
    class Judge(Evaluator[Any, Any, Any]):
        async def evaluate(self, ctx: EvaluatorContext[Any, Any, Any]) -> bool:  # pragma: no cover
            return True

    Judge.__module__ = module
    return Judge


def test_order_init_state_and_module_are_part_of_the_identity() -> None:
    """Found in the fourth review: each pair had one reliable identity, and 180 cached judgments
    certified as ADMISSIBLE a judge that, judged afresh, was INADMISSIBLE."""
    pairs = [
        (_FirstKey({'strict': 0, 'lax': 0}), _FirstKey({'lax': 0, 'strict': 0})),  # a judge can read dict order
        (_InitState(False), _InitState(True)),  # state set in __post_init__, outside the fields
        (_judge_class_in('strict_rules')(), _judge_class_in('lax_rules')()),  # one class name, two modules
    ]
    for a, b in pairs:
        ia, ib = judge_identity(a), judge_identity(b)
        assert identity_reliable(ia) and identity_reliable(ib)
        assert fingerprint(ia) != fingerprint(ib), (ia, ib)
    assert judge_identity(_InitState(True)) == judge_identity(_InitState(True))
    assert judge_identity(_FirstKey({'a': 0, 'b': 0})) == judge_identity(_FirstKey({'a': 0, 'b': 0}))


async def test_init_state_does_not_share_a_cache() -> None:
    from pydantic_evals_admissibility._cache import JudgmentCache, certify_judge_cached

    cache = JudgmentCache()
    await certify_judge_cached(_InitState(False), CASES, cache=cache, controls=())
    _, other = await certify_judge_cached(_InitState(True), CASES, cache=cache, controls=())
    assert other.hits == 0


def test_editing_an_evaluators_code_changes_its_identity() -> None:
    """Found while closing the fourth review: a custom evaluator's identity held its class name and
    fields but not its code, so an edited `evaluate` would reuse verdicts cached for the old one."""

    def make(strict: bool) -> Any:
        if strict:

            @dataclasses.dataclass
            class Judge(Evaluator[Any, Any, Any]):
                async def evaluate(self, ctx: EvaluatorContext[Any, Any, Any]) -> bool:  # pragma: no cover
                    return ctx.output == ctx.expected_output

        else:

            @dataclasses.dataclass
            class Judge(Evaluator[Any, Any, Any]):
                async def evaluate(self, ctx: EvaluatorContext[Any, Any, Any]) -> bool:  # pragma: no cover
                    return True

        return Judge()

    strict, lax = judge_identity(make(True)), judge_identity(make(False))
    assert strict['evaluator'] == lax['evaluator'] and strict['!module'] == lax['!module']
    assert strict['!code'] != lax['!code'] and identity_reliable(strict)
    assert judge_identity(make(True)) == strict


def _evaluator_pair(how: str, always: bool) -> Any:
    """Two evaluators of one class name and module that differ only in how `always` reaches `evaluate`."""
    if how == 'closure':

        @dataclasses.dataclass
        class Judge(Evaluator[Any, Any, Any]):
            async def evaluate(self, ctx: EvaluatorContext[Any, Any, Any]) -> bool:  # pragma: no cover
                return always

    elif how == 'default':

        @dataclasses.dataclass
        class Judge(Evaluator[Any, Any, Any]):
            async def evaluate(
                self, ctx: EvaluatorContext[Any, Any, Any], flag: bool = always
            ) -> bool:  # pragma: no cover
                return flag

    else:

        @dataclasses.dataclass
        class Judge(Evaluator[Any, Any, Any]):
            flag: ClassVar[bool] = always

            async def evaluate(self, ctx: EvaluatorContext[Any, Any, Any]) -> bool:  # pragma: no cover
                return self.flag

    return Judge()


class _Private(pydantic.BaseModel):
    _always: bool = pydantic.PrivateAttr(default=False)


@dataclasses.dataclass
class _CodeField(Evaluator[Any, Any, Any]):
    code: bool = False

    async def evaluate(self, ctx: EvaluatorContext[Any, Any, Any]) -> bool:  # pragma: no cover
        return self.code


def test_fifth_review_closures_defaults_class_attributes_private_state_and_key_clashes() -> None:
    """Found in the fifth review: each pair shared one reliable identity, and 180 cached verdicts
    certified as ADMISSIBLE a judge that, judged afresh, was INADMISSIBLE."""
    for how in ('closure', 'default', 'class attribute'):
        a, b = judge_identity(_evaluator_pair(how, False)), judge_identity(_evaluator_pair(how, True))
        assert identity_reliable(a) and fingerprint(a) != fingerprint(b), how
    on = _Private()
    on._always = True
    assert judge_identity(_Holder(on)) != judge_identity(_Holder(_Private()))
    # A field named like the identity's own keys cannot overwrite them: the generated digest lives
    # under `!code`, and a field that would clash with `evaluator` or `opaque` makes it unreliable.
    assert judge_identity(_CodeField(True)) != judge_identity(_CodeField(False))


@dataclasses.dataclass
class _InitInput:
    flag: dataclasses.InitVar[bool]

    def __post_init__(self, flag: bool) -> None:
        self.seen = flag


def test_cache_keys_keep_dataclass_state_outside_fields_and_private_attributes() -> None:
    from pydantic_evals_admissibility._cache import _canonical  # pyright: ignore[reportPrivateUsage]

    assert _canonical(_InitInput(True)) != _canonical(_InitInput(False))
    on = _Private()
    on._always = True
    assert _canonical(on) != _canonical(_Private())


def _behaviour_pair(how: str, always: bool) -> Any:
    """Two values of one class name that differ only in how a method returns `always`."""
    if how == 'special method':

        @dataclasses.dataclass
        class Judge(Evaluator[Any, Any, Any]):
            def __bool__(self) -> bool:  # pragma: no cover
                return always

            async def evaluate(self, ctx: EvaluatorContext[Any, Any, Any]) -> bool:  # pragma: no cover
                return bool(self)

        return Judge()

    class Config:
        @staticmethod
        def permissive() -> bool:  # pragma: no cover
            return always

    class Lookup(dict[str, bool]):
        def __getitem__(self, key: str) -> bool:  # pragma: no cover
            return always

    return _Holder(Config) if how == 'class as setting' else _Holder(Lookup(always=False))


class _Presence(pydantic.BaseModel):
    always: bool = False


def test_sixth_review_class_behaviour_and_field_presence_are_part_of_the_identity() -> None:
    """Found in the sixth review: two classes of one name were told apart by name only, so their
    methods (a `__bool__`, a class held as a setting, a dict subclass's `__getitem__`) could
    differ under one reliable identity; and `Config()` and `Config(always=False)` collided."""
    for how in ('special method', 'class as setting', 'container subclass'):
        a, b = judge_identity(_behaviour_pair(how, False)), judge_identity(_behaviour_pair(how, True))
        assert identity_reliable(a) and fingerprint(a) != fingerprint(b), how
    assert judge_identity(_Holder(_Presence())) != judge_identity(_Holder(_Presence(always=False)))


def test_cache_keys_keep_input_class_behaviour_and_field_presence() -> None:
    from pydantic_evals_admissibility._cache import _canonical  # pyright: ignore[reportPrivateUsage]

    def make(always: bool) -> Any:
        @dataclasses.dataclass
        class Input:
            @property
            def flag(self) -> bool:  # pragma: no cover
                return always

        return Input()

    assert _canonical(make(True)) != _canonical(make(False))
    assert _canonical(_Presence()) != _canonical(_Presence(always=False))


def _seventh_pair(how: str, always: bool) -> Any:
    """Two values of one class name that differ only in `always`, held as the seventh review held them."""
    if how in ('staticmethod', 'classmethod', 'property'):
        wrap: Any = {'staticmethod': staticmethod, 'classmethod': classmethod, 'property': property}[how]

        def truth(*_: Any) -> bool:  # pragma: no cover
            return always

        @dataclasses.dataclass
        class Judge(Evaluator[Any, Any, Any]):
            __bool__ = wrap(truth)

            async def evaluate(self, ctx: EvaluatorContext[Any, Any, Any]) -> bool:  # pragma: no cover
                return True

        return Judge()
    if how == 'enum':
        import enum

        class Config(enum.Enum):
            SWITCH = always

        return _Holder(Config.SWITCH)
    if how == 'model_dump':

        class Dumps:
            def __init__(self) -> None:
                self.always = always

            def model_dump(self) -> dict[str, Any]:  # pragma: no cover
                return {}

        return _Holder(Dumps())

    class Meta(type):
        def permissive(cls) -> bool:  # pragma: no cover
            return always

    class Config(metaclass=Meta):
        pass

    return _Holder(Config)


def test_seventh_review_wrapped_methods_enum_values_dumps_and_metaclasses() -> None:
    """Found in the seventh review: each pair shared one reliable identity."""
    for how in ('staticmethod', 'classmethod', 'property', 'enum', 'metaclass'):
        a, b = judge_identity(_seventh_pair(how, False)), judge_identity(_seventh_pair(how, True))
        assert identity_reliable(a) and fingerprint(a) != fingerprint(b), how
    # Another library's `model_dump` is not trusted to say everything: such an object is opaque.
    assert not identity_reliable(judge_identity(_seventh_pair('model_dump', False)))


def test_cache_keys_never_match_unidentified_input_code_and_keep_factories_and_fold() -> None:
    import collections
    import datetime

    from pydantic_evals_admissibility._cache import _canonical  # pyright: ignore[reportPrivateUsage]

    def make(always: list[bool]) -> Any:
        @dataclasses.dataclass
        class Input:
            @property
            def flag(self) -> bool:  # pragma: no cover
                return always[0]

        return Input()

    assert _canonical(make([True])) != _canonical(make([True]))  # mutable closure: never the same key

    def factory(always: bool) -> Any:
        def default() -> bool:  # pragma: no cover
            return always

        return collections.defaultdict(default)

    assert _canonical(factory(True)) != _canonical(factory(False))
    assert _canonical(factory(True)) == _canonical(factory(True))
    moment = datetime.datetime(2026, 11, 1, 1, 30)
    assert _canonical(moment) != _canonical(moment.replace(fold=1))
    assert _canonical(moment) == _canonical(datetime.datetime(2026, 11, 1, 1, 30))
