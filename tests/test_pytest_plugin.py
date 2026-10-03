"""The pytest plugin: a suite uses a judge only once it is certified, certifies it once, and says what it trusted."""

from __future__ import annotations

import asyncio
import json
import os
from importlib.metadata import entry_points
from pathlib import Path

import pytest

# Imported here, before any inner run: pytester drops the modules an in-process run imports, and an
# import hook installed by one of the package's dependencies (beartype's, via py-key-value) breaks
# when its modules are dropped and imported again.
import pydantic_evals_admissibility.pytest_plugin  # noqa: F401  # isort: skip

pytest_plugins = ['pytester']

PLUGIN = 'pydantic_evals_admissibility.pytest_plugin'
HERE = Path(__file__).parent

# The inner suites' shared code: the package's own scripted judges, named, and a counter of judge calls. A
# `FunctionModel` with no name of its own has no reliable identity, so the plugin would certify it on every use.
SCRIPTED = """
import dataclasses

from judges import judge as scripted_judge, oracle, yes_man
from pydantic_ai.models.function import FunctionModel
from test_certify import CASES

CALLS = []


def judge(decide):
    made = scripted_judge(decide)
    return dataclasses.replace(made, model=FunctionModel(made.model.function, model_name=f'scripted:{decide.__name__}'))


def unnamed(decide):
    made = scripted_judge(decide)
    return dataclasses.replace(made, model=FunctionModel(made.model.function))


def counted(output, expected):
    CALLS.append(output)
    return oracle(output, expected)
"""


def _installed() -> bool:
    """Whether pytest will load the plugin itself (the `pytest11` entry point), so `-p` would load it twice."""
    if os.environ.get('PYTEST_DISABLE_PLUGIN_AUTOLOAD'):
        return False
    return any(ep.value == PLUGIN for ep in entry_points(group='pytest11'))


def run(pytester: pytest.Pytester, *args: str) -> pytest.RunResult:
    pytester.syspathinsert(HERE)
    pytester.syspathinsert(HERE.parent)
    pytester.makepyfile(scripted=SCRIPTED)
    plugin = () if _installed() else ('-p', PLUGIN)
    # pytest-pretty, if installed, replaces the summary line `assert_outcomes` reads.
    return pytester.runpytest(*plugin, '-p', 'no:cacheprovider', '-p', 'no:pretty', *args)


def test_an_admissible_judge_passes_and_is_certified_once_for_every_test(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        test_inner="""
        from scripted import CALLS, CASES, counted, judge

        def test_one(certify_judge):
            assert certify_judge(judge(counted), CASES) is not None

        def test_two(certify_judge):
            certify_judge(judge(counted), CASES)  # a new object, the same configuration: no new certificate

        def test_three(certify_judge, judge_certificates):
            certify_judge(judge(counted), CASES, decision='report')
            (entry,) = judge_certificates.entries.values()
            assert len(CALLS) == entry.certificate.calls == entry.calls > 0
            assert entry.uses == {'gate': [True, True], 'report': [True]}
        """
    )
    result = run(pytester)
    result.assert_outcomes(passed=3)
    result.stdout.fnmatch_lines(
        [
            '*= judge certificates =*',
            'LLMJudge(scripted:counted): ADMISSIBLE  fingerprint *',
            '  checks: acceptance PASS, rejection PASS, invariance PASS, stability PASS',
            '  calls: * (certified in this session)',
            '  used for: gate qualified (asked 2 times); report qualified (asked 1 time)',
        ]
    )


def test_an_inadmissible_judge_fails_with_the_table_and_the_advice(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        test_inner="""
        from scripted import CASES, judge, yes_man

        def test_gate(certify_judge):
            certify_judge(judge(yes_man), CASES)
            raise AssertionError('never reached: the judge is not qualified')
        """
    )
    result = run(pytester)
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(
        [
            'judge-admissibility: LLMJudge(scripted:yes_man) is not qualified to gate (certified in this session, *',
            '- verdict: INADMISSIBLE (rejection FAIL); gate needs ADMISSIBLE',
            '- checks: rejection FAIL',
            'LLMJudge(scripted:yes_man): INADMISSIBLE',
            'rejection * FAIL *',
            'what to try:',
            '- *',
        ]
    )
    result.stdout.no_fnmatch_line('*never reached*')
    result.stdout.fnmatch_lines(
        ['LLMJudge(scripted:yes_man): INADMISSIBLE  fingerprint *', '*gate NOT qualified (asked 1 time)']
    )


def test_skip_mode_and_the_marker(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        test_inner="""
        import pytest
        from scripted import CASES, judge, oracle, yes_man

        def test_broken_judge_is_skipped(certify_judge):
            certify_judge(judge(yes_man), CASES, name='yes-man')

        def test_sound_judge_may_gate(certify_judge):
            certify_judge(judge(oracle), CASES)

        @pytest.mark.judge_certified(decision='gate', context={'slice': 'refunds'})
        def test_but_not_on_a_slice_it_never_measured(certify_judge):
            certify_judge(judge(oracle), CASES)
        """
    )
    result = run(pytester, '--judge-unqualified=skip', '-rs')
    result.assert_outcomes(passed=1, skipped=2)
    result.stdout.fnmatch_lines(["*context_slice: the certificate did not measure slice 'refunds'*"])


def test_saved_certificates_are_used_in_ci_and_refused_for_another_configuration(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        test_inner="""
        import dataclasses, os
        from scripted import CALLS, CASES, counted, judge

        def make():
            sound = judge(counted)
            return dataclasses.replace(sound, rubric=os.environ.get('RUBRIC', sound.rubric))

        def test_gate(certify_judge):
            certify_judge(make(), CASES, name='capitals')

        def test_unknown(certify_judge):
            if os.environ.get('UNKNOWN'):
                certify_judge(make(), CASES, name='never-certified')

        def test_calls():
            assert bool(CALLS) == bool(os.environ.get('EXPECT_CALLS'))
        """
    )
    certificates = pytester.path / 'certs.json'
    with pytest.MonkeyPatch.context() as env:
        env.setenv('EXPECT_CALLS', '1')
        made = run(pytester, '--judge-recertify', f'--judge-certificates={certificates}')
    made.assert_outcomes(passed=3)
    made.stdout.fnmatch_lines([f'certificates written to {certificates}'])
    saved = json.loads(certificates.read_text())['certificates']
    assert [s['name'] for s in saved] == ['capitals'] and saved[0]['certificate']['verdict'] == 'ADMISSIBLE'

    # CI: the saved certificate is used, and the judge is not called.
    ci = run(pytester, f'--judge-certificates={certificates}')
    ci.assert_outcomes(passed=3)
    ci.stdout.fnmatch_lines(
        ['capitals: ADMISSIBLE  fingerprint *', '  calls: 0 (saved certificate, * calls when made *)']
    )

    # The rubric changed since: the saved certificate is about another judge, and is refused.
    with pytest.MonkeyPatch.context() as env:
        env.setenv('RUBRIC', 'The answer is polite.')
        env.setenv('UNKNOWN', '1')
        changed = run(pytester, f'--judge-certificates={certificates}')
    changed.assert_outcomes(passed=1, failed=2)
    changed.stdout.fnmatch_lines(
        [
            'judge-admissibility: capitals is not qualified to gate (saved certificate, *',
            '- coverage: the certificate is for another configuration: rubric changed',
        ]
    )
    changed.stdout.fnmatch_lines([f'judge-admissibility: {certificates} has no certificate for never-certified*'])

    # Other cases than the certificate was made on: refused, not silently reused.
    pytester.makepyfile(
        test_inner="""
        from scripted import CASES, counted, judge

        def test_fewer_cases(certify_judge):  # the same judge as the saved certificate, on fewer cases
            certify_judge(judge(counted), CASES[:20], name='capitals')
        """
    )
    fewer = run(pytester, f'--judge-certificates={certificates}')
    fewer.assert_outcomes(failed=1)
    fewer.stdout.fnmatch_lines(['*was made on other judgments than this test asks for*'])

    missing = run(pytester, '--judge-certificates=missing.json')
    missing.assert_outcomes(failed=1)
    missing.stdout.fnmatch_lines(['*no certificate file at *missing.json; make it with --judge-recertify'])


def test_a_session_fixture_and_an_async_test_share_one_certificate(pytester: pytest.Pytester) -> None:
    pytester.makeconftest(
        """
        import pytest
        from scripted import CASES, counted, judge

        @pytest.fixture(scope='session')
        def anyio_backend():
            return 'asyncio'

        @pytest.fixture(scope='session')
        def shared_judge(judge_certificates):
            return judge_certificates.certify(judge(counted), CASES, decision='promote')
        """
    )
    pytester.makepyfile(
        test_inner="""
        import pytest
        from scripted import CALLS, CASES, counted, judge

        def test_sync(shared_judge):
            assert shared_judge is not None

        @pytest.mark.anyio
        async def test_async_inside_a_running_loop(certify_judge, judge_certificates):
            await certify_judge.acertify(judge(counted), CASES)
            (entry,) = judge_certificates.entries.values()
            assert len(CALLS) == entry.calls and set(entry.uses) == {'promote', 'gate'}
        """
    )
    result = run(pytester)
    result.assert_outcomes(passed=2)


def test_ini_options(pytester: pytest.Pytester) -> None:
    pytester.makeini('[pytest]\njudge_unqualified = skip\n')
    pytester.makepyfile(
        test_inner="""
        from scripted import CASES, judge, yes_man

        def test_gate(certify_judge):
            certify_judge(judge(yes_man), CASES)
        """
    )
    run(pytester).assert_outcomes(skipped=1)
    pytester.makeini('[pytest]\njudge_unqualified = warn\n')
    result = run(pytester)
    result.stderr.fnmatch_lines(['*judge_unqualified must be fail or skip*'])


def test_the_example_runs(pytester: pytest.Pytester) -> None:
    example = HERE.parent / 'examples' / 'test_with_certified_judge.py'
    pytester.makepyfile(test_example=example.read_text())
    result = run(pytester)
    result.assert_outcomes(passed=3)
    result.stdout.fnmatch_lines(
        [
            'capital-judge: ADMISSIBLE  fingerprint *',
            '  calls: * (certified in this session)',
            '  used for: gate qualified (asked 1 time); promote qualified (asked 1 time)',
            '  warning: promote: slices: no slices*',
        ]
    )


def test_a_certificate_is_not_reused_when_what_the_judge_is_shown_changes(pytester: pytest.Pytester) -> None:
    """The cache key is the materialized plan: a control of the same name and type that makes other outputs, another
    slice function with the same qualname, or another batch size is another certificate."""
    pytester.makepyfile(
        test_inner="""
        from judges import lenient_on_empty
        from pydantic_evals_admissibility import Rewrite, WhitespaceReformat
        from scripted import CASES, judge, oracle

        def controls(rewrite):
            return [Rewrite(rewrite, name='bad'), WhitespaceReformat()]

        def test_a_wrong_answer_control(certify_judge):
            certify_judge(judge(lenient_on_empty), CASES, controls=controls(lambda o: 'Atlantis'))

        def test_an_empty_answer_control_of_the_same_name(certify_judge):
            certify_judge(judge(lenient_on_empty), CASES, controls=controls(lambda o: ''))
            raise AssertionError('never reached: an empty answer is what this judge passes')

        def test_slices_and_batch_size(certify_judge, judge_certificates):
            before = len(judge_certificates.entries)
            first_or_last_letter = [lambda case: case.name[0], lambda case: case.name[-1]]
            for slice_by in first_or_last_letter:
                certify_judge(judge(oracle), CASES, decision='report', slice_by=slice_by)
            certify_judge(judge(oracle), CASES, decision='report', batch_size=10)
            certify_judge(judge(oracle), CASES, decision='report', batch_size=10)  # the same: reused
            assert len(judge_certificates.entries) == before + 3
        """
    )
    result = run(pytester)
    result.assert_outcomes(passed=2, failed=1)
    result.stdout.fnmatch_lines(['*- verdict: INADMISSIBLE (rejection FAIL); gate needs ADMISSIBLE'])
    result.stdout.no_fnmatch_line('*never reached*')


def test_the_plan_is_keyed_keeping_types_order_and_subclass_state() -> None:
    from pydantic_evals_admissibility._cache import _canonical, repr_fallback

    class Flagged(dict[str, int]):
        flag = False

    on, off = Flagged(a=1), Flagged(a=1)
    on.flag = True
    values = [1, 1.0, '1', True, [1, 2], [2, 1], (1, 2), {'a': 1, 'b': 2}, {'b': 2, 'a': 1}, None, 'None', on, off]
    assert len({json.dumps(_canonical(v, repr_fallback)) for v in values}) == len(values)
    assert _canonical(object(), repr_fallback) == _canonical(object(), repr_fallback)  # no memory address


def test_a_judge_whose_identity_is_not_reliable_is_certified_every_time(pytester: pytest.Pytester) -> None:
    """A `FunctionModel` with no name: its identity cannot tell it apart from another, and the summary says so."""
    pytester.makepyfile(
        test_inner="""
        from scripted import CALLS, CASES, counted, unnamed

        def test_one(certify_judge):
            certify_judge(unnamed(counted), CASES, name='unnamed')

        def test_two(certify_judge, judge_certificates):
            certify_judge(unnamed(counted), CASES, name='unnamed')
            (entry,) = judge_certificates.entries.values()
            assert entry.certifications == 2 and len(CALLS) == entry.calls == 2 * entry.certificate.calls
        """
    )
    certificates = pytester.path / 'certs.json'
    result = run(pytester, '--judge-recertify', f'--judge-certificates={certificates}')
    result.assert_outcomes(passed=2)
    result.stdout.fnmatch_lines(
        [
            '  calls: * (certified 2 times in this session: its identity is not reliable, so its certificate is never*',
            '  not identified: model: *',
        ]
    )
    assert not certificates.exists()  # never saved: it could never be reused


def test_an_async_test_certifies_on_its_own_loop(pytester: pytest.Pytester) -> None:
    """The judge may need the test's loop (a future, a client bound to it) and its context variables."""
    pytester.makepyfile(
        test_inner="""
        import asyncio, contextvars
        import pytest
        from pydantic_ai.messages import ModelResponse, ToolCallPart
        from pydantic_ai.models.function import FunctionModel
        from pydantic_evals.evaluators import LLMJudge
        from scripted import CASES

        REQUEST_ID = contextvars.ContextVar('REQUEST_ID', default=None)
        LOOP, WRONG = [], []

        async def model(messages, info):
            if asyncio.get_running_loop() is not LOOP[0] or REQUEST_ID.get() != 'test':
                WRONG.append(REQUEST_ID.get())
                raise RuntimeError('not on the test loop')
            future = LOOP[0].create_future()
            LOOP[0].call_soon(future.set_result, True)  # only the test's loop can resolve it
            passed = await future
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {'reason': '-', 'pass': passed})])

        @pytest.fixture
        def anyio_backend():
            return 'asyncio'

        @pytest.mark.anyio
        async def test_on_the_running_loop(certify_judge):
            LOOP.append(asyncio.get_running_loop())
            REQUEST_ID.set('test')
            judge = LLMJudge(rubric='r', model=FunctionModel(model, model_name='scripted:on-loop'))
            await certify_judge.acertify(judge, CASES, decision='report')
            assert not WRONG, WRONG[:1]

        @pytest.mark.anyio
        async def test_the_sync_call_says_to_await(certify_judge):
            with pytest.raises(RuntimeError, match='acertify'):
                other = LLMJudge(rubric='other', model=FunctionModel(model, model_name='scripted:on-loop'))
                certify_judge(other, CASES)
        """
    )
    run(pytester).assert_outcomes(passed=2)


def test_names_are_qualified(pytester: pytest.Pytester) -> None:
    """The plugin loads into every pytest run: it must not take generic names like `--recertify` or `certify`."""
    pytester.makepyfile(
        test_inner="""
        def test_generic_fixture_is_free(request):
            assert 'certify' not in request.fixturenames

        def test_qualified_names(certify_judge, judge_certificates):
            assert callable(certify_judge) and callable(certify_judge.acertify)
        """
    )
    run(pytester).assert_outcomes(passed=2)
    run(pytester, '--recertify').stderr.fnmatch_lines(['*unrecognized arguments: --recertify*'])
    pytester.makeini('[pytest]\njudge_recertify = true\njudge_certificates = certs.json\n')
    pytester.makepyfile(
        test_inner="""
        from scripted import CASES, judge, oracle

        def test_gate(certify_judge):
            certify_judge(judge(oracle), CASES)
        """
    )
    run(pytester).stdout.fnmatch_lines(['certificates written to *certs.json'])


def test_xdist_workers_send_their_certificates_to_the_controller(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each worker certifies what it uses; the controller prints one summary and writes the file once."""
    pytest.importorskip('xdist')
    monkeypatch.setenv('PYTHONPATH', os.pathsep.join([str(HERE), str(HERE.parent), str(pytester.path)]))
    pytester.makepyfile(scripted=SCRIPTED)
    pytester.makepyfile(
        test_inner="""
        import pytest
        from scripted import CASES, judge, oracle

        @pytest.mark.xdist_group('a')
        def test_a(certify_judge):
            certify_judge(judge(oracle), CASES, name='judge-a')

        @pytest.mark.xdist_group('b')
        def test_b(certify_judge):
            certify_judge(judge(oracle), CASES, name='judge-b')

        def test_shared(certify_judge):
            certify_judge(judge(oracle), CASES, name='judge-shared', decision='report')
        """
    )
    certificates = pytester.path / 'certs.json'
    plugin = () if _installed() else ('-p', PLUGIN)
    args = ('-p', 'no:cacheprovider', '-p', 'no:pretty', '-n', '2', '--judge-recertify')
    result = pytester.runpytest_subprocess(*plugin, *args, '--dist=loadgroup', f'--judge-certificates={certificates}')
    result.assert_outcomes(passed=3)
    result.stdout.fnmatch_lines(['*= judge certificates =*'])
    for name in ('judge-a', 'judge-b', 'judge-shared'):
        result.stdout.fnmatch_lines([f'{name}: ADMISSIBLE  fingerprint *'])
    saved = json.loads(certificates.read_text())['certificates']
    assert sorted(s['name'] for s in saved) == ['judge-a', 'judge-b', 'judge-shared']
    assert not list(pytester.path.glob('.certs.json.*'))  # no temporary file left behind

    # Every test on every worker: the shared judge is certified once per worker, and the summary says so.
    each = pytester.runpytest_subprocess(*plugin, *args, '--dist=each', f'--judge-certificates={certificates}')
    each.assert_outcomes(passed=6)
    each.stdout.fnmatch_lines(
        [
            'judge-shared: ADMISSIBLE  fingerprint *',
            '*',
            '  calls: * (certified 2 times in this session, once per xdist worker)',
            '  used for: report qualified (asked 2 times)',
        ]
    )


def test_the_certificate_is_made_of_the_judgments_its_key_hashes(pytester: pytest.Pytester) -> None:
    """A control that makes other outputs when asked again is planned once: the key hashes the plan the judge is
    certified on. (It once hashed one plan and certified another: empty answers in the key, 'Atlantis' in the
    certificate, so a judge that passes empty answers was ADMISSIBLE, and a control that is always empty reused it.)"""
    pytester.makepyfile(
        test_inner="""
        from judges import lenient_on_empty
        from pydantic_evals_admissibility import Rewrite, WhitespaceReformat
        from scripted import CASES, judge

        def empty_then_atlantis():
            seen = []

            def rewrite(output):
                seen.append(output)
                return '' if len(seen) <= len(CASES) else 'Atlantis'

            return rewrite

        def test_a_control_that_changes_between_calls(certify_judge):
            controls = [Rewrite(empty_then_atlantis(), name='bad'), WhitespaceReformat()]
            certify_judge(judge(lenient_on_empty), CASES, controls=controls)
            raise AssertionError('never reached: the judge was certified on the empty answers it passes')

        def test_a_control_that_is_always_empty(certify_judge, judge_certificates):
            controls = [Rewrite(lambda output: '', name='bad'), WhitespaceReformat()]
            certify_judge(judge(lenient_on_empty), CASES, controls=controls)
            raise AssertionError('never reached: an empty answer is what this judge passes')
        """
    )
    result = run(pytester)
    result.assert_outcomes(failed=2)
    result.stdout.no_fnmatch_line('*never reached*')
    result.stdout.fnmatch_lines(
        [
            'LLMJudge(scripted:lenient_on_empty): INADMISSIBLE  fingerprint *',
            '*',
            '  calls: * (certified in this session)',
            '  used for: gate NOT qualified (asked 2 times)',
        ]
    )


def test_concurrent_requests_share_one_certification(pytester: pytest.Pytester) -> None:
    """Two tasks of one test (or a sync call from a thread, alongside) asking for the same judge at once: one
    certification, the other waits for it. A judge whose identity is not reliable is never shared."""
    pytester.makepyfile(
        test_inner="""
        import asyncio
        import pytest
        from pydantic_evals_admissibility import _identity
        from scripted import CALLS, CASES, counted, judge

        @pytest.fixture
        def anyio_backend():
            return 'asyncio'

        @pytest.mark.anyio
        async def test_two_tasks(certify_judge, judge_certificates):
            make = lambda: judge(counted)
            await asyncio.gather(certify_judge.acertify(make(), CASES), certify_judge.acertify(make(), CASES))
            (entry,) = judge_certificates.entries.values()
            assert entry.certifications == 1 and len(CALLS) == entry.calls == entry.certificate.calls

        @pytest.mark.anyio
        async def test_a_task_and_a_thread(certify_judge, judge_certificates):
            make = lambda: judge(counted)
            sliced = dict(slice_by=lambda case: case.name[0], decision='report')
            await asyncio.gather(
                certify_judge.acertify(make(), CASES, **sliced),
                asyncio.to_thread(certify_judge, make(), CASES, **sliced),
            )
            entry = list(judge_certificates.entries.values())[-1]
            assert len(judge_certificates.entries) == 2 and entry.certifications == 1
            assert entry.uses == {'report': [True, True]}

        @pytest.mark.anyio
        async def test_unreliable_is_not_shared(certify_judge, judge_certificates, monkeypatch):
            monkeypatch.setattr(_identity, 'identity_reliable', lambda identity: False, raising=False)
            make = lambda: judge(counted)
            await asyncio.gather(*(certify_judge.acertify(make(), CASES, batch_size=10) for _ in range(2)))
            entry = list(judge_certificates.entries.values())[-1]
            assert not entry.reliable and entry.certifications == 2
        """
    )
    run(pytester).assert_outcomes(passed=3)


def test_a_waiter_certifies_itself_when_the_certification_it_shares_is_cancelled() -> None:
    """A cancelled certification is not the waiters' failure: one of them certifies, and the key is free again."""
    from judges import judge, oracle
    from test_certify import CASES

    from pydantic_evals_admissibility.pytest_plugin import JudgeCertificates

    registry = JudgeCertificates()
    calls: list[int] = []
    original = registry._request  # pyright: ignore[reportPrivateUsage]

    def request(*args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        made = original(*args, **kwargs)  # type: ignore[arg-type]
        run = made.run

        async def counted() -> object:
            calls.append(1)
            if len(calls) == 1:
                await asyncio.sleep(10)  # the first certification hangs, and is cancelled
            return await run()

        made.run = counted
        made.reliable = True
        return made

    registry._request = request  # type: ignore[method-assign]

    async def main() -> None:
        first = asyncio.create_task(registry.acertify(judge(oracle), CASES, decision='report'))
        await asyncio.sleep(0)
        second = asyncio.create_task(registry.acertify(judge(oracle), CASES, decision='report'))
        await asyncio.sleep(0.01)
        first.cancel()
        assert await second is not None
        with pytest.raises(asyncio.CancelledError):
            await first

    asyncio.run(main())
    (entry,) = registry.entries.values()
    assert len(calls) == 2 and entry.certifications == 1
    assert not registry._inflight  # pyright: ignore[reportPrivateUsage]
