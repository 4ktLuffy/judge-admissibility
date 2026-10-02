"""The canary: arms assigned by Logfire itself, decided by the unpaired gate, applied to the variable."""

from __future__ import annotations

import random

import logfire
import pytest
from logfire.testing import CaptureLogfire
from logfire.variables.config import LabeledValue, LatestVersion, Rollout, VariableConfig, VariablesConfig
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from pydantic_evals_admissibility import GateRules, decide_unpaired, finish_canary, start_canary

FAST = GateRules(resamples=2000)


def test_identical_arms_are_rarely_acted_on() -> None:
    actions = 0
    for t in range(200):
        rng = random.Random(t)
        a = [rng.random() < 0.5 for _ in range(150)]
        b = [rng.random() < 0.5 for _ in range(150)]
        actions += decide_unpaired(a, b, rules=FAST).decision != 'INCONCLUSIVE'
    assert actions / 200 <= 0.05, actions


def test_a_better_arm_with_enough_traffic_is_promoted() -> None:
    promoted = 0
    for t in range(50):
        rng = random.Random(1000 + t)
        a = [rng.random() < 0.5 for _ in range(400)]
        b = [rng.random() < 0.65 for _ in range(400)]
        promoted += decide_unpaired(a, b, rules=FAST).decision == 'PROMOTE'
    assert promoted / 50 >= 0.9, promoted


@pytest.fixture
def prompt_var(capfire: CaptureLogfire, request: pytest.FixtureRequest):  # type: ignore[no-untyped-def]
    # Logfire assigns arms by hashing the variable name with the request key, so a fixed name per
    # test keeps the split, and the test's outcome, the same on every run. (A random name made
    # one in about forty runs of the identical-arms test a legitimate 2.5% false positive.)
    name = f'canary_{request.node.name}'
    config = VariablesConfig(
        variables={
            name: VariableConfig(
                name=name,
                labels={'production': LabeledValue(version=1, serialized_value='"current"')},
                rollout=Rollout(labels={'production': 1.0}),
                overrides=[],
                latest_version=LatestVersion(version=1, serialized_value='"current"'),
            )
        }
    )
    logfire.configure(
        send_to_logfire=False,
        console=False,
        variables=logfire.LocalVariablesOptions(config=config),
        additional_span_processors=[SimpleSpanProcessor(capfire.exporter)],
    )
    return logfire.var(name, type=str, default='code default')


def run_traffic(var, quality: dict[str, float], requests: int, seed: int) -> dict[str, list[bool]]:  # type: ignore[no-untyped-def]
    """Logfire picks each request's arm; the arm's served prompt decides how often it succeeds."""
    rng = random.Random(seed)
    arms: dict[str, list[bool]] = {'production': [], 'candidate': []}
    for i in range(requests):
        with var.get(targeting_key=f'request-{seed}-{i}') as resolved:
            arms[resolved.label].append(rng.random() < quality[resolved.value])
    return arms


def served_to_everyone(var) -> set[str]:  # type: ignore[no-untyped-def]
    values = set()
    for i in range(50):
        with var.get(targeting_key=f'check-{i}') as resolved:
            values.add(resolved.value)
    return values


def test_a_better_candidate_is_promoted_to_all_traffic(prompt_var) -> None:  # type: ignore[no-untyped-def]
    start_canary(prompt_var.name, 'better', weight=0.5)
    arms = run_traffic(prompt_var, {'current': 0.5, 'better': 0.75}, requests=600, seed=1)
    assert 200 < len(arms['candidate']) < 400  # Logfire really split the traffic
    result = decide_unpaired(arms['production'], arms['candidate'], rules=FAST)
    assert finish_canary(result, prompt_var.name) == 'promoted'
    assert served_to_everyone(prompt_var) == {'better'}


def test_a_worse_candidate_is_rolled_back(prompt_var) -> None:  # type: ignore[no-untyped-def]
    start_canary(prompt_var.name, 'worse', weight=0.5)
    arms = run_traffic(prompt_var, {'current': 0.6, 'worse': 0.35}, requests=600, seed=2)
    result = decide_unpaired(arms['production'], arms['candidate'], rules=FAST)
    assert finish_canary(result, prompt_var.name) == 'rolled back'
    assert served_to_everyone(prompt_var) == {'current'}


def test_too_little_traffic_keeps_the_canary_running(prompt_var) -> None:  # type: ignore[no-untyped-def]
    start_canary(prompt_var.name, 'same', weight=0.5)
    arms = run_traffic(prompt_var, {'current': 0.5, 'same': 0.5}, requests=60, seed=3)
    result = decide_unpaired(arms['production'], arms['candidate'], rules=FAST)
    assert finish_canary(result, prompt_var.name) == 'continued'
    assert served_to_everyone(prompt_var) == {'current', 'same'}  # still split
