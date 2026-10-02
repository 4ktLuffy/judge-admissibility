"""`promote` against Logfire's local in-memory variable provider: no account, real SDK."""

from __future__ import annotations

import uuid

import logfire
import pytest
from logfire.testing import CaptureLogfire
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from logfire.variables.config import LabeledValue, LatestVersion, Rollout, VariableConfig, VariablesConfig

from pydantic_evals_admissibility import GateResult, promote


@pytest.fixture
def prompt_var(capfire: CaptureLogfire):  # type: ignore[no-untyped-def]
    # Logfire registers variables process-wide, so each test gets its own name.
    name = f'agent_prompt_{uuid.uuid4().hex[:8]}'
    config = VariablesConfig(
        variables={
            name: VariableConfig(
                name=name,
                labels={'production': LabeledValue(version=1, serialized_value='"v1"')},
                rollout=Rollout(labels={'production': 1.0}),
                overrides=[],
                latest_version=LatestVersion(version=1, serialized_value='"v1"'),
            )
        }
    )
    logfire.configure(
        send_to_logfire=False,
        console=False,
        variables=logfire.LocalVariablesOptions(config=config),
        # Re-configuring replaces capfire's exporter, so hand it back explicitly.
        additional_span_processors=[SimpleSpanProcessor(capfire.exporter)],
    )
    return logfire.var(name, type=str, default='code default')


def served(var) -> str:  # type: ignore[no-untyped-def]
    with var.get() as resolved:
        return resolved.value


@pytest.mark.parametrize('decision', ['REJECT', 'INCONCLUSIVE', 'REFUSED'])
def test_nothing_but_promote_changes_the_served_prompt(prompt_var, decision: str) -> None:  # type: ignore[no-untyped-def]
    assert promote(GateResult(decision, 'test'), prompt_var.name, 'v2') is False  # type: ignore[arg-type]
    assert served(prompt_var) == 'v1'


def test_promote_serves_the_candidate_as_a_new_version(prompt_var, capfire: CaptureLogfire) -> None:  # type: ignore[no-untyped-def]
    result = GateResult('PROMOTE', 'test', mean_gain=0.2, interval=(0.1, 0.3), cases=40, p_better=0.001, p_worse=1.0)
    assert promote(result, prompt_var.name, 'v2') is True
    assert served(prompt_var) == 'v2'
    spans = [s for s in capfire.exporter.exported_spans_as_dict() if s['attributes'].get('gate.decision')]
    assert spans and spans[0]['attributes']['gate.decision'] == 'PROMOTE'
    assert spans[0]['attributes']['gate.p_better'] == 0.001
