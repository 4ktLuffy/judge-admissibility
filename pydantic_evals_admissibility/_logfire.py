"""Apply a gated change to a Logfire managed variable, and only a gated one.

Logfire's managed variables hold an agent's prompt or config remotely; a label such as
`production` points at the version that serves. `promote` writes the candidate as a new version
and moves the label to it when the gate said PROMOTE, and does nothing otherwise. Either way it
records the decision and its evidence on a span, so a prompt change in Logfire can be traced to
the comparison that justified it, and a refused one to the reason it was refused.
"""

from __future__ import annotations

import json
from typing import Any

from ._gate import GateResult


def promote(
    result: GateResult,
    variable: str,
    value: Any,
    *,
    label: str = 'production',
    provider: Any = None,
) -> bool:
    """Point `label` of `variable` at `value` if `result.decision` is PROMOTE. Returns whether it did.

    Args:
        result: The gate's decision for this candidate.
        variable: The managed variable's name.
        value: The candidate value; serialized as JSON, as Logfire stores variable values.
        label: The label to move.
        provider: The variable provider; by default, the one `logfire.configure(variables=...)` set.
    """
    import logfire
    from logfire.variables.config import LabeledValue, LatestVersion

    attributes = {
        'gate.decision': result.decision,
        'gate.reason': result.reason,
        'gate.mean_gain': result.mean_gain,
        'gate.p_better': result.p_better,
        'gate.p_worse': result.p_worse,
        'gate.cases': result.cases,
        'variable': variable,
        'label': label,
    }
    with logfire.span('gate {variable} {label}: {gate.decision}', **attributes):
        if result.decision != 'PROMOTE':
            return False
        provider = provider or logfire.DEFAULT_LOGFIRE_INSTANCE.config.get_variable_provider()
        current = provider.get_variable_config(variable)
        version = (current.latest_version.version if current.latest_version else 0) + 1
        serialized = json.dumps(value)
        labels = {**current.labels, label: LabeledValue(version=version, serialized_value=serialized)}
        provider.update_variable(
            variable,
            current.model_copy(
                update={'labels': labels, 'latest_version': LatestVersion(version=version, serialized_value=serialized)}
            ),
        )
        logfire.info('promoted {variable} to version {version}', variable=variable, version=version)
        return True


def start_canary(
    variable: str,
    value: Any,
    *,
    weight: float = 0.1,
    label: str = 'candidate',
    production: str = 'production',
    provider: Any = None,
) -> int:
    """Serve `value` to `weight` of traffic under `label`, the rest staying on `production`.

    Returns the candidate's version. Which arm served a request is `resolved.label` from
    `logfire.var(...).get(targeting_key=...)`, so outcomes can be attributed to arms.
    """
    import logfire
    from logfire.variables.config import LabeledValue, LatestVersion, Rollout

    if not 0 < weight < 1:
        raise ValueError('weight must be between 0 and 1')
    provider = provider or logfire.DEFAULT_LOGFIRE_INSTANCE.config.get_variable_provider()
    current = provider.get_variable_config(variable)
    version = (current.latest_version.version if current.latest_version else 0) + 1
    serialized = json.dumps(value)
    with logfire.span('canary {variable}: start at {weight}', variable=variable, weight=weight, version=version):
        provider.update_variable(
            variable,
            current.model_copy(
                update={
                    'labels': {**current.labels, label: LabeledValue(version=version, serialized_value=serialized)},
                    'latest_version': LatestVersion(version=version, serialized_value=serialized),
                    'rollout': Rollout(labels={production: 1 - weight, label: weight}),
                }
            ),
        )
    return version


def finish_canary(
    result: GateResult,
    variable: str,
    *,
    label: str = 'candidate',
    production: str = 'production',
    provider: Any = None,
) -> str:
    """Promote the canary to all traffic on PROMOTE, roll it back on REJECT or REFUSED.

    INCONCLUSIVE leaves it running: more traffic is the only thing that can decide it. Returns
    `'promoted'`, `'rolled back'` or `'continued'`, and records the decision on a span.
    """
    import logfire
    from logfire.variables.config import Rollout

    provider = provider or logfire.DEFAULT_LOGFIRE_INSTANCE.config.get_variable_provider()
    current = provider.get_variable_config(variable)
    attributes = {
        'gate.decision': result.decision,
        'gate.reason': result.reason,
        'gate.mean_gain': result.mean_gain,
        'gate.p_better': result.p_better,
        'gate.p_worse': result.p_worse,
        'gate.requests': result.cases,
    }
    with logfire.span('canary {variable}: {gate.decision}', variable=variable, **attributes):
        if result.decision == 'INCONCLUSIVE':
            return 'continued'
        labels = dict(current.labels)
        candidate = labels.pop(label)
        if result.decision == 'PROMOTE':
            labels[production] = candidate
        outcome = 'promoted' if result.decision == 'PROMOTE' else 'rolled back'
        provider.update_variable(
            variable,
            current.model_copy(update={'labels': labels, 'rollout': Rollout(labels={production: 1.0})}),
        )
        return outcome
