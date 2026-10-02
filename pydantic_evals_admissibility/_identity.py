"""What a certificate certified: the judge's configuration, so it cannot vouch for another one.

A certificate covers one judge configuration. Change the rubric, the model, what the judge sees,
its settings, or the pydantic-evals version that builds its prompt, and the evidence is about a
different judge. `judge_identity` records those settings when a judge is certified, and
`Certificate.covers(judge)` compares them before a certificate is used to trust its scores.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any


def _model_name(model: Any) -> str | None:
    if model is None or isinstance(model, str):
        return model
    system, name = getattr(model, 'system', None), getattr(model, 'model_name', None)
    if name is not None:
        return f'{system}:{name}' if system else str(name)
    return type(model).__qualname__


def _plain(value: Any) -> Any:
    """A JSON-stable form of a setting: dataclasses and models as dicts, the rest as strings."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        value = dataclasses.asdict(value)
    elif not isinstance(value, type) and hasattr(value, 'model_dump'):
        value = value.model_dump()
    return json.loads(json.dumps(value, sort_keys=True, default=str))


def judge_identity(judge: Any) -> dict[str, Any]:
    """The settings that make a judge the judge it is, as plain data.

    For an `LLMJudge`: its rubric, model, what it is shown, model settings, score and assertion
    configuration, and the pydantic-evals version that writes its prompt. For any other
    evaluator: its class and its dataclass fields, or its `repr`.
    """
    from importlib.metadata import PackageNotFoundError, version

    try:
        evals_version = version('pydantic-evals')
    except PackageNotFoundError:  # pragma: no cover
        evals_version = None
    kind = type(judge).__qualname__
    if kind == 'LLMJudge':
        fields = {
            'rubric': judge.rubric,
            'model': _model_name(judge.model),
            'include_input': judge.include_input,
            'include_expected_output': judge.include_expected_output,
            'model_settings': _plain(judge.model_settings),
            'score': _plain(judge.score),
            'assertion': _plain(judge.assertion),
        }
    elif dataclasses.is_dataclass(judge) and not isinstance(judge, type):
        values = {f.name: getattr(judge, f.name) for f in dataclasses.fields(judge)}
        fields = {k: _model_name(v) if k == 'model' else _plain(v) for k, v in values.items()}
    else:
        fields = {'repr': repr(judge)}
    return {'evaluator': kind, **fields, 'pydantic_evals': evals_version}


def fingerprint(identity: dict[str, Any]) -> str:
    """A short, stable hash of an identity, for logs and certificate files."""
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]


def differences(certified: dict[str, Any], judge: dict[str, Any]) -> list[str]:
    """The settings that differ, by name: what changed since the certificate was issued."""
    return sorted(k for k in certified.keys() | judge.keys() if certified.get(k) != judge.get(k))
