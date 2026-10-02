"""A certificate covers the judge it certified, as configured then, and nothing else."""

from __future__ import annotations

import dataclasses
import json

import pytest
from judges import judge, oracle
from test_certify import CASES

from pydantic_evals_admissibility import Certificate, InadmissibleJudge, certify_judge, decide, judge_identity


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
