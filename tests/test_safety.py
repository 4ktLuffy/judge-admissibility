"""PIIDetector: each detector's true positives and hard negatives, YAML specs, a real Dataset, and a certificate."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import EvaluationReason, EvaluatorContext
from pydantic_evals.otel._errors import SpanTreeRecordingError

from pydantic_evals_admissibility import JudgeCase, WhitespaceReformat, certify_judge
from pydantic_evals_admissibility._safety import (
    BIAS_JUDGE_RUBRIC,
    PII_CATEGORIES,
    PII_JUDGE_RUBRIC,
    TOXICITY_JUDGE_RUBRIC,
    PIIDetector,
    PIIInjection,
    PIILookalike,
    detect_pii,
    generate_lookalike,
    generate_pii,
    iban_valid,
    luhn_valid,
    pii_controls,
    redact,
)

# Secret-shaped strings are assembled at run time, so no scanner mistakes this file for a leak.
TOKEN = 'Zq8' + 'xW2' * 12 + 'kP'

POSITIVES = [
    ('email', 'Write to maria.garcia@gmail.com for help.', 'maria.garcia@gmail.com'),
    ('email', 'cc: O.Brien+orders@mail.acme-corp.co.uk', 'O.Brien+orders@mail.acme-corp.co.uk'),
    ('phone', 'Call (415) 867-5309 after six.', '(415) 867-5309'),
    ('phone', 'Phone: 1-800-273-8255', '1-800-273-8255'),
    ('phone', 'reach me on +44 20 7946 0958 please', '+44 20 7946 0958'),
    ('phone', 'mobile +33 6 12 34 56 78', '+33 6 12 34 56 78'),
    ('phone', 'Büro: 030 1234 5678', '030 1234 5678'),
    ('payment_card', 'card 4111 1111 1111 1111 expires soon', '4111 1111 1111 1111'),
    ('payment_card', 'Amex 3782-822463-10005', '3782-822463-10005'),
    ('payment_card', 'number 5555555555554444.', '5555555555554444'),
    ('iban', 'IBAN DE89 3704 0044 0532 0130 00 please', 'DE89 3704 0044 0532 0130 00'),
    ('iban', 'Pay GB82WEST12345698765432.', 'GB82WEST12345698765432'),
    ('us_ssn', 'SSN 219-09-9999 on file', '219-09-9999'),
    ('ip_address', 'request from 8.8.4.4.', '8.8.4.4'),
    ('ip_address', 'v6 peer 2001:4860:4860::8888 seen', '2001:4860:4860::8888'),
    ('secret', f'key=sk-proj-{TOKEN} in the env', f'sk-proj-{TOKEN}'),
    ('secret', 'AKIA' + 'Q7W3E9R2T5Y8U1I4' + ' leaked', 'AKIA' + 'Q7W3E9R2T5Y8U1I4'),
    ('secret', 'token ' + 'ghp_' + TOKEN[:36], 'ghp_' + TOKEN[:36]),
    ('secret', '-----BEGIN RSA PRIVATE KEY-----\nMIIE...', '-----BEGIN RSA PRIVATE KEY-----'),
]

HARD_NEGATIVES = [
    'Order 4111111111111112 shipped.',  # 16 digits, fails Luhn
    'Order #4111111111111111 shipped.',  # passes Luhn, but an order number by context
    'Tracking number 9400 1000 0000 0000 0000 00',
    'Upgrade to version 10.2.3.4 tonight.',
    'Requires python==3.12.1.0',
    'Gateway 192.168.1.1 and loopback 127.0.0.1; docs use 203.0.113.7.',
    'Email jane@example.com or ops@service.test.',
    'Clone git@github.com:pydantic/pydantic-ai.git',
    'Delivered on January 5, 2026; today is 2026-03-03.',
    'Refund: $1,234.56 minus 15% = $1,049.38. Answer: $1049.38',
    'Request id 123e4567-e89b-12d3-a456-426614174000',
    'commit 9fceb02d0ae598e95dc970b74767f19372d61af8',
    'Meeting at 12:30:45, MAC 00:1A:2B:3C:4D:5E',
    'Set OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx',
    'AWS docs example: AKIAIOSFODNN7EXAMPLE',
    'Invoice no. 123-45-6789 attached.',
    'Not an IBAN: DE00 1234 5678 9012 3456 78',
    'ISBN 978-0-306-40615-7',
    'Population 8 336 817 and 2 000 000 visitors',
    'Use the coupon SAVE2026 at checkout; call center hours 9-18.',
]


def ctx(output: object) -> EvaluatorContext[object, object, object]:
    return EvaluatorContext(
        name='c', inputs=None, metadata=None, expected_output=None, output=output, duration=0.0,
        _span_tree=SpanTreeRecordingError('none'), attributes={}, metrics={},
    )  # fmt: skip


@pytest.mark.parametrize(('category', 'text', 'value'), POSITIVES)
def test_true_positive(category: str, text: str, value: str) -> None:
    found = detect_pii(text)
    assert [(f.category, text[f.start : f.end]) for f in found] == [(category, value)]


@pytest.mark.parametrize('text', HARD_NEGATIVES)
def test_hard_negative(text: str) -> None:
    assert detect_pii(text) == []


def test_checksums() -> None:
    assert luhn_valid('4111111111111111') and not luhn_valid('4111111111111112')
    assert iban_valid('DE89 3704 0044 0532 0130 00') and not iban_valid('DE88 3704 0044 0532 0130 00')
    assert not iban_valid('ZZ89 3704 0044 0532 0130 00')  # unknown country
    assert not iban_valid('DE89 3704 0044 0532 0130 0')  # wrong length for DE


def test_generated_values_are_found_and_lookalikes_are_not() -> None:
    rng = random.Random(7)
    for category in PII_CATEGORIES:
        for _ in range(100):
            value = generate_pii(category, rng)
            text = f'On file: {value}. Thanks!'
            assert [(f.category, text[f.start : f.end]) for f in detect_pii(text)] == [(category, value)]
    for _ in range(300):
        kind, value = generate_lookalike(rng)
        assert detect_pii(f'Reference: {value}.') == [], kind


def test_reason_names_categories_never_values() -> None:
    text = 'Mail maria.garcia@gmail.com, card 4111 1111 1111 1111, call (415) 867-5309.'
    result = PIIDetector().evaluate(ctx(text))
    verdict = result['pii_free']
    assert isinstance(verdict, EvaluationReason) and verdict.value is False
    assert verdict.reason is not None
    for secret in ('maria', 'gmail', '4111', '867', '5309'):
        assert secret not in verdict.reason
    assert '1 email' in verdict.reason and '1 payment_card' in verdict.reason and '1 phone' in verdict.reason
    assert result['pii_free_found'] == 'email,payment_card,phone'
    clean = PIIDetector().evaluate(ctx('Answer: yes'))
    assert clean['pii_free'] == EvaluationReason(True, 'no PII found (all categories)')
    assert clean['pii_free_found'] == 'none'


def test_categories_allowlist_patterns_code_and_reserved() -> None:
    text = 'Contact support@northwind.io or (415) 867-5309.'
    assert [f.category for f in detect_pii(text, categories=['phone'])] == ['phone']
    assert [f.category for f in detect_pii(text, allowlist=['SUPPORT@northwind.io'])] == ['phone']
    assert detect_pii(text, allow_patterns=[r'.*@northwind\.io', r'\(415\).*']) == []
    assert detect_pii('test card 4111-1111-1111-1111', allowlist=['4111 1111 1111 1111']) == []
    code = 'Example:\n```python\nsend("admin@corp.io")\n```\nand `root@corp.io` inline; real: bob@corp.io'
    assert len(detect_pii(code)) == 3
    found = detect_pii(code, ignore_code=True)
    assert [code[f.start : f.end] for f in found] == ['bob@corp.io']
    assert [f.category for f in detect_pii('jane@example.com 10.0.0.1', ignore_reserved=False)] == [
        'email',
        'ip_address',
    ]
    with pytest.raises(ValueError, match='unknown PII categories'):
        detect_pii('x', categories=['passport'])
    with pytest.raises(ValueError, match='unknown PII categories'):
        PIIDetector(categories=['passport'])


def test_a_card_number_is_not_also_a_phone_even_when_cards_are_off() -> None:
    assert detect_pii('card 4111 1111 1111 1111', categories=['phone']) == []


def test_structured_outputs_and_output_path() -> None:
    class Reply(BaseModel):
        text: str
        card: int

    reply = Reply(text='Refund issued.', card=4111111111111111)
    assert PIIDetector().evaluate(ctx(reply))['pii_free_found'] == 'payment_card'
    assert PIIDetector().evaluate(ctx({'notes': ['ok', 'mail bob@corp.io']}))['pii_free_found'] == 'email'
    only_text = PIIDetector(output_path='output.text')
    assert only_text.evaluate(ctx({'text': 'Refund issued.', 'card': 4111111111111111}))['pii_free_found'] == 'none'


def test_redact() -> None:
    text = 'Mail bob@corp.io or call (415) 867-5309.'
    assert redact(text, detect_pii(text)) == 'Mail [EMAIL] or call [PHONE].'


def test_rubrics_are_plain_strings_for_llmjudge() -> None:
    for rubric in (PII_JUDGE_RUBRIC, TOXICITY_JUDGE_RUBRIC, BIAS_JUDGE_RUBRIC):
        assert isinstance(rubric, str) and len(rubric) > 100


def test_yaml_round_trip_and_handwritten_spec(tmp_path: Path) -> None:
    detector = PIIDetector(
        categories=['email', 'phone'], allowlist=['help@northwind.io'], ignore_code=True, evaluation_name='no_pii'
    )
    dataset = Dataset[str, str, Any](name='pii', cases=[Case(name='a', inputs='q')], evaluators=[detector])
    path = tmp_path / 'pii.yaml'
    dataset.to_file(path, custom_evaluator_types=[PIIDetector])
    assert (tmp_path / 'pii_schema.json').exists()  # the spec is expressible in the dataset's JSON schema
    loaded = Dataset[str, str, Any].from_file(path, custom_evaluator_types=[PIIDetector])
    assert loaded.evaluators == [detector]

    handwritten = tmp_path / 'hand.yaml'
    handwritten.write_text(
        'name: hand\ncases:\n- name: a\n  inputs: q\nevaluators:\n'
        '- PIIDetector\n'
        '- PIIDetector:\n    categories: [payment_card]\n    allow_patterns: ["4111.*"]\n'
    )
    plain, configured = Dataset[str, str, Any].from_file(handwritten, custom_evaluator_types=[PIIDetector]).evaluators
    assert plain == PIIDetector()
    assert configured == PIIDetector(categories=('payment_card',), allow_patterns=('4111.*',))


async def test_inside_a_real_dataset_evaluate() -> None:
    replies = {
        'clean': 'Your refund of $42.50 is on its way. Answer: $42.50',
        'leaky': 'Refunded to card 4111 1111 1111 1111. Answer: $42.50',
        'lookalike': 'Order 4111111111111112 refunded. Answer: $42.50',
    }

    def agent(name: str) -> str:
        return replies[name]

    dataset = Dataset[str, str, None](
        name='replies', cases=[Case(name=name, inputs=name) for name in replies], evaluators=[PIIDetector()]
    )
    report = await dataset.evaluate(agent, progress=False)
    verdicts = {case.name: case.assertions['pii_free'].value for case in report.cases}
    assert verdicts == {'clean': True, 'leaky': False, 'lookalike': True}
    labels = {case.name: case.labels['pii_free_found'].value for case in report.cases}
    assert labels['leaky'] == 'payment_card'
    assert '4111' not in str(report.cases[1].assertions['pii_free'].reason)


async def test_detector_certifies_on_clean_replies_with_pii_controls() -> None:
    replies = [
        f'Hi! Your {item} return is approved; the refund of ${10 + i}.{i:02d} reaches you in 5-7 business days. '
        f'Answer: yes'
        for i, item in enumerate(['blender', 'kettle', 'tent', 'lamp', 'speaker'] * 4)
    ]
    cases = [JudgeCase(f'c{i}', inputs='q', output=reply) for i, reply in enumerate(replies)]
    controls = (*pii_controls(), WhitespaceReformat())
    certificate = await certify_judge(PIIDetector(), cases, controls=controls, repeats=2)
    assert certificate.verdict == 'ADMISSIBLE', certificate.table()
    rejection = next(c for c in certificate.checks if c.name == 'rejection')
    assert {f.name for f in rejection.families} == {f'pii_{c}' for c in PII_CATEGORIES}
    # A detector blind to one category fails on that family, not on average.
    blind = await certify_judge(
        PIIDetector(categories=[c for c in PII_CATEGORIES if c != 'phone']), cases, controls=controls, repeats=1
    )
    assert blind.verdict == 'INADMISSIBLE'
    failed = next(c for c in blind.checks if c.name == 'rejection')
    assert [f.name for f in failed.families if f.status == 'FAIL'] == ['pii_phone']


def test_controls_skip_non_string_outputs_and_mixed_injection() -> None:
    rng = random.Random(0)
    case = JudgeCase('a', inputs=None, output={'x': 1})
    assert PIIInjection('email').make(case, [case], rng) is None
    assert PIILookalike().make(case, [case], rng) is None
    text_case = JudgeCase('b', inputs=None, output='All good. Answer: yes')
    mixed = PIIInjection()
    assert mixed.name == 'pii_any'
    made = [mixed.make(text_case, [text_case], rng) for _ in range(40)]
    assert all(isinstance(m, str) and detect_pii(m) for m in made)


# Regressions from review (S1-S4): each reproduces the reported failure.


@pytest.mark.parametrize(
    ('text', 'category', 'value'),
    [
        ('Call +14158675309.', 'phone', '+14158675309'),
        ('WhatsApp +447700900123 any time', 'phone', '+447700900123'),
        ('Email: _alice@gmail.com.', 'email', '_alice@gmail.com'),
    ],
)
def test_compact_international_phones_and_underscore_emails(text: str, category: str, value: str) -> None:
    assert [(f.category, text[f.start : f.end]) for f in detect_pii(text)] == [(category, value)]


def test_short_plus_numbers_are_not_compact_phones() -> None:
    assert detect_pii('Score +12345678 points') == []


def test_mapping_keys_are_scanned() -> None:
    result = PIIDetector().evaluate(ctx({'alice@gmail.com': 'customer'}))
    assert result['pii_free_found'] == 'email'


def test_allowlist_compares_digits_only_for_numbers() -> None:
    assert [f.category for f in detect_pii('bob123456@gmail.com', allowlist=['alice123456@gmail.com'])] == ['email']
    assert detect_pii('call (415) 867-5309', allowlist=['415.867.5309']) == []
    assert detect_pii('IBAN DE89 3704 0044 0532 0130 00', allowlist=['de89370400440532013000']) == []


def test_missing_output_path_gives_no_verdict_not_a_pass() -> None:
    detector = PIIDetector(output_path='output.reply')
    assert detector.evaluate(ctx({'answer': 'alice@gmail.com'})) == {}
    assert detector.evaluate(ctx({'reply': 'alice@gmail.com'}))['pii_free_found'] == 'email'
