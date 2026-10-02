"""Turn a certificate's failures into what to change.

A certificate says a judge is not evidence; the pattern of its checks usually says why. Each rule
below comes from a failure this package observed on a real judge, and says what it is in the
judge's configuration that most likely causes it. The advice is a hypothesis to re-certify, not
a fix: change one thing, certify again.
"""

from __future__ import annotations

import re
from typing import Any

from ._certify import Certificate, Check
from ._stats import wilson

_MENTIONS_INPUT = re.compile(r'\b(question|input|query|request|prompt|asked|context|user)\b', re.I)
_MENTIONS_EXPECTED = re.compile(r'\b(expected|reference|correct answer|ground truth|gold)\b', re.I)


def _cases_needed(check: Check, *, limit: int = 100_000) -> int | None:
    """Judgments needed for the interval to clear the threshold at the observed rate, if it can."""
    rate = check.rate
    if rate is None or rate <= check.threshold:
        return None
    n = max(check.trials, 1)
    while n <= limit:
        if wilson(round(rate * n), n)[0] >= check.threshold:
            return n
        n = int(n * 1.25) + 1
    return None


def diagnose(certificate: Certificate, judge: Any = None) -> list[str]:
    """Plain-language reasons the certificate is not ADMISSIBLE, and what to try first.

    Args:
        certificate: The certificate to explain.
        judge: The judge it certified. With an `LLMJudge`, the advice can point at its rubric and
            its `include_input` / `include_expected_output` settings.
    """
    checks = {c.name: c for c in certificate.checks}
    advice: list[str] = []
    rubric = getattr(judge, 'rubric', '') or ''
    sees_input = getattr(judge, 'include_input', None)
    sees_expected = getattr(judge, 'include_expected_output', None)

    def status(name: str) -> str | None:
        return checks[name].status if name in checks else None

    acceptance, rejection = checks.get('acceptance'), checks.get('rejection')
    if status('acceptance') == 'FAIL' and status('rejection') == 'PASS':
        advice.append(
            f'It fails answers that are right ({acceptance.successes}/{acceptance.trials} passed) as readily as '
            'wrong ones: it cannot confirm anything, so its pass rate measures the judge, not the agent.'
            if acceptance
            else 'It fails answers that are right.'
        )
        if sees_input is False and _MENTIONS_INPUT.search(rubric):
            advice.append(
                'The rubric refers to the question, but `include_input=False`: the judge never sees it. '
                'Set `include_input=True`.'
            )
        if sees_expected is False and _MENTIONS_EXPECTED.search(rubric):
            advice.append(
                'The rubric refers to an expected answer, but `include_expected_output=False`. '
                'Set `include_expected_output=True`.'
            )
    if status('rejection') == 'FAIL' and rejection:
        detail = rejection.detail
        advice.append(f'It passes answers that cannot be right ({detail}).')
        if 'mismatched_output' in detail and sees_input is False:
            advice.append(
                "It accepts another question's answer, and with `include_input=False` it cannot tell which "
                'question was asked. Set `include_input=True`.'
            )
        elif 'mismatched_output' in detail and sees_expected is False:
            advice.append(
                "It accepts another question's answer even though it sees the question: it cannot verify the "
                'answer itself. If your dataset has expected outputs, set `include_expected_output=True`.'
            )
    if status('invariance') == 'FAIL':
        if status('stability') in ('FAIL', 'UNVALIDATED'):
            advice.append(
                'Its verdict changes when only whitespace changes, but it also disagrees with itself on the very '
                'same answer, so this is most likely noise rather than formatting sensitivity. Lower its '
                'temperature (`model_settings={"temperature": 0}`) or use a stronger model, then certify again.'
            )
        else:
            advice.append(
                'It gives the same answer the same verdict every time, yet a whitespace-only change flips it: it '
                'is sensitive to formatting. Normalise outputs before judging, or say in the rubric that '
                'formatting does not matter.'
            )
    elif status('stability') == 'FAIL':
        advice.append('It disagrees with itself on the same answer. Lower its temperature or use a stronger model.')
    if status('human_agreement') == 'FAIL':
        advice.append(
            'It passes the controls but disagrees with people: the rubric is not what your labellers mean. '
            'Read the disagreements and rewrite the rubric, one criterion at a time.'
        )
    for check in certificate.checks:
        if check.status == 'UNVALIDATED' and 'interval straddles' in check.detail:
            needed = _cases_needed(check)
            if needed is not None:
                advice.append(
                    f'`{check.name}` is {check.rate:.2f} against a bar of {check.threshold:.2f} but '
                    f'{check.trials} judgments cannot show it; about {needed} would, at the same rate.'
                )
            else:
                advice.append(
                    f'`{check.name}` is {check.rate:.2f}, at or below its bar of {check.threshold:.2f}: '
                    'more cases will not make it pass.'
                )
    return advice
