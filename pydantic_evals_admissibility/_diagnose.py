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

_MENTIONS_INPUT = re.compile(
    r"\b(question|input|query|prompt|asked|context|user'?s? (?:question|request|message|query))\b", re.I
)  # not a bare 'user': 'appropriate for user display' says nothing about the input
_ABOUT_CORRECTNESS = re.compile(r'\b(correct\w*|accura\w*|right|true|answers?|factual\w*|valid)\b', re.I)
_ABOUT_STYLE = re.compile(r'\b(style|tone|friendly|polite|format\w*|concise|second-person|grammar|wording)\b', re.I)
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
        n += 1 if n < 1000 else n // 10  # exact for any realistic size, coarse beyond it
    return None


def _consistent_rejections(certificate: Certificate) -> tuple[list[str], bool]:
    """Known-good answers the judge fails every time: maybe the answers, not the judge.

    A noisy judge fails good answers at random. One that fails the same few answers on every
    repeat, and passes the rest every time, is being specific, and may be right: the "known-good"
    answers in a dataset do not always meet its own rubric. Found on pydantic-ai's example
    dataset, where both a weak and a strong judge failed the same three expected outputs every
    time, with reasons that hold up ("impersonal, not second person").
    """
    by_case: dict[str, list[bool]] = {}
    variant_passed: set[str] = set()
    for j in certificate.judgments:
        if j.role.startswith('reference#') and j.passed is not None:
            by_case.setdefault(j.case, []).append(j.passed)
        # A must-hold variant of the case's own answer (not another case's, as `mismatched_output` is)
        # means the same thing; if the judge passed it, it contradicts its own failures of the original.
        if j.role.startswith('must_hold:') and j.role != 'must_hold:mismatched_output' and j.passed:
            variant_passed.add(j.case)
    repeated = {case: v for case, v in by_case.items() if len(v) > 1}
    always_failed = [case for case, v in repeated.items() if not any(v)]
    always_passed = [case for case, v in repeated.items() if all(v)]
    consistent = len(always_failed) + len(always_passed)
    if not always_failed or consistent < 0.8 * len(repeated) or len(always_passed) < 0.3 * len(repeated):
        # Nothing failed consistently; or too many mixed cases (noise, not specificity); or it passes
        # almost nothing, and a dataset is not wrong everywhere: then it is the judge.
        return [], False
    suspect = [case for case in always_failed if case not in variant_passed]
    contradicted = [case for case in always_failed if case in variant_passed]
    advice = []
    if suspect:
        names = ', '.join(repr(c) for c in suspect[:5]) + (' ...' if len(suspect) > 5 else '')
        advice.append(
            f'It fails the same known-good answers every time ({len(suspect)} of {len(repeated)}: {names}) and '
            f'is consistent on {consistent} of {len(repeated)} cases. A judge that consistent may be right, or '
            'those answers may share something it gets wrong every time: read them against the rubric. Fix or '
            'drop the ones that do not meet it; if they do meet it, the judge has a blind spot there, whatever '
            'its overall rate. Other cases borrow them as controls too, so settle them before reading '
            '`invariance`.'
        )
    if contradicted:
        names = ', '.join(repr(c) for c in contradicted[:5]) + (' ...' if len(contradicted) > 5 else '')
        advice.append(
            f'It also fails {names} every time, but passed the same answer with only its formatting changed: '
            'there the judge contradicts itself, so that failure is the judge, not the answer.'
        )
    return advice, bool(suspect)  # with no answer left to suspect, the usual advice about the judge stands


def diagnose(certificate: Certificate, judge: Any = None) -> list[str]:
    """Plain-language reasons the certificate is not ADMISSIBLE, and what to try first.

    Some advice applies to an ADMISSIBLE certificate too: known-good answers it fails every time
    can be hiding behind a passing overall rate.

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
    consistency_advice, data_suspect = _consistent_rejections(certificate)
    if status('acceptance') == 'FAIL' and status('rejection') in ('PASS', 'UNVALIDATED'):
        if data_suspect:
            advice.append(
                f'It passed {acceptance.successes} of {acceptance.trials} judgments of the answers marked good, '
                'but it fails the same ones every time (below): check those answers before the judge.'
                if acceptance
                else 'It fails some of the answers marked good, the same ones every time (below).'
            )
        else:
            advice.append(
                f'It fails answers that are right ({acceptance.successes}/{acceptance.trials} passed) as readily '
                'as wrong ones: it cannot confirm anything, so its pass rate measures the judge, not the agent.'
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
        empty = re.search(r'empty_output (\d+)/(\d+)', detail)
        # A judge of correctness that passes another question's answer is broken; only a rubric
        # about style can make that control the mistake. With no rubric to read, say it as a maybe.
        about_style = bool(_ABOUT_STYLE.search(rubric)) if rubric else True
        about_correctness = bool(_ABOUT_CORRECTNESS.search(rubric)) and not _ABOUT_STYLE.search(rubric)
        if (
            'mismatched_output' in detail
            and empty
            and empty.group(1) == empty.group(2)
            and about_style
            and not about_correctness
        ):
            advice.append(
                "It rejects every empty answer and fails only on other cases' answers. If the rubric grades "
                "style, tone or format rather than correctness, another case's answer can satisfy it, and the "
                "control is wrong, not the judge: use `MismatchedOutput(kind='must_hold')` and `Rewrite` controls "
                'that break the rubric itself.'
            )
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
    if status('invariance') == 'FAIL' and not data_suspect:  # with suspect answers as donors, it would mislead
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
        controls_hold = all(status(n) in (None, 'PASS') for n in ('acceptance', 'rejection', 'invariance'))
        if controls_hold:
            advice.append(
                'It passes the controls but disagrees with people: the rubric is not what your labellers mean. '
                'Read the disagreements and rewrite the rubric, one criterion at a time.'
            )
        else:
            advice.append(
                'It also disagrees with people beyond chance; fix the failures above first, then certify again '
                'before reading anything into the disagreement.'
            )
    if status('slices') == 'FAIL' and 'slices' in checks:
        s = checks['slices']
        worst = s.detail.removeprefix('worst first: ').split(', ')[0]
        advice.append(
            f'It is below the bar on one kind of case ({worst}; at most {s.interval[1]:.2f} against '
            f'{s.threshold:.2f}), whatever its overall acceptance: its verdicts on that kind are not evidence. '
            'Read its reasons on those cases. If they need work it does not do reliably without reasoning '
            '(dates, arithmetic), try a judge that reasons, or a rubric that spells out the rule for that kind.'
        )
    advice += consistency_advice
    for check in certificate.checks:
        if data_suspect and check.name in ('acceptance', 'invariance'):
            continue  # both are computed from the answers that need checking first
        if check.status == 'UNVALIDATED' and 'interval straddles' in check.detail:
            needed = _cases_needed(check)
            if needed is not None:
                advice.append(
                    f'`{check.name}` is {check.rate:.2f} against a bar of {check.threshold:.2f} but '
                    f'{check.trials} judgments cannot show it; {needed} would, at the same rate.'
                )
            else:
                advice.append(
                    f'`{check.name}` is {check.rate:.2f}, at or below its bar of {check.threshold:.2f}: '
                    'more cases will not make it pass.'
                )
    return advice
