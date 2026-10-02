"""Certify a pydantic-evals judge before its scores count.

A judge's verdicts are evidence only once the judge has been shown to fail what it should fail,
pass what it should pass, and give the same verdict when nothing that matters has changed.
"""

from ._canary import CanaryMonitor, JudgeCanary
from ._cases import HumanLabel, JudgeCase
from ._certify import (
    Certificate,
    Check,
    InadmissibleJudge,
    Judgment,
    Thresholds,
    assertion_of,
    certify_judge,
)
from ._controls import DEFAULT_CONTROLS, Control, EmptyOutput, MismatchedOutput, WhitespaceReformat
from ._diagnose import diagnose
from ._gate import Decision, GateResult, GateRules, decide, decide_unpaired, detectable_gain
from ._logfire import finish_canary, promote, start_canary
from ._pairwise import (
    PairCase,
    PairThresholds,
    PairVerdict,
    PairwiseJudge,
    both_orders,
    certify_pairwise,
    first_position_rate,
)
from ._reports import compare_reports, outcomes
from ._stats import cohen_kappa, wilson

__all__ = (
    'CanaryMonitor',
    'JudgeCanary',
    'DEFAULT_CONTROLS',
    'Certificate',
    'Check',
    'Control',
    'Decision',
    'GateResult',
    'GateRules',
    'compare_reports',
    'decide',
    'decide_unpaired',
    'detectable_gain',
    'diagnose',
    'outcomes',
    'finish_canary',
    'first_position_rate',
    'promote',
    'start_canary',
    'EmptyOutput',
    'HumanLabel',
    'InadmissibleJudge',
    'JudgeCase',
    'Judgment',
    'MismatchedOutput',
    'PairCase',
    'PairThresholds',
    'PairVerdict',
    'PairwiseJudge',
    'Thresholds',
    'WhitespaceReformat',
    'assertion_of',
    'both_orders',
    'certify_judge',
    'certify_pairwise',
    'cohen_kappa',
    'wilson',
)
