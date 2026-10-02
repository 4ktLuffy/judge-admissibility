"""Certify a pydantic-evals judge before its scores count.

A judge's verdicts are evidence only once the judge has been shown to fail what it should fail,
pass what it should pass, and give the same verdict when nothing that matters has changed.
"""

from ._bridge import Effect, JudgeBridge, compare_judge_reports, compare_judges
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
    recertify,
)
from ._controls import (
    DEFAULT_CONTROLS,
    Control,
    EmptyOutput,
    EvidenceRewrite,
    MismatchedOutput,
    Rewrite,
    WhitespaceReformat,
)
from ._dataset import (
    DatasetJudgeCertificate,
    EmptyProse,
    ProseWhitespace,
    certify_dataset,
    controls_for,
    dataset_report,
    map_prose,
    rubric_kind,
)
from ._diagnose import diagnose
from ._gate import Decision, GateResult, GateRules, decide, decide_unpaired, detectable_gain
from ._identity import judge_identity
from ._logfire import finish_canary, promote, start_canary
from ._pairwise import (
    PairCase,
    PairThresholds,
    PairVerdict,
    PairwiseJudge,
    both_orders,
    certify_pairwise,
    first_position_rate,
    recertify_pairwise,
)
from ._reports import compare_reports, outcomes
from ._stats import cohen_kappa, wilson

__all__ = (
    'DatasetJudgeCertificate',
    'EmptyProse',
    'ProseWhitespace',
    'certify_dataset',
    'controls_for',
    'dataset_report',
    'map_prose',
    'rubric_kind',
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
    'judge_identity',
    'outcomes',
    'finish_canary',
    'first_position_rate',
    'recertify_pairwise',
    'promote',
    'start_canary',
    'EmptyOutput',
    'EvidenceRewrite',
    'HumanLabel',
    'InadmissibleJudge',
    'JudgeCase',
    'Judgment',
    'MismatchedOutput',
    'PairCase',
    'PairThresholds',
    'PairVerdict',
    'PairwiseJudge',
    'Rewrite',
    'Thresholds',
    'WhitespaceReformat',
    'assertion_of',
    'both_orders',
    'certify_judge',
    'recertify',
    'certify_pairwise',
    'cohen_kappa',
    'wilson',
    'Effect',
    'JudgeBridge',
    'compare_judge_reports',
    'compare_judges',
)
