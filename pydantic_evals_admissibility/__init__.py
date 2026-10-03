"""Certify a pydantic-evals judge before its scores count.

A judge's verdicts are evidence only once the judge has been shown to fail what it should fail,
pass what it should pass, and give the same verdict when nothing that matters has changed.
"""

from ._abstain import AbstainingJudge, AbstainVerdict, certify_abstention
from ._apprentice import ApprenticeResult, Proposal, ReviewedCase, apprentice
from ._bridge import Effect, JudgeBridge, compare_judge_reports, compare_judges
from ._cache import CacheStats, JudgmentCache, certify_judge_cached, uncached_judgments
from ._canary import CanaryMonitor, JudgeCanary
from ._cases import HumanLabel, JudgeCase
from ._certify import (
    Certificate,
    Check,
    FamilyResult,
    InadmissibleJudge,
    Judgment,
    Thresholds,
    assertion_of,
    certify_judge,
    recertify,
)
from ._cited import CitedJudge, CitedVerdict, certify_citations, verify_citations
from ._controls import (
    DEFAULT_CONTROLS,
    Control,
    EmptyOutput,
    EvidenceRewrite,
    MismatchedOutput,
    Rewrite,
    WhitespaceReformat,
)
from ._conversation import RecoveryProfile, insert_turns, recovery_profile
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
from ._disagree import Rating, clarify_disagreements, find_disagreements, ratings_from_certificate
from ._distill import Clause, HybridJudge, JudgeComparison, compare_hybrid, model_calls, split_rubric
from ._evidence_budget import EvidenceBudget, check_budget, evidence_budget
from ._gate import Decision, GateResult, GateRules, decide, decide_unpaired, detectable_gain
from ._identity import identity_reliable, judge_identity
from ._impact import ComparisonRecord, Impact, ImpactReport, decision_impact
from ._jury import Jury, JuryComparison, JuryUndecided, compare_jury, jury_certificate
from ._ledger import ExposedCases, FeedbackLedger
from ._logfire import finish_canary, promote, start_canary
from ._mutation import (
    DEFAULT_MUTANTS,
    EMPTIED,
    LAST_SENTENCE_DROPPED,
    NUMBER_CHANGED,
    TRUNCATED,
    YES_NO_FLIPPED,
    KillRate,
    Mutant,
    MutantSummary,
    MutationOutcome,
    MutationReport,
    UncaughtDefects,
    field_dropped,
    mutation_report,
    observed_by,
)
from ._outcomes import OutcomeReport, PassRateEstimate, outcome_calibration, recalibrated_pass_rate
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
from ._ppi import AuditSample, PPIEstimate, audit_sample, ppi_pass_rate
from ._qualify import Qualification, UnqualifiedJudge, qualify
from ._report_eval import CertifiedJudgeReport, CertifyJudgeReport, certify_for_report, result_kinds
from ._reports import compare_reports, outcomes
from ._review import ReviewPlan, ReviewResult
from ._routing import RoutedJudge, certify_routing, replay_routing
from ._stats import cohen_kappa, wilson
from ._steering import DEFAULT_NEUTRAL, SteeringResult, SteeringVerdict, assess_steering
from ._stress import ATTACKS, Attack, StressResult, stress_judge
from ._trace_controls import duplicate_call, hindsight_pair, retried_with_changed_arguments, transient_retry
from ._trace_export import certificate_attributes, certify_judge_traced, log_certificate
from ._witness import Witness, minimize_witness

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
    'FamilyResult',
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
    'identity_reliable',
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
    'ReviewPlan',
    'ReviewResult',
    'ATTACKS',
    'Attack',
    'StressResult',
    'stress_judge',
    'Witness',
    'minimize_witness',
    'DEFAULT_NEUTRAL',
    'SteeringResult',
    'SteeringVerdict',
    'assess_steering',
    'ComparisonRecord',
    'Impact',
    'ImpactReport',
    'decision_impact',
    'ExposedCases',
    'FeedbackLedger',
    'DEFAULT_MUTANTS',
    'EMPTIED',
    'LAST_SENTENCE_DROPPED',
    'NUMBER_CHANGED',
    'TRUNCATED',
    'YES_NO_FLIPPED',
    'KillRate',
    'Mutant',
    'MutantSummary',
    'MutationOutcome',
    'MutationReport',
    'UncaughtDefects',
    'field_dropped',
    'mutation_report',
    'observed_by',
    'Qualification',
    'UnqualifiedJudge',
    'qualify',
    'OutcomeReport',
    'PassRateEstimate',
    'outcome_calibration',
    'recalibrated_pass_rate',
    'duplicate_call',
    'hindsight_pair',
    'retried_with_changed_arguments',
    'transient_retry',
    'CitedJudge',
    'CitedVerdict',
    'certify_citations',
    'verify_citations',
    'AbstainVerdict',
    'AbstainingJudge',
    'certify_abstention',
    'RoutedJudge',
    'certify_routing',
    'replay_routing',
    'EvidenceBudget',
    'check_budget',
    'evidence_budget',
    'ApprenticeResult',
    'Proposal',
    'ReviewedCase',
    'apprentice',
    'Rating',
    'clarify_disagreements',
    'find_disagreements',
    'ratings_from_certificate',
    'Clause',
    'HybridJudge',
    'JudgeComparison',
    'compare_hybrid',
    'model_calls',
    'split_rubric',
    'RecoveryProfile',
    'insert_turns',
    'recovery_profile',
    'CertifiedJudgeReport',
    'CertifyJudgeReport',
    'certify_for_report',
    'result_kinds',
    'certificate_attributes',
    'certify_judge_traced',
    'log_certificate',
    'AuditSample',
    'PPIEstimate',
    'audit_sample',
    'ppi_pass_rate',
    'Jury',
    'JuryComparison',
    'JuryUndecided',
    'compare_jury',
    'jury_certificate',
    'CacheStats',
    'JudgmentCache',
    'certify_judge_cached',
    'uncached_judgments',
)
