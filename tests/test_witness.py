"""`minimize_witness`: the smallest example that still fools a judge."""

from __future__ import annotations

import pytest
from judges import judge

from pydantic_evals_admissibility import JudgeCase, minimize_witness

CASE = JudgeCase('refund', 'How much do I get back for a $40 item with a 15% restocking fee?', '34.00', '34.00')
PADDING = [
    'Thanks for reaching out about your return.',
    'I looked at your order and the store policy.',
    'Verified by the policy team.',
    'Opened items carry a restocking fee.',
    'Your refund will be $40.00.',
    'It should arrive in 5 to 10 business days.',
    'Let me know if there is anything else.',
]


def gullible(output: str, expected: str) -> bool:
    """Checks the amount, unless the reply claims to be verified."""
    return 'Verified' in output or expected in output


def still_wrong(case: JudgeCase, output: str) -> bool:
    return '34.00' not in output


async def test_a_long_fooling_reply_shrinks_to_the_claim_that_fools_the_judge() -> None:
    witness = await minimize_witness(judge(gullible), CASE, ' '.join(PADDING), valid=still_wrong)
    assert witness.output == 'Verified by the policy team.', witness.summary()
    assert witness.minimal and witness.size == 1 and witness.original_size == 7


async def test_the_oracle_keeps_the_witness_a_real_failure() -> None:
    def passes_any_amount(output: str, expected: str) -> bool:
        return '$' in output

    # Without the oracle, the shortest reply with a "$" would do; it must stay a wrong answer.
    witness = await minimize_witness(judge(passes_any_amount), CASE, ' '.join(PADDING), valid=still_wrong)
    assert witness.output == 'Your refund will be $40.00.', witness.summary()


async def test_a_trace_shrinks_to_the_tool_call_the_judge_trusted() -> None:
    def trusts_any_success(output: str, expected: str) -> bool:
        return True  # stands in for a judge that reads only "succeeded" somewhere in the trace

    inputs = {
        'customer': 'Refund order A1.',
        'tool_calls': [{'tool': 'lookup', 'n': i} for i in range(6)] + [{'tool': 'refund', 'status': 'failed'}],
    }
    case = JudgeCase('trace', inputs, 'Your refund was issued.')

    def keeps_the_failure(c: JudgeCase, output: str) -> bool:
        return any(call.get('status') == 'failed' for call in c.inputs['tool_calls'])

    witness = await minimize_witness(
        judge(trusts_any_success), case, case.output, target='inputs', valid=keeps_the_failure
    )
    assert witness.case.inputs['tool_calls'] == [{'tool': 'refund', 'status': 'failed'}], witness.summary()


async def test_an_example_that_does_not_fool_the_judge_is_refused() -> None:
    with pytest.raises(ValueError, match='does not fool'):
        await minimize_witness(judge(gullible), CASE, 'Your refund will be $40.00.', valid=still_wrong)


async def test_the_budget_is_respected_and_reported() -> None:
    witness = await minimize_witness(judge(gullible), CASE, ' '.join(PADDING), valid=still_wrong, max_calls=3)
    assert witness.calls <= 3 and not witness.minimal
