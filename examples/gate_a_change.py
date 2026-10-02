"""Gate a prompt change on two pydantic-evals reports, then apply it only if it holds up.

Runs offline: the "agents" are plain functions and the variable provider is Logfire's local
in-memory one. Swap in your agent, your dataset and `logfire.configure()` for real use.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import logfire
from logfire.variables.config import LabeledValue, LatestVersion, Rollout, VariableConfig, VariablesConfig
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext

from pydantic_evals_admissibility import GateRules, compare_reports, detectable_gain, outcomes, promote


@dataclass
class Correct(Evaluator[int, int, None]):
    def evaluate(self, ctx: EvaluatorContext[int, int, None]) -> dict[str, bool]:
        return {'correct': ctx.output == ctx.expected_output}


dataset = Dataset[int, int, None](
    name='doubling',
    cases=[Case(name=f'case-{i}', inputs=i, expected_output=2 * i) for i in range(60)],
    evaluators=[Correct()],
)


def current_prompt(x: int) -> int:  # stands in for the agent with the current prompt
    return 2 * x if x % 3 else x


def candidate_prompt(x: int) -> int:  # stands in for the agent with the proposed prompt
    return 2 * x


async def main() -> bool:
    logfire.configure(
        send_to_logfire=False,
        console=False,
        variables=logfire.LocalVariablesOptions(
            config=VariablesConfig(
                variables={
                    'agent_prompt': VariableConfig(
                        name='agent_prompt',
                        labels={'production': LabeledValue(version=1, serialized_value='"current"')},
                        rollout=Rollout(labels={'production': 1.0}),
                        overrides=[],
                        latest_version=LatestVersion(version=1, serialized_value='"current"'),
                    )
                }
            )
        ),
    )
    prompt = logfire.var('agent_prompt', type=str, default='current')

    baseline = await dataset.evaluate(current_prompt, repeat=2, progress=False)
    print('smallest gain this dataset can see:', detectable_gain(outcomes(baseline, 'correct')))

    candidate = await dataset.evaluate(candidate_prompt, repeat=2, progress=False)
    result = compare_reports(baseline, candidate, assertion='correct', rules=GateRules())
    print(result.summary())

    promoted = promote(result, 'agent_prompt', 'candidate')
    with prompt.get() as served:
        print('serving:', served.value)
    return promoted


if __name__ == '__main__':
    asyncio.run(main())
