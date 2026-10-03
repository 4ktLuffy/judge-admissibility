"""Scripted judges for the certificate's own controls.

Each is a real `LLMJudge` whose model is a `FunctionModel` that reads the `<Output>` and
`<ExpectedOutput>` sections of the prompt `LLMJudge` builds, so the code under test is the
same code a real judge runs; only the model's decision is scripted.

The model is named after its decide function, as a user names a function-backed judge: the name is
what `judge_identity` trusts, so it must tell the judges of this suite apart. It is the function's
qualified name plus the best-effort digest of its code and what it closes over (stable here: the
decide functions do not change during a run), so `judge(oracle)` and `judge(yes_man)`, and two
closures from one factory over different values, have different names. A decide function whose
digest is not complete (it closes over an rng or a counter, as `coin` does) gets no name: its
judge is not reliable, as a user's unnamed `FunctionModel` is not. `named=False` gives any judge
pydantic-ai's default name, to test that path.
"""

from __future__ import annotations

import random
import re
from collections.abc import Callable

from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_evals.evaluators import LLMJudge

from pydantic_evals_admissibility._identity import _function_digest  # pyright: ignore[reportPrivateUsage]

Decide = Callable[[str, str], bool]


def _section(prompt: str, tag: str) -> str:
    match = re.search(rf'<{tag}>\n(.*?)\n</{tag}>', prompt, re.S)
    return match.group(1) if match else ''


def judge(decide: Decide, *, named: bool = True) -> LLMJudge:
    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        prompt = ''.join(
            part.content
            for message in messages
            for part in getattr(message, 'parts', [])
            if isinstance(part, UserPromptPart) and isinstance(part.content, str)
        )
        passed = decide(_section(prompt, 'Output'), _section(prompt, 'ExpectedOutput'))
        tool = info.output_tools[0]
        return ModelResponse(
            parts=[ToolCallPart(tool.name, {'reason': 'scripted', 'pass': passed, 'score': float(passed)})]
        )

    return LLMJudge(
        rubric='The output answers the question correctly.',
        model=FunctionModel(model, model_name=_name(decide) if named else None),
        include_expected_output=True,
    )


def _name(decide: Decide) -> str | None:
    opaque: list[str] = []
    digest = _function_digest(decide, opaque)
    return None if opaque else f'scripted:{decide.__module__}.{decide.__qualname__}#{digest}'


def oracle(output: str, expected: str) -> bool:
    """Passes exactly the answers that contain the expected answer, ignoring whitespace."""
    return bool(output.strip()) and ' '.join(expected.split()) in ' '.join(output.split())


def yes_man(output: str, expected: str) -> bool:
    return True


def no_man(output: str, expected: str) -> bool:
    return False


def exact(output: str, expected: str) -> bool:
    """Correct, but whitespace-sensitive: a reformatted answer fails."""
    return output == expected


def coin(seed: int) -> Decide:
    rng = random.Random(seed)
    return lambda output, expected: rng.random() < 0.5


def lenient_on_empty(output: str, expected: str) -> bool:
    """Right on real answers, but accepts an empty one."""
    return not output.strip() or oracle(output, expected)
