"""Use the Codex CLI as a Pydantic AI model, with no API key: as an `LLMJudge`'s model, or an agent's.

`LLMJudge` builds its prompt and system prompt as usual; this `FunctionModel` hands them to
`codex exec`, replacing Codex's own agent instructions with the judge's system prompt through
`model_instructions_file`, asks for the judge's output as JSON against a schema, and returns it
as the judge's output tool call. Runs on the ChatGPT account Codex is signed in with.

`experimental_instructions_file` looks like the same setting but is ignored: told to reply only
"BANANA", Codex answers the question instead. `model_instructions_file` is honoured.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import tempfile
from pathlib import Path

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

SCHEMA = {
    'type': 'object',
    'additionalProperties': False,
    'properties': {'reason': {'type': 'string'}, 'pass': {'type': 'boolean'}, 'score': {'type': 'number'}},
    'required': ['reason', 'pass', 'score'],
}
WORKDIR = Path(__file__).resolve().parent.parent / 'results' / 'scratch' / 'codex-judge'
TOKENS: list[int] = []  # tokens Codex reports per call, for the cost line in the results


def _text(messages: list[ModelMessage], kind: type) -> str:
    return '\n'.join(
        part.content
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, kind) and isinstance(part.content, str)
    )


async def run_codex(instructions: str, prompt: str, *, model: str, effort: str, schema: Path | None, timeout: float) -> str:
    """One `codex exec` call with `instructions` in place of Codex's own agent prompt; returns its last message."""
    WORKDIR.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(instructions.encode()).hexdigest()[:16]
    instructions_file = WORKDIR / f'instructions-{digest}.md'
    if not instructions_file.exists():
        # Atomic, so a concurrent call never reads a half-written file.
        partial = instructions_file.with_suffix(f'.{id(prompt)}.tmp')
        partial.write_text(instructions)
        partial.replace(instructions_file)
    with tempfile.NamedTemporaryFile(dir=WORKDIR, suffix='.out', delete=False) as out:
        out_path = Path(out.name)
    args = [
        'codex', 'exec', '-m', model, '-c', f'model_reasoning_effort="{effort}"',
        '-c', f'model_instructions_file="{instructions_file}"',
        '--skip-git-repo-check', '--sandbox', 'read-only', '--ephemeral', '--color', 'never',
        '-o', str(out_path),
    ]  # fmt: skip
    if schema is not None:
        args += ['--output-schema', str(schema)]
    process = await asyncio.create_subprocess_exec(
        *args, prompt, stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT, cwd=WORKDIR,
    )  # fmt: skip
    try:
        log, _ = await asyncio.wait_for(process.communicate(), timeout)
    except TimeoutError:
        process.kill()
        raise
    text = log.decode(errors='replace')
    if 'tokens used' in text:
        try:
            TOKENS.append(int(text.split('tokens used', 1)[1].split()[0].replace(',', '')))
        except ValueError:
            pass
    if process.returncode != 0:
        raise RuntimeError(f'codex exited {process.returncode}: {text[-300:]}')
    reply = out_path.read_text()
    out_path.unlink(missing_ok=True)
    return reply


def codex_model(model: str = 'gpt-5.6-luna', effort: str = 'none', timeout: float = 120) -> FunctionModel:
    """A judge model: returns `LLMJudge`'s structured verdict."""
    WORKDIR.mkdir(parents=True, exist_ok=True)
    schema = WORKDIR / 'schema.json'
    schema.write_text(json.dumps(SCHEMA))

    async def judge(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        instructions = (info.instructions or '') + '\n' + _text(messages, SystemPromptPart)
        instructions += '\nReply with the JSON verdict only: reason, pass, score.'
        reply = await run_codex(
            instructions, _text(messages, UserPromptPart), model=model, effort=effort, schema=schema, timeout=timeout
        )
        tool = info.output_tools[0]
        return ModelResponse(parts=[ToolCallPart(tool.name, json.loads(reply))], model_name=f'codex:{model}')

    return FunctionModel(judge, model_name=f'codex:{model}')


def codex_text_model(model: str = 'gpt-5.6-luna', effort: str = 'none', timeout: float = 120) -> FunctionModel:
    """An agent model: the agent's instructions replace Codex's, and the reply is plain text."""

    async def agent(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        instructions = ((info.instructions or '') + '\n' + _text(messages, SystemPromptPart)).strip()
        reply = await run_codex(
            instructions or 'Answer the question.', _text(messages, UserPromptPart),
            model=model, effort=effort, schema=None, timeout=timeout,
        )  # fmt: skip
        return ModelResponse(parts=[TextPart(reply)], model_name=f'codex:{model}')

    return FunctionModel(agent, model_name=f'codex:{model}')
