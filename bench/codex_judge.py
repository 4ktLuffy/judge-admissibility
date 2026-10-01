"""Use the Codex CLI as the model behind a real `LLMJudge`, with no API key.

`LLMJudge` builds its prompt and system prompt as usual; this `FunctionModel` hands them to
`codex exec`, replacing Codex's own agent instructions with the judge's system prompt (which
also cuts each call from ~12,600 tokens to ~600), asks for the judge's output as JSON against a
schema, and returns it as the judge's output tool call. Runs on the ChatGPT account Codex is
signed in with.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import tempfile
from pathlib import Path

from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, SystemPromptPart, ToolCallPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

SCHEMA = {
    'type': 'object',
    'additionalProperties': False,
    'properties': {'reason': {'type': 'string'}, 'pass': {'type': 'boolean'}, 'score': {'type': 'number'}},
    'required': ['reason', 'pass', 'score'],
}
# Not the system temp dir: Codex silently ignores an instructions file under `/var/folders/...`
# and falls back to its own ~12,600-token agent prompt. Measured, not guessed.
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


def codex_model(model: str = 'gpt-5.6-luna', effort: str = 'none', timeout: float = 120) -> FunctionModel:
    WORKDIR.mkdir(parents=True, exist_ok=True)
    schema = WORKDIR / 'schema.json'
    schema.write_text(json.dumps(SCHEMA))

    async def judge(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        instructions = (info.instructions or '') + '\n' + _text(messages, SystemPromptPart)
        instructions += '\nReply with the JSON verdict only: reason, pass, score.'
        digest = hashlib.sha256(instructions.encode()).hexdigest()[:16]
        instructions_file = WORKDIR / f'instructions-{digest}.md'
        if not instructions_file.exists():
            # Atomic: a concurrent call that finds a half-written file would make Codex ignore it
            # and fall back to its 12,600-token agent prompt.
            partial = instructions_file.with_suffix(f'.{id(messages)}.tmp')
            partial.write_text(instructions)
            partial.replace(instructions_file)
        with tempfile.NamedTemporaryFile(dir=WORKDIR, suffix='.json', delete=False) as out:
            out_path = Path(out.name)
        process = await asyncio.create_subprocess_exec(
            'codex', 'exec', '-m', model, '-c', f'model_reasoning_effort="{effort}"',
            '-c', f'experimental_instructions_file="{instructions_file}"',
            '--skip-git-repo-check', '--sandbox', 'read-only', '--ephemeral', '--color', 'never',
            '--output-schema', str(schema), '-o', str(out_path), _text(messages, UserPromptPart),
            stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
            cwd=WORKDIR,
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
        verdict = json.loads(out_path.read_text())
        out_path.unlink(missing_ok=True)
        tool = info.output_tools[0]
        return ModelResponse(parts=[ToolCallPart(tool.name, verdict)], model_name=f'codex:{model}')

    return FunctionModel(judge, model_name=f'codex:{model}')
