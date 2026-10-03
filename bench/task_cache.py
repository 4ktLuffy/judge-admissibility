"""Re-score without re-running the task: what `evaluate_cached` saves, counted in model calls.

    PYTHONPATH=.:bench <venv>/bin/python bench/task_cache.py [--no-codex]

`--no-codex` re-measures stage A only and keeps the Codex stage already in the results file.

pydantic-ai issues #1350 and #3314 ask to run new evaluators on a dataset without paying for the
task again. Each stage evaluates one dataset four times through `evaluate_cached` and counts the
model calls the task's agent makes:

1. first evaluation (every case runs),
2. re-scoring with a new evaluator added (nothing should run),
3. one case's inputs edited (that case should run, and only it),
4. an unchanged rerun (nothing should run).

Stage A is a pydantic-ai `Agent` on a scripted `FunctionModel` with a call counter (no model):
the four steps above, plus what the identity rules cost: the same task without a `version` (it
reads a module-level `Agent`, which cannot be identified by value, so it is never cached), a
bumped version, and two evaluations at once on one cache. Stage B is the same four steps with the
agent on Codex (`gpt-5.6-luna`, no reasoning), 4 cases, so at most 5 Codex calls. If Codex is out
of credits, the model falls back to `claude -p --model claude-haiku-4-5 --effort low`; every
output's model is recorded either way.

Writes `results/task_cache.json`.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext

from pydantic_evals_admissibility._task_cache import (
    TaskCache,
    TaskCacheStats,
    evaluate_cached,
    task_identity,
    task_identity_reliable,
)

RESULTS = Path(__file__).resolve().parent.parent / 'results' / 'task_cache.json'

QUESTIONS = {
    'refund-window': 'How many days do I have to ask for a refund?',
    'reset-password': 'How do I reset my password?',
    'change-plan': 'Can I switch from the monthly to the yearly plan?',
    'cancel': 'How do I cancel my subscription?',
    'invoice': "Where can I download last month's invoice?",
    'two-factor': 'How do I turn on two-factor authentication?',
    'export': 'Can I export my data as CSV?',
    'delete-account': 'How do I delete my account?',
}
EDITED = ('refund-window', 'How many days after delivery can I still ask for a refund?')


@dataclasses.dataclass
class Answered(Evaluator[Any, Any, Any]):
    """The reply is not empty."""

    def evaluate(self, ctx: EvaluatorContext[Any, Any, Any]) -> bool:
        return bool(str(ctx.output).strip())


@dataclasses.dataclass
class Concise(Evaluator[Any, Any, Any]):
    """The evaluator added when re-scoring: the reply is at most `words` words."""

    words: int = 60

    def evaluate(self, ctx: EvaluatorContext[Any, Any, Any]) -> bool:
        return len(str(ctx.output).split()) <= self.words


def dataset(questions: dict[str, str], *, rescore: bool = False) -> Dataset[Any, Any, Any]:
    evaluators: list[Evaluator[Any, Any, Any]] = [Answered()]
    if rescore:
        evaluators.append(Concise())
    cases = [Case(name=name, inputs=question) for name, question in questions.items()]
    return Dataset(name='support-questions', cases=cases, evaluators=evaluators)


# --- the agents -----------------------------------------------------------------------------------

MODEL_CALLS: list[dict[str, Any]] = []
"""Every model call the task's agent made: which model answered which question."""


def _question(messages: list[ModelMessage]) -> str:
    return next(
        part.content
        for message in messages
        for part in getattr(message, 'parts', ())
        if isinstance(part, UserPromptPart) and isinstance(part.content, str)
    )


async def scripted(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    question = _question(messages)
    MODEL_CALLS.append({'model': 'scripted-support-v1', 'question': question})
    await asyncio.sleep(0.02)  # a model call takes time; a served output should not
    return ModelResponse(parts=[TextPart(f'Support answer to: {question}')], model_name='scripted-support-v1')


INSTRUCTIONS = (
    'You are a support agent for a subscription software product. Answer the customer in at most two '
    'sentences. If you do not know a policy detail, say what the customer should check instead.'
)
SCRIPTED_AGENT = Agent(FunctionModel(scripted, model_name='scripted-support-v1'), instructions=INSTRUCTIONS)


async def scripted_answer(question: str) -> str:
    return (await SCRIPTED_AGENT.run(question)).output


FALLBACK = {'active': False, 'reason': None}


async def _claude(question: str) -> str:
    """`claude -p` with a small model at low effort: used only once Codex is out of credits."""
    proc = await asyncio.create_subprocess_exec(
        'claude', '-p', '--model', 'claude-haiku-4-5', '--effort', 'low', '--system-prompt', INSTRUCTIONS, question,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )  # fmt: skip
    out, err = await asyncio.wait_for(proc.communicate(), timeout=180)
    if proc.returncode:
        raise RuntimeError(f'claude -p failed: {err.decode()[-400:]}')
    return out.decode().strip()


def _out_of_credits(exc: BaseException) -> bool:
    text = str(exc).lower()
    return any(s in text for s in ('credit', 'quota', 'usage limit', 'rate limit reached', 'insufficient'))


def make_codex_agent() -> Agent[None, str]:
    from codex_judge import codex_text_model

    codex = codex_text_model(effort='none')

    async def counted(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        question = _question(messages)
        if not FALLBACK['active']:
            try:
                response = await codex.function(messages, info)  # pyright: ignore[reportOptionalCall,reportGeneralTypeIssues]
                MODEL_CALLS.append({'model': codex.model_name, 'question': question})
                return response
            except Exception as exc:
                if not _out_of_credits(exc):
                    raise
                FALLBACK.update(active=True, reason=str(exc)[-300:])
        reply = await _claude(question)
        MODEL_CALLS.append({'model': 'claude-haiku-4-5@low', 'question': question})
        return ModelResponse(parts=[TextPart(reply)], model_name='claude-haiku-4-5@low')

    return Agent(FunctionModel(counted, model_name=codex.model_name), instructions=INSTRUCTIONS)


CODEX_AGENT: Agent[None, str] | None = None


async def codex_answer(question: str) -> str:
    assert CODEX_AGENT is not None
    return (await CODEX_AGENT.run(question)).output


# --- measuring --------------------------------------------------------------------------------


async def step(label: str, ds: Dataset[Any, Any, Any], task: Any, cache: TaskCache, **kw: Any) -> dict[str, Any]:
    before = len(MODEL_CALLS)
    started = time.perf_counter()
    report = await evaluate_cached(ds, task, cache=cache, progress=False, max_concurrency=2, **kw)
    seconds = time.perf_counter() - started
    stats = TaskCacheStats.from_report(report)
    served = [c for c in report.cases if c.attributes['task_cache']['cached']]
    return {
        'step': label,
        'model_calls': len(MODEL_CALLS) - before,
        'task_runs': stats.runs,
        'served_from_cache': stats.hits,
        'ran': list(stats.ran),
        'failures': [f.error_message for f in report.failures],
        'assertions': {c.name: {k: v.value for k, v in sorted(c.assertions.items())} for c in report.cases},
        'outputs': {c.name: c.output for c in report.cases},
        'served_marked_cached': len(served),
        'mean_task_seconds_served': round(sum(c.task_duration for c in served) / len(served), 6) if served else None,
        'mean_original_task_seconds_served': (
            round(sum(c.attributes['task_cache']['original_task_duration'] for c in served) / len(served), 6)
            if served
            else None
        ),
        'wall_seconds': round(seconds, 3),
    }


async def four_steps(task: Any, cache: TaskCache, questions: dict[str, str], **kw: Any) -> list[dict[str, Any]]:
    edited = {**questions, EDITED[0]: EDITED[1]} if EDITED[0] in questions else questions
    steps = [
        await step('1 first evaluation', dataset(questions), task, cache, **kw),
        await step('2 re-score with a new evaluator (Concise)', dataset(questions, rescore=True), task, cache, **kw),
        await step(f'3 one case edited ({EDITED[0]})', dataset(edited, rescore=True), task, cache, **kw),
        await step('4 unchanged rerun', dataset(edited, rescore=True), task, cache, **kw),
    ]
    first, rescored = steps[0]['outputs'], steps[1]['outputs']
    steps[1]['outputs_identical_to_first'] = first == rescored
    steps[2]['unedited_outputs_identical'] = all(
        steps[2]['outputs'][n] == first[n] for n in questions if n != EDITED[0]
    )
    return steps


def summary(steps: list[dict[str, Any]]) -> dict[str, int]:
    return {s['step']: s['model_calls'] for s in steps}


async def stage_scripted(cache_dir: Path) -> dict[str, Any]:
    identity = task_identity(scripted_answer)
    out: dict[str, Any] = {
        'model': 'scripted FunctionModel (no model calls)',
        'identity_without_version': {
            'reliable': task_identity_reliable(identity),
            'opaque': identity.get('opaque', []),
        },
    }
    with TaskCache(cache_dir / 'scripted.sqlite') as cache:
        steps = await four_steps(scripted_answer, cache, QUESTIONS, version='support-v1')
        out['steps'] = steps
        out['model_calls_by_step'] = summary(steps)
        uncached_total = len(QUESTIONS) * len(steps)
        out['model_calls_total'] = sum(s['model_calls'] for s in steps)
        out['model_calls_without_cache'] = uncached_total
        bumped = await step('version bumped to support-v2', dataset(QUESTIONS), scripted_answer, cache,
                            version='support-v2')  # fmt: skip
        out['version_bump'] = {'model_calls': bumped['model_calls'], 'task_runs': bumped['task_runs']}
    # Without a version the task is not reliably identified: never served, never stored.
    with TaskCache(cache_dir / 'scripted-unversioned.sqlite') as cache:
        unversioned = [await step(f'no version, run {i}', dataset(QUESTIONS), scripted_answer, cache) for i in (1, 2)]
        out['without_version'] = {
            'model_calls_by_run': [s['model_calls'] for s in unversioned],
            'stored': len(cache),
        }
    # Two evaluations at once on one cache: each case runs once between them.
    with TaskCache(cache_dir / 'scripted-concurrent.sqlite') as cache:
        before = len(MODEL_CALLS)
        await asyncio.gather(
            step('concurrent A', dataset(QUESTIONS), scripted_answer, cache, version='support-v1'),
            step('concurrent B', dataset(QUESTIONS, rescore=True), scripted_answer, cache, version='support-v1'),
        )
        out['two_concurrent_evaluations'] = {'model_calls': len(MODEL_CALLS) - before, 'cases': len(QUESTIONS)}
    return out


async def stage_codex(cache_dir: Path) -> dict[str, Any]:
    global CODEX_AGENT
    CODEX_AGENT = make_codex_agent()
    questions = dict(list(QUESTIONS.items())[:4])
    first_call = len(MODEL_CALLS)
    with TaskCache(cache_dir / 'codex.sqlite') as cache:
        steps = await four_steps(codex_answer, cache, questions, version='support-codex-v1')
    calls = MODEL_CALLS[first_call:]
    return {
        'model': 'codex:gpt-5.6-luna@none',
        'fallback': dict(FALLBACK),
        'models_used': sorted({c['model'] for c in calls}),
        'calls': calls,
        'steps': steps,
        'model_calls_by_step': summary(steps),
        'model_calls_total': len(calls),
        'model_calls_without_cache': len(questions) * len(steps),
    }


async def main() -> None:
    results: dict[str, Any] = {'python': sys.version.split()[0]}
    with tempfile.TemporaryDirectory() as tmp:
        results['scripted'] = await stage_scripted(Path(tmp))
        if '--no-codex' not in sys.argv:
            results['codex'] = await stage_codex(Path(tmp))
        elif RESULTS.exists() and 'codex' in (kept := json.loads(RESULTS.read_text())):
            results['codex'] = kept['codex']  # the measured Codex stage is kept, not re-run
    RESULTS.write_text(json.dumps(results, indent=2, default=repr) + '\n')
    for stage in ('scripted', 'codex'):
        if stage in results:
            r = results[stage]
            print(
                stage, r['model_calls_by_step'], f'total {r["model_calls_total"]} vs {r["model_calls_without_cache"]}'
            )
    s = results['scripted']
    print('without version:', s['without_version'], 'version bump:', s['version_bump'])
    print('concurrent:', s['two_concurrent_evaluations'])


if __name__ == '__main__':
    asyncio.run(main())
