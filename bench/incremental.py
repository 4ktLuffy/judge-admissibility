"""Incremental re-certification: what re-certifying after a small dataset edit costs with the cache.

    PYTHONPATH=.:bench <venv>/bin/python bench/incremental.py

1. Scripted (no model): a judge that reads the reply's `Answer:` line is certified on the 40
   support cases (3 repeats), 4 replies are reworded, and it is certified again from the cache.
   Measured: judge calls on the second run against a full run, and whether the cached certificate
   equals an uncached `certify_judge` on the edited cases.
2. Real (Codex `gpt-5.6-luna`, no reasoning, at most 2 calls in flight): the same on 10 support
   cases with 1 repeat, 2 replies reworded. Measured: calls on the second run, and that every
   verdict served from the cache is the first run's verdict, reason included. Then an unchanged
   third run, and what a change of judge would cost (counted, not run).

Agent replies come from `results/support-cache.json` (written by `support_eval.py`); no agent is
called. The real stage runs once per cache key format (`CACHE_FORMAT`); its numbers are kept in
`results/incremental.json` and re-measured when the key format changes, since old keys no longer
match.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from codex_judge import TOKENS, codex_model
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_evals.evaluators import LLMJudge
from support_eval import AGENT_PROMPT, CACHE, RUBRIC, fmt
from support_task import SupportCase, cases, is_correct

from pydantic_evals_admissibility import JudgeCase, certify_judge
from pydantic_evals_admissibility._cache import CACHE_FORMAT, JudgmentCache, certify_judge_cached, uncached_judgments
from pydantic_evals_admissibility._identity import identity_reliable, judge_identity

ROOT = Path(__file__).parent.parent
OUT = ROOT / 'results' / 'incremental.json'
DB = ROOT / 'results' / 'scratch' / 'incremental.sqlite'
REWORD = 'Thanks for reaching out! '


def judge_cases(cs: list[SupportCase]) -> list[JudgeCase]:
    """Each case's first correct cached reply, as `support_eval.py` picks them."""
    replies: dict[str, str] = json.loads(CACHE.read_text())
    out = []
    for c in cs:
        mine = [replies[hashlib.sha256(f'{AGENT_PROMPT}|{c.inputs}|{r}'.encode()).hexdigest()] for r in range(2)]
        good = next((r for r in mine if is_correct(c, r)), f'Answer: {fmt(c, c.answer)}')
        out.append(JudgeCase(c.name, c.inputs, good, expected_output=c.answer))
    return out


def reword(cs: list[JudgeCase], names: set[str]) -> list[JudgeCase]:
    """A wording edit: the reply is still right, but it is not the text that was judged."""
    return [dataclasses.replace(c, output=REWORD + c.output) if c.name in names else c for c in cs]


def scripted_judge() -> LLMJudge:
    """Passes a reply whose last `Answer:` line holds the expected answer. Deterministic."""

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        prompt = ''.join(
            p.content
            for m in messages
            for p in getattr(m, 'parts', [])
            if isinstance(p, UserPromptPart) and isinstance(p.content, str)
        )
        output = re.search(r'<Output>\n(.*?)\n</Output>', prompt, re.S)
        expected = re.search(r'<ExpectedOutput>\n(.*?)\n</ExpectedOutput>', prompt, re.S)
        lines = re.findall(r'answer\W*:\s*(.+)', output.group(1), re.I) if output else []
        passed = bool(lines and expected and expected.group(1).strip() in lines[-1].replace('$', ''))
        tool = info.output_tools[0]
        return ModelResponse(parts=[ToolCallPart(tool.name, {'reason': 'scripted', 'pass': passed, 'score': 1.0})])

    # Named: an unnamed function-backed model is never cached. The name is ours to bump if the code changes.
    return LLMJudge(
        rubric=RUBRIC,
        model=FunctionModel(model, model_name='scripted:last-answer-line-v1'),
        include_expected_output=True,
    )


async def scripted() -> dict[str, Any]:
    cs = judge_cases(cases())
    edited_names = {cs[i].name for i in (3, 14, 25, 36)}
    edited = reword(cs, edited_names)
    judge, cache = scripted_judge(), JudgmentCache()
    first, cold = await certify_judge_cached(judge, cs, cache=cache)
    second, warm = await certify_judge_cached(judge, edited, cache=cache)
    full = await certify_judge(judge, edited)
    changed_judge = dataclasses.replace(judge, include_input=True)
    return {
        'cache_format': CACHE_FORMAT,
        'identity_reliable': identity_reliable(judge_identity(judge)),
        'cases': len(cs),
        'edited': sorted(edited_names),
        'planned': first.planned,
        'first_run_calls': cold.misses,
        'second_run_calls': warm.misses,
        'second_run_saved': warm.calls_saved,
        'second_run_missed': [list(m) for m in warm.missed],
        'identical_to_uncached': second == full and second.to_dict() == full.to_dict(),
        'verdict': second.verdict,
        'calls_after_judge_change': len(uncached_judgments(changed_judge, edited, cache=cache)),
    }


async def real() -> dict[str, Any]:
    cs = judge_cases(cases()[::4])
    edited_names = {cs[0].name, cs[5].name}
    edited = reword(cs, edited_names)
    judge = LLMJudge(rubric=RUBRIC, model=codex_model(effort='none'), include_input=True)
    DB.parent.mkdir(parents=True, exist_ok=True)
    DB.unlink(missing_ok=True)  # a fresh cache, so the first run is a full run
    out: dict[str, Any] = {
        'cache_format': CACHE_FORMAT,
        'identity_reliable': identity_reliable(judge_identity(judge)),
        'cases': [c.name for c in cs],
        'edited': sorted(edited_names),
        'repeats': 1,
    }
    with JudgmentCache(DB) as cache:
        runs = []
        for label, dataset in (('first', cs), ('after edit', edited), ('unchanged rerun', edited)):
            before = len(TOKENS)
            certificate, stats = await certify_judge_cached(judge, dataset, cache=cache, repeats=1, max_concurrency=2)
            runs.append((certificate, stats))
            out[label] = {
                'judge_calls': stats.misses,
                'codex_calls_with_token_report': len(TOKENS) - before,
                'tokens': sum(TOKENS[before:]),
                'served_from_cache': stats.hits,
                'errors': stats.errors,
                'missed': [list(m) for m in stats.missed],
                'certificate': certificate.to_dict(),
            }
            print(
                f'{label}: {stats.misses} calls, {stats.hits} from cache, {stats.errors} errors\n{certificate.table()}'
            )
            if label == 'first' and stats.errors > 2:
                out['stopped'] = 'judge errors on the first run (out of credits?)'
                return out
        first, (second, warm) = runs[0][0], runs[1]
        before = {(j.case, j.role, j.output): (j.passed, j.reason) for j in first.judgments}
        missed = set(warm.missed)
        reused = [j for j in second.judgments if (j.case, j.role) not in missed]
        out['reused_verdicts'] = len(reused)
        out['reused_identical_to_first_run'] = sum(before.get((j.case, j.role, j.output)) == (j.passed, j.reason)
                                                   for j in reused)  # fmt: skip
        out['full_run_calls'] = first.planned
        stronger = LLMJudge(rubric=RUBRIC, model=codex_model(effort='low'), include_input=True)
        out['calls_if_effort_changed_to_low'] = len(uncached_judgments(stronger, edited, cache=cache, repeats=1))
    return out


async def main() -> None:
    results: dict[str, Any] = json.loads(OUT.read_text()) if OUT.exists() else {}
    results['scripted'] = await scripted()
    print(json.dumps({k: v for k, v in results['scripted'].items() if k != 'second_run_missed'}, indent=2))
    OUT.write_text(json.dumps(results, indent=2, default=str))
    if results.get('codex', {}).get('cache_format') != CACHE_FORMAT:
        results['codex'] = await real()
        OUT.write_text(json.dumps(results, indent=2, default=str))
    codex = results['codex']
    print({k: codex.get(k) for k in ('reused_verdicts', 'reused_identical_to_first_run', 'full_run_calls')})


if __name__ == '__main__':
    asyncio.run(main())
