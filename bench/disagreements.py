"""Where do competent judges read a rubric differently? Offline, from verdicts already paid for.

    PYTHONPATH=.:bench <venv>/bin/python bench/disagreements.py [--offline]

Two rubrics, from saved certificates (no judge calls):

- **Pydantic's example dataset** (`time_range_v2.yaml`, rubric "...in a second-person or friendly
  style"): three raters, Codex `gpt-5.6-luna` without reasoning and with high reasoning
  (`results/pydantic_example_judge.json`), and the high-reasoning judge again, run by
  `certify_dataset` (`results/certify_pydantic_dataset.json`), 3 repeats each. An item is the
  user-facing text (`explanation` or `error_message`) with whitespace collapsed: the rubric is
  about that text alone, the judge does not see the input, and the two runs print the same output
  differently (a dict and a model repr). So a whitespace control rates the same item as its
  answer, and so does another case's borrowed answer.
- **The support task** (`results/support_eval.json`, a correctness rubric): the four certified
  judges and ground truth as a fifth rater. An item is the case and the reply, since the same
  reply can be right for one question and wrong for another.

Then Codex (no reasoning, one call) proposes one sentence for the Pydantic rubric's most contested
cluster. The support task gets none: its disputes are one judge that cannot see the question
against everyone else, which no sentence in the rubric settles. Saved to
`results/disagreements.json`; `--offline` skips the calls.
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Any

from codex_judge import TOKENS, codex_text_model
from pydantic_ai import Agent

from pydantic_evals_admissibility._certify import Judgment
from pydantic_evals_admissibility._disagree import (
    Cluster,
    DisagreementReport,
    Rating,
    clarify_disagreements,
    find_disagreements,
    normalized,
    ratings_from_certificate,
)

ROOT = Path(__file__).parent.parent
OUT = ROOT / 'results' / 'disagreements.json'
_TEXT = re.compile(
    r"""(?:'(?:explanation|error_message)':\s*|(?:explanation|error_message)=)(['"])(.*?)(?<!\\)\1""", re.S
)

CLARIFIER = (
    'You maintain the rubric of an LLM judge. Judges disagreed on the outputs below: some failed them, some passed '
    'them, each with a reason. Write the shortest single sentence that, appended to the rubric, would make every '
    'careful judge reach the same verdict on outputs like these. Do not decide which side is right on the merits '
    'unless the rubric already does; say which reading the rubric intends, in its own terms. Reply with the '
    'sentence only.'
)


def user_text(j: Judgment) -> str | None:
    """The explanation or error message a style rubric is about; '' for an emptied one."""
    output = j.output if isinstance(j.output, str) else repr(j.output)
    match = _TEXT.search(output)
    # Literal escapes from the repr (a whitespace control adds a newline) are whitespace too.
    return ' '.join(match.group(2).replace('\\n', ' ').replace('\\t', ' ').split()) if match else None


def pydantic_ratings() -> tuple[str, list[Rating]]:
    example = json.loads((ROOT / 'results' / 'pydantic_example_judge.json').read_text())
    dataset = json.loads((ROOT / 'results' / 'certify_pydantic_dataset.json').read_text())
    ratings: list[Rating] = []
    for label, cert in example['judges'].items():
        ratings += ratings_from_certificate(cert, rater=f'{label} (hand controls)', item=user_text)
    for cert in dataset['judges']:
        if cert.get('judgments'):
            ratings += ratings_from_certificate(cert, rater='reasoning high (certify_dataset)', item=user_text)
    # The dataset's author is a rater too: every expected output is an answer they marked good.
    marked = {user_text(Judgment(j['case'], j['role'], j['output'], j['passed'])) for j in
              next(iter(example['judges'].values()))['judgments'] if j['role'] == 'reference#0'}  # fmt: skip
    ratings += [Rating(item, 'dataset author (expected_output)', True) for item in sorted(m for m in marked if m)]
    return example['rubric'], ratings


def support_ratings() -> tuple[str, list[Rating]]:
    data = json.loads((ROOT / 'results' / 'support_eval.json').read_text())

    def key(j: Judgment) -> str:
        return f'{j.case} | {normalized(j.output)}'

    ratings: list[Rating] = []
    for label, cert in data['certificates'].items():
        ratings += ratings_from_certificate(cert, rater=label, item=key)
    # Ground truth rates each item once: the labelled replies by their label, the known-good answers and their
    # whitespace copies as right, the borrowed answers (another question's, with a different answer) as wrong.
    truth: dict[str, bool] = {}
    for cert in data['certificates'].values():
        for j in cert['judgments']:
            role, item = j['role'], f'{j["case"]} | {normalized(j["output"])}'
            if role.startswith('human:'):
                truth[item] = role == 'human:1'
            elif role.startswith(('reference#', 'must_hold:')):
                truth.setdefault(item, True)
            elif role.startswith('must_fail:'):
                truth.setdefault(item, False)
    ratings += [Rating(item, 'ground truth', passed) for item, passed in sorted(truth.items())]
    rubric = "The reply correctly answers the customer's question according to the store policy."
    return rubric, ratings


async def codex_clarify(rubric: str, cluster: Cluster) -> str:
    sides = []
    for reading in (cluster.fail, cluster.passing):
        quotes = '\n'.join(f'- ({rater}) {reason}' for rater, reason in reading.quotes[:4])
        sides.append(f'Judges who gave {reading.verdict} said:\n{quotes}')
    outputs = '\n'.join(f'- {i.item}' for i in cluster.items[:6])
    prompt = f'Rubric:\n{rubric}\n\nContested outputs:\n{outputs}\n\n' + '\n\n'.join(sides)
    agent = Agent(codex_text_model(effort='none', timeout=300), instructions=CLARIFIER)
    return (await agent.run(prompt)).output.strip()


async def run(
    name: str, rubric: str, ratings: list[Rating], saved: dict[str, Any], *, offline: bool, clarify: bool
) -> dict[str, Any]:
    report: DisagreementReport = find_disagreements(ratings)
    old = saved.get(name, {}).get('clusters', [])
    cached = {tuple(c['items']): c['clarification'] for c in old if c.get('clarification')}
    if any(tuple(i.item for i in c.items) in cached for c in report.clusters):
        report = await clarify_disagreements(
            report, rubric, lambda r, c: cached.get(tuple(i.item for i in c.items), ''), max_calls=1
        )
    elif not offline and clarify:
        report = await clarify_disagreements(report, rubric, codex_clarify, max_calls=1)
    print(f'\n===== {name}\nrubric: {rubric}\n\n{report.table()}', flush=True)
    return {'rubric': rubric, 'ratings': len(ratings), **report.to_dict()}


async def main() -> None:
    offline = '--offline' in sys.argv
    saved: dict[str, Any] = json.loads(OUT.read_text()) if OUT.exists() else {}
    results: dict[str, Any] = {}
    # The support task's disputes turn out to be about who sees the question (below), so no sentence is asked for.
    for name, (rubric, ratings), clarify in (
        ('pydantic example dataset', pydantic_ratings(), True),
        ('support task', support_ratings(), False),
    ):
        results[name] = await run(name, rubric, ratings, saved, offline=offline, clarify=clarify)
        OUT.write_text(json.dumps(results, indent=2, default=str))
    results['codex_calls'] = len(TOKENS)
    OUT.write_text(json.dumps(results, indent=2, default=str))
    print(f'\n{len(TOKENS)} Codex calls this run')


if __name__ == '__main__':
    asyncio.run(main())
