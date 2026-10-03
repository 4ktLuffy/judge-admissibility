"""Similarity evaluators: normalized edit distance in code, and embedding cosine through a provider.

Both compare the output with a reference: the case's `expected_output` by default, or any field
named by a dotted path (`'metadata.reference'`, `'inputs.answer'`, `'expected_output.text'`). The
path is explicit configuration, not a guess at the dataset's shape, so the same spec works on
typed cases and on YAML-loaded dicts. A case whose reference is missing is skipped (no result),
as `EqualsExpected` does.

Each returns two results: the raw score, and a pass/fail assertion against `threshold`. The
assertion is what `certify_judge` reads, so a similarity check can be certified like any judge.

A high string similarity is not a correct answer. On this repo's support task a correct reply in
different words scores lower than another case's answer in the same template; see
`bench/safety_evaluators.py` and `results/safety_evaluators.json` before using it as a verdict.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Literal

from pydantic_core import to_json
from pydantic_evals.evaluators import EvaluationReason, Evaluator, EvaluatorContext

_MISSING = object()
_ROOTS = ('output', 'expected_output', 'inputs', 'metadata')


def lookup(ctx: EvaluatorContext[Any, Any, Any], path: str) -> Any:
    """The value at a dotted `path` in the evaluator context, or None when any step is missing.

    The first step is one of `output`, `expected_output`, `inputs`, `metadata`. Later steps are
    mapping keys, attribute names, or list indices, so `'inputs.messages.0.content'` works on a
    YAML-loaded dict and on a Pydantic model alike.
    """
    root, *rest = path.split('.')
    if root not in _ROOTS:
        raise ValueError(f'a path starts with one of {", ".join(_ROOTS)}; got {path!r}')
    value: Any = getattr(ctx, root)
    for step in rest:
        if value is None:
            return None
        if isinstance(value, Mapping):
            value = value.get(step, _MISSING)  # pyright: ignore[reportUnknownMemberType]
        elif isinstance(value, Sequence) and not isinstance(value, str) and step.lstrip('-').isdigit():
            index = int(step)
            value = value[index] if -len(value) <= index < len(value) else _MISSING
        else:
            value = getattr(value, step, _MISSING)
        if value is _MISSING:
            return None
    return value


def as_text(value: Any) -> str:
    """A string as is; anything else as JSON, so a structured output is compared by its content."""
    if isinstance(value, str):
        return value
    return to_json(value, fallback=str).decode()


def levenshtein(a: str, b: str) -> int:
    """Edit distance: insertions, deletions and substitutions, each costing one."""
    if a == b:
        return 0
    # A shared prefix and suffix cost nothing; dropping them keeps long, similar texts cheap.
    start = 0
    while start < min(len(a), len(b)) and a[start] == b[start]:
        start += 1
    end = 0
    while end < min(len(a), len(b)) - start and a[-1 - end] == b[-1 - end]:
        end += 1
    a, b = a[start : len(a) - end], b[start : len(b) - end]
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def string_similarity(
    a: str,
    b: str,
    *,
    method: Literal['levenshtein', 'difflib'] = 'levenshtein',
    case_sensitive: bool = False,
    normalize_whitespace: bool = True,
) -> float:
    """1.0 for identical texts, 0.0 for texts with nothing in common.

    `levenshtein`: one minus the edit distance over the longer text's length. `difflib`:
    `SequenceMatcher.ratio()`, which counts matching blocks and is faster on long texts.
    """
    if normalize_whitespace:
        a, b = ' '.join(a.split()), ' '.join(b.split())
    if not case_sensitive:
        a, b = a.casefold(), b.casefold()
    if not a and not b:
        return 1.0
    if method == 'difflib':
        return SequenceMatcher(None, a, b, autojunk=False).ratio()
    return 1.0 - levenshtein(a, b) / max(len(a), len(b))


@dataclass(repr=False)
class StringSimilarity(Evaluator[object, object, object]):
    """Normalized edit distance between the output and a reference, with a pass/fail threshold.

    Returns `{name: score, f'{name}_pass': assertion}`. With the defaults, the output is compared
    with `expected_output`, case and runs of whitespace ignored.
    """

    threshold: float = 0.8
    """The assertion passes when the score is at least this."""
    reference_path: str = 'expected_output'
    """Where the reference is: a dotted path, see `lookup`."""
    output_path: str = 'output'
    """Which part of the output to compare, for a structured output: a dotted path."""
    method: Literal['levenshtein', 'difflib'] = 'levenshtein'
    case_sensitive: bool = False
    normalize_whitespace: bool = True
    evaluation_name: str | None = field(default=None)

    def get_default_evaluation_name(self) -> str:
        return self.evaluation_name if isinstance(self.evaluation_name, str) else 'string_similarity'

    def evaluate(self, ctx: EvaluatorContext[object, object, object]) -> dict[str, float | EvaluationReason]:
        reference, output = lookup(ctx, self.reference_path), lookup(ctx, self.output_path)
        if reference is None or output is None:
            return {}
        score = string_similarity(
            as_text(output),
            as_text(reference),
            method=self.method,
            case_sensitive=self.case_sensitive,
            normalize_whitespace=self.normalize_whitespace,
        )
        name = self.get_default_evaluation_name()
        passed = score >= self.threshold
        reason = (
            f'{self.method} similarity {score:.3f} {">=" if passed else "<"} {self.threshold} to {self.reference_path}'
        )
        return {name: score, f'{name}_pass': EvaluationReason(value=passed, reason=reason)}


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity; 0.0 when either vector is all zeros."""
    if len(a) != len(b):
        raise ValueError(f'embeddings of different sizes: {len(a)} and {len(b)}')
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


@dataclass(repr=False)
class EmbeddingSimilarity(Evaluator[object, object, object]):
    """Cosine similarity of the output's and the reference's embeddings, with a pass/fail threshold.

    `model` is a Pydantic AI embedding model, by name (`'openai:text-embedding-3-small'`,
    `'sentence-transformers:all-MiniLM-L6-v2'`) or as an `EmbeddingModel` instance. The provider's
    SDK is imported only when the first case is evaluated, as `LLMJudge` does for chat models, so
    this class costs nothing to import or to load from YAML. Returns `{name: score, f'{name}_pass':
    assertion}`.

    Cosine scores are not comparable across models: a threshold chosen for one model means
    nothing for another, which is why the model is part of a certificate's identity.
    """

    model: Any
    """An embedding model name with its provider prefix, or an `EmbeddingModel`."""
    threshold: float = 0.8
    reference_path: str = 'expected_output'
    output_path: str = 'output'
    evaluation_name: str | None = field(default=None)

    def get_default_evaluation_name(self) -> str:
        return self.evaluation_name if isinstance(self.evaluation_name, str) else 'embedding_similarity'

    def _embedder(self) -> Any:
        cached = getattr(self, '_cached_embedder', None)
        # Keyed on the model object itself: assigning a new `model` must not reuse the old provider.
        if cached is not None and cached[0] is self.model:
            return cached[1]
        try:
            from pydantic_ai.embeddings import Embedder
        except ImportError as error:  # pragma: no cover - pydantic-ai-slim without embeddings
            raise ImportError(
                'EmbeddingSimilarity needs `pydantic_ai.embeddings` (pydantic-ai-slim with embeddings support), '
                "plus the provider's extra, for example `pip install 'pydantic-ai-slim[openai]'`"
            ) from error
        embedder = Embedder(self.model)
        self._cached_embedder = (self.model, embedder)  # not a field: never serialized, compared or fingerprinted
        return embedder

    async def evaluate(self, ctx: EvaluatorContext[object, object, object]) -> dict[str, float | EvaluationReason]:
        reference, output = lookup(ctx, self.reference_path), lookup(ctx, self.output_path)
        if reference is None or output is None:
            return {}
        result = await self._embedder().embed_documents([as_text(output), as_text(reference)])
        score = cosine(result.embeddings[0], result.embeddings[1])
        name = self.get_default_evaluation_name()
        passed = score >= self.threshold
        model = getattr(self.model, 'model_name', self.model)
        reason = f'cosine {score:.3f} {">=" if passed else "<"} {self.threshold} to {self.reference_path} ({model})'
        return {name: score, f'{name}_pass': EvaluationReason(value=passed, reason=reason)}
