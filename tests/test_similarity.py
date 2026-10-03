"""StringSimilarity and EmbeddingSimilarity: the scores, the field paths, YAML specs, and a real Dataset.

No real embedding model is called here: `TestEmbeddingModel` (pydantic-ai's, every vector all
ones) and a deterministic bag-of-words model stand in for one.
"""

from __future__ import annotations

import hashlib
import itertools
import math
import random
import re
from collections.abc import Sequence
from pathlib import Path

import pytest
from pydantic_ai.embeddings import EmbeddingModel, EmbeddingResult, TestEmbeddingModel
from pydantic_ai.embeddings.result import EmbedInputType
from pydantic_ai.embeddings.settings import EmbeddingSettings
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import EvaluationReason, EvaluatorContext
from pydantic_evals.otel._errors import SpanTreeRecordingError

from pydantic_evals_admissibility._similarity import (
    EmbeddingSimilarity,
    StringSimilarity,
    cosine,
    levenshtein,
    lookup,
    string_similarity,
)


def ctx(output: object, expected: object = None, inputs: object = None, metadata: object = None):  # type: ignore[no-untyped-def]
    return EvaluatorContext(
        name='c', inputs=inputs, metadata=metadata, expected_output=expected, output=output, duration=0.0,
        _span_tree=SpanTreeRecordingError('none'), attributes={}, metrics={},
    )  # fmt: skip


def brute_levenshtein(a: str, b: str) -> int:
    if not a or not b:
        return len(a) + len(b)
    return min(
        brute_levenshtein(a[1:], b) + 1,
        brute_levenshtein(a, b[1:]) + 1,
        brute_levenshtein(a[1:], b[1:]) + (a[0] != b[0]),
    )


def test_levenshtein_known_values_and_against_brute_force() -> None:
    assert levenshtein('kitten', 'sitting') == 3
    assert levenshtein('', 'abc') == 3 and levenshtein('abc', 'abc') == 0
    assert levenshtein('flaw', 'lawn') == 2
    rng = random.Random(0)
    for _ in range(200):
        a = ''.join(rng.choice('abc') for _ in range(rng.randint(0, 6)))
        b = ''.join(rng.choice('abc') for _ in range(rng.randint(0, 6)))
        assert levenshtein(a, b) == brute_levenshtein(a, b) == levenshtein(b, a)


def test_string_similarity_normalization() -> None:
    assert string_similarity('Answer: Yes', 'answer:   yes\n') == 1.0
    assert string_similarity('Answer: Yes', 'answer: yes', case_sensitive=True) < 1.0
    assert string_similarity('', '') == 1.0
    assert string_similarity('abc', 'xyz') == 0.0
    assert string_similarity('kitten', 'sitting') == pytest.approx(1 - 3 / 7)
    assert 0 < string_similarity('kitten', 'sitting', method='difflib') < 1


def test_lookup_paths() -> None:
    c = ctx({'reply': 'hi'}, inputs={'messages': [{'content': 'q0'}, {'content': 'q1'}]}, metadata={'gold': 'g'})
    assert lookup(c, 'output.reply') == 'hi'
    assert lookup(c, 'inputs.messages.1.content') == 'q1'
    assert lookup(c, 'inputs.messages.-1.content') == 'q1'
    assert lookup(c, 'metadata.gold') == 'g'
    assert lookup(c, 'metadata.missing') is None and lookup(c, 'inputs.messages.5.content') is None
    with pytest.raises(ValueError, match='a path starts with'):
        lookup(c, 'gold')


def test_string_similarity_evaluator() -> None:
    result = StringSimilarity().evaluate(ctx('Answer: yes', 'answer: yes'))
    assert result['string_similarity'] == 1.0
    verdict = result['string_similarity_pass']
    assert isinstance(verdict, EvaluationReason) and verdict.value is True
    low = StringSimilarity(threshold=0.9).evaluate(ctx('Answer: no', 'Answer: yes'))
    assert isinstance(low['string_similarity_pass'], EvaluationReason)
    assert low['string_similarity_pass'].value is False
    assert StringSimilarity().evaluate(ctx('x')) == {}  # no reference: skipped, as EqualsExpected does
    by_field = StringSimilarity(reference_path='metadata.gold', output_path='output.reply', evaluation_name='sim')
    assert by_field.evaluate(ctx({'reply': 'Paris'}, metadata={'gold': 'paris'}))['sim'] == 1.0
    structured = StringSimilarity().evaluate(ctx({'a': 1}, {'a': 1}))
    assert structured['string_similarity'] == 1.0


class BagOfWords(EmbeddingModel):
    """Deterministic: each word hashed into one of 64 dimensions. Shared words, high cosine."""

    @property
    def model_name(self) -> str:
        return 'bag-of-words'

    @property
    def system(self) -> str:
        return 'test'

    async def embed(
        self, inputs: str | Sequence[str], *, input_type: EmbedInputType, settings: EmbeddingSettings | None = None
    ) -> EmbeddingResult:
        texts, _ = self.prepare_embed(inputs, settings)
        vectors = []
        for text in texts:
            vector = [0.0] * 64
            for word in re.findall(r'\w+', text.lower()):
                vector[int(hashlib.sha256(word.encode()).hexdigest(), 16) % 64] += 1.0
            vectors.append(vector)
        return EmbeddingResult(
            embeddings=vectors, inputs=texts, input_type=input_type, model_name='bag-of-words', provider_name='test'
        )


def test_cosine() -> None:
    assert cosine([1, 0], [1, 0]) == 1.0 and cosine([1, 0], [0, 1]) == 0.0
    assert cosine([0, 0], [1, 1]) == 0.0
    assert cosine([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)
    assert cosine([1, 1], [1, -1]) == pytest.approx(0.0, abs=1e-12)
    assert cosine([3, 4], [4, 3]) == pytest.approx(24 / 25)
    with pytest.raises(ValueError, match='different sizes'):
        cosine([1], [1, 2])
    assert all(math.isfinite(cosine(a, b)) for a, b in itertools.product([[1.0, 2.0], [0.0, 0.0]], repeat=2))


async def test_embedding_similarity_with_test_models() -> None:
    same = EmbeddingSimilarity(model=TestEmbeddingModel())  # every vector all ones: cosine 1.0
    result = await same.evaluate_async(ctx('anything', 'else'))
    assert result['embedding_similarity'] == pytest.approx(1.0)

    words = EmbeddingSimilarity(model=BagOfWords(), threshold=0.5)
    close = await words.evaluate_async(ctx('the refund is 42 dollars', 'refund is 42 dollars'))
    far = await words.evaluate_async(ctx('the warranty covers defects', 'refund is 42 dollars'))
    assert close['embedding_similarity'] > 0.8 > far['embedding_similarity']
    assert isinstance(close['embedding_similarity_pass'], EvaluationReason)
    assert close['embedding_similarity_pass'].value is True
    assert isinstance(far['embedding_similarity_pass'], EvaluationReason)
    assert far['embedding_similarity_pass'].value is False
    assert await words.evaluate_async(ctx('x')) == {}


def test_embedding_model_is_not_built_until_used() -> None:
    evaluator = EmbeddingSimilarity(model='openai:text-embedding-3-small', threshold=0.7)
    assert '_cached_embedder' not in evaluator.__dict__
    assert evaluator.build_serialization_arguments() == {'model': 'openai:text-embedding-3-small', 'threshold': 0.7}


def test_yaml_round_trip(tmp_path: Path) -> None:
    evaluators = [
        StringSimilarity(threshold=0.6, reference_path='metadata.gold'),
        EmbeddingSimilarity(model='openai:text-embedding-3-small', threshold=0.75),
    ]
    dataset = Dataset[str, str, dict[str, str]](
        name='sim', cases=[Case(name='a', inputs='q', metadata={'gold': 'g'})], evaluators=evaluators
    )
    path = tmp_path / 'sim.yaml'
    dataset.to_file(path, custom_evaluator_types=[StringSimilarity, EmbeddingSimilarity])
    assert (tmp_path / 'sim_schema.json').exists()
    loaded = Dataset[str, str, dict[str, str]].from_file(
        path, custom_evaluator_types=[StringSimilarity, EmbeddingSimilarity]
    )
    assert loaded.evaluators == evaluators


async def test_inside_a_real_dataset_evaluate() -> None:
    answers = {'paris': 'Paris', 'rome': 'Rom', 'oslo': 'Stockholm'}

    def agent(city: str) -> str:
        return answers[city]

    dataset = Dataset[str, str, None](
        name='cities',
        cases=[Case(name=c, inputs=c, expected_output=c.title()) for c in answers],
        evaluators=[StringSimilarity(threshold=0.7), EmbeddingSimilarity(model=BagOfWords(), threshold=0.9)],
    )
    report = await dataset.evaluate(agent, progress=False)
    passed = {c.name: c.assertions['string_similarity_pass'].value for c in report.cases}
    assert passed == {'paris': True, 'rome': True, 'oslo': False}
    assert report.cases[0].scores['string_similarity'].value == 1.0
    embedded = {c.name: c.assertions['embedding_similarity_pass'].value for c in report.cases}
    assert embedded == {'paris': True, 'rome': False, 'oslo': False}


async def test_changing_the_model_rebuilds_the_embedder() -> None:
    """Regression (review E1): a new `model` must not reuse the embedder of the old one."""
    evaluator = EmbeddingSimilarity(model=TestEmbeddingModel(), threshold=0.5)
    first = await evaluator.evaluate_async(ctx('cat', 'dog'))
    assert first['embedding_similarity'] == pytest.approx(1.0)
    evaluator.model = BagOfWords()
    again = await evaluator.evaluate_async(ctx('cat', 'dog'))
    fresh = await EmbeddingSimilarity(model=BagOfWords(), threshold=0.5).evaluate_async(ctx('cat', 'dog'))
    assert again == fresh
    assert again['embedding_similarity'] < 0.5
