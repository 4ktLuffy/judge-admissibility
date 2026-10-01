# pydantic-evals-admissibility

Certify a [pydantic-evals](https://ai.pydantic.dev/evals/) judge before its scores count.

> **A judge's verdict is evidence only once the judge has been shown to fail what it should fail.**

An `LLMJudge` that passes everything produces a perfect score on every dataset. Nothing in an
eval report distinguishes that judge from a good one: both say "pass" on the cases that should
pass, and nobody looks at the cases that should not have passed, because the dataset does not
contain any. Pydantic's own writing says judges "often flip [their] verdict" when the order of
what they see changes, and that they have to be calibrated. This package measures part of that
(repeat consistency and agreement with people; not order swaps yet), with controls built from the
cases you already have.

```python
from pydantic_evals.evaluators import LLMJudge
from pydantic_evals_admissibility import JudgeCase, certify_judge

cases = [JudgeCase(name=c.name, inputs=c.inputs, output=good_answer, expected_output=c.expected_output) for c in ...]
certificate = await certify_judge(LLMJudge(rubric='The output correctly answers the question.', include_input=True), cases)
print(certificate.table())
if not certificate.admissible:
    ...  # do not report this judge's scores
```

## What it checks

| Check | The judge must... | Built from |
|---|---|---|
| `acceptance` | pass the known-good answers | your cases |
| `rejection` | fail answers that cannot be right | another case's answer (`mismatched_output`), an empty answer (`empty_output`) |
| `invariance` | keep its verdict when nothing that matters changes | the same answer, whitespace reformatted |
| `stability` | give the same verdict when asked again | each answer judged `repeats` times |
| `human_agreement` | agree with people beyond chance (Cohen's kappa) | `HumanLabel`s, in any form (they could come from a Logfire annotation export) |

Each rate carries a 95% Wilson interval and is decided on it, not on the point estimate:

- **PASS** when the interval's lower bound clears the threshold;
- **FAIL** when its upper bound is below it, which is evidence the judge is below the bar;
- **UNVALIDATED** otherwise. Twelve out of twelve is a lower bound of 0.76: not a failure, and
  not yet a pass. The certificate says "needs more cases" instead of rounding up.

The certificate is **ADMISSIBLE** only if every check passes, **INADMISSIBLE** if any fails,
and **UNVALIDATED** otherwise. `acceptance` and `rejection` are both required because each
alone is satisfied by a constant judge: one that always passes clears acceptance, invariance
and stability, and one that always fails clears rejection.

## The package's own controls

`tests/test_certify.py` drives the real `LLMJudge` with scripted models and checks that the
certificate admits a sound judge and refuses each kind of broken one:

| Judge | Certificate |
|---|---|
| oracle: passes exactly the right answers | ADMISSIBLE |
| always passes | INADMISSIBLE: rejection |
| always fails | INADMISSIBLE: acceptance (it *passes* rejection) |
| right, but whitespace-sensitive | INADMISSIBLE: invariance |
| coin flip | INADMISSIBLE: stability |
| right, but accepts an empty answer | INADMISSIBLE, and the rejection detail names `empty_output 0/30` |
| raises on every call | INADMISSIBLE: an error is not a verdict |
| oracle on 3 or 12 cases | UNVALIDATED |

## A real judge: `LLMJudge`'s default configuration is inadmissible

`bench/certify_codex_judge.py` certifies a real `LLMJudge`, with its own system prompt, whose model
is Codex (`gpt-5.6-luna`, no reasoning, through `bench/codex_judge.py`, no API key). It uses 20
known-good answers judged twice each and the rubric *"The output correctly answers the question."*

| `LLMJudge` configuration | Certificate | acceptance | rejection | invariance | stability |
|---|---|---|---|---|---|
| default, `include_input=False` | **INADMISSIBLE** | **0/40** | 40/40 | 20/20 | 20/20 |
| `include_input=True` | ADMISSIBLE | 40/40 | 40/40 | 20/20 | 20/20 |

With `include_input=False` the judge never sees the question its rubric refers to, so it cannot
confirm any answer and fails all of them, right or wrong. In an eval report that judge's 0% pass
rate would read as a broken agent, when the judge is what is broken. `acceptance` is the check
that catches it; `rejection`, `invariance` and `stability` all pass, which is why no single check
is a certificate. With the question in view, the same model is admissible on every check.

**Correction.** An earlier version of this section reported the default judge passing 19 of 20
answers to a different question. That run had a bug: the adapter passed `LLMJudge`'s system
prompt through a Codex setting that Codex ignores, so the model judged without it. The same
bug made this README claim the adapter cut each call to about 620 tokens; the token counts Codex
reports do not show that, and a real judge call here costs about 1,300 to 2,000 tokens. Both are
fixed: the adapter now uses `model_instructions_file`, checked by instructing the model to reply
with a fixed word and confirming it does.

## Limits

- The controls are generic. `mismatched_output` assumes another case's answer is wrong for
  this case, which fails on datasets where many cases share an answer; the donor is drawn only
  from cases with a different answer, and a case with none is skipped.
- `human_agreement` uses kappa's point estimate, without an interval.
- A certificate covers one judge configuration (model, rubric, flags, settings) on one set of
  cases. Change any of them and certify again.
