# What judge certification found in pydantic-ai's own evals material

Measured with this repository's `certify_dataset` and `certify_judge` against pydantic-ai `main`
(66951321, which has the same eval example and docs as the checkout the runs used, 6bc07cf). The
judge model is Codex `gpt-5.6-luna`, since the dataset leaves the model unset. Every number
below comes from a saved result in `results/`, and every run can be repeated from the bench
script named next to it.

Each finding is labelled by intent, after reading the surrounding code, docs and history:
**unintended** (the material asks for one thing and does another), **simplification** (an
illustrative snippet that would mislead only if copied as is), or **not a fault**.

## 1. The time-range example's expected outputs fail its own judge (unintended)

`examples/pydantic_ai_examples/evals/datasets/time_range_v2.yaml` (and `v1`, which has the same
10 expected outputs) carries a dataset-wide judge:

> Ensure the explanation or error_message fields are truly appropriate for user display, in a
> second-person or friendly style.

Certified on the dataset as shipped (`bench/certify_pydantic_dataset.py`,
`results/certify_pydantic_dataset.json`; 3 repeats), the judge failed 4 of the 10 expected
outputs on every repeat. Its reasons, verbatim from the first repeat:

| Case | Expected output | Judge's reason |
|---|---|---|
| Impossible range | "Conflicting time instructions: 2025 and 2020 cannot both apply." | "uses a terse, technical style and does not address the user in a second-person or friendly manner" |
| Confusing relative references | "Conflicting instructions: 'yesterday' versus 'last year' could not be reconciled." | "impersonal and does not address the user in a second-person or friendly style" |
| Ambiguous mention | "We interpret the mention of early May as extraneous, focusing on 'last week or so' from the current time." | "uses an internal, impersonal phrasing ('We interpret') rather than addressing the user directly" |
| No mention | "No timeframe could be inferred from your request." | "impersonal and technical" (see finding 2) |

**Why it is unintended.** `example_01_generate_dataset.py` asked `generate_dataset` for the
judge itself: "a dataset-wide LLMJudge evaluator to ensure that the 'explanation' or
'error_message' fields are appropriate to be displayed to the user (e.g., written in second
person, etc.)". The same generation wrote the expected outputs, and nothing checks generated
expected outputs against generated judges.

**Fix, measured.** The first three rewritten in second person, nothing else changed
(`bench/data/time_range_v2.second_person.yaml`; `bench/certify_pydantic_dataset.py
--second-person`, `results/certify_pydantic_dataset.second_person.json`):

| Case | As shipped | Rewritten |
|---|---|---|
| Impossible range | failed 3 of 3 | passed 3 of 3 |
| Confusing relative references | failed 3 of 3 | passed 3 of 3 |
| Ambiguous mention | failed 3 of 3 | passed 3 of 3 |

The rewrites, all under the dataset's `UserMessageIsConcise` limit of 50 words:

- "You asked for logs from both 2025 and 2020, which can't both apply. Which year did you mean?"
- "You asked for logs from yesterday and from last year, and I couldn't tell which you meant.
  Which one would you like?"
- "You mentioned both 'last week or so' and early May, so I've shown you the last week and set
  early May aside." (It still addresses ignoring early May, as that case's own judge asks.)

A wider fix, for pydantic-evals rather than this dataset: run the generated judges over the
generated expected outputs before `generate_dataset` returns, or recommend `certify_dataset` on
anything it produces.

## 2. The rubric is ambiguous, and the judge decides differently on identical text (unintended)

"Second-person or friendly" does not say whether a friendly first-person plural ("We interpret
...") or a passive sentence that mentions "your request" qualifies. On two expected outputs the
judge's verdict changes between runs on the same text:

- **No mention** ("No timeframe could be inferred from your request."): failed 3 of 3 in the
  first run, passed 2 of 3 in the second. Its reasons disagree with each other: "impersonal and
  technical; it does not address the user in second person" against "directly addresses the user
  with 'your request'". In the first run it also passed the same text with only its spacing
  changed.
- **Ambiguous elliptical mention** ("We interpret 'around the start of last quarter' as ..."):
  passed 2 of 3, then 1 of 3. One reason calls "We interpret" "a friendly first-person plural
  style"; another says it does not address the user directly.

**Fix.** Say which is meant, for example "written to the user in the second person ('you',
'your request'); first-person plural ('we interpret') does not count", and the expected outputs
can then be written to it.

## 3. Ten cases cannot certify the judge either way (not a fault, but worth knowing)

With 10 cases every check stays UNVALIDATED: the certificate cannot show the judge is reliable,
nor that it is not. At the rate observed after the fix (8 of 10 first judgments passed),
`diagnose` says 82 cases would be needed to show acceptance above 0.7. The dataset is an
example, so this is expected; it is the reason the findings above are stated per case, not as
a verdict on the judge.

## 4. The LLM-judge guide's "Better" rubric cannot see the question (simplification)

`docs/evals/evaluators/llm-judge.md`, "Best Practices", "1. Be Specific in Rubrics", shows Bad
(`'Good answer'`), Better
(`LLMJudge(rubric='Response accurately answers the question without hallucinating facts')`) and
Best (a fuller rubric with `include_input=True`). The "Better" step asks about the question,
which the default `include_input=False` does not show the judge. The guide does explain
`include_input` a few sections earlier and adds it in "Best", so this reads as a deliberate
simplification. It matters only if the middle step is copied.

The guide ships no cases for it, so it was measured on this repository's support task instead
(`bench/docs_rubrics.py`, `results/docs_rubrics.json`; 40 correct replies, and the same 40
replies each moved to another customer's question, one judgment each):

| The "Better" rubric | Correct replies passed | Replies to another customer's question passed |
|---|---|---|
| as the guide writes it (`include_input=False`) | 10 of 40 | 13 of 40 |
| with `include_input=True` | 37 of 40 | 0 of 40 |

As written, the judge cannot tell a right answer from a wrong one: it passed the wrong ones a
little more often than the right ones. With the question shown it separates them completely.
The certificate rows: acceptance FAIL (10/40, interval 0.13 to 0.41) as written, PASS (37/40)
with the question. The run used only the mismatched-answer control, so neither certificate
reaches ADMISSIBLE; the comparison is the point.

**Fix.** Add `include_input=True` to the "Better" example.

## Looked at and not reported

- The docs' other `LLMJudge` examples ship 0 to 3 cases each, below the 10 a certificate needs,
  so none could be certified on pydantic-ai's own cases. Docs snippets illustrate an API; this
  is not a fault.
- The per-case judge in the time-range dataset ("confirm the explanation addresses ignoring
  early May") grades a single case, too few to certify.
- `example_01_generate_dataset.py` leaves `include_input` at its default on purpose ("Leave the
  model and include_input arguments to LLMJudge as their default values"). For the style judge,
  which needs only the output, that is right.
