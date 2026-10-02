# pydantic-evals-admissibility

**Find out whether your LLM judge can be trusted before you trust its scores.**

An `LLMJudge` that passes everything scores 100% on a dataset of good answers, and the report
looks exactly like one from a judge that works. This package tests the judge itself, with
controls built from the cases you already have: answers it must fail, changes it must ignore,
orderings it must not care about. A judge's verdicts count only once it has been shown to fail
what it should fail.

What it found on real judges (Codex `gpt-5.6-luna`; every number is reproduced in `bench/`):

- **A strong judge, a realistic task, `LLMJudge`'s default settings:** grading a support agent's
  replies against a store policy, a reasoning judge with the default `include_input=False`
  passed **10 of 10 answers that belonged to other customers' questions**. Certification caught it
  after 70 of 280 calls. With `include_input=True` the same judge agreed with ground truth on
  79 of 80 replies.
- **Ask for the reason before the verdict.** A bug in this repository's Codex adapter sorted
  the judge's output schema, so judges gave the verdict first. A judge without reasoning then
  certified ADMISSIBLE overall while passing **0 of 10 correct "no" answers**, its own reasons
  saying they were right; slicing the certificate by kind of case caught it. A comparison judge
  picked **whichever answer came first 68% of the time** and was right 53%. With the reason first,
  as `LLMJudge` defines it: 10 of 10, and no position preference (48%) at 85%. Judges with
  reasoning were unaffected. Every result is re-run; `bench/field_order.py` shows both.
- **Self-improving prompts:** in one round of optimization, 4 of 5 proposed prompts were worse
  than the original; an uncertified judge overstated every prompt by 11 to 23 points. Gating on a
  certified judge, then confirming on fresh questions, promoted the one real improvement
  (p = 0.015), the same call ground truth makes.
- **Pydantic's own example dataset, in one call:** `certify_dataset` on pydantic-ai's
  `time_range_v2.yaml`, loaded with `Dataset.from_file` and no controls written by hand. The
  judge failed **the same 3 of its 10 "good" answers every time**, and its reasons hold up: they
  are impersonal, while the rubric asks for "second-person or friendly". On a fourth it
  contradicted itself: it failed *"...inferred from your request"* three times as not second
  person, then passed the same text with doubled spaces. `diagnose` tells the two apart.

```python
results = await certify_dataset(dataset)  # every LLMJudge in a pydantic-evals Dataset, controls chosen per rubric
certificate = await certify_judge(judge, cases, batch_size=10)  # is it evidence? stops early if clearly not
certificate.raise_unless_admissible(judge)  # fail CI, with what's wrong and what to try
result = compare_reports(baseline, candidate, assertion='LLMJudge', certificate=certificate)  # is the change real?
```

**The core** is three functions: `certify_judge` (pointwise judges), `certify_pairwise`
(comparison judges, including position bias) and `diagnose` (what to change when either fails).
`certify_dataset` runs `certify_judge` on every judge in a `Dataset`.
**Built on them:** `decide` / `compare_reports` (is a change real?), `detectable_gain` (can your
dataset even see it?), and, for Logfire users, `promote`, the canary rollout and `JudgeCanary`
(keep checking a judge in production). The reasoning behind every choice is in
[DESIGN.md](DESIGN.md).

## Install

```bash
pip install "pydantic-evals-admissibility[logfire] @ git+https://github.com/4ktLuffy/judge-admissibility"
```

The `logfire` extra is only needed for `promote` and the canary functions. Tested against the
released pydantic-evals 2.52.0 and logfire 5.1.1. Until pydantic-evals declares `sniffio` (its
online evaluation imports it), also `pip install sniffio` to use `JudgeCanary` online.

## What it checks

| Check | The judge must... | Built from |
|---|---|---|
| `acceptance` | pass the known-good answers | your cases |
| `rejection` | fail answers that cannot be right | another case's answer (`mismatched_output`), an empty answer (`empty_output`) |
| `invariance` | keep its verdict when nothing that matters changes | the same answer, whitespace reformatted |
| `stability` | give the same verdict when asked again | each answer judged `repeats` times |
| `human_agreement` | agree with people beyond chance (Cohen's kappa) | `HumanLabel`s, in any form (they could come from a Logfire annotation export) |
| `slices` (opt-in) | not be below the bar on any one kind of case | `slice_by`: a name for each case's kind |

Each rate carries a 95% Wilson interval and is decided on it, not on the point estimate:

- **PASS** when the interval's lower bound clears the threshold;
- **FAIL** when its upper bound is below it, which is evidence the judge is below the bar;
- **UNVALIDATED** otherwise. Twelve out of twelve is a lower bound of 0.76: not a failure, and
  not yet a pass. The certificate says "needs more cases" instead of rounding up.

The certificate is **ADMISSIBLE** only if every check passes, **INADMISSIBLE** if any fails,
and **UNVALIDATED** otherwise. `acceptance` and `rejection` are both required because each
alone is satisfied by a constant judge: one that always passes clears acceptance, invariance
and stability, and one that always fails clears rejection.

## Cheaper certificates, and what to do when one fails

**Stop early when a judge has clearly failed.** `certify_judge(..., batch_size=10)` judges ten
cases at a time and stops as soon as a check fails beyond doubt; a judge that is not failing runs
to the end and is judged exactly as if all at once, so stopping early costs a sound judge nothing.
Early looks use a wider interval (the confidence level split across the looks), so peeking does not
fail a sound judge by chance. `certificate.calls` says how many judgments were actually made.

Measured with scripted judges whose verdicts are fixed per answer, so both modes see the same
verdicts (60 cases, 2 repeats, batches of 15, 100 seeds each, `bench/sequential_check.py`):

| Judge | Same verdict as judging everything | Calls used |
|---|---|---|
| broken (passes 60% of wrong answers) | 100/100 | 26% |
| borderline bad (25%; the bar is 20%) | 100/100 | 89% |
| borderline good (10%) | 100/100 | 100% |
| sound | 100/100 | 100% |

On a real judge, the Codex `LLMJudge` that failed certification on the arithmetic task: the
sequential certificate reached the same verdict, INADMISSIBLE, after 140 of 280 calls
(`results/judge_vs_truth.sequential.json`).

Stopping early for success as well was tried and dropped: it certified sound judges less often
(74/100 against 87/100) to save 18% of calls.

**Say why, and what to try.** `diagnose(certificate, judge)` turns the pattern of failed checks
into plain advice. On the two real failures in this README it says:

- default `LLMJudge`: *"It fails answers that are right (0/40 passed) as readily as wrong ones ...
  The rubric refers to the question, but `include_input=False`: the judge never sees it. Set
  `include_input=True`."*
- the plain judge on the arithmetic task: *"Its verdict changes when only whitespace changes, but
  it also disagrees with itself on the very same answer, so this is most likely noise rather than
  formatting sensitivity. Lower its temperature ..."*, and that more cases will not make its
  acceptance or stability pass, since their rates are below the bar.

The advice is a hypothesis to re-certify, not a fix. `certificate.raise_unless_admissible(judge)`
raises `InadmissibleJudge` with the table and the advice, for a test or a CI step that should fail
when a judge stops being evidence.

## A strong judge on a realistic task

`bench/support_eval.py`: a support agent (Codex, no reasoning) answers 40 customer questions
from short store policies (return windows, restocking fees, shipping thresholds, warranties, with
distracting details); answers are computed by code and balanced between yes and no. The agent got
72 of 80 replies right. Four judges were certified on those 80 replies, ground truth as labels,
sequentially in batches of 10:

| Judge (Codex) | Certificate | Calls | Agreement with ground truth |
|---|---|---|---|
| `gpt-5.6-luna`, no reasoning, `include_input=True` | ADMISSIBLE | 280 of 280 | 80/80, kappa 1.00 |
| `gpt-reserve`, no reasoning, `include_input=True` | ADMISSIBLE | 280 of 280 | 78/80, kappa 0.86 |
| `gpt-5.6-luna`, reasoning high, `include_input=True` | ADMISSIBLE | 280 of 280 | 79/80, kappa 0.93 |
| `gpt-5.6-luna`, reasoning high, default `include_input=False` | **INADMISSIBLE** | **70 of 280** | 15/20, kappa 0.34 |

(The reasoning judge's acceptance is 76/80 because 4 calls timed out; a judgment that errors
does not count as a pass.)

The default-configured judge passed every one of the 10 answers borrowed from other customers'
questions (each with a different correct answer, so wrong by construction). `diagnose` said why:
*"It accepts another question's answer, and with `include_input=False` it cannot tell which
question was asked. Set `include_input=True`."*

On the arithmetic task the same default made the judge fail everything instead (0 of 40 right
answers passed). Both are inadmissible; which way a blind judge fails depends on the task, which
is the argument for measuring rather than assuming.

The comparison judges, asked to pick between the right answer and a plausible mistake (the refund
without the restocking fee, the wrong side of the shipping threshold), both ways round:

| Comparison judge | Accuracy | Same pick both ways | Chose the answer shown first |
|---|---|---|---|
| no reasoning | 0.96 (77/80) | 39/40 | 0.51 |
| reasoning high | 0.96 (77/80) | 39/40 | 0.51 |

(The same totals, from different mistakes: the two judges were wrong on different pairs.)

### Slices: a passing rate can hide a kind of case

`certify_judge(..., slice_by=...)` names the kind of each case and adds a `slices` check, which
fails when the judge is shown to be below the bar on one kind of case. `recertify` applies it to a
certificate already paid for. Sliced by kind and correct answer (`return: no`, `refund`, ...),
every judge above passes it. It found something only when the judge was broken: through the adapter
bug described under comparison judges, both judges without reasoning were asked for the verdict
before the reason, and both certified ADMISSIBLE overall (70/80 accepted) while passing **0 of 10
correct "no" answers** to return-window questions. Their reasons said the answers were right,
after the verdict had been given: *"...98 days later, so the answer 'no' is correct."* Sliced,
both certificates were INADMISSIBLE.

Each slice is decided with an exact (Clopper-Pearson) interval, split across the slices. A sound
judge exactly at the bar fails the check at most 1.9% of the time (simulated: 2 to 12 slices
of 10 to 40 cases each); a slice it always gets wrong is caught every time with 10 cases.

## Pydantic's own example judge

`bench/pydantic_example_judge.py` certifies the judge that pydantic-ai's example evals
(`examples/pydantic_ai_examples/evals/datasets/time_range_v2.yaml`) apply to every case:

> Ensure the explanation or error_message fields are truly appropriate for user display, in a
> second-person or friendly style.

It is a style rubric, so the default controls are the wrong ones: another case's friendly
explanation is still friendly. The controls are written for it with `Rewrite`: an empty
explanation and a raw debug string must fail; another case's explanation
(`MismatchedOutput(kind='must_hold')`) and extra spaces must not change the verdict. The
dataset's 10 expected outputs are the answers marked good.

| Judge (Codex `gpt-5.6-luna`) | Certificate | Good answers passed | Must-fail controls rejected |
|---|---|---|---|
| no reasoning | INADMISSIBLE | 15/30 | 20/20 |
| reasoning high | UNVALIDATED | 18/30 | 20/20 |

Both judges failed the same three expected outputs every time they were asked (the one without
reasoning failed two more), with reasons that hold up against the rubric's words:

- *"Conflicting time instructions: 2025 and 2020 cannot both apply."*: "clear but impersonal and
  does not address the user in a second-person or friendly style"
- *"Conflicting instructions: 'yesterday' versus 'last year' could not be reconciled."*: the same
- *"We interpret the mention of early May as extraneous ..."*: first person, not second; this one
  is arguable

So the low acceptance is at least partly the dataset: some of its answers marked good do not meet
its own rubric as written. `diagnose` now says so instead of blaming the judge when a judge fails
the same answers on every repeat and passes most others every time. With 10 cases, the reasoning
judge could not be certified either way. Its numbers came out identical in two runs, before and
after the schema fix described under comparison judges. Two things from this run changed the package: rubric-specific controls (`Rewrite`,
`MismatchedOutput(kind='must_hold')`), and the doctor checking the data before the judge.

A third-person rewrite was left out as a control on purpose: the rubric says "second-person **or**
friendly", so a friendly third-person explanation meets it.

## One call for a whole dataset: `certify_dataset`

```python
dataset = Dataset[Inputs, Output, None].from_file(path, custom_evaluator_types=...)
results = await certify_dataset(dataset)  # optionally model=..., to certify on a cheaper model
print(dataset_report(results))
```

It finds every `LLMJudge` in the dataset: those in `evaluators`, and those attached to single
cases, grouped by rubric. Each case's `expected_output` is a known-good answer. Two decisions it
makes for you, and reports:

- **What the rubric grades.** A rubric about style or tone (read from its words; pass `controls=`
  to override) gets controls for style: another case's answer must *not* change the verdict, since
  it is just as friendly. A rubric about correctness gets the default controls.
- **Where the controls apply.** In structured outputs, the empty and whitespace controls change
  only the prose fields (strings with a space in them), never timestamps or ids, so a
  "whitespace only" change cannot change the answer.

On pydantic-ai's example dataset (`bench/certify_pydantic_dataset.py`, their `main` at
`6bc07cf`, Codex `gpt-5.6-luna` reasoning high, 3 repeats), against the hand-written controls of the
previous section:

| Controls | Calls | Good answers passed | Must-fail rejected | Certificate |
|---|---|---|---|---|
| written by hand for this rubric | 70 | 18/30 | 20/20 | UNVALIDATED |
| chosen by `certify_dataset` | 60 | 17/30 | 10/10 | UNVALIDATED |

The same verdict, without writing a control, and the same numbers again in a second run after
the schema fix. The judge failed four answers on all three tries.
Three are the three the hand-written run found ("Ambiguous mention", "Impossible range",
"Confusing relative references"); the doctor says to read those against the rubric. The fourth,
"No mention" (*"No timeframe could be inferred from your request."*), is second person, and the
judge passed it once its spaces were doubled, now saying it "refers directly to 'your request'"
(and again in the second run).
The doctor reports that one as the judge contradicting itself, not as bad data. The one judge attached to a single case was reported and
not certified: one case cannot certify anything.

The automatic controls are weaker than hand-written ones. For a style rubric the only answer that
must fail under any rubric is an empty one, so with 10 cases `rejection` cannot pass (10/10 has a
lower bound of 0.72; the doctor says 16 cases would do it). A control that breaks your rubric
(`Rewrite`), as in the previous section, adds the evidence.

## Comparison judges: does the order decide?

Pydantic's own warning: *"Swap the order of two answers and an LLM judge will often flip its
verdict on the same pair, favoring whichever it saw first."* `certify_pairwise(compare, pairs)`
measures exactly that. Every pair has a known better answer and is shown both ways round; the
judge must pick the better one (`accuracy`) and the same one either way (`order_consistency`),
and when it flips, the certificate says which position it followed. `PairwiseJudge(rubric,
model)` is a comparison judge on any Pydantic AI model with a typed verdict.

Measured on Codex `gpt-5.6-luna`, no reasoning (`bench/pairwise_codex.py`): 54 pairs of real
agent replies to the same arithmetic question, one right and one wrong by ground truth, each asked
both ways (108 calls). The output schema is `PairVerdict`: `reason`, then `choice`.

| Pairs as the agent wrote them | Accuracy | Same pick both ways | Chose the answer shown first |
|---|---|---|---|
| full replies, with the working | 0.85 (92/108) | 48/54 | 0.50, interval [0.41, 0.59] |
| final answer line only | 0.85 (92/108) | 46/54 | 0.48, interval [0.39, 0.57] |

No position preference, and the same accuracy whether or not the working is shown.
`both_orders(compare)`, which asks both ways and answers only when the two agree, raised accuracy
to 0.90 (43/48, abstaining on 6 pairs) with the working and 0.91 (42/46, abstaining on 8) without.

**An earlier version of this section was wrong.** It reported that the judge picked whichever
answer came first in 68% of presentations and was right 53% of the time. Those runs went through
a bug in this repository's Codex adapter, which sorted the schema's keys and so asked for `choice`
before `reason`: the judge chose before it thought. On the same 54 pairs, same model:

| Final answer line only | Accuracy | Same pick both ways | Chose the answer shown first |
|---|---|---|---|
| choice first (the bug) | 0.53 (57/108) | 29/54 | 0.68, interval [0.58, 0.76] |
| reason first | 0.85 (92/108) | 46/54 | 0.48, interval [0.39, 0.57] |

The position bias was real, and the field order caused it: a judge without reasoning that is
asked for its verdict first commits to it before writing a word of reasoning, and here it fell
back on position. Keep the verdict after the reason in any judge's output type, and in anything
that rewrites its schema.
`bench/field_order.py` prints every certificate the bug touched next to its re-run.

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

## From a certificate to a decision: gating changes

A certificate says whether a judge's verdicts are evidence. The rest of the package uses that
to decide whether a change, such as a new prompt, should ship.

```python
from pydantic_evals_admissibility import GateRules, compare_reports, promote

baseline = await dataset.evaluate(agent_with_current_prompt, repeat=2)
candidate = await dataset.evaluate(agent_with_new_prompt, repeat=2)

result = compare_reports(baseline, candidate, assertion='LLMJudge', certificate=certificate)
print(result.summary())  # PROMOTE / REJECT / INCONCLUSIVE / REFUSED
promote(result, 'agent_prompt', new_prompt)  # moves the Logfire label only on PROMOTE
```

- **`decide` / `compare_reports`** compare the two versions case by case (a paired sign-flip
  test, exact when nothing changed). A judge without an ADMISSIBLE certificate gets **REFUSED**:
  its scores are not evidence, so there is nothing to decide. `compare_reports` reads two ordinary
  pydantic-evals reports, groups repeats by source case and counts a crashed run as a fail.
- **`GateRules().for_candidates(k)`** splits the significance level when an optimizer picks the
  best of `k` proposals; otherwise the best of several noisy candidates wins by luck.
- **`detectable_gain(outcomes)`** answers, before anything runs, how big an improvement the
  dataset can see at all.
- **`promote`** applies a PROMOTE to a Logfire managed variable (a new version, the
  `production` label moved) and records the decision and its evidence on a span either way.
- **`start_canary` / `decide_unpaired` / `finish_canary`** do the same for live traffic: serve
  the candidate to a share of requests, compare the arms, then promote, roll back, or keep
  waiting when the traffic cannot tell yet.
- **`JudgeCanary`** keeps checking a judge after it is certified, on the traffic it grades. On a
  sampled share of calls it also asks the judge about an empty answer, which it must fail.
  `borrow=True` adds another request's answer as a control; it is off by default because on
  this package's task a borrowed answer was in fact right for 20-27% of same-kind questions, which
  made a sound judge look unhealthy. Use it only where answers are specific to their question.
  `every=n` checks exactly every n-th call instead of sampling, so a monitor sees a known number
  of checks; the monitor says HEALTHY only once the evidence supports it (16 straight rejections
  at the default 0.8 bar). Works with `Dataset.evaluate` and with online evaluation; online, the
  result lands in Logfire as `judge_canary_rejected`.

How well the gate itself behaves, measured in this package's tests (40 cases, 2 runs each,
200 trials for A/A and 50 for the others):

| True situation | PROMOTE | INCONCLUSIVE | REJECT |
|---|---|---|---|
| identical versions | 5 (2.5%) | 193 | 2 (1%) |
| better by 0.1 | 15 | 35 | 0 |
| better by 0.3 | 49 | 1 | 0 |
| worse by 0.1 | 0 | 45 | 5 |
| worse by 0.3 | 0 | 1 | 49 |

It never called a better version worse or promoted a worse one. Small gains are mostly
INCONCLUSIVE, which is the honest answer at this size: on the real Codex baseline below,
`detectable_gain` says 40 questions with 2 runs each reliably see only a 25-point gain.

## Experiment: one round of prompt optimization, decided three ways

`bench/optimize.py` runs one round of a self-improving loop on a task with answers computed by
Python (`bench/task.py`: arithmetic, letter counts, weekday arithmetic, string transformations).
The agent, the judges and the optimizer are all Codex `gpt-5.6-luna` with no reasoning. Codex
proposes five prompts from the baseline and the failures the judge flagged; each runs twice on
40 training questions. Ground truth is never used to decide anything: it is the referee, and the
40 held-out questions are scored only after the decisions are made.

Two judges, certified first on the agent's real replies with ground truth as the labels:

| Judge | Certificate | Agreement with ground truth on 80 real replies |
|---|---|---|
| plain `LLMJudge(include_input=True)` | INADMISSIBLE | passed 7 of 39 wrong answers, failed 6 of 41 right ones |
| reference `LLMJudge(include_input=True, include_expected_output=True)` | ADMISSIBLE | 80 of 80 |

On the 40 training questions:

| Prompt | Plain judge | Certified judge | Truth | Gate on plain judge | Gate on certified judge (level split 5 ways) |
|---|---|---|---|---|---|
| baseline | 0.625 | 0.550 | 0.513 | | |
| candidate 0 | 0.500 | 0.275 | 0.275 | REFUSED | REJECT |
| candidate 1 | 0.575 | 0.412 | 0.425 | REFUSED | INCONCLUSIVE |
| candidate 2 | 0.800 | 0.688 | 0.688 | REFUSED | INCONCLUSIVE (p = 0.018, needs < 0.005) |
| candidate 3 | 0.512 | 0.350 | 0.350 | REFUSED | REJECT |
| candidate 4 | 0.400 | 0.250 | 0.250 | REFUSED | REJECT |

What it shows, including where it did not go the way I expected:

- **The uncertified judge's numbers were wrong, its ranking was not.** It overstated every
  prompt by 11 to 23 points, evenly, so "keep the highest-scoring prompt" still picked
  candidate 2, which is genuinely better: 0.588 vs 0.425 on the held-out questions (p = 0.015).
  The danger in an uncertified judge here is anything that trusts its number: a release bar,
  a regression threshold, a claim that the agent is 80% accurate when it is 69%.
- **Most proposals were worse than the prompt they replaced.** Four of five, by 9 to 26 points of
  truth. All four worse prompts got the model to
  reply tersely, median 16 to 18 characters against the baseline's 46, and two of them asked
  for "internal" or "silent" reasoning outright. At no reasoning effort the written working is
  the model's only reasoning, and the terse replies came with lower accuracy (this run shows the
  association, not the cause). The one better prompt said "prioritize correctness over brevity"
  and got longer replies (median 67 characters).
- **The gate rejected all three clearly worse prompts and promoted nothing bad, but missed the
  good one.** With 40 questions and the level split five ways it could not separate a measured 14-point
  gain from noise, as `detectable_gain` predicted (it puts this dataset's reliable floor at 25
  points).
- **Selecting and confirming on different questions fixes that.** `bench/confirm.py` tests only
  the selected candidate, once, on the 40 held-out questions, scored by the certified judge:
  PROMOTE (gain +0.16 per case, p = 0.015), the same decision ground truth gives. The certified
  judge agreed with ground truth on all 160 held-out replies.
- **The certified judge found a bug in my ground truth.** It passed "−27 pens" (a U+2212 minus)
  for an expected -27, which my checker had failed. The checker is fixed and tested; no row of the
  baseline changed.

The numbers above are from a re-run over the call cache after that fix, with no new calls. A few
training-set judge rates differ from the first run by about 0.01: identical replies share one
cache entry, and the first run had judged them separately, not always alike.


## Limits

- The controls are generic. `mismatched_output` assumes another case's answer is wrong for
  this case, which fails on datasets where many cases share an answer; the donor is drawn only
  from cases with a different answer, and a case with none is skipped.
- `human_agreement` uses kappa's point estimate, without an interval.
- A certificate covers one judge configuration (model, rubric, flags, settings) on one set of
  cases. Change any of them and certify again.
