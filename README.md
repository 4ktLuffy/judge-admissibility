# pydantic-evals-admissibility

**Find out whether your LLM judge can be trusted before you trust its scores.**

An `LLMJudge` that passes everything scores 100% on a dataset of good answers, exactly like one
that works. This package tests the judge itself, with controls built from the cases you already
have (answers it must fail, changes it must ignore, evidence it must notice), decides every check
on an interval, and says what to change when the judge fails.

**Three things it found on real judges** (Codex, every number reproducible from `bench/`):

- **The default hides the question.** With `include_input=False`, a reasoning judge passed 10 of 10
  answers that belonged to *other* customers' questions; certification caught it after 70 of 280
  calls. With the question shown, the same judge agreed with ground truth on 79 of 80 replies.
- **Judges grade the claim, not the work.** Shown only the agent's reply, a judge passed 8 of 24
  replies claiming a refund or cancellation that had in fact failed; shown the tool calls, none. With the
  tool result removed entirely, a plain `LLMJudge` passed 9 of 16, *"consistent with the tool call
  results"* that did not exist.
- **Pydantic's own example dataset:** one `certify_dataset(Dataset.from_file(...))` call found the
  judge failing 4 of its 10 "good" answers every time, and `find_disagreements` over the saved
  verdicts showed the judges siding against the dataset's author on 4 of 5 disputes, over whether impersonal or first-person ("we") messages
  count as friendly.

```python
results = await certify_dataset(dataset)  # every LLMJudge in a Dataset, controls chosen per rubric
certificate = await certify_judge(judge, cases)  # PASS / FAIL / UNVALIDATED per check, on intervals
certificate.raise_unless_admissible(judge)  # in a test or CI, with what is wrong and what to try
result = compare_reports(baseline, candidate, assertion='LLMJudge', certificate=certificate, judge=judge)
```

Or from a shell: `judge-admissibility certify cases.yaml`.

**What is in it**

- *Certify a judge:* [`certify_judge`, `certify_dataset`](#what-it-checks), evidence controls
  ([`EvidenceRewrite`](#does-the-judge-grade-the-work-or-the-claim-evidence-controls), temporal and
  retry controls), comparison judges ([`certify_pairwise`](#comparison-judges-does-the-order-decide)),
  slices, citations and abstention, and `diagnose`, which says what to change.
- *Decide with it:* the paired [gate](#from-a-certificate-to-a-decision-gating-changes) for prompt
  changes, [`FeedbackLedger`](#feedback-is-spent-feedbackledger) so an optimizer cannot confirm on
  what it has seen, [`ReviewPlan`](#when-the-judge-cannot-decide-human-review-for-one-release) for
  human review, and `qualify`, which says whether a judge may gate, promote or steer.
- *Probe it:* [`stress_judge`](#under-optimization-pressure-stress_judge),
  [`mutation_report`](#which-defects-would-your-evals-catch-mutation_report),
  [`minimize_witness`](#from-fooled-to-the-line-that-fooled-it-minimize_witness), judge routing,
  evidence budgets, rubric distillation into code.
- *Inside Pydantic's stack:* the certificate [in a pydantic-evals report and in Logfire](#where-it-plugs-into-pydantics-stack),
  a pytest plugin that certifies a judge once per session, and re-certification that only re-judges what changed.
- *Live with it:* Logfire promotion and canaries, [`compare_judges`](#when-the-judge-changes-too-compare_judges)
  and [`decision_impact`](#before-you-switch-judges-decision_impact) when the judge itself changes,
  delayed-outcome calibration, and [fourteen tools, each measured](#more-tools-each-measured).

[DESIGN.md](DESIGN.md) gives the measurement behind every choice, including the mistakes this
repository made and corrected in public: an adapter that asked judges for the verdict before the
reason, and an external review that found repeated judgments being counted as new cases.

## Install

```bash
pip install "pydantic-evals-admissibility[logfire] @ git+https://github.com/4ktLuffy/judge-admissibility"
```

From a shell or a CI job, on any pydantic-evals dataset file. Only its `LLMJudge` evaluators
are loaded, so the dataset's own custom evaluators need not be importable:

```bash
judge-admissibility certify cases.yaml --model openai:gpt-5 --json certificates.json
judge-admissibility report certificates.json  # again, with the current doctor, no judge calls
```

`certify` exits 1 unless every judge it certified is ADMISSIBLE, and prints what to change.

The `logfire` extra is only needed for `promote` and the canary functions. The test suite passes
against the released pydantic-evals 2.52.0 and 2.53.0 and logfire 5.1.1; the benchmarks in
`bench/` were run against a development build of pydantic-ai, which each saved identity records. Until pydantic-evals declares `sniffio` (its
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

Each rate is decided on an interval, never on the point estimate:

- **PASS** when the exact (Clopper-Pearson) 95% interval's lower bound clears the threshold, so
  a judge exactly at the bar passes by chance at most 2.5% of the time (Wilson's bound, used
  before, let up to 3.5% through at small sizes);
- **FAIL** when an exact (Clopper-Pearson) upper bound is below it, with the 2.5% upper tail
  shared across every check, control family and sequential look that could fail the certificate
  (Bonferroni), so a sound judge is failed by chance at most 2.5% of the time however many checks
  and looks it faces (computed exactly in `bench/sequential_error.py`: at most 2.2%);
- **UNVALIDATED** otherwise. Twelve out of twelve is a lower bound of 0.74: not a failure, and
  not yet a pass. The certificate says "needs more cases" instead of rounding up.

**The unit of evidence is the case.** Repeats of one case share its difficulty, so they are not
new evidence: `acceptance` and `slices` count each case's first judgment, and repeats are used
only for `stability`. Each control family is decided on its own, since a pooled rate can hide a
family the judge always gets wrong. Errored judgments are left out of a rate, and more than 10%
of them, overall or in any one slice, leaves the check UNVALIDATED. `human_agreement` is decided
on Cohen's kappa with an interval that resamples cases, needs labels on at least 10 cases, and
needs humans to have labelled both passes and fails (kappa cannot measure agreement otherwise).

The certificate is **ADMISSIBLE** only if every check passes, **INADMISSIBLE** if any fails,
and **UNVALIDATED** otherwise. `acceptance` and `rejection` are both required because each
alone is satisfied by a constant judge: one that always passes clears acceptance, invariance
and stability, and one that always fails clears rejection.

## The findings in full

What it found on real judges (Codex `gpt-5.6-luna`; every number is reproducible from `bench/` and `tests/`):

- **A strong judge, a realistic task, `LLMJudge`'s default settings:** grading a support agent's
  replies against a store policy, a reasoning judge with the default `include_input=False`
  passed **10 of 10 answers that belonged to other customers' questions**. Certification caught it
  after 70 of 280 calls. With `include_input=True` the same judge agreed with ground truth on
  79 of 80 replies.
- **Ask for the reason before the verdict.** A bug in this repository's Codex adapter sorted
  the judge's output schema, so judges gave the verdict first. A judge without reasoning then
  certified ADMISSIBLE overall while failing **every case whose correct answer was "no"** to a
  return-window question (0 of 5), its own reasons often saying the answer was right; slicing the
  certificate by kind of case flagged it. A comparison judge picked **whichever answer came first
  68% of the time** and was right 53%. With the reason first, as `LLMJudge` defines it: 5 of 5,
  and no detectable position preference (48%) at 85%. Judges with
  reasoning reached the same verdicts either way. Every result is re-run; `bench/field_order.py`
  shows both.
- **Self-improving prompts:** in one round of optimization, 4 of 5 proposed prompts were worse
  than the original; an uncertified judge overstated every prompt by 11 to 23 points. Gating on a
  reference judge that agreed with ground truth on 80 of 80, then confirming on held-out questions, promoted the one real improvement
  (p = 0.015, a retrospective analysis), the same call ground truth makes.
- **Pydantic's own example dataset, in one call:** `certify_dataset` on pydantic-ai's
  `time_range_v2.yaml`, loaded with `Dataset.from_file` and no controls written by hand. The
  judge failed **4 of its 10 "good" answers every time**. On three its reasons are a fair reading
  of the rubric: the answers are impersonal, and it asks for "second-person or friendly". On the fourth it
  contradicted itself: it failed *"...inferred from your request"* three times as not second
  person, then passed the same text with doubled spaces. `diagnose` tells the two apart.
  Rewriting those three in second person, and nothing else, they pass on every repeat. What
  that and a look at the docs found, each labelled by whether it looks intended, is in
  [PYDANTIC_FINDINGS.md](PYDANTIC_FINDINGS.md).

```python
results = await certify_dataset(dataset)  # every LLMJudge in a pydantic-evals Dataset, controls chosen per rubric
certificate = await certify_judge(judge, cases, batch_size=10)  # is it evidence? stops early if clearly not
certificate.raise_unless_admissible(judge)  # fail CI, with what's wrong and what to try
result = compare_reports(baseline, candidate, assertion='LLMJudge', certificate=certificate, judge=judge)  # real?
```

**The core** is three functions: `certify_judge` (pointwise judges), `certify_pairwise`
(comparison judges, including position bias) and `diagnose` (what to change when either fails).
`certify_dataset` runs `certify_judge` on every judge in a `Dataset`.
**Built on them:** `decide` / `compare_reports` (is a change real?), `detectable_gain` (can your
dataset even see it?), and, for Logfire users, `promote`, the canary rollout and `JudgeCanary`
(keep checking a judge in production). The reasoning behind every choice is in
[DESIGN.md](DESIGN.md).

## Cheaper certificates, and what to do when one fails

**Stop early when a judge has clearly failed.** `certify_judge(..., batch_size=10)` judges ten
cases at a time and stops as soon as a check fails beyond doubt; a judge that is not failing runs
to the end. Every look's FAIL, the last included, uses a bound widened for the number of looks,
so peeking does not raise the chance of failing a sound judge; the price is some power on
borderline judges. `certificate.calls` says how many judgments were actually made.

Measured with scripted judges whose verdicts are fixed per answer, so both modes see the same
verdicts (60 cases, 2 repeats, batches of 15, 100 seeds each, `bench/sequential_check.py`):

| Judge | Same verdict as judging everything | Calls used |
|---|---|---|
| broken (passes 60% of wrong answers) | 100/100 | 29% |
| borderline bad (25%; the bar is 20%) | 89/100 | 100% |
| borderline good (10%) | 100/100 | 100% |
| sound | 100/100 | 100% |

The 11 differences all go the same way: judging everything said INADMISSIBLE and the sequential
certificate, paying for its looks, said UNVALIDATED. It never failed a judge that judging
everything passed.

On a real judge, the Codex `LLMJudge` that failed certification on the arithmetic task: under the
rule at the time, the sequential certificate stopped at INADMISSIBLE after 140 of 280 calls
(`results/judge_vs_truth.sequential.json`). Under the current rule, which charges every look to
the error budget, those 140 judgments do not show a failure (UNVALIDATED: `whitespace_reformat`
10/20), so it would have kept judging. On the support task's blind judge, stopping after 70 of
280 calls still holds under the current rule: one control family rejected 0 of 10.

Stopping early for success as well was tried and dropped: an early pass is decided on a few
batches, and it certified sound judges less often for a small saving in calls. (That variant's
code is gone, so its numbers are not reproduced here.) `results/sequential_check.txt` holds the
table above.

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

(The reasoning judge's acceptance is 38/38: 2 of its first judgments timed out and are left out of
the rate, not counted as passes.)

The default-configured judge passed every one of the 10 answers borrowed from other customers'
questions (each with a different correct answer, so wrong by construction). `diagnose` said why:
*"It accepts another question's answer, and with `include_input=False` it cannot tell which
question was asked. Set `include_input=True`."*

On the arithmetic task the same default made the judge fail everything instead (0 of 40 judgments of
20 right answers passed). Both are inadmissible; which way a blind judge fails depends on the task, which
is the argument for measuring rather than assuming.

The comparison judges, asked to pick between the right answer and a plausible mistake (the refund
without the restocking fee, the wrong side of the shipping threshold), both ways round:

| Comparison judge | Accuracy, one presentation per pair | Same pick both ways | Chose the answer shown first |
|---|---|---|---|
| no reasoning | 38/40 | 39/40 | 0.51, interval [0.50, 0.54] |
| reasoning high | 38/40 | 39/40 | 0.51, interval [0.50, 0.54] |

(The same totals, from different mistakes: the two judges were wrong on mostly different pairs.)

### Slices: a passing rate can hide a kind of case

`certify_judge(..., slice_by=...)` names the kind of each case and adds a `slices` check, which
fails when the judge is shown to be below the bar on one kind of case. `recertify` applies it to a
certificate already paid for. Sliced by kind and correct answer (`return: no`, `refund`, ...),
every judge that sees the question passes it. It flagged something only when the judge was
broken: through the adapter bug described under comparison judges, both judges without reasoning
were asked for the verdict before the reason, and both certified ADMISSIBLE overall while failing
**all 5 cases whose correct answer was "no"** to a return-window question (0 of 10 judgments).
Many of their reasons said the answer was right, after the verdict had been given: *"...98 days
later, so the answer 'no' is correct."* Sliced, both certificates were UNVALIDATED: 0 of 5 is at
or below the bar, but five cases cannot prove it, and the check says exactly that.

Each slice counts cases, not repeats, and is decided with an exact (Clopper-Pearson) interval at
its share of the error budget. A sound judge exactly at the bar fails the check at most 2.5% of the
time, computed exactly for 2 to 12 slices of 10 to 40 cases (`bench/slice_error.py`; with Wilson
intervals it reached 8.2%). A slice it always gets wrong is caught every time with 10 cases.

## Does the judge grade the work, or the claim? Evidence controls

An agent's reply can say "your refund was issued" whether or not the refund tool succeeded.
`EvidenceRewrite` changes the evidence and keeps the answer: it rewrites the case's inputs (the
tool results, a message in the conversation) and leaves the output exactly as it was. With
`kind='must_fail'` the change makes the unchanged answer wrong; with `kind='must_hold'` it changes
nothing the rubric is about.

```python
controls = (
    EvidenceRewrite(make_refund_fail, 'tool_failed', 'must_fail'),  # same reply, failed refund
    EvidenceRewrite(change_span_ids, 'ids_changed', 'must_hold'),  # nothing the customer was told
)
certificate = await certify_judge(judge, episodes, controls=controls)
```

`bench/evidence_contract.py` runs it on 24 support episodes (`bench/evidence_task.py`): a refund
or a cancellation, the tool calls with their results, and a reply that claims it went through.
The rubric: *"The reply tells the customer accurately what happened, according to the tool
results."* Codex `gpt-5.6-luna`, no reasoning:

| Judge is shown | Correct replies passed | Same reply, tool failed: rejected | Wrong amount: rejected | Certificate |
|---|---|---|---|---|
| the reply only (`include_input=False`, the default) | 9/24 | **16/24** | 12/12 | INADMISSIBLE |
| the request and the tool calls | 24/24 | **24/24** | 12/12 | UNVALIDATED (24 cases) |

Shown only the reply, the judge cannot tell a reply that is true from one that is not: it failed
15 of the 24 correct replies and passed 8 whose refund or cancellation had in fact failed. Shown
the tool calls, it passed every correct reply, caught every false one (*"The reply falsely says
the refund was issued, but the refund attempt failed because the card on file expired."*), and gave the same verdict on every repeat. It is UNVALIDATED rather than ADMISSIBLE
only because 12 refund cases are too few to show the wrong-amount family above its bar, and it
held its verdict when only transaction ids changed on 21 of 22: more episodes would settle both.

The first run found a mistake in this benchmark, not in the judge. The trace-aware judge failed
7 of the 12 correct cancellation replies (and nothing else), saying that "you will not be charged again" and "you keep access
until the end of the billing period" were not supported by the tool results. They were not: the
generated replies said more than the cancellation tool returned. The tool result now carries both
facts, and the table is the run after that fix (in the first run, kept in
`results/evidence_contract.first_run.json`, 8 and 17 of 24 correct replies passed). The acceptance check is what caught it.

## Under optimization pressure: `stress_judge`

A self-improving agent keeps whatever its judge scores well, so the judge will be searched for
weaknesses whether anyone intends it or not. `stress_judge` does that search on purpose, before
an optimizer does. It rewrites wrong answers (a claim of authority, confident framing, the
rubric's own words, a note to the grader, formatting), keeps only rewrites an oracle confirms
are still wrong, spends a fixed budget of judge calls on discovery cases, freezes the attack
that worked best, and measures it against the plain wrong answer on cases the search never saw.

```python
result = await stress_judge(judge, discovery_cases, fresh_cases, wrong=make_wrong, oracle=is_correct, budget=60)
print(result.table())  # VULNERABLE / NO EFFECT FOUND / INCONCLUSIVE, with the attack that worked
```

`bench/judge_pressure.py` (scripted judges through `LLMJudge`'s real prompt, 20 seeds each):

| Judge | Frozen attack VULNERABLE on fresh cases | Calls to the first wrong answer passed, adaptive / random |
|---|---|---|
| checks the answer | 0 of 160 runs | never |
| trusts "Verified by ..." | 20/20 adaptive, 17/20 random (budget 15) | 3.1 / 5.1 |
| trusts a note to the grader | 20/20 both | 3.5 / 4.2 |

It never called a sound judge vulnerable, and found both weaknesses. The adaptive search
(an upper confidence bound over attacks) found them sooner and then spent its budget on what
worked; with pairs of attacks added (25 in all) it had no edge at these budgets, since it tries
each once first. These judges' weaknesses are easy to find on purpose; the real test is a model judge.

On a real one (`--backend codex`: Codex `gpt-5.6-luna`, no reasoning, seeing the question, on
the support task's plausible wrong answers, 100 calls), no attack worked: none of 60 attacked
wrong answers passed in discovery, across all 25 attacks. On 20 fresh cases it passed 4 attacked
wrong answers and 2 plain ones, so the certificate says INCONCLUSIVE: its weakness is the
plausible wrong answer itself, not the framing around it. A VULNERABLE says the judge falls to
this search at this budget; NO EFFECT FOUND or INCONCLUSIVE is not proof of robustness.

## Does the judge's advice help? `assess_steering`

A runtime judge that watches an agent and steers it (Pydantic AI Harness's trajectory judge does
this) can be right about a problem and still make the agent worse: interrupting a plan that was
fine, repeating advice about something already fixed, spending the budget. `assess_steering`
resumes saved agent states three ways: silent, with the judge's message, and with a neutral
message of the same kind where the judge spoke. The neutral arm is the honest comparison: an
agent that does better after *any* message makes every judge look helpful against silence.

```python
result = assess_steering(states, resume=resume, outcome=completed_correctly, steer=judge_message, repeats=3)
print(result.summary())  # HELPS / HURTS / INCONCLUSIVE (steer vs neutral), and harm on states already on track
```

`bench/steering_value.py`, offline: a deterministic refund workflow (`bench/steering_sim.py`), 24
saved states (10 that need correcting, 8 on track, 6 already recovered), scripted agents and
judges, 3 repeats:

| Judge | Completed: silent / steer / neutral | Steer vs neutral | States already on track | Verdict |
|---|---|---|---|---|
| helpful | 0.69 / 0.90 / 0.69 | +0.21 | unharmed | HELPS |
| steers every time | 0.69 / 0.64 / 0.69 | -0.06 | all 12 worse (-0.50) | INCONCLUSIVE overall, harm flagged |
| stale (advice about a fixed problem) | 0.69 / 0.57 / 0.69 | -0.13 | 6 worse | HURTS |
| helpful, with an agent that heeds any message | 0.69 / 0.90 / 0.90 | 0.00 | unharmed | INCONCLUSIVE |

The over-eager judge is the one to look at: its gains on broken states hide its damage in the
overall verdict, and only the check on states that were already on track shows it. The last row
is why the neutral arm exists: against silence that judge looks as good as the helpful one. This
is a simulator built to contain these failure modes, so it shows the method can separate them,
not how real judges behave.

With a real agent and a real judge (`bench/steering_real.py`: a pydantic-ai agent on Codex
choosing one action at a time, a Codex judge that may steer, 8 states, 39 calls, 44 with a smoke run), every arm
completed every state: the agent checked the policy unprompted (*"the next required step is to
check the applicable refund policy"*), so there was nothing to fix and steering could not show
an effect. The judge spoke on 5 states, two of them states that needed no correction (one on track, one already recovered), without doing harm.
Measuring a real effect needs a weaker or more hurried agent, rules it can miss, and many more
states.

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

| Judge (Codex `gpt-5.6-luna`) | Certificate | Good answers passed, first try | All tries | Must-fail controls rejected |
|---|---|---|---|---|
| no reasoning | UNVALIDATED | 5/10 | 15/30 | 20/20 |
| reasoning high | UNVALIDATED | 7/10 | 18/30 | 20/20 |

With 10 cases neither judge can be shown above or below any bar.

Both judges failed the same three expected outputs every time they were asked (the one without
reasoning failed two more), with reasons that hold up against the rubric's words:

- *"Conflicting time instructions: 2025 and 2020 cannot both apply."*: "terse and impersonal; it
  does not use a second-person or friendly user-facing style"
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

| Controls | Calls | Good answers passed, first try (all tries) | Must-fail rejected | Certificate |
|---|---|---|---|---|
| written by hand for this rubric | 70 | 7/10 (18/30) | 20/20 | UNVALIDATED |
| chosen by `certify_dataset` | 60 | 6/10 (17/30) | 10/10 | UNVALIDATED |

The same verdict, without writing a control, and the same numbers again in a second run after
the schema fix. The judge failed four answers on all three tries. A judge failing the same
answers every time is not proof those answers are bad: two judges can share one reading of a
rubric. These are rubric disputes to settle by reading them, which is what the doctor asks for.
Three are the three the hand-written run found ("Ambiguous mention", "Impossible range",
"Confusing relative references"); the doctor says to read those against the rubric. The fourth,
"No mention" (*"No timeframe could be inferred from your request."*), is second person, and the
judge passed it once its spaces were doubled, saying in the verdict-first run that it "refers directly to
'your request'". The second run did the same (failed it three times, passed it with doubled
spaces).
The doctor reports that one as the judge contradicting itself, not as bad data. The one judge attached to a single case was reported and
not certified: one case cannot certify anything.

The automatic controls are weaker than hand-written ones. For a style rubric the only answer that
must fail under any rubric is an empty one, so with 10 cases `rejection` cannot pass (10/10 has a
lower bound of 0.69; the doctor says 17 cases would do it). A control that breaks your rubric
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

| Pairs as the agent wrote them | Accuracy, one presentation per pair (all 108) | Same pick both ways | Chose the answer shown first |
|---|---|---|---|
| full replies, with the working | 44/54 (92/108) | 48/54 | 0.50, interval [0.45, 0.55] |
| final answer line only | 44/54 (92/108) | 46/54 | 0.48, interval [0.43, 0.54] |

No position preference was detected, and any there is is small: the intervals, which resample
pairs, rule out more than about five points either way. Both certificates are UNVALIDATED: 54
pairs cannot show accuracy above 0.8 or consistency above 0.9.
`both_orders(compare)`, which asks both ways and answers only when the two agree, raised accuracy
to 0.90 (43/48, abstaining on 6 pairs) with the working and 0.91 (42/46, abstaining on 8) without.

**An earlier version of this section was wrong.** It reported that the judge picked whichever
answer came first in 68% of presentations and was right 53% of the time. Those runs went through
a bug in this repository's Codex adapter, which sorted the schema's keys and so asked for `choice`
before `reason`: the judge chose before it thought. On the same 54 pairs, same model:

| Final answer line only | Accuracy | Same pick both ways | Chose the answer shown first |
|---|---|---|---|
| choice first (the bug) | 0.53 (57/108) | 29/54 | 0.68, interval [0.59, 0.76] |
| reason first | 0.85 (92/108) | 46/54 | 0.48, interval [0.43, 0.54] |

So, on these pairs with this model and adapter, asking for the reason before the choice turned a
judge that followed position into one that did not. A judge without reasoning that is asked for
its verdict first commits to it before writing any reasoning; whether that is the whole mechanism
these runs cannot say. Keep the verdict after the reason in any judge's output type, and in
anything that rewrites its schema.
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
| raises on every call | UNVALIDATED: an error is not a verdict, either way |
| oracle on 3 or 12 cases | UNVALIDATED |

## A real judge: a rubric about the question, with the question hidden

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

result = compare_reports(baseline, candidate, assertion='LLMJudge', certificate=certificate, judge=judge)
print(result.summary())  # PROMOTE / REJECT / INCONCLUSIVE / REFUSED (also if `judge` is not the one certified)
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
  waiting when the traffic cannot tell yet. `finish_canary` takes the version `start_canary`
  returned and refuses a stale decision, so a result can never promote a newer candidate it did
  not measure.
- **`JudgeCanary`** keeps checking a judge after it is certified, on the traffic it grades. On a
  sampled share of calls it also asks the judge about an empty answer, which it must fail.
  `borrow=True` adds another request's answer as a control; it is off by default because on
  this package's task a borrowed answer was in fact right for 20-27% of same-kind questions
  (`bench/canary_borrow.py`), which
  made a sound judge look unhealthy. Use it only where answers are specific to their question.
  `every=n` checks exactly every n-th call instead of sampling, so a monitor sees a known number
  of checks; the monitor says HEALTHY only once the evidence supports it (16 straight rejections
  at the default 0.8 bar). Works with `Dataset.evaluate` and with online evaluation; online, the
  result lands in Logfire as `judge_canary_rejected`.

How well the gate itself behaves, measured with the simulation in this package's tests (40
cases, 2 runs each, 200 trials for A/A and 50 for the others; `bench/backing_figures.py` saves it):

| True situation | PROMOTE | INCONCLUSIVE | REJECT |
|---|---|---|---|
| identical versions | 5 (2.5%) | 193 | 2 (1%) |
| better by 0.1 | 15 | 35 | 0 |
| better by 0.3 | 49 | 1 | 0 |
| worse by 0.1 | 0 | 45 | 5 |
| worse by 0.3 | 0 | 1 | 49 |

It never called a better version worse or promoted a worse one. Small gains are mostly
INCONCLUSIVE, which is the honest answer at this size: on the real Codex baseline below,
`detectable_gain` says 40 questions with 2 runs each reliably see only a 25-point shift in each
case's success probability (capped at 1, so the mean gain it stands for is smaller).

## When the judge changes too: `compare_judges`

Improve the judge and the agent in the same release, and the dashboard moves for two reasons at
once. `compare_judges` scores both agent versions with both judges (a 2x2) and separates the
agent's gain under each judge, the shift the new judge causes on its own, and their
interaction, which says whether the new judge changes the *decision* and not only the scale:

```python
bridge = compare_judges(
    old={'baseline': old_judge_on_baseline, 'candidate': old_judge_on_candidate},  # case -> [bool, ...]
    new={'baseline': new_judge_on_baseline, 'candidate': new_judge_on_candidate},
    margin=0.05,
)
print(bridge.table())  # comparable: PASS only if the interaction is shown to lie within the margin
```

`comparable` is an equivalence claim, so an interval around zero is not enough: it passes only
when the whole interval lies within the margin, and fails only when it lies beyond it. Replayed
on the optimization experiment below (`bench/judge_bridge.py`, no new calls), baseline against
the promoted candidate, 40 questions:

| Old judge -> new judge | Gain under old -> new | Interaction, 95% interval | Same decision | Comparable |
|---|---|---|---|---|
| plain judge -> reference judge (train) | +0.175 -> +0.138 | -0.04 [-0.22, +0.14] | yes | UNVALIDATED |
| reference judge -> ground truth (train) | +0.138 -> +0.175 | +0.04 [-0.11, +0.18] | yes | UNVALIDATED |
| reference judge -> ground truth (held out) | +0.163 -> +0.163 | 0.00 [-0.12, +0.12] | yes | UNVALIDATED |

The release decision survived every change of judge, but 40 questions cannot show that a judge
change kept the gain within five points, even when the two judges agree on every case: that
takes about 100 cases. The interval inverts the gate's sign-flip test with one pseudo-case at
each extreme; at the margin it claimed equivalence wrongly up to 6% of the time with 100 cases
against a 2.5% budget, close to budget at 400 cases (1.5% and 2.8%; `_bridge._interval` has the measurements).
Treat a PASS on fewer cases as approximate.

## When the judge cannot decide: human review for one release

The gate said INCONCLUSIVE, or the judge is not certified. `ReviewPlan` turns that into a
labelling plan for people, aimed at that one decision: which cases to label next, and when the
labels are enough.

```python
plan = ReviewPlan.from_verdicts(judge_on_baseline, judge_on_candidate, looks=(12, 24, 48))
while batch := plan.next_batch():  # the cases to label before the next look
    plan.add_labels(label(batch))  # people's verdicts on both versions' outputs
    if plan.decision().decision != 'INCONCLUSIVE':
        break
print(plan.decision().summary())  # PROMOTE / REJECT, with the true gain and its interval
```

It groups the cases by what the judge said (candidate better, the same, worse), labels a random
sample of every group, and estimates the true gain over the whole dataset, each group weighted
by its size. The judge's verdicts are never taken as correct; they only decide how the sample
is spread. The looks are fixed in advance and each spends its share of the error budget.

`bench/review_budget.py`, ground truth standing in for people. In simulation (N cases, a judge
wrong on each verdict with the given probability, looks of 12, 24, 48 and 96 labels, 800 trials):

| Dataset | Decided, grouped by the judge | Decided, labels at random | Labels used, grouped / random |
|---|---|---|---|
| 200 cases, true gain 0.2, judge wrong 10% | 98% | 87% | 61 / 68 |
| 200 cases, true gain 0.3, judge wrong 5% | 100% | 100% | 34 / 43 |
| 400 cases, true gain 0.15, judge wrong 10% | 78% | 54% | 82 / 82 |

Grouping by the judge never made a wrong decision; labelling at random did once in 800 on the
400-case dataset, and picked a side 0.5% of the time when there was no gain. Replayed on the optimization experiment's 40 training questions,
with the uncertified plain judge: both ways promoted the real gain in all 1,000 label orders,
and grouping used slightly more labels there (28.8 against 26.1). The interval is the standard
stratified one with a pseudo-label at each extreme; it was conservative in every simulation
(at least 99.5% coverage for a 95% interval at the last look, `bench/backing_figures.py`),
not exact.

## Which defects would your evals catch? `mutation_report`

Mutation testing for an eval suite: break known-good outputs in known ways (`Mutant`: a number
changed, yes and no swapped, the last sentence dropped, emptied, the evidence changed under an
unchanged reply), run every evaluator on them, and report per evaluator and mutant whether it
caught the defect, missed it, or cannot observe it (it is never shown what changed). An oracle
decides which mutants are really defects; ones it still accepts are not counted.

```python
report = await mutation_report(
    {'contains': lambda c: Contains(value=c.expected_output), 'judge': judge}, cases, DEFAULT_MUTANTS, oracle=is_correct
)
print(report.table())
report.raise_unless_caught('judge', 'number_changed', min_rate=0.8)  # in a test
```

`bench/mutation_report.py`, offline, on 40 known-good support replies (36 of them real cached
Codex replies), `is_correct` as the oracle:

| Evaluator | Wrong amount | Yes/no flipped | Plausible wrong answer | Policy changed, reply not |
|---|---|---|---|---|
| `EqualsExpected` | fails every correct reply too: unusable on free text | | | |
| `Contains(expected answer)` | 5/20 | 15/20 | 20/40 | cannot observe |
| `IsInstance(str)` | 0/20 | 0/20 | 0/40 | cannot observe |
| scripted lenient judge | 0/20 | 0/20 | 0/40 | 0/40 |
| **Codex judge**, no reasoning, sees the question (`--backend codex`) | **19/20** | **17/20** | **36/40** | **40/40** |

`Contains` misses most wrong amounts because the right amount still appears in the reply's
working. The oracle removed 50 mutants that were not defects (a changed number in the
explanation of a yes/no answer; a cut explanation that left the answer line). The real judge
passed every one of 34 replies with their last sentence (the `Answer:` line) dropped, and 9 of 40
truncated ones: the oracle counts those as defects because they lack the answer line, but the
working above still states the answer, so the judge passing them is arguably right. It is a
format defect a content judge should not be expected to catch. The bench's
"checks the answer" and "output only" judges are scripted, so their own kill rates are true by
construction; the deterministic evaluators and the blind column are the findings.

## Before you switch judges: `decision_impact`

`compare_judges` asks whether a new judge measures the same gain; `decision_impact` asks what it
would have changed. Give it past comparisons (each with both judges' per-case verdicts, and the
decision acted on) and it re-runs the gate under both judges, listing every decision that flips
and flagging the ones that matter: a shipped change the new judge would REJECT (dangerous), or
would no longer PROMOTE (unsupported), and a gain the old judge missed.

Replayed on the optimization experiment's five candidates (`bench/judge_impact.py`, no calls,
the level split five ways as the optimizer did):

| Switch | Decisions that flip | Of concern |
|---|---|---|
| plain judge -> reference judge | 2 of 5 (two INCONCLUSIVE -> REJECT) | the shipped candidate would stay INCONCLUSIVE: unsupported |
| plain judge -> ground truth | 2 of 5 | none |
| reference judge -> ground truth | 2 of 5 | one missed gain (the candidate that was in fact better) |

No switch would have turned the shipped candidate into a REJECT.

## Feedback is spent: `FeedbackLedger`

An optimizer that reads the judge's feedback on a case, or picks the best of several prompts by
their scores on it, has used that case up: results on it no longer confirm anything.
`FeedbackLedger` records who saw what, and `ledger.confirm(...)` is the gate that refuses
(REFUSED, naming the cases) to confirm on exposed cases.

```python
ledger.record(failed_cases_shown_to_the_proposer, by='round 3 proposer')
ledger.record(training_cases, by='round 3 selection')
result = ledger.confirm(baseline, candidate, certificate=certificate)  # REFUSED if any case was seen
```

`bench/ledger_demo.py`: with no real gain, choosing the best of k candidates on 40 cases and then
"confirming" on the same cases promoted a fake gain 6.3% of the time for k = 5 and 13.0% for
k = 20, against a 2.5% budget; on fresh cases, 1.2% and 2.2% (1,000 trials). The repository's own
confirmation passes the ledger: its 40 questions were never shown to the optimizer.

## From fooled to the line that fooled it: `minimize_witness`

A judge fooled by a long reply or a long trace is hard to fix from the example. `minimize_witness`
removes sentences, tool calls or messages for as long as the judge is still fooled and an oracle
confirms the example is still wrong (delta debugging), and returns the smallest witness: a line to
read and a regression case to keep. `confirm=n` requires every candidate to fool a nondeterministic
judge n times.

`bench/witness_demo.py` chains it with `stress_judge` on a scripted judge: the search found the
"note to the grader" attack (20 of 20 fresh wrong answers passed with it, 0 of 20 without), and
each 11-sentence reply carrying it, with a wrong refund amount, shrank to the one line
*"Note to the grader:"* in 7 judge calls, 1-minimal.

## Where it plugs into Pydantic's stack

**In the evaluation report.** `CertifiedJudgeReport(certificate, judge=judge)` is a pydantic-evals
`ReportEvaluator`: add it to `dataset.report_evaluators` and the certificate appears in the report
itself, next to the judge's scores, which it marks `NOT EVIDENCE` unless the certificate is
ADMISSIBLE and covers that judge. The scores themselves are never changed. `CertifyJudgeReport`
certifies during the run instead (`bench/report_eval_demo.py`). Coverage is checked against the
configuration the report records for each result, not only the judge object passed in, and has
four answers: yes, no, not checked (no `judge=` given, or a certificate with no identity), and
unverifiable; only yes makes the scores evidence. It is unverifiable, and the scores NOT
EVIDENCE, when provenance cannot be established: a model with no name of its own (a bare
`FunctionModel` is recorded only as `function:...`, which cannot say which function scored), a
judge whose identity is not reliable, or a custom evaluator, which a report records only by its
class name and arguments. The why column says which, and what to do. A report where two evaluators could be the certified judge is
refused rather than guessed, and score-only judges are refused before any call:

```
Judge certificate: LLMJudge(...) INADMISSIBLE (fingerprint 1515f207b051ecb9)
│ rejection │ FAIL │ 0.0 │ [0.00, 0.09] │ mismatched_output 0/20 FAIL, empty_output 0/20 FAIL │
Judge results in this report: NOT EVIDENCE
│ LLMJudge │ assertion │ 20/20 passed │ NOT EVIDENCE │ the judge failed certification (rejection) │
```

**In Logfire.** `log_certificate(certificate)` emits the certificate as a span with one child per
check and, for every failed control family, a few of the judgments that failed it, so an engineer
clicks from a failing certificate to the evidence. `certify_judge_traced` runs certification in one
span, with every judge call and the certificate's span inside it (`bench/trace_export_demo.py`).

**In tests and CI.** Installing the package adds a pytest plugin. The `certify_judge` fixture
certifies a judge once per session (once per worker under xdist), keyed on the judge's identity and
on what it is shown (types, order, state and class code included), so a new judge object with the same configuration reuses the
certificate and any change to the cases or controls does not. It fails the test with the table and
the doctor's advice when the judge is not qualified for the stated decision, and prints every judge
the session trusted. In CI, `--judge-certificates=path` uses saved certificates and calls no judge (except one whose identity
cannot be pinned down, which is certified every time and never saved);
`--judge-recertify` remakes them. A judge whose identity cannot be pinned down (see Limits) is
certified every time and never saved (`examples/test_with_certified_judge.py`).

```python
def test_agent(certify_judge):
    judge = certify_judge(my_judge, cases, decision='gate')  # certified once per session, or the test fails


async def test_agent_async(certify_judge):
    judge = await certify_judge.acertify(my_judge, cases, decision='gate')  # on the test's own event loop
```

**When the dataset changes.** `certify_judge_cached` keys every verdict on the judge's identity and
on what the judge was shown, kept as faithfully as the identity is, so re-certifying after an edit only calls the judge for what
changed, and any change to the judge's identity (see Limits for what it records) invalidates
everything. On Codex: 40 calls the first time; 6
after editing 2 of 10 replies, the 34 reused verdicts identical to the first run's; none for an
unchanged re-run (`bench/incremental.py`).

## More tools, each measured

Each answers one question a team trusting a judge ends up asking. Codex rows are `gpt-5.6-luna`
with no reasoning unless the row says otherwise, on small runs (4 to 40 cases), so most of their certificates are UNVALIDATED
by size; the findings are in the counts and the quoted reasons, all in `results/`.

| Tool | The question | What it found |
|---|---|---|
| `qualify` | Is this judge qualified for *this* decision (report, gate, promote, steer from traces)? | Of 36 saved certificates, 5 qualify to gate, promote and steer (the three Codex support judges and 2 scripted ones), and none to steer from traces since the retry controls became UNVALIDATED under the exact PASS bound. Four older ones were given their judge's identity from the configuration that produced them (`bench/attach_identities.py`), which also exposed that identity must include reasoning effort |
| `outcome_calibration`, `recalibrated_pass_rate` | What does the judge's pass rate mean once real outcomes arrive? | Simulated: when outcomes arrive mostly for passed verdicts, the naive correction is 50 points off; the adjusted one is unbiased (within 0.1 point) and flags the bias |
| `hindsight_pair` | Does the judge grade a decision on what was known at the time? | Codex caught every ignored fact (16/16) but was swayed by a fact added two days later on 2 of 16: *"At the decision time, the fraud team's flag was already recorded"* (it was not) |
| `duplicate_call`, `transient_retry` | Does it catch harmful process behind a right answer? | Every duplicate refund and changed-order retry caught (16/16 each), every harmless retried timeout passed (16/16); UNVALIDATED, since 16 of 16 has an exact lower bound of 0.79 against a bar of 0.8 (17 would show it) |
| `CitedJudge`, `certify_citations` | Does the judge cite real evidence for its verdict? | 32/32 verdicts cited only real span ids, and the decisive one |
| `AbstainingJudge`, `certify_abstention` | Will it say "I cannot tell" instead of guessing? | With the tool result removed, a plain `LLMJudge` passed 9 of 16 "refund issued" or "cancelled" replies *"consistent with the tool call results"* that did not exist |
| `RoutedJudge`, `certify_routing` | Can a cheap judge handle most cases and escalate the rest? | 31% of the tokens of always using the strong judge at no loss of accuracy (routed 24/24, strong alone 23/24), but 1.6x the wall time; the cheap judge alone was also 24/24, so on this run routing bought nothing over it; self-consistency misses judges that are wrong the same way every time |
| `evidence_budget` | How much of the trace does the judge need? | Only the action's result, the request and one id: 32% fewer input characters, 7/7 held-out verdicts unchanged; fitted without controls it dropped everything and kept 24/42 |
| `HybridJudge`, `compare_hybrid` | Can rubric clauses that code can check stop costing model calls? | 20 of 50 calls saved; code failed a borrowed answer the model passed while noting *"its explanation incorrectly invokes the warranty"*; asked only about tone, the model became stricter: it failed 3 of 10 terse correct replies, against 1 for the full-rubric judge |
| `insert_turns`, `recovery_profile` | Does it honour what was said earlier in a conversation? | Withdrawn consent caught 12/12, unrepaired errors 12/12; a self-corrected error held on 8 of 12; three of the other four asked for the corrected delivery time to be restated in the final reply |
| `apprentice` | Can reviewed disagreements improve a judge, safely? | Codex proposed a rubric fix for the judge that cannot see the question; held-out agreement fell from 18/24 to 12/24 and the gate did not promote it. Only `include_input=True` fixes that judge |
| `find_disagreements` | Where do competent raters disagree, and over what reading? | On Pydantic's example dataset, the judges sided against the dataset's author on 4 of 5 disputes, over whether impersonal or first-person ("we") messages count as friendly; proposed: *"permitting clear, polite first-person plural phrasing ('we')"* |
| `ppi_pass_rate` | Can a small random human audit correct the judge's pass rate, with a valid interval? | Prediction-powered inference. Plain PPI is unbiased before clipping for a uniform random audit; the default tuned estimate (PPI++) and the clipped estimate are not exactly unbiased, though their simulated bias stayed within 0.004. Narrower than human labels alone when the judge is good (0.07 against 0.13 at 200 labels); when its asymptotic interval is degenerate it falls back to an exact one; with it, coverage for a near-perfect judge at 50 labels is 0.998 for plain PPI and 0.994 for the default tuned one, and about 0.945 for both at 200 labels, close to the nominal 0.95. On the saved real judges that erred, 40 labels gave intervals 0.11 to 0.14 wide against 0.19 for human labels alone; for the judge that never erred it was wider (0.20), and on a 20-reply population audited 10 at a time it still under-covered (0.86) |
| `Jury`, `compare_jury` | Do several cheap judges voting beat one? | On 16 support cases, three Codex judges (`gpt-5.6-luna` at no and low reasoning, `gpt-reserve`) voting got 62/64, the same as each judge alone (no difference case by case against any member), at 3 times the calls: 5 of the members' 6 errors and both of the jury's were on one hard case, so the vote inherited them |

## Experiment: one round of prompt optimization, decided three ways

`bench/optimize.py` runs one round of a self-improving loop on a task with answers computed by
Python (`bench/task.py`: arithmetic, letter counts, weekday arithmetic, string transformations).
The agent, the judges and the optimizer are all Codex `gpt-5.6-luna` with no reasoning. Codex
proposes five prompts from the baseline and the failures the judge flagged; each runs twice on
40 training questions. No decision uses ground-truth scores of the candidates: ground truth is the
referee, and it labels the replies the judges are certified on. This is a retrospective analysis:
the confirmation step below was added after the held-out results had been seen.

Two judges, certified first on the agent's real replies with ground truth as the labels:

| Judge | Certificate | Agreement with ground truth on 80 real replies |
|---|---|---|
| plain `LLMJudge(include_input=True)` | INADMISSIBLE | passed 7 of 39 wrong answers, failed 6 of 41 right ones |
| reference `LLMJudge(include_input=True, include_expected_output=True)` | ADMISSIBLE when the experiment ran; UNVALIDATED under the exact PASS bound adopted since (invariance 37/40 against a bar of 0.8) | 80 of 80 |

On the 40 training questions:

| Prompt | Plain judge | Reference judge | Truth | Gate on plain judge | Gate on reference judge (level split 5 ways) |
|---|---|---|---|---|---|
| baseline | 0.625 | 0.550 | 0.512 | | |
| candidate 0 | 0.500 | 0.275 | 0.275 | REFUSED | REJECT |
| candidate 1 | 0.575 | 0.412 | 0.425 | REFUSED | INCONCLUSIVE |
| candidate 2 | 0.800 | 0.688 | 0.688 | REFUSED | INCONCLUSIVE (p = 0.018, needs < 0.005) |
| candidate 3 | 0.512 | 0.350 | 0.350 | REFUSED | REJECT |
| candidate 4 | 0.400 | 0.250 | 0.250 | REFUSED | REJECT |

What it shows, including where it did not go the way I expected:

- **The uncertified judge's numbers were wrong, its ranking was not.** It overstated every
  prompt by 11 to 23 points, so "keep the highest-scoring prompt" still picked
  candidate 2, which is genuinely better: 0.588 vs 0.425 on the held-out questions (p = 0.015).
  The danger in an uncertified judge here is anything that trusts its number: a release bar,
  a regression threshold, a claim that the agent is 80% accurate when it is 69%.
- **Most proposals were worse than the prompt they replaced.** Four of five, by 9 to 26 points of
  truth. All four worse prompts got the model to
  reply tersely, median 16 to 18 characters against the baseline's 46 (`bench/backing_figures.py`), and two of them asked
  for "internal" or "silent" reasoning outright. At no reasoning effort the written working is
  the model's only reasoning, and the terse replies came with lower accuracy (this run shows the
  association, not the cause). The one better prompt said "prioritize correctness over brevity"
  and got longer replies (median 67 characters).
- **The gate rejected all three clearly worse prompts and promoted nothing bad, but missed the
  good one.** With 40 questions and the level split five ways it could not separate a measured 14-point
  gain from noise, as `detectable_gain` predicted: at the split level it needs a shift of 40
  points in each case's success probability (capped at 1) to be seen reliably.
- **Selecting and confirming on different questions can fix that.** `bench/confirm.py` tests only
  the selected candidate, once, on the 40 held-out questions, scored by the reference judge:
  PROMOTE (gain +0.16 per case, one-sided Monte Carlo p = 0.015), the same decision ground truth
  gives. Retrospective: it was run after the held-out results were seen, so it shows the
  procedure, not a prospective test of it. The certified
  judge agreed with ground truth on all 160 held-out replies.
- **The reference judge found a bug in my ground truth.** It passed "−27 pens" (a U+2212 minus)
  for an expected -27, which my checker had failed. The checker is fixed and tested; no row of the
  baseline changed.

The numbers above are from a re-run over the call cache after that fix, with no new calls. A few
training-set judge rates differ from the first run by about 0.01: identical replies share one
cache entry, and the first run had judged them separately, not always alike.


## Limits

- **The automatic controls can be wrong for your task.** `mismatched_output` assumes another
  case's answer is wrong for this one; it is not for "name a prime" (another case's "3" is a
  prime too). Donors are drawn only from cases with a different expected answer, which helps but
  does not prove the donor wrong. Whitespace changes are not meaning-preserving for code or YAML
  answers, and an empty answer satisfies a rubric that only forbids something. `certify_dataset`
  guesses a rubric's kind from its words. For anything but plain prose answers, pass the controls
  that fit your rubric (`Rewrite`, `MismatchedOutput(kind=...)`).
- **The canary is a negative-control sentinel, not a drift detector.** It checks that the judge
  still fails an empty answer on live traffic. A judge that fails everything passes it; its
  health is a rolling window, not a time-uniform test; and requests from one user are treated
  as independent.
- **A certificate covers one judge configuration, as far as its identity can see.** The identity
  records the rubric, the model by name, what the judge is shown, its settings and the
  pydantic-evals version; for any other evaluator, its class and module, its fields (a dict's
  order included), what it stores outside them, and its class's code: the methods its author
  wrote (bytecode, defaults, closed-over values), its class attributes and its metaclass (so also
  the Python version); for a judge built on a Python function, the function's code and the plain
  values it closes over; judges inside judges by their own identities. Python objects can hide
  behaviour in more places than any list covers: seven adversarial review rounds each found a
  place this missed, each now closed with a regression test (DESIGN.md), and an eighth may find
  another. Inputs whose code cannot be pinned down get cache keys that never match, so they are
  judged afresh every time. It does not follow
  what a function reaches through module globals (a helper, a prompt file), and it cannot see
  outside the judge: a provider adapter that reorders the output schema, as this repository's
  did, changes the judge without changing its identity. What it captures but cannot identify by value
  (captured lists, dicts or objects, a function held as a setting, a function-backed model with no
  name of its own, an evaluator that is not a dataclass) is marked opaque, and `identity_reliable`
  is then False. Nothing that reuses evidence trusts such a judge: the cache and the pytest plugin
  certify it afresh and never save it, a jury refuses to borrow its certificate, the report
  evaluator calls its coverage unverifiable, and `qualify` warns that its certificate holds only
  for the very object it was made from. A model object with a name of its own is identified by
  that name; give it one that changes when it does.
- **The evidence comes from synthetic tasks and Codex models**, through one CLI adapter: support,
  arithmetic, evidence, conversation, temporal and retry tasks, plus Pydantic's own example dataset.
  Most real runs are small (4 to 40 cases), so many of their certificates are UNVALIDATED by size.
  Native providers, human-adjudicated cases and larger runs would test it further.
