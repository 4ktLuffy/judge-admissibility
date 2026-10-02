# Why it works this way

Every choice below was either forced by a measurement in this repository or made to avoid a
failure one would have caused. Each answer says which.

## The idea

**Why certify a judge at all?** A judge that says "pass" to everything scores 100% on a dataset
of good answers, and so does a judge that works. Eval datasets are made of good answers, so the
report cannot tell the two apart. The only way to tell is to show the judge answers whose right
verdict is already known, including ones it must fail. That is what a control is.

**Why two checks that look like opposites (acceptance and rejection)?** Each alone is passed by
a constant judge. "Always pass" clears acceptance; "always fail" clears rejection. Only both
together rule out a judge that is not reading the answer. The real default `LLMJudge` measured
here failed 40 of 40 right answers, and it would have passed a rejection-only test.

## The statistics

**Why Wilson intervals instead of the rate?** Ten passes out of ten is a rate of 1.0, but it is
also what a judge that is right 75% of the time produces about one run in twenty. A Wilson interval
says how low the true rate could plausibly be: 10/10 has a lower bound of 0.72. Wilson rather
than the textbook normal interval because the normal one breaks near 0 and 1, which is exactly
where a good judge lives.

**Why three outcomes (PASS, FAIL, UNVALIDATED) instead of two?** Because "not enough evidence"
is not the same as "bad". The first version failed any check whose lower bound missed the bar,
and so it failed a perfect judge on 12 cases (12/12, lower bound 0.76). Now a check fails only
when its *upper* bound is below the bar, which is evidence the judge is bad, and otherwise says
how many cases would settle it (`diagnose` computes this: 16 for 12/12 at a 0.8 bar).

**Why does the gate use a sign-flip permutation test and not the bootstrap interval?** The
first version decided with a bootstrap interval. The gate's own A/A test, comparing identical
prompts, caught it rejecting 6% of the time against a 5% budget: with 40 binary cases the
bootstrap interval is too narrow. The sign-flip test is exact when the two versions are the same
(each case's gain is then as likely negative as positive), so its false-call rate is what it says
it is. Measured after the change: 2.5% false promote, 1% false reject over 200 trials.

**Why is the comparison paired?** Each case is compared with itself. A hard question drags the
old and the new prompt down together instead of adding noise to the difference, so the same
number of cases can see a smaller real gain.

**Why divide the significance level by the number of candidates?** An optimizer that tries five
prompts and keeps any that passes a 5% test gets several chances to be fooled. With the level
split five ways, the chance of promoting any of five identical prompts fell from 9/200 to 3/200.

**Why select on one set of questions and confirm on another?** Splitting the level five ways
makes each test stricter, and at 40 questions it missed a real 14-point gain. Testing only the
one selected prompt, once, on questions it was not chosen on, needs no split. It promoted the
right prompt at p = 0.015, the same decision ground truth gives.

**Why was the regression guard switched off by default?** It rejected a candidate when more
than 10% of cases got worse. With two noisy runs per case a truly better prompt still shows
chance drops on many cases, and the guard rejected a prompt that was 0.1 better 13 times in 50.
It stays available for scores that are not noisy.

**What is `detectable_gain` for?** To say, before anything runs, whether the dataset can see the
improvement you are hoping for. On the real baseline here, 40 questions with 2 runs each reliably
see only a 25-point shift in each case's success probability (capped at 1, so the mean gain is
smaller). A 5-point "improvement" on such a set is not evidence of anything.

**Why is the case the unit, and not the judgment?** Judging a case again tells you about the
judge's noise, not about another case. An external review showed the cost of pooling them: a
judge right on 16 of 20 cases, judged ten times each, went from 16/20 (UNVALIDATED) to 160/200
(ADMISSIBLE) without seeing a new case, and six slices of ten cases judged three times each
failed a sound judge 25% of the time. `acceptance` and `slices` now use each case's first
judgment; the repeats feed `stability`, which is about exactly that noise. Comparison judges
count one presentation per pair, the better answer first in alternate pairs.

**Why is each control family decided on its own?** A pooled rejection rate measures a mixture.
Nine families rejected perfectly and one never rejected pool to 270/300, which passes. The
certificate claims the judge rejects each kind of wrong answer, so each family must pass.

**Why share one error budget across the FAIL decisions, and not the PASS ones?** A certificate is
ADMISSIBLE only if every check passes: being wrongly admitted needs every check to be wrong at
once, so a PASS needs no correction. It is INADMISSIBLE if any one check fails, so each check and
family is another chance to fail a sound judge by luck; their FAILs split one 2.5% tail.

**Why is an errored judgment missing, not a failure?** Scoring a timeout as a failure fails a sound
judge on a slow day; dropping it silently lets a judge that errors on the hard cases pass on the
easy ones. So errors are left out of the rate and counted, and above 10% the check is
UNVALIDATED. The same applies to human agreement, whose kappa is now decided on an interval that
resamples cases, since a case's labelled outputs are not independent.

## The controls

**Why must a borrowed answer have a different expected answer?** `mismatched_output` uses
another case's correct answer as a wrong one. On a task whose answers are often "yes" or "no",
another case's "yes" is right half the time, and a sound judge that passes it is not at fault.
When cases carry `expected_output`, the donor's must differ.

**Why does the live canary use empty answers by default and not borrowed ones?** Live traffic has
no known answers to compare. Measured from cached replies (`bench/canary_borrow.py`): a borrowed
answer from a question of the same kind was in fact correct 20% of the time for weekdays and 27%
for letter counts. The
canary marked a sound judge down for agreeing with a correct answer. Empty answers are always
wrong.

**Why can the canary check every n-th call instead of sampling?** At a 25% rate a fixed seed gave
2 checks in 40 calls, a 0.1% draw. A monitor that must see a known number of checks per window
should not depend on luck for it.

**Why ask a comparison judge both ways round?** Because the order can decide, and you cannot know
in advance whether it will. Asked for its choice before its reason, the Codex comparison judge
chose whichever answer came first in 68% of presentations; asked for its reason first, the same
judge on the same pairs showed no preference (48%). Asking both ways and answering only when the
two agree raised accuracy from 0.85 to 0.90 at the cost of abstaining on 6 of 54 pairs.

**Why must the verdict come after the reason in a judge's output?** Structured output is written
in schema order. A judge without reasoning asked for `pass` (or `choice`) first commits before it
has written any reasoning. Measured by accident, through a bug in this repository's Codex adapter
that sorted the schema's keys: on the support task the no-reasoning judge passed 70 of 80 good
answers verdict-first and 80 of 80 reason-first, and the comparison judge's accuracy went from
0.53 to 0.85 on the arithmetic pairs. `LLMJudge` and `PairwiseJudge` both put the reason first; a provider adapter or a
custom output type that reorders it undoes that silently. Judges with reasoning were unaffected:
they reached the same verdicts in both orders.

## When the controls or the data are the problem

**Why must controls be chosen for the rubric?** The default controls assume the rubric is about
correctness, where another case's answer is wrong. Under a style rubric it is just as well
written, and a sound judge passes it. With the defaults, a sound style judge in the tests is
INADMISSIBLE (0 of 40 borrowed answers rejected); with `MismatchedOutput(kind='must_hold')` and a
`Rewrite` control that breaks the rubric itself, it is ADMISSIBLE.

**Why does the doctor check the data before the judge?** On pydantic-ai's example dataset, two
judges of different strength failed the same three "good" answers on every repeat, and their
reasons held up: those answers do not meet the rubric as written. A noisy judge fails good answers
at random; one that fails the same few every time and passes most others every time is being
specific. Then the advice is to read those answers first, and the doctor stops drawing
conclusions (noise, cases needed) from numbers those answers feed into. It does not blame the data
when the judge passes almost nothing, because a dataset is not wrong everywhere.

**Why does the doctor stop blaming an answer the judge passed when reformatted?** `certify_dataset` on
the same dataset found a fourth answer the judge failed every time: *"No timeframe could be
inferred from your request."*, failed as "not second person". With its spaces doubled the judge
passed it (in the first run citing "your request"; a second run repeated both). A whitespace change is a must-hold control: it means the same
answer, so passing it contradicts the three failures. Consistency over repeats points at the data
only when nothing the judge did contradicts it. A contradiction does not say which verdict was
wrong, only that the judge is inconsistent on that answer, so the doctor reports it as the judge's
problem to look at, not as bad data. Another case's answer (`mismatched_output`) does not count, because it is a different
answer.

**Why does `certify_dataset` change only prose fields?** A structured output holds timestamps
and ids next to its explanation. Emptying or re-spacing the whole thing would test whether the
judge notices a broken timestamp, not whether it reads the text the rubric is about, and a
"whitespace only" change to an id is not whitespace only. So the controls touch strings with a
space in them and leave the rest as it was.

**Why slice a certificate by kind of case?** An overall rate is an average over kinds of case,
and a judge can be wrong on every case of one kind while the average passes. Measured, through
the field-order bug: two no-reasoning judges accepted 70 of 80 good answers, comfortably over the
bar, and none of the 5 cases whose correct answer was "no" to a return-window question. `slice_by` is opt-in
because only you know which kinds matter. Its `PASS` is weaker than the other checks': it means no
slice is shown to be below the bar, not that each slice is shown above it. To certify one kind,
certify its cases alone.

**Why an exact interval for slices, when the other checks use Wilson?** Slices are small and
many, and each is a chance to fail a sound judge. With Wilson intervals, a judge exactly at the
bar on six slices of ten fails the check 6.2% of the time against a 5% budget, and up to 8.2% in
other layouts. With exact Clopper-Pearson intervals, split across the slices, it is at most 2.5%
in every layout computed (2 to 12 slices of 10 to 40 cases, `bench/slice_error.py`), and a slice
the judge always gets wrong is still caught every time. The price: a slice at 0.2 is caught 68%
of the time with 10 cases instead of 88%.

**Why change the evidence and not the answer?** Every other control changes the output: a
borrowed answer, an empty one. A judge that grades the agent's claim passes those tests whenever
the claim is well written. Changing a tool result while keeping the reply word for word isolates
one question: does the verdict depend on what actually happened? Shown only the reply, a Codex
judge passed half the false "refund issued" claims; shown the tool calls, none.

**Why group the cases a person labels by the judge's verdict?** Because the judge, right or
wrong, sorts cases by how likely they are to have changed. Sampling every group at random keeps
the estimate unbiased whatever the judge's mistakes; weighting each group by its size gives the
dataset's true gain. Labelling more densely where the judge saw a difference sounded better and
was not, in simulation, when the judge's mistakes were spread evenly; it stays an option, not
the default. On a small real dataset grouping did not help either (28.8 labels against 26.1),
which the README reports next to the simulation where it did.

## The sequential certificate

**Why stop early only for failure?** A broken judge shows it in the first batch; a sound one has
to be watched to the end to earn a PASS. Stopping early for success too was tried and dropped: it
certified sound judges less often for a small saving (the variant is gone, so no numbers are
claimed for it). Stopping only for failure gives the same verdict as judging everything in 389
of 400 paired runs, at 29% of the calls for a broken judge; in the other 11 a borderline-bad judge
was UNVALIDATED instead of INADMISSIBLE, the cost of paying for the looks.

**Why are the looks stricter?** Looking after every batch and stopping when the evidence looks
bad gives a sound judge several chances to look bad by luck. Every look's FAIL, the last
included, uses an exact interval widened for the number of looks (Bonferroni). An earlier version
used the normal interval at the last look; the review computed that this spent the budget twice
(4.1% false FAILs at the bar against 2.2% judging all at once). Splitting the budget with Wilson
intervals still leaked (3.4%), because Wilson undercovers at these sizes; with exact bounds it is
at most 2.2% in every layout computed (`bench/sequential_error.py`). A PASS, only possible at the
end, uses the usual Wilson interval.

## Mistakes that shaped it

Every one of these was caught by a control or a check; three had already been reported publicly, and
were corrected where they were reported:

- A Codex setting that silently ignored the judge's instructions made the default `LLMJudge` look
  like it passed wrong answers. With the instructions delivered, it fails right answers instead.
- The Codex adapter sorted the output schema's keys, so judges gave the verdict before the reason.
  It made a no-reasoning judge fail every correct "no" to a return-date question while its own
  reasons said the answer was right, and it produced a 68% position bias that this README
  reported. Flagged by slicing a certificate by kind of case; every affected result was re-run.
- An external review (Codex `gpt-6-astra`) found that repeats were counted as if they were new
  cases, which narrowed every interval; that pooled control families could hide one that always
  failed; that the last sequential look spent the error budget a second time; and that errored
  judgments were scored as failures. Each is fixed with a regression test, and every saved
  certificate was decided again (`bench/recertify_all.py`). No headline verdict changed; one
  other did (Pydantic's example judge without reasoning, INADMISSIBLE to UNVALIDATED on 10 cases).
- A ground-truth checker failed "−27" written with a Unicode minus. The certified judge was
  right and the checker was wrong.
- The gate's bootstrap made too many false calls; the regression guard rejected real gains; the
  canary's borrowed answers were sometimes right; a test was flaky because Logfire's traffic split
  depends on the variable's name. Each is fixed and its fix measured.
