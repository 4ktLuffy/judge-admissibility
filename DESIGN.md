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
see only a 25-point gain. A 5-point "improvement" on such a set is not evidence of anything.

## The controls

**Why must a borrowed answer have a different expected answer?** `mismatched_output` uses
another case's correct answer as a wrong one. On a task whose answers are often "yes" or "no",
another case's "yes" is right half the time, and a sound judge that passes it is not at fault.
When cases carry `expected_output`, the donor's must differ.

**Why does the live canary use empty answers by default and not borrowed ones?** Live traffic has
no known answers to compare. Measured from cached replies: a borrowed answer from a question of
the same kind was in fact correct 20% of the time for weekdays and 27% for letter counts. The
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
0.53 to 0.85. `LLMJudge` and `PairwiseJudge` both put the reason first; a provider adapter or a
custom output type that reorders it undoes that silently. Judges with reasoning were unaffected:
their certificates came out the same in both orders.

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

**Why does the doctor clear an answer the judge passed when reformatted?** `certify_dataset` on
the same dataset found a fourth answer the judge failed every time: *"No timeframe could be
inferred from your request."*, failed as "not second person". With its spaces doubled the judge
passed it, citing "your request". A whitespace change is a must-hold control: it means the same
answer, so passing it contradicts the three failures. Consistency over repeats points at the data
only when nothing the judge did contradicts it; a contradicted answer is reported as the judge's
mistake. Another case's answer (`mismatched_output`) does not count, because it is a different
answer.

**Why does `certify_dataset` change only prose fields?** A structured output holds timestamps
and ids next to its explanation. Emptying or re-spacing the whole thing would test whether the
judge notices a broken timestamp, not whether it reads the text the rubric is about, and a
"whitespace only" change to an id is not whitespace only. So the controls touch strings with a
space in them and leave the rest as it was.

**Why slice a certificate by kind of case?** An overall rate is an average over kinds of case,
and a judge can be wrong on every case of one kind while the average passes. Measured, through
the field-order bug: two no-reasoning judges accepted 70 of 80 good answers, comfortably over the
bar, and 0 of the 10 that were a correct "no" to a return-window question. `slice_by` is opt-in
because only you know which kinds matter. Its `PASS` is weaker than the other checks': it means no
slice is shown to be below the bar, not that each slice is shown above it. To certify one kind,
certify its cases alone.

**Why an exact interval for slices, when the other checks use Wilson?** Slices are small and
many, and each is a chance to fail a sound judge. With Wilson intervals, a judge exactly at the
bar on six slices of ten failed the check 6.3% of the time against a 5% budget. With exact
Clopper-Pearson intervals, split across the slices, it was at most 1.9% in every layout simulated
(2 to 12 slices of 10 to 40 cases), and a slice the judge always gets wrong was still caught every
time. The price: a slice at 0.2 is caught 68% of the time with 10 cases instead of 89%.

## The sequential certificate

**Why stop early only for failure?** A broken judge shows it in the first batch; a sound one has
to be watched to the end to earn a PASS. Stopping early for success too was tried: it certified
sound judges less often (74/100 against 87/100) to save 18% of calls. Stopping only for failure
gives the same verdict as judging everything in 400 of 400 paired runs, at 26% of the calls for a
broken judge.

**Why are early looks stricter?** Looking after every batch and stopping when the evidence looks
bad gives a sound judge several chances to look bad by luck. Each early look uses an interval
widened for the number of looks (Bonferroni). The final look uses the normal one, because a judge
that got that far is judged as if all at once.

## Mistakes that shaped it

Every one of these was caught by a control or a check before a number was reported:

- A Codex setting that silently ignored the judge's instructions made the default `LLMJudge` look
  like it passed wrong answers. With the instructions delivered, it fails right answers instead.
- The Codex adapter sorted the output schema's keys, so judges gave the verdict before the reason.
  It made a no-reasoning judge fail every correct "no" to a return-date question while its own
  reasons said the answer was right, and it produced a 68% position bias that this README
  reported. Found by slicing a certificate by kind of case; every affected result was re-run.
- A ground-truth checker failed "−27" written with a Unicode minus. The certified judge was
  right and the checker was wrong.
- The gate's bootstrap made too many false calls; the regression guard rejected real gains; the
  canary's borrowed answers were sometimes right; a test was flaky because Logfire's traffic split
  depends on the variable's name. Each is fixed and its fix measured.
