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

**Why ask a comparison judge both ways round?** Because the order can decide. With only the
final answers shown, the Codex comparison judge chose whichever came first in 68% of
presentations. Asking both ways and answering only when they agree raised accuracy from 0.77 to
0.85 where there was a signal to recover, and could not help where there was none.

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
- A ground-truth checker failed "−27" written with a Unicode minus. The certified judge was
  right and the checker was wrong.
- The gate's bootstrap made too many false calls; the regression guard rejected real gains; the
  canary's borrowed answers were sometimes right; a test was flaky because Logfire's traffic split
  depends on the variable's name. Each is fixed and its fix measured.
