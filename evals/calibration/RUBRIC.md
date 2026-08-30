# Judge calibration — labelling rubric (ADR-0027)

This mirrors `evals/judge.py`'s `JUDGE_SYSTEM_PROMPT` word-for-word. A human
labelling `items.jsonl` grades against the exact same two criteria the judge
model is instructed to use — a different rubric would make "does the human
agree with the judge" an unanswerable question, since they'd be answering
different questions.

## The two criteria (verbatim from `JUDGE_SYSTEM_PROMPT`)

- **faithful**: true only if EVERY factual claim in the answer is actually
  supported by the provided context excerpts (the same text the answering
  system's citations quoted). A claim not backed by the excerpts, or a
  citation that misquotes them, makes this false.
- **relevant**: true only if the answer actually addresses the question
  asked AND agrees in substance with the reference answer given (a
  differently worded but substantively correct answer still counts as
  relevant; a partial, evasive, or substantively wrong answer does not).

Note: for the 10 benign/red-team items in this calibration set, `reference`
is empty (no hand-written reference answer exists for them) — for those
items, "relevant" collapses to "does the answer actually address the
question asked", since there's no reference to agree with.

## Blind-labelling protocol

1. **Open `items.jsonl` only.** Do not open `judge_verdicts.jsonl` yet.
2. For each item, read `question`, `answer`, `contexts`, and `reference` (if
   present). Decide `faithful` (true/false) and `relevant` (true/false)
   against the two criteria above, exactly as if you were the judge model.
3. Write your verdict plus **one line of why** into your labels file
   (`labels.jsonl`, same shape as `labels.template.jsonl`: `id`, `faithful`,
   `relevant`, `why`) — the same discipline `JudgeVerdict.reasoning` already
   forces onto the judge: name the specific gap, don't rubber-stamp.
4. **Only after every item is labelled**, open `judge_verdicts.jsonl` and
   compare. Do not go back and change a label after seeing the judge's
   verdict — that would anchor the label to the judge instead of measuring
   agreement with it. If you disagree with your own earlier label on
   reflection, note it in the `why` field rather than silently changing the
   boolean.
5. Run `uv run python -m evals.run_judge_calibration` to get raw agreement,
   Cohen's kappa, and the confusion matrix (including the dangerous
   judge-yes/human-no cell) once labelling is done.

Labelling first and revealing second is the entire point of keeping
`items.jsonl` and `judge_verdicts.jsonl` as two separate files — merging them
into one file would make blind labelling impossible to enforce.
