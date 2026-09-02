# evals/run_judge_calibration.py — reads the three calibration files
# (evals/calibration/items.jsonl, judge_verdicts.jsonl, labels*.jsonl) and
# reports whether evals/judge.py's LLM-as-judge agrees with a careful human
# reading the same 20 items (ADR-0027, lesson 22).
#
# Per rubric dimension (`faithful`, `relevant` — scored separately, they can
# calibrate differently): raw agreement, Cohen's kappa (plain Python, no new
# dependency), the 2x2 confusion matrix, and the DANGEROUS cell count
# (judge says yes, human says no — a hallucination the gate waved through as
# clean) printed first, not buried in an aggregate. n=20 is a smoke
# calibration, not a validated instrument — that caveat is printed every
# time, never only in a docstring someone can skip past.
#
# Report-only by default: exits non-zero ONLY when --kappa-min is given and
# not met by either dimension.
#     uv run python -m evals.run_judge_calibration
#     uv run python -m evals.run_judge_calibration --kappa-min 0.6
from __future__ import annotations

import argparse
import json
from pathlib import Path

CALIBRATION_DIR = Path(__file__).parent / "calibration"
ITEMS_PATH = CALIBRATION_DIR / "items.jsonl"
VERDICTS_PATH = CALIBRATION_DIR / "judge_verdicts.jsonl"
LABELS_PATH = CALIBRATION_DIR / "labels.jsonl"
LABELS_PROVISIONAL_PATH = CALIBRATION_DIR / "labels.provisional.jsonl"

DIMENSIONS = ("faithful", "relevant")


def _load_jsonl(path: Path) -> dict[str, dict]:
    """Keyed by `id` — every calibration file (items/verdicts/labels) uses
    `id` as its join key, same convention golden_answers.jsonl/redteam.jsonl
    already use. A line starting with `#` is a comment, skipped rather than
    parsed as JSON — `labels.provisional.jsonl` opens with one marking
    itself PROVISIONAL (ADR-0027) without breaking this loader."""
    rows: dict[str, dict] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                row = json.loads(line)
                rows[row["id"]] = row
    return rows


def load_calibration(
    items_path: Path, verdicts_path: Path, labels_path: Path
) -> tuple[dict[str, dict], dict[str, dict], dict[str, dict]]:
    """Loads the three files and checks they line up by id — a missing or
    extra id in any one of them is a data-integrity bug, not something to
    silently skip past (a report computed on a mismatched subset would be
    lying about what it measured)."""
    items = _load_jsonl(items_path)
    verdicts = _load_jsonl(verdicts_path)
    labels = _load_jsonl(labels_path)

    item_ids, verdict_ids, label_ids = set(items), set(verdicts), set(labels)
    if not (item_ids == verdict_ids == label_ids):
        raise ValueError(
            "calibration files do not line up by id:\n"
            f"  in items but not verdicts: {sorted(item_ids - verdict_ids)}\n"
            f"  in items but not labels:   {sorted(item_ids - label_ids)}\n"
            f"  in verdicts but not items: {sorted(verdict_ids - item_ids)}\n"
            f"  in labels but not items:   {sorted(label_ids - item_ids)}"
        )
    for item_id, label in labels.items():
        for dim in DIMENSIONS:
            if label.get(dim) is None:
                raise ValueError(
                    f"item {item_id!r} has no human label for {dim!r} — "
                    f"{labels_path} is incomplete (labels.template.jsonl ships nulls "
                    "on purpose, they must be filled in before this report runs)"
                )
    return items, verdicts, labels


def raw_agreement(judge_vals: list[bool], human_vals: list[bool]) -> float:
    n = len(judge_vals)
    if n == 0:
        return 0.0
    return sum(j == h for j, h in zip(judge_vals, human_vals, strict=True)) / n


def cohens_kappa(judge_vals: list[bool], human_vals: list[bool]) -> float:
    """Cohen's kappa for two binary raters: (po - pe) / (1 - pe), po = raw
    agreement, pe = agreement expected by chance from each rater's own
    marginal yes-rate. Plain Python per lesson 22's "implement the formula,
    no new dependency" instruction.

    Edge case: pe == 1.0 only when both raters' marginals are fully
    concentrated on the SAME single class (both all-yes, or both all-no) —
    that forces po == 1.0 too (every item is that class for both raters), so
    (po - pe) / (1 - pe) is a genuine 0/0, not a bug. Agreement is total in
    that case, so kappa is defined as 1.0 rather than raising
    ZeroDivisionError."""
    n = len(judge_vals)
    if n == 0:
        return 0.0
    po = raw_agreement(judge_vals, human_vals)
    p_judge_yes = sum(judge_vals) / n
    p_human_yes = sum(human_vals) / n
    pe = p_judge_yes * p_human_yes + (1 - p_judge_yes) * (1 - p_human_yes)
    if pe == 1.0:
        return 1.0 if po == 1.0 else 0.0
    return (po - pe) / (1 - pe)


def confusion_matrix(judge_vals: list[bool], human_vals: list[bool]) -> dict[str, int]:
    cm = {
        "judge_yes_human_yes": 0,
        "judge_yes_human_no": 0,
        "judge_no_human_yes": 0,
        "judge_no_human_no": 0,
    }
    for j, h in zip(judge_vals, human_vals, strict=True):
        cm[f"judge_{'yes' if j else 'no'}_human_{'yes' if h else 'no'}"] += 1
    return cm


def dangerous_cell_count(judge_vals: list[bool], human_vals: list[bool]) -> int:
    """judge=yes, human=no — the categorically worse error (lesson 22): a
    hallucination the gate waved through as faithful. The opposite error
    (judge=no, human=yes) just costs a wasted rerun or a blocked PR."""
    return sum(1 for j, h in zip(judge_vals, human_vals, strict=True) if j and not h)


def print_report(
    items: dict[str, dict], verdicts: dict[str, dict], labels: dict[str, dict]
) -> list[str]:
    print("\n=== judge calibration ===\n")
    print(f"n = {len(items)} — a smoke calibration, NOT a validated psychometric instrument.")
    print("With n=20, the confidence interval around kappa is wide; treat this as")
    print("'the judge isn't obviously broken', not a final trust number.\n")

    ids = sorted(items)
    failures: list[str] = []
    for dim in DIMENSIONS:
        judge_vals = [bool(verdicts[i][dim]) for i in ids]
        human_vals = [bool(labels[i][dim]) for i in ids]

        dangerous = dangerous_cell_count(judge_vals, human_vals)
        ra = raw_agreement(judge_vals, human_vals)
        kappa = cohens_kappa(judge_vals, human_vals)
        cm = confusion_matrix(judge_vals, human_vals)

        print(f"--- {dim} ---")
        print(
            f"DANGEROUS cell (judge=yes, human=no): {dangerous} / {len(ids)}  <- check this first"
        )
        print(f"raw agreement: {ra:.3f}")
        print(f"Cohen's kappa: {kappa:.3f}")
        print(
            "confusion matrix: "
            f"yes/yes={cm['judge_yes_human_yes']} "
            f"yes/no={cm['judge_yes_human_no']} "
            f"no/yes={cm['judge_no_human_yes']} "
            f"no/no={cm['judge_no_human_no']}"
        )
        print()
        failures.append((dim, kappa))
    return failures


def _pick_labels_path(explicit: Path | None) -> tuple[Path, bool]:
    """Default `labels.jsonl` if present else `labels.provisional.jsonl`
    (the coder's own labels, ADR-0027's provisional-until-Jay-reviews
    stage) — `--labels` overrides both. Returns (path, is_provisional)."""
    if explicit is not None:
        return explicit, explicit == LABELS_PROVISIONAL_PATH
    if LABELS_PATH.exists():
        return LABELS_PATH, False
    if LABELS_PROVISIONAL_PATH.exists():
        return LABELS_PROVISIONAL_PATH, True
    raise SystemExit(f"no labels file found — expected {LABELS_PATH} or {LABELS_PROVISIONAL_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labels",
        type=Path,
        default=None,
        help="Defaults to labels.jsonl if it exists, else labels.provisional.jsonl.",
    )
    parser.add_argument(
        "--kappa-min",
        type=float,
        default=None,
        help="Fail (exit non-zero) if either dimension's kappa is below this. Unset = report only.",
    )
    args = parser.parse_args()

    labels_path, provisional = _pick_labels_path(args.labels)
    print(f"using labels: {labels_path}")
    if provisional:
        print(
            "\n!!! PROVISIONAL LABELS — these are the coder agent's own labels, not Jay's. !!!\n"
            "!!! Replace with evals/calibration/labels.jsonl before trusting this number. !!!\n"
        )

    items, verdicts, labels = load_calibration(ITEMS_PATH, VERDICTS_PATH, labels_path)
    per_dim_kappa = print_report(items, verdicts, labels)

    if args.kappa_min is not None:
        failures = [
            f"{dim}: kappa={k:.3f} below --kappa-min={args.kappa_min:.3f}"
            for dim, k in per_dim_kappa
            if k < args.kappa_min
        ]
        if failures:
            raise SystemExit("; ".join(failures) + " — failing as a CI gate")


if __name__ == "__main__":
    main()
