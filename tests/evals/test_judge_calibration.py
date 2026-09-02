# tests/evals/test_judge_calibration.py — plumbing + arithmetic tests for
# the judge-calibration report (ADR-0027). No network, no LLM, no DB:
# `cohens_kappa`/`confusion_matrix`/`dangerous_cell_count` are checked
# against hand-computed numbers (same "check the formula before trusting the
# vibes" precedent tests/evals/test_redteam.py already set for ASR/FPR), and
# `load_calibration` is checked against small fixture jsonl files written to
# `tmp_path`.
import json

import pytest

from evals.run_judge_calibration import (
    cohens_kappa,
    confusion_matrix,
    dangerous_cell_count,
    load_calibration,
    raw_agreement,
)

# --- Cohen's kappa -----------------------------------------------------


def test_kappa_perfect_agreement_is_one():
    # Not all-one-class, so this exercises the ordinary formula path, not
    # the pe==1.0 edge case below.
    judge = [True, True, False, False]
    human = [True, True, False, False]
    assert cohens_kappa(judge, human) == 1.0
    assert raw_agreement(judge, human) == 1.0


def test_kappa_chance_level_agreement_is_zero():
    # po=0.5, pe=0.5*0.5+0.5*0.5=0.5 -> kappa=(0.5-0.5)/(1-0.5)=0. Hand
    # picked so both raters' marginal yes-rate is 50/50 and agreement
    # exactly matches what chance alone predicts.
    judge = [True, True, False, False]
    human = [True, False, True, False]
    assert raw_agreement(judge, human) == 0.5
    assert cohens_kappa(judge, human) == pytest.approx(0.0)


def test_kappa_all_one_class_both_all_yes_no_zero_division_error():
    judge = [True, True, True, True]
    human = [True, True, True, True]
    assert cohens_kappa(judge, human) == 1.0  # doesn't raise, po==1 -> defined as 1.0


def test_kappa_all_one_class_both_all_no_no_zero_division_error():
    judge = [False, False, False]
    human = [False, False, False]
    assert cohens_kappa(judge, human) == 1.0


def test_kappa_empty_list_is_zero_not_a_crash():
    assert cohens_kappa([], []) == 0.0
    assert raw_agreement([], []) == 0.0


def test_kappa_partial_agreement_hand_computed():
    # 8 items: 6 agree, 2 disagree in the same direction (judge=yes,
    # human=no both times). po=6/8=0.75. p_judge_yes=6/8=0.75,
    # p_human_yes=4/8=0.5. pe=0.75*0.5+0.25*0.5=0.5. kappa=(0.75-0.5)/0.5=0.5.
    judge = [True, True, True, True, True, True, False, False]
    human = [True, True, True, True, False, False, False, False]
    assert raw_agreement(judge, human) == pytest.approx(0.75)
    assert cohens_kappa(judge, human) == pytest.approx(0.5)


# --- confusion matrix / dangerous cell ----------------------------------


def test_confusion_matrix_and_dangerous_cell_on_fixed_list():
    judge = [True, True, False, False, True]
    human = [True, False, True, False, False]
    cm = confusion_matrix(judge, human)
    assert cm == {
        "judge_yes_human_yes": 1,
        "judge_yes_human_no": 2,  # indices 1 and 4
        "judge_no_human_yes": 1,  # index 2
        "judge_no_human_no": 1,  # index 3
    }
    # The dangerous cell (judge says yes, human says no) is counted
    # separately, not just read off the matrix — pin both stay in sync.
    assert dangerous_cell_count(judge, human) == cm["judge_yes_human_no"] == 2


def test_dangerous_cell_zero_when_judge_never_false_positive():
    judge = [False, False, True]
    human = [False, True, True]
    assert dangerous_cell_count(judge, human) == 0


# --- load_calibration: files must line up by id -------------------------


def _write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_load_calibration_happy_path(tmp_path):
    items_path = tmp_path / "items.jsonl"
    verdicts_path = tmp_path / "judge_verdicts.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    _write_jsonl(items_path, [{"id": "a01", "question": "q", "answer": "a"}])
    _write_jsonl(verdicts_path, [{"id": "a01", "faithful": True, "relevant": True}])
    _write_jsonl(labels_path, [{"id": "a01", "faithful": True, "relevant": False, "why": "ok"}])

    items, verdicts, labels = load_calibration(items_path, verdicts_path, labels_path)
    assert set(items) == set(verdicts) == set(labels) == {"a01"}


def test_load_calibration_tolerates_a_leading_comment_line(tmp_path):
    # labels.provisional.jsonl opens with a "# PROVISIONAL ..." line
    # (ADR-0027) — the loader must skip it, not choke on invalid JSON.
    items_path = tmp_path / "items.jsonl"
    verdicts_path = tmp_path / "judge_verdicts.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    _write_jsonl(items_path, [{"id": "a01"}])
    _write_jsonl(verdicts_path, [{"id": "a01", "faithful": True, "relevant": True}])
    labels_path.write_text(
        "# PROVISIONAL — not real labels\n"
        '{"id": "a01", "faithful": true, "relevant": true, "why": "ok"}\n',
        encoding="utf-8",
    )

    items, verdicts, labels = load_calibration(items_path, verdicts_path, labels_path)
    assert set(labels) == {"a01"}


def test_load_calibration_missing_id_from_labels_raises_clear_error(tmp_path):
    items_path = tmp_path / "items.jsonl"
    verdicts_path = tmp_path / "judge_verdicts.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    _write_jsonl(items_path, [{"id": "a01"}, {"id": "a02"}])
    _write_jsonl(
        verdicts_path,
        [
            {"id": "a01", "faithful": True, "relevant": True},
            {"id": "a02", "faithful": True, "relevant": True},
        ],
    )
    _write_jsonl(
        labels_path, [{"id": "a01", "faithful": True, "relevant": True, "why": "ok"}]
    )  # a02 missing

    with pytest.raises(ValueError, match="a02"):
        load_calibration(items_path, verdicts_path, labels_path)


def test_load_calibration_extra_id_in_verdicts_raises_clear_error(tmp_path):
    items_path = tmp_path / "items.jsonl"
    verdicts_path = tmp_path / "judge_verdicts.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    _write_jsonl(items_path, [{"id": "a01"}])
    _write_jsonl(
        verdicts_path,
        [
            {"id": "a01", "faithful": True, "relevant": True},
            {"id": "extra99", "faithful": True, "relevant": True},
        ],
    )
    _write_jsonl(labels_path, [{"id": "a01", "faithful": True, "relevant": True, "why": "ok"}])

    with pytest.raises(ValueError, match="extra99"):
        load_calibration(items_path, verdicts_path, labels_path)


def test_load_calibration_null_label_raises_clear_error(tmp_path):
    items_path = tmp_path / "items.jsonl"
    verdicts_path = tmp_path / "judge_verdicts.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    _write_jsonl(items_path, [{"id": "a01"}])
    _write_jsonl(verdicts_path, [{"id": "a01", "faithful": True, "relevant": True}])
    _write_jsonl(labels_path, [{"id": "a01", "faithful": None, "relevant": True, "why": ""}])

    with pytest.raises(ValueError, match="a01"):
        load_calibration(items_path, verdicts_path, labels_path)
