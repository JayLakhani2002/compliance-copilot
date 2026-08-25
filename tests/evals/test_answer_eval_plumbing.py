# tests/evals/test_answer_eval_plumbing.py — plumbing tests for the
# answer-quality gate (ADR-0017). No network, no real LLM: judge() is
# exercised against a fake chat-model double (same minimal `with_structured_
# output(...).invoke(...)` contract graph/nodes.py's answer_node depends
# on), and the aggregation/gate math is checked with plain constructed
# QuestionOutcome rows — no golden set, no DB, no API call.
from compliance_copilot.graph.state import AnswerSchema, CitationError
from evals.judge import JudgeVerdict, judge
from evals.run_answer_eval import (
    GoldenAnswer,
    QuestionOutcome,
    _outcome_from_citation_error,
    _outcome_from_verdict,
    aggregate,
)


class _FakeJudgeLLM:
    """Test double for the chat model `judge()` takes: `.with_structured_
    output(schema, method=...)` returns something whose `.invoke(messages)`
    hands back a preset verdict, regardless of the prompt — the same shape
    `ChatOpenAI(...).with_structured_output(...)` returns for real."""

    def __init__(self, verdict: JudgeVerdict) -> None:
        self._verdict = verdict
        self.last_messages = None

    def with_structured_output(self, schema, method="json_schema"):  # noqa: ARG002
        return self

    def invoke(self, messages):
        self.last_messages = messages
        return self._verdict


def test_judge_returns_the_fake_llms_preset_verdict():
    preset = JudgeVerdict(faithful=True, relevant=False, reasoning="answers a different question")
    fake_llm = _FakeJudgeLLM(preset)
    answer = AnswerSchema(answer="Some answer.", citations=[])

    result = judge(
        "a question", answer, contexts=["some context"], reference="a reference", llm=fake_llm
    )

    assert result is preset
    assert fake_llm.last_messages is not None


def _outcome(id_: str, faithful: bool, relevant: bool = True) -> QuestionOutcome:
    return QuestionOutcome(
        id=id_,
        faithful=faithful,
        relevant=relevant,
        n_citations=1,
        note="",
        is_citation_error=False,
    )


def test_aggregate_faithfulness_gate_math_9_of_10_passes_at_0_9():
    outcomes = [_outcome(str(i), faithful=i != 0) for i in range(10)]  # 9/10 faithful
    aggregates = aggregate(outcomes)
    assert aggregates["faithfulness"] == 0.9
    assert aggregates["faithfulness"] >= 0.9  # the gate check run_answer_eval.main() applies


def test_aggregate_faithfulness_gate_math_8_of_10_fails_at_0_9():
    outcomes = [_outcome(str(i), faithful=i not in (0, 1)) for i in range(10)]  # 8/10 faithful
    aggregates = aggregate(outcomes)
    assert aggregates["faithfulness"] == 0.8
    assert aggregates["faithfulness"] < 0.9


def test_citation_error_counts_as_faithful_and_relevant_false():
    golden = GoldenAnswer(id="x01", question="q", expected_anchors=["ai_act:art_1"], reference="r")
    outcome = _outcome_from_citation_error(golden, CitationError("bad citation"))

    assert outcome.faithful is False
    assert outcome.relevant is False
    assert outcome.is_citation_error is True
    assert outcome.note.startswith("CitationError:")

    aggregates = aggregate([outcome])
    assert aggregates["citation_error_rate"] == 1.0
    assert aggregates["faithfulness"] == 0.0


def test_outcome_from_verdict_carries_citation_count_and_reasoning():
    answer = AnswerSchema(
        answer="x",
        citations=[{"regulation": "ai_act", "anchor": "art_6", "quote": "q" * 25}],
    )
    verdict = JudgeVerdict(faithful=True, relevant=True, reasoning="looks right")
    golden = GoldenAnswer(id="x02", question="q", expected_anchors=["ai_act:art_6"], reference="r")

    outcome = _outcome_from_verdict(golden, answer, verdict)

    assert outcome.n_citations == 1
    assert outcome.note == "looks right"
    assert outcome.is_citation_error is False
