# tests/test_graph.py — unit tests for the LangGraph graph
# (src/compliance_copilot/graph/, ADR-0014). No network, no DB, no API key:
# `FakeLLM` stands in for `runtime.context.llm` (a real `ChatAnthropic` +
# `with_structured_output` can't be faked with LangChain's own fake chat
# models — see nodes.py's `make_llm` and ADR-0014's references — so this is
# a tiny hand-written double implementing just `.invoke(messages) ->
# AnswerSchema`), and monkeypatching `compliance_copilot.graph.nodes.retrieve`
# (the retriever function nodes.py imports, not the graph node) replaces the
# DB-backed lookup with hand-made `RetrievedChunk`s. This lets the whole
# compiled graph run end-to-end with `session=None, embeddings=None`.
import pytest

from compliance_copilot.graph import AnswerSchema, Citation, CitationError, GraphContext
from compliance_copilot.graph.build import build_graph
from compliance_copilot.retriever import RetrievedChunk

ARTICLES = [
    RetrievedChunk(
        anchor="art_6",
        regulation="ai_act",
        kind="article",
        number=6,
        title="Classification rules for high-risk AI systems",
        text="An AI system shall be considered high-risk where it is a safety component.",
        distance=0.1,
        part=0,
    ),
    RetrievedChunk(
        anchor="art_9",
        regulation="ai_act",
        kind="article",
        number=9,
        title="Risk management system",
        text="A risk management system shall be established for high-risk AI systems.",
        distance=0.2,
        part=0,
    ),
]

RECITALS = [
    RetrievedChunk(
        anchor="rct_1",
        regulation="ai_act",
        kind="recital",
        number=1,
        title=None,
        text="This Regulation aims to improve the functioning of the internal market.",
        distance=0.15,
        part=0,
    ),
]


class FakeLLM:
    """Stands in for `runtime.context.llm` — the only contract answer_node()
    depends on is `.invoke(messages) -> AnswerSchema` (see nodes.py's
    module docstring and ADR-0014)."""

    def __init__(self, response: AnswerSchema):
        self._response = response
        self.messages: list[tuple[str, str]] | None = None

    def invoke(self, messages):
        self.messages = messages  # captured for the prompt-content assertions
        return self._response


def _fake_retrieve(question, k, *, kinds, session, embeddings):
    """Replaces `compliance_copilot.retriever.retrieve` as imported into
    nodes.py — returns the module-level ARTICLES/RECITALS fixtures instead
    of querying a real DB, keyed on the same `kinds` filter retrieve_node
    passes (ADR-0013's article/recital split)."""
    return ARTICLES if kinds == ("article",) else RECITALS


@pytest.fixture(autouse=True)
def patch_retrieve(monkeypatch):
    monkeypatch.setattr("compliance_copilot.graph.nodes.retrieve", _fake_retrieve)


def _run(llm: FakeLLM, question: str = "What is a high-risk AI system?"):
    graph = build_graph()
    context = GraphContext(session=None, embeddings=None, llm=llm)
    return graph.invoke({"question": question}, context=context)


def test_happy_path_returns_answer_with_retrieved_context():
    answer = AnswerSchema(
        answer="A high-risk AI system is one used as a safety component.",
        citations=[Citation(regulation="ai_act", anchor="art_6", quote="is a safety component")],
    )
    state = _run(FakeLLM(answer))

    assert len(state["articles"]) == 2
    assert len(state["recitals"]) == 1
    assert state["answer"].citations[0].anchor == "art_6"


def test_citation_to_anchor_not_retrieved_raises_citation_error():
    answer = AnswerSchema(
        answer="...",
        citations=[Citation(regulation="ai_act", anchor="art_99", quote="anything")],
    )
    question = "a very specific question that must not leak into the error"
    with pytest.raises(CitationError) as excinfo:
        _run(FakeLLM(answer), question=question)

    message = str(excinfo.value)
    assert "art_99" in message
    assert question not in message


def test_citation_to_recital_anchor_raises_citation_error():
    """Recitals are retrieved (as context) but must never be citable —
    ADR-0013/ADR-0014's "articles are the only citation source" rule."""
    answer = AnswerSchema(
        answer="...",
        citations=[Citation(regulation="ai_act", anchor="rct_1", quote="internal market")],
    )
    with pytest.raises(CitationError, match="rct_1"):
        _run(FakeLLM(answer))


def test_quote_not_found_in_cited_chunk_raises_citation_error():
    answer = AnswerSchema(
        answer="...",
        citations=[Citation(regulation="ai_act", anchor="art_6", quote="something never said")],
    )
    with pytest.raises(CitationError, match="art_6"):
        _run(FakeLLM(answer))


def test_quote_with_different_whitespace_and_case_still_passes():
    """The verbatim check is whitespace-collapsed and case-folded (nodes.py's
    `_normalise`) — a real quote that only differs by spacing/case is not a
    misquote."""
    answer = AnswerSchema(
        answer="...",
        citations=[Citation(regulation="ai_act", anchor="art_6", quote="IS A  SAFETY\ncomponent")],
    )
    state = _run(FakeLLM(answer))  # must not raise
    assert state["answer"] is answer


def test_prompt_contains_article_text_recital_heading_and_question():
    answer = AnswerSchema(answer="...", citations=[])
    llm = FakeLLM(answer)
    question = "What is a high-risk AI system?"
    _run(llm, question=question)

    human_content = llm.messages[1][1]
    assert ARTICLES[0].text in human_content
    assert "do not cite" in human_content
    assert question in human_content


def test_zero_citations_with_cannot_answer_text_is_accepted():
    answer = AnswerSchema(answer="The provided excerpts do not answer this question.", citations=[])
    state = _run(FakeLLM(answer))  # must not raise
    assert state["answer"].citations == []


def test_quote_from_a_non_first_part_of_a_multi_part_anchor_passes(monkeypatch):
    """Oversize articles split into multiple parts sharing one anchor
    (chunker.py) can come back as two separate retrieved rows for the same
    (regulation, anchor). Grouping by key (not a plain last-wins dict —
    reviewer round 1) must accept a quote verbatim in *either* part."""
    parts = [
        RetrievedChunk(
            anchor="art_3",
            regulation="ai_act",
            kind="article",
            number=3,
            title="Definitions",
            text="Part zero text mentions safety component here.",
            distance=0.1,
            part=0,
        ),
        RetrievedChunk(
            anchor="art_3",
            regulation="ai_act",
            kind="article",
            number=3,
            title="Definitions",
            text="Part one text is totally different and unrelated.",
            distance=0.2,
            part=1,
        ),
    ]

    def fake_retrieve(question, k, *, kinds, session, embeddings):
        return parts if kinds == ("article",) else []

    monkeypatch.setattr("compliance_copilot.graph.nodes.retrieve", fake_retrieve)

    answer = AnswerSchema(
        answer="...",
        citations=[
            Citation(regulation="ai_act", anchor="art_3", quote="mentions safety component here")
        ],
    )
    state = _run(FakeLLM(answer))  # must not raise — quote is verbatim in part 0
    assert state["answer"] is answer


def test_empty_quote_raises_citation_error():
    answer = AnswerSchema(
        answer="...",
        citations=[Citation(regulation="ai_act", anchor="art_6", quote="")],
    )
    with pytest.raises(CitationError, match="too short"):
        _run(FakeLLM(answer))


def test_short_quote_raises_citation_error():
    answer = AnswerSchema(
        answer="...",
        citations=[Citation(regulation="ai_act", anchor="art_6", quote="a saf")],
    )
    with pytest.raises(CitationError, match="too short"):
        _run(FakeLLM(answer))


def test_curly_quote_in_source_matches_straight_quote_in_citation(monkeypatch):
    """The real corpus uses curly quotes (U+2018/U+2019) around defined
    terms (chunker.py's definition-boundary regex matches '‘' for the same
    reason); the model may reproduce a quote with straight ASCII quotes —
    that difference alone must not cause a false rejection (reviewer round
    1)."""
    curly_chunk = RetrievedChunk(
        anchor="art_3",
        regulation="ai_act",
        kind="article",
        number=3,
        title="Definitions",
        text="For the purposes of this Regulation, ‘deployer’ means a person.",
        distance=0.1,
        part=0,
    )

    def fake_retrieve(question, k, *, kinds, session, embeddings):
        return [curly_chunk] if kinds == ("article",) else []

    monkeypatch.setattr("compliance_copilot.graph.nodes.retrieve", fake_retrieve)

    answer = AnswerSchema(
        answer="...",
        citations=[
            Citation(regulation="ai_act", anchor="art_3", quote="'deployer' means a person")
        ],
    )
    state = _run(FakeLLM(answer))  # must not raise
    assert state["answer"] is answer
