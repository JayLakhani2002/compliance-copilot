# tests/evals/test_cost_report_plumbing.py — plumbing tests for the cost
# report (ADR-0029). No network, no real LLM, no DB: `estimate_cost` is
# exercised against fixed usage dicts (same idiom test_answer_eval_
# plumbing.py already uses for `judge()`/`aggregate()`), and `aggregate`/
# `print_report` are checked against hand-built `QuestionCost` rows.
from compliance_copilot.costing import PRICES, estimate_cost
from compliance_copilot.graph.nodes import _DEFAULT_MODELS as _ANSWER_MODELS
from compliance_copilot.guards.classifier import _DEFAULT_MODELS as _CLASSIFIER_MODELS
from compliance_copilot.router import _DEFAULT_MODELS as _ROUTER_MODELS
from compliance_copilot.settings import settings
from evals.run_cost_report import QuestionCost, aggregate, print_report


def test_estimate_cost_basic_math_no_cache():
    # 1,000,000 in / 1,000,000 out at gpt-4.1-mini's $0.40/$1.60 per MTok.
    usage = {"gpt-4.1-mini": {"input_tokens": 1_000_000, "output_tokens": 1_000_000}}
    result = estimate_cost(usage)
    assert result["usd_total"] == 0.40 + 1.60
    assert result["eur_total"] == (0.40 + 1.60) * settings.eur_usd_rate
    assert result["by_model"]["gpt-4.1-mini"]["cached_tokens"] == 0


def test_estimate_cost_honours_cached_input_discount():
    # 1,000,000 input tokens, half of them cached, at gpt-4.1-nano's
    # $0.10 standard / $0.025 cached rate per MTok. No output tokens.
    usage = {
        "gpt-4.1-nano": {
            "input_tokens": 1_000_000,
            "output_tokens": 0,
            "input_token_details": {"cache_read": 500_000},
        }
    }
    result = estimate_cost(usage)
    expected = (500_000 / 1_000_000) * 0.10 + (500_000 / 1_000_000) * 0.025
    assert result["usd_total"] == expected
    assert result["by_model"]["gpt-4.1-nano"]["cached_tokens"] == 500_000


def test_estimate_cost_embeddings_have_no_output_price():
    usage = {"text-embedding-3-small": {"input_tokens": 1_000_000, "output_tokens": 0}}
    result = estimate_cost(usage)
    assert result["usd_total"] == 0.02


def test_estimate_cost_unpriced_model_raises_keyerror():
    try:
        estimate_cost({"gpt-9000-turbo": {"input_tokens": 1, "output_tokens": 1}})
    except KeyError as exc:
        assert "gpt-9000-turbo" in str(exc)
    else:
        raise AssertionError("expected KeyError for an unpriced model")


def test_prices_covers_every_model_make_llm_and_friends_can_produce():
    # Every model id the app's own per-provider default tables can hand to
    # a chat-model constructor must have a PRICES row, or a real run would
    # crash on a legitimate, already-shipped model choice.
    all_model_ids = set(_ANSWER_MODELS.values()) | set(_CLASSIFIER_MODELS.values())
    all_model_ids |= set(_ROUTER_MODELS.values())
    all_model_ids.add(settings.embedding_model)
    missing = all_model_ids - set(PRICES)
    assert not missing, f"PRICES is missing rows for: {missing}"


def _cost(
    id_: str, eur: float, input_tokens: int, cached_tokens: int, by_model: dict | None = None
) -> QuestionCost:
    return QuestionCost(
        id=id_,
        usd=eur / settings.eur_usd_rate,
        eur=eur,
        input_tokens=input_tokens,
        output_tokens=100,
        cached_tokens=cached_tokens,
        embedding_tokens=20,
        degraded=False,
        by_model=by_model or {},
    )


def test_aggregate_computes_eur_per_100_and_cached_fraction():
    costs = [_cost("q1", 0.01, 1000, 500), _cost("q2", 0.02, 1000, 0)]
    result = aggregate(costs)
    assert result["n"] == 2
    assert result["eur_per_question"] == 0.015
    assert result["eur_per_100"] == 1.5
    assert result["cached_fraction"] == 500 / 2000


def test_aggregate_empty_list_is_zero_not_a_crash():
    result = aggregate([])
    assert result == {
        "n": 0,
        "eur_per_question": 0.0,
        "eur_per_100": 0.0,
        "cached_fraction": 0.0,
        "avg_embedding_tokens": 0.0,
        "n_degraded": 0,
        "by_model": {},
    }


def test_aggregate_pools_dated_snapshot_ids_under_the_bare_model_name():
    # Two questions, both answered by a dated gpt-4.1-mini snapshot — a
    # real OpenAI response shape (costing.py's own `_price_key` comment) —
    # must land in ONE "gpt-4.1-mini" row, not two separate snapshot rows.
    row = {"usd": 0.001, "input_tokens": 100, "output_tokens": 50, "cached_tokens": 20}
    costs = [
        _cost("q1", 0.001, 100, 20, by_model={"gpt-4.1-mini-2025-04-14": dict(row)}),
        _cost("q2", 0.001, 100, 20, by_model={"gpt-4.1-mini-2025-04-14": dict(row)}),
    ]
    result = aggregate(costs)
    assert set(result["by_model"]) == {"gpt-4.1-mini"}
    assert result["by_model"]["gpt-4.1-mini"]["input_tokens"] == 200


def test_print_report_runs_without_raising_on_a_degraded_row(capsys):
    costs = [_cost("q1", 0.01, 1000, 0)]
    costs[0].degraded = True
    print_report(costs, aggregate(costs))
    out = capsys.readouterr().out
    assert "degraded" in out
