"""Router accuracy >= 90% on the labelled set of 30 questions (rules router always; LLM router when available)."""
import pandas as pd
import pytest

from bikeops.common import ROOT
from bikeops.desk import llm, router

LABELS = pd.read_csv(ROOT / "tests" / "router_labels.csv")


def test_labelled_set_size():
    assert len(LABELS) == 30 and set(LABELS.intent) == set(router.INTENTS)


def test_rule_router_accuracy():
    acc = (LABELS.question.map(router.rule_route) == LABELS.intent).mean()
    assert acc >= 0.9, acc


@pytest.mark.parametrize("provider", ["anthropic", "gemini", "ollama"])
def test_llm_router_accuracy(provider):
    if not llm.available(provider):
        pytest.skip(f"{provider} not configured")
    pred = [router.route(q, provider)[0] for q in LABELS.question]
    assert (pd.Series(pred) == LABELS.intent).mean() >= 0.9
