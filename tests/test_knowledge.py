import pytest

from clinic_voice_agent.faq_eval import evaluate, write_report
from clinic_voice_agent.knowledge import (
    BM25Retriever,
    Passage,
    load_passages,
    load_retriever,
    slugify,
    tokenize,
)


@pytest.fixture
def tiny():
    return BM25Retriever(
        [
            Passage("hours", "Opening hours", "We are open Monday to Friday from 8 to 18."),
            Passage("parking", "Parking", "Street parking is available next to the entrance."),
            Passage("vaccinations", "Vaccinations", "The nurse gives flu and travel vaccines."),
        ]
    )


def test_tokenize_drops_stopwords_and_normalises():
    assert tokenize("Where can I park my car?") == ["park", "park"]  # car -> parking -> stem "park"
    assert tokenize("Buses and flu shots") == ["bus", "flu", "vaccin"]
    assert tokenize("glasses") == ["glass"]


def test_search_ranks_the_matching_passage_first(tiny):
    assert tiny.search("is there parking for my car")[0].passage.id == "parking"
    assert tiny.search("can I get a flu jab")[0].passage.id == "vaccinations"
    assert tiny.search("when are you open")[0].passage.id == "hours"


def test_search_returns_nothing_for_unrelated_question(tiny):
    assert tiny.search("concert tickets") == []


def test_min_score_filters_weak_hits(tiny):
    hits = tiny.search("flu", k=3)
    assert len(hits) == 1
    assert tiny.search("flu", min_score=hits[0].score) == []


def test_load_passages_splits_sections(tmp_path):
    kb = tmp_path / "kb.md"
    kb.write_text("# Title\nintro ignored\n## First topic\nline one\nline two\n\n## Second\nx\n")
    passages = load_passages(kb)
    assert [p.id for p in passages] == ["first-topic", "second"]
    assert passages[0].text == "line one line two"


def test_duplicate_titles_rejected(tmp_path):
    kb = tmp_path / "kb.md"
    kb.write_text("## Same\na\n## Same\nb\n")
    with pytest.raises(ValueError, match="duplicate"):
        load_passages(kb)


def test_shipped_knowledge_base_is_consistent():
    retriever = load_retriever()
    assert len(retriever.passages) >= 20
    assert slugify("Out-of-hours care") == "out-of-hours-care"
    assert retriever.search("what are your opening hours")[0].passage.id == "opening-hours"


def test_evaluate_and_report(tiny, tmp_path):
    questions = [("flu jab please", "vaccinations"), ("parking", "parking"), ("tickets", None)]
    results, summary = evaluate(tiny, questions)
    assert summary["hit@1"] == 1.0
    assert summary["thresholds"][0]["unanswerable_rejected"] == 1.0
    write_report(results, summary, tmp_path)
    assert (tmp_path / "faq_retrieval.md").read_text().startswith("# FAQ retrieval")


def test_unknown_gold_id_raises(tiny):
    with pytest.raises(ValueError, match="not in knowledge base"):
        evaluate(tiny, [("x", "missing-id")])
