"""Evaluate the FAQ retriever on caller-style questions.

Metrics on answerable questions: Hit@1, Hit@3 and MRR@3. Unanswerable questions (gold `null`)
measure whether a score threshold can tell "no answer in the knowledge base" apart from a weak
match. Results are reported for several thresholds instead of picking one on this small set.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from clinic_voice_agent.config import KNOWLEDGE_DIR
from clinic_voice_agent.knowledge import BM25Retriever

DEFAULT_EVAL = KNOWLEDGE_DIR / "faq_eval.yaml"
THRESHOLDS = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0)


@dataclass
class QuestionResult:
    question: str
    gold: str | None
    ranked: list[str]
    scores: list[float]
    rank: int | None  # 1-based rank of the gold passage within the top 3, None if missed


def load_questions(path: Path | str = DEFAULT_EVAL) -> list[tuple[str, str | None]]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return [(item["q"], item["gold"]) for item in data["questions"]]


def evaluate(retriever: BM25Retriever, questions: list[tuple[str, str | None]], k: int = 3):
    known = {p.id for p in retriever.passages}
    results: list[QuestionResult] = []
    for question, gold in questions:
        if gold is not None and gold not in known:
            raise ValueError(f"gold passage {gold!r} not in knowledge base")
        hits = retriever.search(question, k=k)
        ranked = [h.passage.id for h in hits]
        rank = ranked.index(gold) + 1 if gold in ranked else None
        results.append(
            QuestionResult(question, gold, ranked, [round(h.score, 3) for h in hits], rank)
        )
    return results, summarize(results)


def summarize(results: list[QuestionResult]) -> dict:
    answerable = [r for r in results if r.gold is not None]
    unanswerable = [r for r in results if r.gold is None]
    n = len(answerable)
    summary = {
        "n_answerable": n,
        "n_unanswerable": len(unanswerable),
        "hit@1": sum(r.rank == 1 for r in answerable) / n if n else 0.0,
        "hit@3": sum(r.rank is not None for r in answerable) / n if n else 0.0,
        "mrr@3": sum(1 / r.rank for r in answerable if r.rank) / n if n else 0.0,
        "thresholds": [],
    }
    for t in THRESHOLDS:
        top = lambda r: r.scores[0] if r.scores else 0.0  # noqa: E731
        answered_ok = sum(r.rank == 1 and top(r) > t for r in answerable)
        rejected = sum(top(r) <= t for r in unanswerable)
        summary["thresholds"].append(
            {
                "min_score": t,
                "answerable_correct_and_kept": answered_ok / n if n else 0.0,
                "unanswerable_rejected": rejected / len(unanswerable) if unanswerable else 0.0,
            }
        )
    return summary


def write_report(results: list[QuestionResult], summary: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"summary": summary, "questions": [asdict(r) for r in results]}
    (out_dir / "faq_retrieval.json").write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# FAQ retrieval (BM25)",
        "",
        "Produced by `cva-faq-eval`. Questions: `knowledge/faq_eval.yaml`.",
        "",
        f"Answerable questions: {summary['n_answerable']} · "
        f"unanswerable: {summary['n_unanswerable']}",
        "",
        "| Hit@1 | Hit@3 | MRR@3 |",
        "|---|---|---|",
        f"| {summary['hit@1']:.3f} | {summary['hit@3']:.3f} | {summary['mrr@3']:.3f} |",
        "",
        "## Score threshold for \"no answer\"",
        "",
        "| min score | answerable: correct top-1 kept | unanswerable: rejected |",
        "|---|---|---|",
    ]
    for row in summary["thresholds"]:
        lines.append(
            f"| {row['min_score']:.1f} | {row['answerable_correct_and_kept']:.3f} "
            f"| {row['unanswerable_rejected']:.3f} |"
        )
    misses = [r for r in results if r.gold is not None and r.rank != 1]
    lines += ["", "## Misses (gold not ranked first)", ""]
    if not misses:
        lines.append("None.")
    for r in misses:
        got = r.ranked[0] if r.ranked else "-"
        lines.append(f"- \"{r.question}\" → gold `{r.gold}`, got `{got}` (rank {r.rank})")
    lines += ["", "## Unanswerable questions: top score", ""]
    for r in results:
        if r.gold is None:
            top = r.scores[0] if r.scores else 0.0
            got = r.ranked[0] if r.ranked else "-"
            lines.append(f"- \"{r.question}\" → `{got}` ({top:.2f})")
    (out_dir / "faq_retrieval.md").write_text("\n".join(lines) + "\n")
