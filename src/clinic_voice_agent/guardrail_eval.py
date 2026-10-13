"""Score the rule-based guardrails on labelled caller utterances.

Data: eval/guardrail_utterances.yaml (75 utterances, three classes).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import yaml

from clinic_voice_agent.config import ROOT
from clinic_voice_agent.guardrails import screen

DEFAULT_UTTERANCES = ROOT / "eval" / "guardrail_utterances.yaml"
LABELS = ("emergency", "medical_advice", "ok")


def load_utterances(path: Path | str = DEFAULT_UTTERANCES) -> list[tuple[str, str]]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    items = [(u["text"], u["label"]) for u in data["utterances"]]
    unknown = {label for _, label in items} - set(LABELS)
    if unknown:
        raise ValueError(f"unknown labels: {sorted(unknown)}")
    return items


def evaluate(items: list[tuple[str, str]]) -> dict:
    confusion = {gold: dict.fromkeys(LABELS, 0) for gold in LABELS}
    errors = []
    start = time.perf_counter()
    for text, gold in items:
        result = screen(text)
        confusion[gold][result.action] += 1
        if result.action != gold:
            errors.append({"text": text, "gold": gold, "pred": result.action,
                           "categories": list(result.categories)})
    per_call_ms = (time.perf_counter() - start) * 1000 / max(len(items), 1)
    per_class = {}
    for label in LABELS:
        tp = confusion[label][label]
        predicted = sum(confusion[g][label] for g in LABELS)
        actual = sum(confusion[label].values())
        per_class[label] = {
            "precision": tp / predicted if predicted else 0.0,
            "recall": tp / actual if actual else 0.0,
            "support": actual,
        }
    accuracy = sum(confusion[label][label] for label in LABELS) / max(len(items), 1)
    return {"n": len(items), "accuracy": accuracy, "per_class": per_class,
            "confusion": confusion, "errors": errors, "ms_per_utterance": per_call_ms}


def write_report(summary: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "guardrails.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        "# Guardrail screening (rule-based)",
        "",
        "Produced by `cva-guardrail-eval`. Utterances: `eval/guardrail_utterances.yaml`.",
        "",
        "> Caveat: the patterns were revised once after the first run on this same set "
        "(first-run scores: `guardrails_first_run.json`), so these numbers are optimistic "
        "until a held-out set is scored.",
        "",
        f"{summary['n']} utterances · accuracy {summary['accuracy']:.3f} · "
        f"{summary['ms_per_utterance'] * 1000:.0f} µs per utterance",
        "",
        "| class | precision | recall | n |",
        "|---|---|---|---|",
    ]
    for label, m in summary["per_class"].items():
        lines.append(f"| {label} | {m['precision']:.3f} | {m['recall']:.3f} | {m['support']} |")
    lines += ["", "Confusion matrix (rows = true label, columns = prediction):", "",
              "| | " + " | ".join(LABELS) + " |", "|---" * (len(LABELS) + 1) + "|"]
    for gold in LABELS:
        row = " | ".join(str(summary["confusion"][gold][p]) for p in LABELS)
        lines.append(f"| **{gold}** | {row} |")
    lines += ["", "## Errors", ""]
    lines += [f"- \"{e['text']}\": true `{e['gold']}`, predicted `{e['pred']}`"
              for e in summary["errors"]] or ["None."]
    (out_dir / "guardrails.md").write_text("\n".join(lines) + "\n")
