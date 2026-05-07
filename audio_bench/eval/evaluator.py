"""Evaluation: compute overall and per-subgroup accuracy."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable


@dataclass
class PredRecord:
    sample_id: str
    youtube_id: str
    qa_type: str
    qa_level: str
    language_type: str
    gold: str  # ground-truth letter
    pred: str | None  # predicted letter, may be None on parse failure
    raw_text: str = ""
    error: str | None = None


def _safe_eq(pred: str | None, gold: str) -> bool:
    if not pred or not gold:
        return False
    return pred.strip().upper() == gold.strip().upper()


def evaluate(records: Iterable[PredRecord]) -> dict:
    """Compute overall accuracy and grouped accuracy by qa_type / qa_level /
    language_type / (qa_type, language_type). Returns a JSON-serialisable dict."""

    records = list(records)
    total = len(records)
    if total == 0:
        return {"total": 0, "overall_accuracy": 0.0, "groups": {}}

    correct = 0
    parse_fail = 0
    infer_fail = 0
    by_type: dict[str, list[bool]] = defaultdict(list)
    by_level: dict[str, list[bool]] = defaultdict(list)
    by_lang: dict[str, list[bool]] = defaultdict(list)
    by_type_lang: dict[str, list[bool]] = defaultdict(list)

    for r in records:
        ok = _safe_eq(r.pred, r.gold)
        correct += int(ok)
        if r.error:
            infer_fail += 1
        elif r.pred is None:
            parse_fail += 1
        by_type[r.qa_type].append(ok)
        by_level[r.qa_level].append(ok)
        by_lang[r.language_type].append(ok)
        by_type_lang[f"{r.qa_type}|{r.language_type}"].append(ok)

    def _acc(items: list[bool]) -> dict:
        n = len(items)
        c = sum(items)
        return {"n": n, "correct": c, "accuracy": (c / n) if n else 0.0}

    return {
        "total": total,
        "correct": correct,
        "overall_accuracy": correct / total,
        "infer_failures": infer_fail,
        "parse_failures": parse_fail,
        "groups": {
            "qa_type": {k: _acc(v) for k, v in sorted(by_type.items())},
            "qa_level": {k: _acc(v) for k, v in sorted(by_level.items())},
            "language_type": {k: _acc(v) for k, v in sorted(by_lang.items())},
            "qa_type_x_language": {k: _acc(v) for k, v in sorted(by_type_lang.items())},
        },
    }


def format_report(metrics: dict, model_name: str = "") -> str:
    """Render a human-readable text report from `evaluate()` output."""
    lines = []
    if model_name:
        lines.append(f"Model: {model_name}")
    lines.append(f"Total samples : {metrics.get('total', 0)}")
    lines.append(f"Overall acc   : {metrics.get('overall_accuracy', 0.0) * 100:.2f}%  "
                 f"({metrics.get('correct', 0)}/{metrics.get('total', 0)})")
    lines.append(f"Infer failed  : {metrics.get('infer_failures', 0)}")
    lines.append(f"Parse failed  : {metrics.get('parse_failures', 0)}")

    for group_name, group in metrics.get("groups", {}).items():
        lines.append(f"\n[{group_name}]")
        for k, v in group.items():
            lines.append(
                f"  {k:<24s}  acc={v['accuracy'] * 100:6.2f}%  "
                f"({v['correct']}/{v['n']})"
            )
    return "\n".join(lines)
