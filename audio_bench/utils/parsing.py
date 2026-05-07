"""Helpers to parse free-form model outputs into A/B/C/D letters."""

from __future__ import annotations

import re

_LETTER_RE_PATTERNS = [
    # Forms like "Answer: B", "答案: C", "选项 D"
    re.compile(r"(?im)^\s*(?:final\s*answer|answer|the\s*answer\s*is|答案|选\s*项|选择)\s*[:：]?\s*[\(\[\{]?\s*([ABCD])\b"),
    re.compile(r"(?im)\b(?:answer\s*is|answer\s*:)\s*[\(\[\{]?\s*([ABCD])\b"),
    # Forms like "(B)" / "[C]" anywhere
    re.compile(r"[\(\[\{]\s*([ABCD])\s*[\)\]\}]"),
    # Lines that are exactly one letter
    re.compile(r"(?m)^\s*([ABCD])\s*[\.\)、\]]?\s*$"),
]

_FINAL_LETTER_RE = re.compile(r"\b([ABCD])\b")


def parse_letter(text: str) -> str | None:
    """Best-effort extraction of an A/B/C/D letter from model output.
    Returns None if nothing plausible is found."""
    if not isinstance(text, str) or not text.strip():
        return None
    s = text.strip()

    for pat in _LETTER_RE_PATTERNS:
        m = pat.search(s)
        if m:
            return m.group(1).upper()

    # Fallback: take the LAST standalone A/B/C/D letter in the text (often the conclusion).
    matches = _FINAL_LETTER_RE.findall(s)
    if matches:
        return matches[-1].upper()
    return None
