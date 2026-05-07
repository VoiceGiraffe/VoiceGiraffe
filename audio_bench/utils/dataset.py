"""Benchmark dataset loader.

The benchmark JSONL has one record per line with fields:
    question, answer, audio, qa_type, qa_level, language_type
The `audio` field is a path whose basename (without extension) is the
YouTube video id, e.g.
    /data/.../youtube_long/Q0drb68Orps.wav  ->  youtube_id = Q0drb68Orps
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field, asdict
from typing import Iterable, Iterator


_OPTION_LINE_RE = re.compile(r"(?ms)^\s*([ABCD])\s*[:：]\s*(.+?)\s*$")


def _split_question_and_options(full_question: str) -> tuple[str, dict]:
    """Split the inline `Q\\nA: ...\\nB: ...\\nC: ...\\nD: ...` text into
    (stem, {A,B,C,D})."""
    if not isinstance(full_question, str):
        return "", {}
    matches = list(_OPTION_LINE_RE.finditer(full_question))
    if len(matches) < 4:
        # No standard options found — return the whole text as stem.
        return full_question.strip(), {}

    options = {m.group(1): m.group(2).strip() for m in matches}
    stem_end = matches[0].start()
    stem = full_question[:stem_end].strip()
    return stem, options


def _youtube_id_from_audio_path(path: str) -> str:
    if not path:
        return ""
    base = os.path.basename(path)
    stem, _ = os.path.splitext(base)
    return stem


@dataclass
class BenchmarkSample:
    """One QA sample in the benchmark."""

    sample_id: str
    youtube_id: str
    audio_path: str  # original path from jsonl (may not exist locally)
    question_full: str  # raw question text including options inline
    question_stem: str  # stripped question (no options)
    options: dict  # {"A": "...", "B": "...", "C": "...", "D": "..."}
    answer: str  # ground-truth letter, one of A/B/C/D
    qa_type: str
    qa_level: str
    language_type: str
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def resolve_local_audio(self, audio_dir: str, exts: Iterable[str] = (".wav", ".m4a", ".mp3", ".flac", ".opus")) -> str | None:
        """Return the local path under audio_dir if the file exists, else None."""
        if not self.youtube_id or not audio_dir:
            return None
        for ext in exts:
            cand = os.path.join(audio_dir, self.youtube_id + ext)
            if os.path.exists(cand):
                return cand
        return None


def load_benchmark(jsonl_path: str) -> list[BenchmarkSample]:
    """Load all samples from a benchmark JSONL file."""
    samples: list[BenchmarkSample] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:  # noqa: BLE001
                raise ValueError(f"line {idx} is not valid JSON: {e}") from e

            full_q = obj.get("question", "") or ""
            stem, options = _split_question_and_options(full_q)
            audio_path = obj.get("audio", "") or ""
            yid = _youtube_id_from_audio_path(audio_path)

            sample = BenchmarkSample(
                sample_id=f"{idx:06d}_{yid}" if yid else f"{idx:06d}",
                youtube_id=yid,
                audio_path=audio_path,
                question_full=full_q,
                question_stem=stem,
                options=options,
                answer=str(obj.get("answer", "")).strip().upper(),
                qa_type=str(obj.get("qa_type", "Unknown")),
                qa_level=str(obj.get("qa_level", "Unknown")),
                language_type=str(obj.get("language_type", "Unknown")),
                extra={k: v for k, v in obj.items() if k not in {
                    "question", "answer", "audio", "qa_type", "qa_level", "language_type"
                }},
            )
            samples.append(sample)
    return samples


def iter_unique_youtube_ids(samples: Iterable[BenchmarkSample]) -> Iterator[str]:
    seen: set[str] = set()
    for s in samples:
        if s.youtube_id and s.youtube_id not in seen:
            seen.add(s.youtube_id)
            yield s.youtube_id
