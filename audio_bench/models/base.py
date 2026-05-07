"""Abstract base for all audio-QA backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class QAInput:
    """One input for the model."""

    audio_path: str
    question: str  # full question text including the four options inline
    language: str = "zh"  # "zh" or "en", used to choose the system prompt
    sample_id: str = ""


@dataclass
class QAOutput:
    """One model output."""

    raw_text: str  # the raw text the model generated
    pred_letter: str | None  # parsed A/B/C/D, may be None if parsing failed
    sample_id: str = ""
    error: str | None = None


class BaseAudioQAModel(ABC):
    """Subclass this to add a new model backend."""

    name: str = "base"

    @abstractmethod
    def load(self) -> None:
        """Load model weights and processor into memory."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, item: QAInput) -> QAOutput:
        """Run inference for one (audio, question) pair."""
        raise NotImplementedError

    def predict_batch(self, items: list[QAInput]) -> list[QAOutput]:
        """Default sequential implementation; backends may override for batching."""
        return [self.predict(it) for it in items]
