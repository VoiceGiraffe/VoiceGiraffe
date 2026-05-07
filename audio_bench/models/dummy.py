"""A trivial backend that always answers 'A'. Useful for pipeline smoke tests
and as a worked-out example for adding new backends."""

from __future__ import annotations

from .base import BaseAudioQAModel, QAInput, QAOutput
from .registry import register_model


@register_model("dummy")
class DummyModel(BaseAudioQAModel):
    """Returns letter 'A' every time. Replace with a real model for actual eval."""

    def __init__(self, fixed_letter: str = "A", **_: object) -> None:
        self.fixed_letter = fixed_letter.upper()

    def load(self) -> None:
        return None

    def predict(self, item: QAInput) -> QAOutput:
        return QAOutput(
            raw_text=f"Dummy answer: {self.fixed_letter}",
            pred_letter=self.fixed_letter,
            sample_id=item.sample_id,
        )
