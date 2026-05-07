"""Placeholder for API-based model backends (e.g., Qwen3.5-Omni-Plus, Gemini).

This file registers API model names in the registry but raises NotImplementedError
when called. Contributors with API access can implement the actual logic.
"""

from __future__ import annotations

from .base import BaseAudioQAModel, QAInput, QAOutput
from .registry import register_model


@register_model("qwen3.5-omni-api")
class Qwen35OmniAPIModel(BaseAudioQAModel):
    """Placeholder: Qwen3.5-Omni API backend (requires API key)."""

    def __init__(self, api_key: str = "", **_: object) -> None:
        self.api_key = api_key

    def load(self) -> None:
        if not self.api_key:
            raise NotImplementedError(
                "Qwen3.5-Omni API backend is not yet implemented. "
                "Contributors with API access can implement this in "
                "audio_bench/models/api_placeholder.py"
            )

    def predict(self, item: QAInput) -> QAOutput:
        raise NotImplementedError(
            "Qwen3.5-Omni API inference not implemented. "
            "Please contribute an implementation."
        )


@register_model("gemini-api")
class GeminiAPIModel(BaseAudioQAModel):
    """Placeholder: Gemini API backend (requires API key)."""

    def __init__(self, api_key: str = "", **_: object) -> None:
        self.api_key = api_key

    def load(self) -> None:
        if not self.api_key:
            raise NotImplementedError(
                "Gemini API backend is not yet implemented. "
                "Contributors with API access can implement this in "
                "audio_bench/models/api_placeholder.py"
            )

    def predict(self, item: QAInput) -> QAOutput:
        raise NotImplementedError(
            "Gemini API inference not implemented. "
            "Please contribute an implementation."
        )
