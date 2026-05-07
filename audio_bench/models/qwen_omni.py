"""Qwen2.5-Omni backend (HuggingFace transformers).

Reference:
    https://huggingface.co/Qwen/Qwen2.5-Omni-7B

This file is intentionally lazy: heavy imports happen inside `load()` so that
the rest of the toolkit (data loader, evaluator, web app) can be used on
machines without GPU / transformers installed.
"""

from __future__ import annotations

import os
from typing import Any

from .base import BaseAudioQAModel, QAInput, QAOutput
from .registry import register_model
from ..utils.parsing import parse_letter


_SYS_PROMPT_EN = (
    "You are a helpful assistant that analyses long audio sequences to answer "
    "multiple-choice questions about audio understanding. Listen to the audio "
    "carefully and choose the single best option. Respond with ONLY one letter: "
    "A, B, C, or D."
)
_SYS_PROMPT_ZH = (
    "你是一位擅长分析长音频的助手，需要根据音频内容回答多项选择题。"
    "请仔细聆听音频，从 A、B、C、D 四个选项中选出唯一最合适的一个。"
    "只输出一个字母：A、B、C 或 D。"
)


def _pick_system_prompt(language: str) -> str:
    lang = (language or "").lower()
    if "zh" in lang or "中文" in language:
        return _SYS_PROMPT_ZH
    return _SYS_PROMPT_EN


@register_model("qwen2.5-omni")
class QwenOmniModel(BaseAudioQAModel):
    """Qwen2.5-Omni-7B audio QA backend."""

    def __init__(
        self,
        model_path: str = "Qwen/Qwen2.5-Omni-7B",
        device: str = "cuda",
        dtype: str = "bfloat16",
        max_new_tokens: int = 16,
        attn_implementation: str | None = None,
        **_: object,
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self.attn_implementation = attn_implementation
        self._model: Any = None
        self._processor: Any = None
        self._torch: Any = None

    def load(self) -> None:
        if self._model is not None:
            return
        # Lazy imports — the rest of the package must be usable without these.
        import torch  # type: ignore

        try:
            from transformers import (  # type: ignore
                Qwen2_5OmniForConditionalGeneration,
                Qwen2_5OmniProcessor,
            )
        except ImportError as e:
            raise ImportError(
                "Qwen2.5-Omni requires transformers>=4.52 with omni support. "
                "Install with: pip install 'transformers>=4.52' accelerate librosa soundfile"
            ) from e

        torch_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }.get(self.dtype, torch.bfloat16)

        kwargs: dict = {"torch_dtype": torch_dtype, "device_map": self.device}
        if self.attn_implementation:
            kwargs["attn_implementation"] = self.attn_implementation

        self._processor = Qwen2_5OmniProcessor.from_pretrained(self.model_path)
        self._model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            self.model_path, **kwargs
        )
        self._model.eval()
        self._torch = torch

    def predict(self, item: QAInput) -> QAOutput:
        try:
            self.load()
        except Exception as e:  # noqa: BLE001
            return QAOutput(
                raw_text="", pred_letter=None, sample_id=item.sample_id,
                error=f"load_failed: {e}",
            )

        if not item.audio_path or not os.path.exists(item.audio_path):
            return QAOutput(
                raw_text="", pred_letter=None, sample_id=item.sample_id,
                error=f"audio_not_found: {item.audio_path}",
            )

        try:
            sys_prompt = _pick_system_prompt(item.language)
            conversation = [
                {"role": "system", "content": [{"type": "text", "text": sys_prompt}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio": item.audio_path},
                        {"type": "text", "text": item.question},
                    ],
                },
            ]

            # Build prompt + tensors via processor.
            text = self._processor.apply_chat_template(
                conversation, add_generation_prompt=True, tokenize=False
            )
            inputs = self._processor(
                text=text,
                audios=[item.audio_path],
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: (v.to(self._model.device) if hasattr(v, "to") else v)
                      for k, v in inputs.items()}

            with self._torch.no_grad():
                gen = self._model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                )

            input_len = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0
            new_tokens = gen[:, input_len:] if input_len else gen
            raw_text = self._processor.batch_decode(
                new_tokens, skip_special_tokens=True
            )[0].strip()

            return QAOutput(
                raw_text=raw_text,
                pred_letter=parse_letter(raw_text),
                sample_id=item.sample_id,
            )
        except Exception as e:  # noqa: BLE001
            return QAOutput(
                raw_text="", pred_letter=None, sample_id=item.sample_id,
                error=f"infer_failed: {type(e).__name__}: {e}",
            )
