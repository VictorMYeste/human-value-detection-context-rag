"""Gemma 3 12B client wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from value_context_rag.utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass
class GemmaConfig:
    model_name: str = "google/gemma-3-12b-it"
    device: Optional[str] = None
    quantization: Optional[str] = None  # "8bit" or "4bit"
    max_new_tokens: int = 64
    temperature: float = 0.0
    top_p: float = 1.0


class GemmaClient:
    def __init__(self, config: GemmaConfig) -> None:
        self.config = config
        self.device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._load_model()

    def _load_model(self) -> None:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError("transformers is required for Gemma") from exc

        LOGGER.debug(
            "Initializing Gemma client (model=%s, device=%s, quantization=%s)",
            self.config.model_name,
            self.device,
            self.config.quantization,
        )
        LOGGER.info("Loading Gemma model %s", self.config.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)

        model_kwargs = {"device_map": "auto" if self.device == "cuda" else None}
        if self.config.quantization == "8bit":
            model_kwargs["load_in_8bit"] = True
        elif self.config.quantization == "4bit":
            model_kwargs["load_in_4bit"] = True

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            **model_kwargs,
        )

        if self.device != "cuda":
            self.model.to(self.device)
        LOGGER.debug("Gemma model loaded")

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> str:
        max_tokens = max_tokens or self.config.max_new_tokens
        temperature = temperature if temperature is not None else self.config.temperature
        top_p = top_p if top_p is not None else self.config.top_p

        LOGGER.debug(
            "Generating with max_tokens=%s temperature=%s top_p=%s prompt_len=%d",
            max_tokens,
            temperature,
            top_p,
            len(prompt),
        )
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                temperature=temperature,
                top_p=top_p,
            )

        generated = output[0][inputs["input_ids"].shape[1] :]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        return text.strip()
