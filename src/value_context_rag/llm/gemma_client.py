"""Decoder-only LLM client wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import warnings

import torch

from value_context_rag.utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass
class GemmaConfig:
    model_name: str = "google/gemma-3-12b-it"
    device: str | None = None
    quantization: str | None = None  # "8bit" or "4bit"
    int8_fp32_cpu_offload: bool = False
    max_new_tokens: int = 64
    temperature: float = 0.0
    top_p: float = 1.0


class GemmaClient:
    def __init__(self, config: GemmaConfig) -> None:
        self.config = config
        self.device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor: Any | None = None
        self._use_chat_template = False
        self._load_model()

    def _load_model(self) -> None:
        auto_image_text_cls: Any | None = None
        try:
            from transformers import (  # type: ignore
                AutoModelForCausalLM,
                AutoProcessor,
                AutoTokenizer,
                BitsAndBytesConfig,
            )
            try:
                from transformers import AutoModelForImageTextToText  # type: ignore

                auto_image_text_cls = AutoModelForImageTextToText
            except Exception:
                auto_image_text_cls = None
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError("transformers is required for Gemma") from exc

        LOGGER.debug(
            "Initializing LLM client (model=%s, device=%s, quantization=%s, int8_cpu_offload=%s)",
            self.config.model_name,
            self.device,
            self.config.quantization,
            self.config.int8_fp32_cpu_offload,
        )
        LOGGER.info("Loading model %s", self.config.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        try:
            self.processor = AutoProcessor.from_pretrained(self.config.model_name)
            processor_tokenizer = getattr(self.processor, "tokenizer", None)
            if processor_tokenizer is not None:
                self.tokenizer = processor_tokenizer
            self._use_chat_template = hasattr(self.processor, "apply_chat_template")
            LOGGER.info("Using processor chat template for prompt formatting")
        except Exception:
            self.processor = None
            self._use_chat_template = (
                hasattr(self.tokenizer, "apply_chat_template")
                and getattr(self.tokenizer, "chat_template", None) is not None
            )
            if self._use_chat_template:
                LOGGER.info("Using tokenizer chat template for prompt formatting")
            else:
                LOGGER.info("No chat template detected; using plain text prompt")

        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs: dict[str, object] = {}
        if self.device == "cuda":
            model_kwargs["device_map"] = "auto"
            # Keep bf16 for better Gemma 3 generation quality.
            model_kwargs["dtype"] = torch.bfloat16

        if self.config.quantization == "8bit":
            # bnb may warn about bf16->fp16 casting; this is expected for int8 kernels.
            warnings.filterwarnings(
                "ignore",
                message=(
                    r"MatMul8bitLt: inputs will be cast from torch\.bfloat16 "
                    r"to float16 during quantization"
                ),
                category=UserWarning,
            )
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=self.config.int8_fp32_cpu_offload,
            )
        elif self.config.quantization == "4bit":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        elif self.config.quantization not in (None, "none"):
            raise ValueError(f"Unsupported Gemma quantization mode: {self.config.quantization}")

        model_loader = AutoModelForCausalLM
        model_name_lc = self.config.model_name.lower()
        if "gemma-3" in model_name_lc and auto_image_text_cls is not None:
            model_loader = auto_image_text_cls
            LOGGER.info("Using AutoModelForImageTextToText for Gemma 3")
        self.model = model_loader.from_pretrained(self.config.model_name, **model_kwargs)
        self.model.eval()

        if self.device != "cuda":
            self.model.to(self.device)
        LOGGER.debug("Gemma model loaded")

    @staticmethod
    def _text_messages(prompt: str) -> list[dict[str, str]]:
        return [{"role": "user", "content": prompt}]

    @staticmethod
    def _rich_text_messages(prompt: str) -> list[dict[str, object]]:
        return [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

    def preview_model_prompt(self, prompt: str) -> str:
        """Return the text prompt as rendered for the model (chat template if available)."""
        if not self._use_chat_template:
            return prompt

        messages = self._text_messages(prompt)
        rich_messages = self._rich_text_messages(prompt)
        if self.processor is not None and hasattr(self.processor, "apply_chat_template"):
            try:
                rendered = self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                )
                return str(rendered)
            except Exception:
                try:
                    rendered = self.processor.apply_chat_template(
                        rich_messages,
                        add_generation_prompt=True,
                        tokenize=False,
                    )
                    return str(rendered)
                except Exception:
                    pass

        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                rendered = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                )
                return str(rendered)
            except Exception:
                try:
                    rendered = self.tokenizer.apply_chat_template(
                        rich_messages,
                        add_generation_prompt=True,
                        tokenize=False,
                    )
                    return str(rendered)
                except Exception:
                    pass

        return prompt

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> str:
        max_tokens = max_tokens or self.config.max_new_tokens
        temperature = (
            temperature if temperature is not None else self.config.temperature
        )
        top_p = top_p if top_p is not None else self.config.top_p

        LOGGER.debug(
            "Generating with max_tokens=%s temperature=%s top_p=%s prompt_len=%d",
            max_tokens,
            temperature,
            top_p,
            len(prompt),
        )
        if self._use_chat_template:
            messages = self._text_messages(prompt)
            rich_messages = self._rich_text_messages(prompt)
            if self.processor is not None and hasattr(self.processor, "apply_chat_template"):
                try:
                    inputs = self.processor.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        tokenize=True,
                        return_dict=True,
                        return_tensors="pt",
                    )
                except Exception:
                    try:
                        inputs = self.processor.apply_chat_template(
                            rich_messages,
                            add_generation_prompt=True,
                            tokenize=True,
                            return_dict=True,
                            return_tensors="pt",
                        )
                    except Exception:
                        try:
                            prompt_text = self.processor.apply_chat_template(
                                messages,
                                add_generation_prompt=True,
                                tokenize=False,
                            )
                        except Exception:
                            prompt_text = self.processor.apply_chat_template(
                                rich_messages,
                                add_generation_prompt=True,
                                tokenize=False,
                            )
                        inputs = self.tokenizer(prompt_text, return_tensors="pt")
            else:
                try:
                    inputs = self.tokenizer.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        tokenize=True,
                        return_dict=True,
                        return_tensors="pt",
                    )
                except Exception:
                    inputs = self.tokenizer.apply_chat_template(
                        rich_messages,
                        add_generation_prompt=True,
                        tokenize=True,
                        return_dict=True,
                        return_tensors="pt",
                    )
        else:
            inputs = self.tokenizer(prompt, return_tensors="pt")

        if isinstance(inputs, torch.Tensor):
            inputs = {"input_ids": inputs}
        if "attention_mask" not in inputs and "input_ids" in inputs:
            inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        eos_ids: list[int] = []
        if self.tokenizer.eos_token_id is not None:
            eos_ids.append(int(self.tokenizer.eos_token_id))
        if hasattr(self.tokenizer, "get_vocab"):
            vocab = self.tokenizer.get_vocab()
            if "<end_of_turn>" in vocab:
                eot_id = int(vocab["<end_of_turn>"])
                if eot_id not in eos_ids:
                    eos_ids.append(eot_id)

        generation_kwargs: dict[str, object] = {
            "max_new_tokens": max_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
            "use_cache": True,
        }
        if eos_ids:
            generation_kwargs["eos_token_id"] = eos_ids if len(eos_ids) > 1 else eos_ids[0]
        if temperature > 0:
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_p"] = top_p

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                **generation_kwargs,
            )

        generated = output[0][inputs["input_ids"].shape[1] :]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        return text.strip()
