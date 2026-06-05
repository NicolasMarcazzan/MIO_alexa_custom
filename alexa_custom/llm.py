from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from alexa_custom.config import LLMConfig

logger = logging.getLogger(__name__)


class LLMEngine:
    def __init__(self, get_llm_config: Callable[[], LLMConfig | None]) -> None:
        self._get_llm_config = get_llm_config
        self._llm = None
        self._loaded_model_path: str | None = None

    def _init_model(self) -> None:
        if self._llm is not None:
            # Check if model path has changed
            config = self._get_llm_config()
            if config and config.model_path == self._loaded_model_path:
                return

        config = self._get_llm_config()
        if not config or not config.enabled:
            return

        logger.info(f"Initializing LLM model from {config.model_path}")
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "The 'llama-cpp-python' package is required for local LLM features, but it is not installed. "
                "You can install it manually using 'pip install llama-cpp-python' or install this project "
                "with the LLM optional dependencies: 'pip install -e .[llm]'"
            )
        import os

        if not os.path.exists(config.model_path):
            raise FileNotFoundError(
                f"LLM model not found at {config.model_path}. "
                f"Run 'alexa-setup --llm' to download it."
            )

        # Clean up old engine instance if model path changed
        if self._llm is not None:
            self._llm = None

        self._llm = Llama(
            model_path=str(config.model_path),
            n_ctx=512,  # Short context is sufficient for voice Q&A
            n_threads=4,  # Optimized for the Snapdragon 801 CPU cores
            verbose=False,
        )
        self._loaded_model_path = config.model_path

    async def generate(self, text: str) -> str:
        """Asynchronously generate a response using llama-cpp-python."""
        # Ensure the model is initialized inside a background thread to prevent blocking the event loop
        await asyncio.to_thread(self._init_model)

        if self._llm is None:
            raise RuntimeError("LLM engine model is not loaded.")

        def _sync_generate() -> str:
            config = self._get_llm_config()
            if not config:
                raise RuntimeError("LLM configuration is missing.")

            messages = [
                {"role": "system", "content": config.system_prompt},
                {"role": "user", "content": text},
            ]
            response = self._llm.create_chat_completion(
                messages=messages,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
            )
            return response["choices"][0]["message"]["content"].strip()

        return await asyncio.to_thread(_sync_generate)
