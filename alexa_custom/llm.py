from __future__ import annotations

import asyncio
import logging
from threading import Lock

logger = logging.getLogger(__name__)


class LLMEngine:
    def __init__(
        self,
        model_path: str,
        system_prompt: str = "Sei un assistente vocale casalingo. Rispondi in modo breve e conciso, massimo 3 frasi. Rispondi in italiano.",
        max_tokens: int = 128,
        temperature: float = 0.7,
    ):
        self._model_path = model_path
        self._system_prompt = system_prompt
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._model = None
        self._lock = Lock()

    def _lazy_init(self):
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            try:
                from llama_cpp import Llama
            except ImportError:
                raise RuntimeError(
                    "llama-cpp-python not installed. Run: pip install llama-cpp-python"
                )
            logger.info(f"Loading LLM model from {self._model_path}")
            self._model = Llama(
                model_path=self._model_path,
                n_ctx=512,
                n_threads=4,
                verbose=False,
            )
            logger.info("LLM model loaded")

    def generate(
        self, text: str, max_tokens: int | None = None, temperature: float | None = None
    ) -> str:
        self._lazy_init()
        try:
            prompt = (
                f"<|im_start|>system\n{self._system_prompt}<|im_end|>\n"
                f"<|im_start|>user\n{text}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            response = self._model.create_completion(
                prompt=prompt,
                max_tokens=max_tokens or self._max_tokens,
                temperature=temperature or self._temperature,
                stop=["<|im_end|>", "<|im_start|>"],
            )
            content = response["choices"][0]["text"].strip()
            return content
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return ""

    async def async_generate(
        self, text: str, max_tokens: int | None = None, temperature: float | None = None
    ) -> str:
        return await asyncio.to_thread(self.generate, text, max_tokens, temperature)
