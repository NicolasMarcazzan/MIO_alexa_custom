from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

# Mock llama_cpp module before importing/testing anything that uses it
mock_llama_mod = MagicMock()
sys.modules["llama_cpp"] = mock_llama_mod

import asyncio
import pytest

from alexa_custom.config import LLMConfig
from alexa_custom.llm import LLMEngine


class TestLLMEngine:
    def test_lazy_initialization(self):
        # Reset the mock
        mock_llama_mod.reset_mock()
        
        config = LLMConfig(enabled=True, model_path="/path/to/model.gguf")
        engine = LLMEngine(lambda: config)
        assert engine._llm is None
        mock_llama_mod.Llama.assert_not_called()

    @pytest.mark.asyncio
    async def test_does_not_initialize_when_disabled(self):
        mock_llama_mod.reset_mock()
        
        config = LLMConfig(enabled=False, model_path="/path/to/model.gguf")
        engine = LLMEngine(lambda: config)
        
        with pytest.raises(RuntimeError, match="LLM engine model is not loaded"):
            await engine.generate("ciao")
        mock_llama_mod.Llama.assert_not_called()

    @pytest.mark.asyncio
    async def test_generate_loads_and_calls_llama(self):
        mock_llama_mod.reset_mock()
        
        config = LLMConfig(
            enabled=True,
            model_path="/path/to/model.gguf",
            system_prompt="test system prompt",
            max_tokens=64,
            temperature=0.8,
        )
        
        engine = LLMEngine(lambda: config)
        
        mock_llama_instance = MagicMock()
        mock_llama_instance.create_chat_completion.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "  Ciao, sono un LLM!  "
                    }
                }
            ]
        }
        mock_llama_mod.Llama.return_value = mock_llama_instance
        
        with patch("os.path.exists", return_value=True):
            response = await engine.generate("hello")
            
            # Check model initialization parameters
            mock_llama_mod.Llama.assert_called_once_with(
                model_path="/path/to/model.gguf",
                n_ctx=512,
                n_threads=4,
                verbose=False,
            )
            
            # Check generation call and formatting
            mock_llama_instance.create_chat_completion.assert_called_once_with(
                messages=[
                    {"role": "system", "content": "test system prompt"},
                    {"role": "user", "content": "hello"},
                ],
                max_tokens=64,
                temperature=0.8,
            )
            
            # Should strip leading/trailing whitespace
            assert response == "Ciao, sono un LLM!"
