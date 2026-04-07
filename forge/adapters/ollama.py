"""
FORGE Ollama Adapter — Interface to Ollama's local inference API.

Provides a clean abstraction over Ollama's REST API at localhost:11434.
Handles both chat completions (with tool support) and simple generation.

Supports:
  - Tool calling via Ollama's native tool format
  - Streaming and non-streaming responses
  - Model health checking
  - Token counting estimation

Dependencies: httpx (async HTTP client)
Depended on by: agent_loop.py, enrichment workers
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger("forge.ollama")

DEFAULT_OLLAMA_URL = "http://localhost:11434"


class OllamaAdapter:
    """
    Adapter for Ollama's local inference API.

    Provides generate() method compatible with FORGE's agent loop.
    Handles connection management, timeouts, and error recovery.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_OLLAMA_URL,
        default_model: str = "gemma4:26b",
        default_timeout: float = 300.0,
    ):
        self._base_url = base_url.rstrip("/")
        self._default_model = default_model
        self._default_timeout = default_timeout
        self._client = httpx.Client(timeout=default_timeout)

    def generate(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        timeout: Optional[float] = None,
        temperature: float = 0.3,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Send a chat completion request to Ollama.

        Args:
            messages: List of message dicts (role + content).
            model: Model name (defaults to gemma4).
            tools: List of tool definitions for function calling.
            timeout: Request timeout in seconds.
            temperature: Sampling temperature (lower = more deterministic).

        Returns:
            Dict with the model's response including message content
            and any tool calls.
        """
        model = model or self._default_model
        timeout = timeout or self._default_timeout

        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": 2048,  # Max output tokens
            },
        }

        if tools:
            payload["tools"] = tools

        try:
            response = self._client.post(
                f"{self._base_url}/api/chat",
                json=payload,
                timeout=timeout,
            )
            response.raise_for_status()
            data = response.json()

            # Log inference stats
            eval_count = data.get("eval_count", 0)
            eval_duration = data.get("eval_duration", 1)
            tokens_per_sec = eval_count / (eval_duration / 1e9) if eval_duration > 0 else 0
            logger.debug(
                "Ollama response: model=%s, tokens=%d, tok/s=%.1f",
                model, eval_count, tokens_per_sec,
            )

            return data

        except httpx.TimeoutException:
            logger.error("Ollama timeout after %.0fs for model %s", timeout, model)
            raise TimeoutError(f"Ollama timed out after {timeout}s")
        except httpx.HTTPStatusError as e:
            logger.error("Ollama HTTP error: %s", e)
            raise ConnectionError(f"Ollama returned {e.response.status_code}")
        except httpx.ConnectError:
            logger.error("Cannot connect to Ollama at %s", self._base_url)
            raise ConnectionError(f"Ollama not running at {self._base_url}")

    def generate_simple(
        self,
        prompt: str,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
        temperature: float = 0.3,
        think: bool = False,
    ) -> str:
        """
        Simple text generation without chat format.

        Useful for one-shot tasks like classification or summarization
        where you don't need tool calling or conversation history.

        Args:
            think: If False, disables Gemma 4's thinking mode for faster inference.
                   Thinking mode uses ~80% of tokens for reasoning, making it ~8x slower.

        Returns the response text directly.
        """
        model = model or self._default_model
        timeout = timeout or self._default_timeout

        payload: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "think": think,
            "options": {
                "temperature": temperature,
                "num_predict": 1024,
            },
        }

        try:
            response = self._client.post(
                f"{self._base_url}/api/generate",
                json=payload,
                timeout=timeout,
            )
            response.raise_for_status()
            return response.json().get("response", "")

        except httpx.TimeoutException:
            raise TimeoutError(f"Ollama timed out after {timeout}s")
        except httpx.ConnectError:
            raise ConnectionError(f"Ollama not running at {self._base_url}")

    def generate_batch(
        self,
        prompts: List[str],
        model: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> List[str]:
        """
        Generate responses for multiple prompts sequentially.

        For batch enrichment where you want one model call per business
        but process them in a tight loop.

        Returns list of response strings, same order as input prompts.
        """
        results = []
        for prompt in prompts:
            try:
                result = self.generate_simple(prompt, model=model, timeout=timeout)
                results.append(result)
            except Exception as e:
                logger.warning("Batch item failed: %s", e)
                results.append("")
        return results

    def is_healthy(self) -> bool:
        """Check if Ollama is running and responsive."""
        try:
            response = self._client.get(
                f"{self._base_url}/api/tags",
                timeout=5.0,
            )
            return response.status_code == 200
        except Exception:
            return False

    def list_models(self) -> List[str]:
        """List available models on the local Ollama instance."""
        try:
            response = self._client.get(
                f"{self._base_url}/api/tags",
                timeout=10.0,
            )
            data = response.json()
            return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []

    def model_info(self, model: Optional[str] = None) -> Dict[str, Any]:
        """Get info about a specific model."""
        model = model or self._default_model
        try:
            response = self._client.post(
                f"{self._base_url}/api/show",
                json={"name": model},
                timeout=10.0,
            )
            return response.json()
        except Exception as e:
            return {"error": str(e)}

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self) -> "OllamaAdapter":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
