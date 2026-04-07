"""
FORGE Claude Adapter -- Interface to Anthropic's Claude API.

Provides the same abstraction as OllamaAdapter but backed by Anthropic's
Messages API.  Handles message format translation, tool-use mapping,
rate-limit retries with exponential backoff, and extended thinking.

Supports:
  - Chat completion with tool support (generate)
  - Single prompt convenience method (generate_simple)
  - Extended thinking mode
  - Sequential batch generation
  - Health check and model listing
  - Exponential backoff on rate limits (3 retries: 5s / 15s / 45s)

Dependencies: anthropic (Anthropic Python SDK)
Depended on by: agent_loop.py, enrichment workers, config.get_adapter()
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional

import anthropic

logger = logging.getLogger("forge.claude")

# Models available via the Anthropic API
CLAUDE_MODELS = [
    "claude-sonnet-4-6",
    "claude-opus-4-6",
    "claude-haiku-3-5",
    "claude-sonnet-4-20250514",
    "claude-haiku-3-5-20241022",
]

# Retry schedule for 429 rate-limit errors (seconds)
_BACKOFF_DELAYS = [5, 15, 45]


class ClaudeAdapter:
    """
    Adapter for Anthropic's Claude API.

    Provides generate() method compatible with FORGE's agent loop.
    Maps Ollama-style message format to Anthropic's Messages API format
    so the rest of the codebase can swap adapters transparently.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: str = "claude-sonnet-4-6",
        default_timeout: float = 120.0,
    ):
        """
        Initialize the Claude adapter.

        Args:
            api_key: Anthropic API key.  Falls back to ANTHROPIC_API_KEY env var.
            default_model: Default model for all requests.
            default_timeout: Request timeout in seconds.
        """
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._default_model = default_model
        self._default_timeout = default_timeout
        self._client = anthropic.Anthropic(
            api_key=self._api_key,
            timeout=default_timeout,
        )
        logger.info(
            "Claude adapter initialized — model=%s, timeout=%.0fs",
            self._default_model,
            self._default_timeout,
        )

    # ── Format translation helpers ──────────────────────────────────────

    @staticmethod
    def _extract_system(messages: List[Dict[str, Any]]) -> tuple[str, List[Dict[str, Any]]]:
        """
        Extract system messages from an Ollama-style message list.

        Ollama keeps system messages inline.  Anthropic requires the system
        prompt as a separate top-level parameter.  This pulls all
        role='system' entries out and concatenates them.

        Returns:
            (system_text, remaining_messages)
        """
        system_parts: list[str] = []
        remaining: list[Dict[str, Any]] = []

        for msg in messages:
            if msg.get("role") == "system":
                system_parts.append(msg.get("content", ""))
            else:
                remaining.append(msg)

        return "\n\n".join(system_parts), remaining

    @staticmethod
    def _convert_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert Ollama-format messages to Anthropic-format messages.

        Ollama uses: {"role": "user"|"assistant", "content": "..."}
        Anthropic uses the same base shape but tool results differ.
        This normalises the format and strips any Ollama-specific keys.
        """
        converted = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # Skip empty messages
            if not content and "tool_calls" not in msg:
                continue

            converted.append({"role": role, "content": content})

        return converted

    @staticmethod
    def _convert_tools(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert Ollama tool format to Anthropic tool_use format.

        Ollama format:
            {"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}

        Anthropic format:
            {"name": ..., "description": ..., "input_schema": ...}
        """
        converted = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool["function"]
                converted.append({
                    "name": func["name"],
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {"type": "object", "properties": {}}),
                })
            elif "name" in tool:
                # Already in Anthropic format
                converted.append(tool)
        return converted

    # ── Core API methods ────────────────────────────────────────────────

    def _call_with_retry(self, func, *args, **kwargs) -> Any:
        """
        Call an API function with exponential backoff on rate-limit (429) errors.

        Retries up to 3 times with delays of 5s, 15s, 45s.
        """
        last_error = None
        for attempt, delay in enumerate(_BACKOFF_DELAYS):
            try:
                return func(*args, **kwargs)
            except anthropic.RateLimitError as e:
                last_error = e
                logger.warning(
                    "Rate limited (attempt %d/%d) — retrying in %ds",
                    attempt + 1,
                    len(_BACKOFF_DELAYS),
                    delay,
                )
                time.sleep(delay)
            except anthropic.AuthenticationError:
                logger.error("Authentication failed — check ANTHROPIC_API_KEY")
                raise
            except anthropic.APIError as e:
                logger.error("Anthropic API error: %s", e)
                raise

        # Final attempt (no sleep after)
        try:
            return func(*args, **kwargs)
        except anthropic.RateLimitError:
            logger.error(
                "Rate limited after %d retries — giving up",
                len(_BACKOFF_DELAYS),
            )
            raise last_error  # type: ignore[misc]

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
        Send a chat completion request to Claude.

        Accepts Ollama-style messages and tools, translates them to
        Anthropic format, and returns a response dict compatible with
        the FORGE agent loop.

        Args:
            messages: List of message dicts (role + content).  System messages
                      are extracted and passed as the system parameter.
            model: Model name (defaults to configured model).
            tools: List of tool definitions (Ollama or Anthropic format).
            timeout: Request timeout in seconds.
            temperature: Sampling temperature.

        Returns:
            Dict with 'message' containing 'role', 'content', and
            optionally 'tool_calls' for compatibility with the agent loop.
        """
        model = model or self._default_model
        timeout = timeout or self._default_timeout

        # Extract system prompt and convert messages
        system_text, user_messages = self._extract_system(messages)
        converted_messages = self._convert_messages(user_messages)

        # Build request kwargs
        request_kwargs: Dict[str, Any] = {
            "model": model,
            "messages": converted_messages,
            "max_tokens": 4096,
            "temperature": temperature,
        }

        if system_text:
            request_kwargs["system"] = system_text

        if tools:
            request_kwargs["tools"] = self._convert_tools(tools)

        if timeout != self._default_timeout:
            # Override client timeout for this request
            request_kwargs["timeout"] = timeout

        def _do_create():
            return self._client.messages.create(**request_kwargs)

        response = self._call_with_retry(_do_create)

        # Convert Anthropic response back to Ollama-compatible format
        result = self._response_to_ollama_format(response)

        # Log usage
        usage = response.usage
        logger.debug(
            "Claude response: model=%s, input_tokens=%d, output_tokens=%d",
            model,
            usage.input_tokens,
            usage.output_tokens,
        )

        return result

    @staticmethod
    def _response_to_ollama_format(response) -> Dict[str, Any]:
        """
        Convert an Anthropic Messages response to Ollama-compatible format.

        Returns a dict with 'message' key containing role, content,
        and tool_calls (if present).
        """
        content_text = ""
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                content_text += block.text
            elif block.type == "tool_use":
                tool_calls.append({
                    "function": {
                        "name": block.name,
                        "arguments": block.input,
                    },
                })

        result: Dict[str, Any] = {
            "message": {
                "role": "assistant",
                "content": content_text,
            },
            "model": response.model,
            "done": True,
        }

        if tool_calls:
            result["message"]["tool_calls"] = tool_calls

        return result

    def generate_simple(
        self,
        prompt: str,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
        temperature: float = 0.3,
        think: bool = False,
    ) -> str:
        """
        Simple text generation from a single prompt.

        Useful for one-shot tasks like classification or summarization
        where you don't need tool calling or conversation history.

        Args:
            prompt: The text prompt.
            model: Override the default model.
            timeout: Override the default timeout.
            temperature: Sampling temperature.
            think: If True, enables extended thinking (Claude's chain-of-thought
                   reasoning mode).  The thinking tokens are consumed but only
                   the final answer text is returned.

        Returns:
            The response text directly.
        """
        model = model or self._default_model
        timeout = timeout or self._default_timeout

        request_kwargs: Dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 4096,
        }

        if think:
            # Extended thinking requires temperature=1 and uses the
            # thinking block type.  Budget tokens control how much
            # reasoning the model can do.
            request_kwargs["temperature"] = 1
            request_kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": 2048,
            }
        else:
            request_kwargs["temperature"] = temperature

        if timeout != self._default_timeout:
            request_kwargs["timeout"] = timeout

        def _do_create():
            return self._client.messages.create(**request_kwargs)

        response = self._call_with_retry(_do_create)

        # Extract text content (skip thinking blocks)
        text_parts = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)

        return "".join(text_parts)

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
        """
        Verify API key works by making a minimal API call.

        Returns True if the key is valid and the API is reachable.
        """
        if not self._api_key:
            return False

        try:
            # Minimal request to verify authentication
            self._client.messages.create(
                model=self._default_model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
            )
            return True
        except anthropic.AuthenticationError:
            logger.warning("Claude health check failed: invalid API key")
            return False
        except anthropic.RateLimitError:
            # Rate limited means the key works -- we're just sending too much
            return True
        except Exception as e:
            logger.warning("Claude health check failed: %s", e)
            return False

    def list_models(self) -> List[str]:
        """
        Return available Claude models.

        The Anthropic API doesn't have a list-models endpoint like Ollama,
        so this returns the known model catalogue.
        """
        return list(CLAUDE_MODELS)

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self) -> "ClaudeAdapter":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
