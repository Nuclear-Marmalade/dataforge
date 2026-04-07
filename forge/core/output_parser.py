"""
FORGE Output Parser — Extracts tool calls and text from model responses.

Handles multiple output formats:
  1. Ollama native tool calling (structured JSON)
  2. Manual JSON extraction from text (fallback for models without native tool support)
  3. Markdown code block extraction

The parser is intentionally forgiving — it tries multiple strategies
before giving up. Local models are less reliable than cloud APIs at
producing perfectly formatted tool calls.

Dependencies: None
Depended on by: agent_loop.py
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger("forge.output_parser")


@dataclass
class ToolCall:
    """A parsed tool call from the model's response."""
    name: str
    arguments: Dict[str, Any]
    id: str = ""  # Optional tool call ID for tracking


class OutputParser:
    """
    Extracts tool calls and text from model responses.

    Tries multiple parsing strategies in order:
    1. Native Ollama tool call format (message.tool_calls)
    2. JSON object with "tool" and "arguments" keys in text
    3. JSON in markdown code blocks
    4. Plain text (no tool calls)
    """

    def extract_tool_calls(self, response: Any) -> List[ToolCall]:
        """
        Extract tool calls from a model response.

        Args:
            response: The raw response from the model adapter.
                     Can be a dict (Ollama format) or string.

        Returns:
            List of ToolCall objects. Empty if no tool calls found.
        """
        # Strategy 1: Ollama native tool calling
        if isinstance(response, dict):
            native_calls = self._parse_native_tool_calls(response)
            if native_calls:
                return native_calls

        # Get the text content
        text = self._get_text(response)
        if not text:
            return []

        # Strategy 2: JSON object with tool/arguments structure
        json_calls = self._parse_json_tool_calls(text)
        if json_calls:
            return json_calls

        # Strategy 3: JSON in code blocks
        code_block_calls = self._parse_code_block_tool_calls(text)
        if code_block_calls:
            return code_block_calls

        return []

    def extract_text(self, response: Any) -> Optional[str]:
        """Extract the plain text content from a model response."""
        return self._get_text(response)

    def _get_text(self, response: Any) -> Optional[str]:
        """Get text content from various response formats."""
        if isinstance(response, str):
            return response
        if isinstance(response, dict):
            # Ollama format
            msg = response.get("message", {})
            if isinstance(msg, dict):
                return msg.get("content", "")
            return response.get("response", response.get("content", ""))
        return None

    def _parse_native_tool_calls(self, response: dict) -> List[ToolCall]:
        """Parse Ollama's native tool call format."""
        calls: List[ToolCall] = []
        msg = response.get("message", {})
        if not isinstance(msg, dict):
            return calls

        tool_calls = msg.get("tool_calls", [])
        for i, tc in enumerate(tool_calls):
            if isinstance(tc, dict):
                func = tc.get("function", {})
                name = func.get("name", "")
                args = func.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {"raw": args}
                if name:
                    calls.append(ToolCall(
                        name=name,
                        arguments=args if isinstance(args, dict) else {},
                        id=f"call_{i}",
                    ))
        return calls

    def _parse_json_tool_calls(self, text: str) -> List[ToolCall]:
        """
        Extract tool calls from JSON objects embedded in text.

        Looks for patterns like:
          {"tool": "tool_name", "arguments": {...}}
        or:
          {"name": "tool_name", "input": {...}}
        """
        calls: List[ToolCall] = []
        # Find all JSON objects in the text
        json_objects = self._extract_json_objects(text)

        for obj in json_objects:
            # Try various key patterns
            name = obj.get("tool") or obj.get("name") or obj.get("function")
            args = obj.get("arguments") or obj.get("input") or obj.get("params") or obj.get("parameters")

            if name and isinstance(name, str):
                if args is None:
                    args = {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {"raw": args}
                calls.append(ToolCall(
                    name=name,
                    arguments=args if isinstance(args, dict) else {},
                    id=f"parsed_{len(calls)}",
                ))

        return calls

    def _parse_code_block_tool_calls(self, text: str) -> List[ToolCall]:
        """Extract tool calls from JSON in markdown code blocks."""
        calls: List[ToolCall] = []
        # Match ```json ... ``` or ``` ... ```
        pattern = r"```(?:json)?\s*\n?(.*?)\n?\s*```"
        matches = re.findall(pattern, text, re.DOTALL)

        for match in matches:
            try:
                obj = json.loads(match.strip())
                if isinstance(obj, dict):
                    name = obj.get("tool") or obj.get("name") or obj.get("function")
                    args = obj.get("arguments") or obj.get("input") or obj.get("params")
                    if name:
                        calls.append(ToolCall(
                            name=name,
                            arguments=args if isinstance(args, dict) else {},
                            id=f"codeblock_{len(calls)}",
                        ))
            except json.JSONDecodeError:
                continue

        return calls

    def _extract_json_objects(self, text: str) -> List[dict]:
        """
        Extract all valid JSON objects from a string.

        Uses bracket matching to find JSON boundaries.
        Handles nested objects and arrays.
        """
        objects = []
        i = 0
        while i < len(text):
            if text[i] == '{':
                # Try to find matching closing brace
                depth = 0
                start = i
                in_string = False
                escape_next = False

                for j in range(i, len(text)):
                    char = text[j]
                    if escape_next:
                        escape_next = False
                        continue
                    if char == '\\':
                        escape_next = True
                        continue
                    if char == '"' and not escape_next:
                        in_string = not in_string
                        continue
                    if in_string:
                        continue
                    if char == '{':
                        depth += 1
                    elif char == '}':
                        depth -= 1
                        if depth == 0:
                            candidate = text[start:j+1]
                            try:
                                obj = json.loads(candidate)
                                if isinstance(obj, dict):
                                    objects.append(obj)
                            except json.JSONDecodeError:
                                pass
                            i = j + 1
                            break
                else:
                    i += 1
            else:
                i += 1

        return objects


def strip_thinking_block(text: str) -> str:
    """
    Strip Gemma 4's <think>...</think> reasoning blocks from output.

    Gemma 4 models use thinking tokens that appear as <think>...</think>.
    The actual output follows after the closing tag.
    """
    if "</think>" in text:
        return text.split("</think>", 1)[-1].strip()
    return text


def extract_json_from_response(text: str) -> Optional[dict]:
    """
    Utility function: extract a single JSON object from model text.

    Used by enrichment tools that expect structured output.
    Handles Gemma 4's thinking blocks by stripping them first.
    Returns the first valid JSON object found, or None.
    """
    # Strip thinking blocks first
    text = strip_thinking_block(text)

    parser = OutputParser()
    objects = parser._extract_json_objects(text)
    return objects[0] if objects else None
