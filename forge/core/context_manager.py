"""
FORGE Context Manager — Conversation state and auto-compaction.

Implements conversation context management with auto-compaction.
  - Maintains conversation history as a list of messages
  - Tracks approximate token count
  - When context approaches the limit, summarizes old messages
  - Preserves the system prompt and recent messages

Critical for local models with limited context windows (8K-32K).
Without compaction, the agent loop would fail after a few turns.

Dependencies: None (uses model adapter for summarization)
Depended on by: agent_loop.py
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("forge.context_manager")


@dataclass
class Message:
    """A single message in the conversation."""
    role: str  # "system", "user", "assistant", "tool"
    content: str
    tool_call_id: Optional[str] = None
    tool_name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to Ollama message format."""
        d: Dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        return d

    def estimated_tokens(self) -> int:
        """Rough token estimate: ~4 chars per token for English text."""
        return max(1, len(self.content) // 4)


class ContextManager:
    """
    Manages conversation history with automatic compaction.

    When the conversation approaches the context window limit,
    older messages are summarized into a compact summary message,
    preserving the system prompt and the most recent messages.

    This is essential for long-running agent loops on local models
    with limited context (8K-32K tokens).
    """

    def __init__(
        self,
        max_tokens: int = 8192,
        compact_threshold: float = 0.75,
        preserve_recent: int = 6,
    ):
        """
        Args:
            max_tokens: Maximum context window size in tokens.
            compact_threshold: Fraction of max_tokens at which to trigger compaction.
            preserve_recent: Number of recent messages to preserve during compaction.
        """
        self._max_tokens = max_tokens
        self._compact_threshold = compact_threshold
        self._preserve_recent = preserve_recent
        self._system_prompt: Optional[str] = None
        self._messages: List[Message] = []
        self._compaction_count = 0

    def set_system_prompt(self, prompt: str) -> None:
        """Set the system prompt. Called once at agent initialization."""
        self._system_prompt = prompt

    def add_user_message(self, content: str) -> None:
        """Add a user message to the conversation."""
        self._messages.append(Message(role="user", content=content))

    def add_assistant_message(self, content: Any) -> None:
        """
        Add an assistant message. Accepts string or dict (raw model response).
        """
        if isinstance(content, dict):
            msg = content.get("message", {})
            text = msg.get("content", "") if isinstance(msg, dict) else str(content)
        else:
            text = str(content)
        self._messages.append(Message(role="assistant", content=text))

    def add_tool_result(self, tool_name: str, tool_call_id: str, result: str) -> None:
        """Add a tool execution result to the conversation."""
        self._messages.append(Message(
            role="tool",
            content=result,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
        ))

    def get_messages(self) -> List[Dict[str, Any]]:
        """
        Get the full message list in Ollama format.

        Returns system prompt + all messages.
        """
        msgs: List[Dict[str, Any]] = []
        if self._system_prompt:
            msgs.append({"role": "system", "content": self._system_prompt})
        for m in self._messages:
            msgs.append(m.to_dict())
        return msgs

    def estimated_tokens(self) -> int:
        """Estimate total tokens in the current context."""
        total = 0
        if self._system_prompt:
            total += len(self._system_prompt) // 4
        for m in self._messages:
            total += m.estimated_tokens()
        return total

    def needs_compaction(self) -> bool:
        """Check if the context is approaching the limit and needs compaction."""
        current = self.estimated_tokens()
        threshold = int(self._max_tokens * self._compact_threshold)
        return current > threshold

    def compact(self, model_adapter: Any = None) -> None:
        """
        Compact the conversation by summarizing older messages.

        If a model_adapter is provided, uses the model to generate a summary.
        Otherwise, falls back to a simple truncation strategy.

        Preserves:
          - System prompt (always)
          - Last N messages (preserve_recent)
          - A summary of everything in between
        """
        if len(self._messages) <= self._preserve_recent + 1:
            return  # Nothing to compact

        self._compaction_count += 1
        old_count = len(self._messages)

        # Split messages: old (to summarize) + recent (to keep)
        split_point = len(self._messages) - self._preserve_recent
        old_messages = self._messages[:split_point]
        recent_messages = self._messages[split_point:]

        # Generate summary
        if model_adapter is not None:
            summary = self._model_summary(old_messages, model_adapter)
        else:
            summary = self._simple_summary(old_messages)

        # Replace conversation with summary + recent
        self._messages = [
            Message(
                role="user",
                content=f"[CONTEXT SUMMARY — compaction #{self._compaction_count}]\n{summary}",
                metadata={"is_compaction": True, "summarized_messages": len(old_messages)},
            ),
        ] + recent_messages

        logger.info(
            "Compacted: %d messages → %d (summarized %d, preserved %d)",
            old_count, len(self._messages), len(old_messages), len(recent_messages),
        )

    def _model_summary(self, messages: List[Message], model_adapter: Any) -> str:
        """Use the model to summarize old messages."""
        try:
            # Build a compact representation of what happened
            content_parts = []
            for m in messages:
                prefix = m.role.upper()
                # Truncate long tool results
                content = m.content[:500] if m.role == "tool" else m.content[:1000]
                content_parts.append(f"[{prefix}] {content}")

            summary_prompt = (
                "Summarize the following conversation history in 3-5 bullet points. "
                "Focus on: what tasks were attempted, what succeeded, what failed, "
                "and what the current state is. Be concise.\n\n"
                + "\n".join(content_parts)
            )

            response = model_adapter.generate(
                messages=[{"role": "user", "content": summary_prompt}],
                model="gemma4",
                timeout=60.0,
            )

            if isinstance(response, dict):
                msg = response.get("message", {})
                return msg.get("content", "") if isinstance(msg, dict) else str(response)
            return str(response)

        except Exception as e:
            logger.warning("Model summary failed, using simple summary: %s", e)
            return self._simple_summary(messages)

    def _simple_summary(self, messages: List[Message]) -> str:
        """Fallback: create a summary without using the model."""
        tool_calls = [m for m in messages if m.role == "tool"]
        user_msgs = [m for m in messages if m.role == "user"]
        assistant_msgs = [m for m in messages if m.role == "assistant"]

        parts = [
            f"Previous conversation ({len(messages)} messages):",
            f"- {len(user_msgs)} user messages",
            f"- {len(assistant_msgs)} assistant responses",
            f"- {len(tool_calls)} tool calls executed",
        ]

        # Include the last user message for context
        if user_msgs:
            last_user = user_msgs[-1].content[:200]
            parts.append(f"- Last task: {last_user}")

        return "\n".join(parts)

    def message_count(self) -> int:
        """Return total message count."""
        return len(self._messages)

    def clear(self) -> None:
        """Clear all messages (preserves system prompt)."""
        self._messages = []
        self._compaction_count = 0
