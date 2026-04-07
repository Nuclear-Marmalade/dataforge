"""
FORGE Agent Loop — The core execution engine.

Implements the core agentic execution loop.
  1. Send conversation + tools to model
  2. Parse response for tool calls
  3. Execute tools
  4. Feed results back
  5. Check stopping conditions
  6. Auto-compact if context too long
  7. Repeat

This is the single most important file in FORGE.
Everything else plugs into this loop.

Dependencies:
  - adapters/ (model interface)
  - tools/ (tool registry)
  - core/context_manager.py (conversation management)
  - core/output_parser.py (extract tool calls from model output)

Depended on by:
  - enrichment/ (uses this loop to run enrichment agents)
  - coordinator.py (spawns multiple loops in parallel)
"""

from __future__ import annotations

import json
import logging
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from forge.core.context_manager import ContextManager
from forge.core.output_parser import OutputParser, ToolCall
from forge.core.tool_registry import ToolRegistry

logger = logging.getLogger("forge.agent_loop")


@dataclass
class AgentConfig:
    """Configuration for an agent run."""
    model: str = "gemma4"
    system_prompt: str = ""
    max_turns: int = 200
    max_retries_per_tool: int = 3
    max_consecutive_errors: int = 5
    context_window: int = 8192  # tokens — conservative for 8B model
    compact_threshold: float = 0.75  # compact when context is 75% full
    timeout_per_turn: float = 120.0  # seconds
    stop_sequences: List[str] = field(default_factory=lambda: ["TASK_COMPLETE", "TASK_FAILED", "NEED_HUMAN"])


@dataclass
class AgentResult:
    """Result of an agent run."""
    status: str  # "completed", "failed", "max_turns", "stopped"
    turns_used: int
    total_time: float
    tool_calls_made: int
    errors: List[str]
    final_output: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class AgentLoop:
    """
    The core FORGE agent execution engine.

    Takes a system prompt, a tool registry, and a model adapter.
    Runs autonomously until a stopping condition is met.

    Pattern derived from Claude Code's runAgent.ts:
    - conversation = [system_prompt, ...messages]
    - Each turn: send to model → parse tool calls → execute → feed back
    - Auto-compact when conversation grows too long
    - Circuit breaker on consecutive errors
    """

    def __init__(
        self,
        model_adapter: Any,  # OllamaAdapter or compatible
        tool_registry: ToolRegistry,
        config: AgentConfig,
        on_turn_complete: Optional[Callable] = None,
    ):
        self._model = model_adapter
        self._tools = tool_registry
        self._config = config
        self._context = ContextManager(
            max_tokens=config.context_window,
            compact_threshold=config.compact_threshold,
        )
        self._parser = OutputParser()
        self._on_turn_complete = on_turn_complete

        # State
        self._turn_count = 0
        self._tool_call_count = 0
        self._consecutive_errors = 0
        self._errors: List[str] = []
        self._running = False

    def run(self, initial_message: str) -> AgentResult:
        """
        Execute the agent loop from an initial user message.

        This is the main entry point. Blocks until the agent stops.

        Args:
            initial_message: The task/prompt to execute.

        Returns:
            AgentResult with status, turns used, errors, and final output.
        """
        start_time = time.time()
        self._running = True
        self._turn_count = 0
        self._tool_call_count = 0
        self._consecutive_errors = 0
        self._errors = []

        # Initialize conversation
        self._context.set_system_prompt(self._config.system_prompt)
        self._context.add_user_message(initial_message)

        # Provide tool definitions to context
        tool_defs = self._tools.get_tool_definitions()

        logger.info(
            "Agent starting — model=%s, tools=%d, max_turns=%d",
            self._config.model, len(tool_defs), self._config.max_turns,
        )

        final_output = None

        try:
            while self._running and self._turn_count < self._config.max_turns:
                self._turn_count += 1

                # ── Step 1: Check if context needs compaction ──
                if self._context.needs_compaction():
                    logger.info("Turn %d: compacting context", self._turn_count)
                    self._context.compact(self._model)

                # ── Step 2: Send conversation to model ──
                try:
                    messages = self._context.get_messages()
                    response = self._model.generate(
                        messages=messages,
                        tools=tool_defs,
                        model=self._config.model,
                        timeout=self._config.timeout_per_turn,
                    )
                    self._consecutive_errors = 0  # Reset on success
                except Exception as e:
                    self._consecutive_errors += 1
                    error_msg = f"Turn {self._turn_count}: model error — {e}"
                    logger.error(error_msg)
                    self._errors.append(error_msg)
                    if self._consecutive_errors >= self._config.max_consecutive_errors:
                        logger.error("Circuit breaker: %d consecutive errors", self._consecutive_errors)
                        break
                    time.sleep(2 ** min(self._consecutive_errors, 5))  # Exponential backoff
                    continue

                # ── Step 3: Parse response for tool calls ──
                tool_calls = self._parser.extract_tool_calls(response)
                text_response = self._parser.extract_text(response)

                # ── Step 4: Check stopping conditions ──
                if self._should_stop(text_response, tool_calls):
                    final_output = text_response
                    logger.info("Turn %d: stopping condition met", self._turn_count)
                    break

                # ── Step 5: If no tool calls, add response and continue ──
                if not tool_calls:
                    self._context.add_assistant_message(text_response or "")
                    final_output = text_response
                    # If model didn't call any tools and didn't hit stop sequence,
                    # it's probably done thinking
                    if text_response:
                        logger.info("Turn %d: model responded without tool calls", self._turn_count)
                        break
                    continue

                # ── Step 6: Execute tool calls ──
                self._context.add_assistant_message(response)

                for tc in tool_calls:
                    self._tool_call_count += 1
                    result = self._execute_tool(tc)
                    self._context.add_tool_result(tc.name, tc.id, result)

                # ── Step 7: Notify callback if registered ──
                if self._on_turn_complete:
                    try:
                        self._on_turn_complete(
                            turn=self._turn_count,
                            tool_calls=len(tool_calls),
                            total_tool_calls=self._tool_call_count,
                        )
                    except Exception:
                        pass  # Callback errors shouldn't kill the loop

        except KeyboardInterrupt:
            logger.warning("Agent interrupted by user")
            self._running = False
        except Exception as e:
            logger.error("Agent fatal error: %s\n%s", e, traceback.format_exc())
            self._errors.append(f"Fatal: {e}")

        elapsed = time.time() - start_time
        status = self._determine_status(final_output)

        result = AgentResult(
            status=status,
            turns_used=self._turn_count,
            total_time=elapsed,
            tool_calls_made=self._tool_call_count,
            errors=self._errors,
            final_output=final_output,
        )

        logger.info(
            "Agent finished — status=%s, turns=%d, tool_calls=%d, time=%.1fs, errors=%d",
            result.status, result.turns_used, result.tool_calls_made,
            result.total_time, len(result.errors),
        )

        return result

    def stop(self) -> None:
        """Gracefully stop the agent loop after current turn completes."""
        self._running = False

    def _execute_tool(self, tool_call: ToolCall) -> str:
        """
        Execute a single tool call with error handling and retries.

        Returns the tool result as a string.
        """
        tool = self._tools.get_tool(tool_call.name)
        if tool is None:
            error = f"Unknown tool: {tool_call.name}"
            logger.warning(error)
            return json.dumps({"error": error})

        retries = 0
        while retries <= self._config.max_retries_per_tool:
            try:
                result = tool.execute(tool_call.arguments)
                if isinstance(result, dict):
                    return json.dumps(result)
                return str(result)
            except Exception as e:
                retries += 1
                error_msg = f"Tool {tool_call.name} failed (attempt {retries}): {e}"
                logger.warning(error_msg)
                if retries > self._config.max_retries_per_tool:
                    self._errors.append(error_msg)
                    return json.dumps({"error": str(e), "retries_exhausted": True})
                time.sleep(1)

        return json.dumps({"error": "max retries exceeded"})

    def _should_stop(self, text: Optional[str], tool_calls: List[ToolCall]) -> bool:
        """Check if the agent should stop based on response content."""
        if not text:
            return False
        for seq in self._config.stop_sequences:
            if seq in text:
                return True
        return False

    def _determine_status(self, final_output: Optional[str]) -> str:
        """Determine the final status of the agent run."""
        if not self._running and self._turn_count < self._config.max_turns:
            return "stopped"
        if self._turn_count >= self._config.max_turns:
            return "max_turns"
        if self._consecutive_errors >= self._config.max_consecutive_errors:
            return "error_circuit_breaker"
        if final_output:
            for seq in self._config.stop_sequences:
                if seq in final_output:
                    if "TASK_COMPLETE" in final_output:
                        return "completed"
                    if "TASK_FAILED" in final_output:
                        return "failed"
                    if "NEED_HUMAN" in final_output:
                        return "needs_human"
            # Model responded but no stop signal — ended without explicit completion
            return "no_stop_signal"
        return "completed"
