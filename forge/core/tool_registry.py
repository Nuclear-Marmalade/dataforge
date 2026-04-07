"""
FORGE Tool Registry — Manages tool definitions and dispatch.

Manages tool definitions, registration, and dispatch.
  - Tools register with a name, description, parameter schema, and execute function
  - Registry provides tool definitions for the model prompt
  - Registry dispatches tool calls to the correct handler

Dependencies: None (this is a leaf module)
Depended on by: agent_loop.py, enrichment workers
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("forge.tool_registry")


@dataclass
class ToolDefinition:
    """Schema for a tool that the model can call."""
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema for parameters


class Tool(ABC):
    """
    Base class for all FORGE tools.

    Subclasses must implement:
      - name: str property
      - description: str property
      - parameters: dict property (JSON schema)
      - execute(arguments: dict) -> Any
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique tool name."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of what this tool does."""
        ...

    @property
    @abstractmethod
    def parameters(self) -> Dict[str, Any]:
        """JSON Schema for the tool's parameters."""
        ...

    @abstractmethod
    def execute(self, arguments: Dict[str, Any]) -> Any:
        """
        Execute the tool with the given arguments.

        Args:
            arguments: Dict matching the parameters schema.

        Returns:
            Result of the tool execution (dict, string, or any serializable type).

        Raises:
            Exception on failure — the agent loop handles retries.
        """
        ...

    def get_definition(self) -> ToolDefinition:
        """Return the tool definition for model prompting."""
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
        )


class SimpleTool(Tool):
    """
    Convenience wrapper for creating tools from functions.

    Usage:
        def my_tool(query: str) -> dict:
            return {"result": query.upper()}

        tool = SimpleTool(
            name="uppercase",
            description="Convert text to uppercase",
            parameters={"type": "object", "properties": {"query": {"type": "string"}}},
            func=my_tool,
        )
    """

    def __init__(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        func: Callable[[Dict[str, Any]], Any],
    ):
        self._name = name
        self._description = description
        self._parameters = parameters
        self._func = func

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters(self) -> Dict[str, Any]:
        return self._parameters

    def execute(self, arguments: Dict[str, Any]) -> Any:
        return self._func(arguments)


class ToolRegistry:
    """
    Registry of available tools.

    Maintains a dictionary of Tool instances indexed by name.
    Provides tool definitions for the model and dispatches calls.
    """

    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool. Overwrites if name already exists."""
        if tool.name in self._tools:
            logger.warning("Overwriting existing tool: %s", tool.name)
        self._tools[tool.name] = tool
        logger.debug("Registered tool: %s", tool.name)

    def register_function(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        func: Callable,
    ) -> None:
        """Convenience method to register a function as a tool."""
        self.register(SimpleTool(name, description, parameters, func))

    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a tool by name. Returns None if not found."""
        return self._tools.get(name)

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """
        Get all tool definitions formatted for the model prompt.

        Returns a list of dicts with name, description, and parameters.
        This format is compatible with Ollama's tool calling API.
        """
        definitions = []
        for tool in self._tools.values():
            definitions.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            })
        return definitions

    def list_tools(self) -> List[str]:
        """Return list of registered tool names."""
        return list(self._tools.keys())

    def count(self) -> int:
        """Return number of registered tools."""
        return len(self._tools)
