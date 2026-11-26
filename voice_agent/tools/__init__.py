"""The tool registry: every tool the LLM may call is declared here; the logic lives in sibling modules.

To add a tool: implement it in its own module (like timers.py), give ToolBox the state it needs, and add a
Tool entry to TOOLS below.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass

from .timers import TimerStore

LABEL = {"type": "string", "description": "What the timer is for, e.g. tea or pasta."}
OPTIONAL_LABEL = {
    "type": "string",
    "description": "What the timer is for. Leave it out only when exactly one timer exists.",
}


def minutes(description: str) -> dict:
    return {"type": "number", "description": description}


def parameters(properties: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": properties, "required": required,
            "additionalProperties": False}


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict                    # JSON schema of the arguments
    handler: Callable[..., dict]        # handler(toolbox, **arguments) -> result dict


TOOLS = [
    Tool(
        "start_timer",
        "Start a new countdown timer. Ask the user for the length first if they did not say it.",
        parameters({"label": LABEL, "minutes": minutes("Length in minutes; may be a fraction.")},
                   ["label", "minutes"]),
        lambda box, **arguments: box.timers.start(**arguments),
    ),
    Tool(
        "list_timers",
        "List every timer with its length and the seconds left. Use it to answer how much time is left.",
        parameters({}, []),
        lambda box: box.timers.list(),
    ),
    Tool(
        "set_timer_length",
        "Change a timer to a new total length, still counted from when it was started.",
        parameters({"label": OPTIONAL_LABEL, "minutes": minutes("The new total length in minutes.")},
                   ["minutes"]),
        lambda box, **arguments: box.timers.set_length(**arguments),
    ),
    Tool(
        "extend_timer",
        "Add time to a running timer.",
        parameters({"label": OPTIONAL_LABEL, "minutes": minutes("Minutes to add.")}, ["minutes"]),
        lambda box, **arguments: box.timers.extend(**arguments),
    ),
    Tool(
        "cancel_timer",
        "Cancel and remove a timer.",
        parameters({"label": OPTIONAL_LABEL}, []),
        lambda box, **arguments: box.timers.cancel(**arguments),
    ),
]


class ToolBox:
    """The state behind the tools for one session, and the bridge between the LLM and the tools."""

    def __init__(self, timers: TimerStore | None = None) -> None:
        self.timers = timers or TimerStore()
        self.tools = {tool.name: tool for tool in TOOLS}

    @property
    def schemas(self) -> list[dict]:
        """The tool list in the form the OpenAI chat API takes as `tools=`."""
        return [
            {"type": "function",
             "function": {"name": tool.name, "description": tool.description,
                          "parameters": tool.parameters}}
            for tool in self.tools.values()
        ]

    def call(self, name: str, arguments_json: str) -> str:
        """Run one tool call from the model and return its result as JSON text. Never raises."""
        tool = self.tools.get(name)
        if tool is None:
            return json.dumps({"error": f"Unknown tool {name}."})
        try:
            arguments = json.loads(arguments_json) if arguments_json.strip() else {}
            if not isinstance(arguments, dict):
                raise ValueError("arguments must be a JSON object")
            result = tool.handler(self, **arguments)
        except (ValueError, TypeError) as error:   # bad JSON, missing or unexpected arguments
            result = {"error": f"Bad arguments for {name}: {error}"}
        return json.dumps(result)
