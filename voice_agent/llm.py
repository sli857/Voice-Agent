import json
import os
import time
from collections.abc import Iterator
from pathlib import Path

import yaml
from openai import OpenAI

from .tools import ToolBox

DEFAULT_CONFIG = Path(__file__).with_name("config.yaml")

# Environment variables that override a key of the "llm" section in the config file.
ENV_OVERRIDES = {"base_url": "LLM_BASE_URL", "model": "LLM_MODEL", "api_key": "LLM_API_KEY"}

# Spoken when the model keeps calling tools without ever answering.
TOO_MANY_TOOL_ROUNDS = "Sorry, I could not finish that. Please try again."


class GPTOssLLM:
    """Client for a gpt-oss model served by an already-running vLLM (OpenAI-compatible) server."""

    def __init__(self, base_url: str, model: str, api_key: str = "EMPTY",
                 max_tokens: int = 512, reasoning_effort: str = "low",
                 system_prompt: str | None = None, max_tool_rounds: int = 4,
                 tools: ToolBox | None = None) -> None:
        self.model = model
        # gpt-oss thinks before it answers, and those thinking tokens count against max_tokens.
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.system_prompt = system_prompt
        # How many requests one user turn may make; every request after the first follows tool calls.
        self.max_tool_rounds = max_tool_rounds
        self.tools = tools
        # The conversation so far: user, assistant and tool messages (no system prompt, no reasoning).
        self.history: list[dict] = []
        # The tool calls made in the latest turn: name, arguments, result and how long the tool took.
        self.last_tool_calls: list[dict] = []
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    @classmethod
    def from_config(cls, path: Path = DEFAULT_CONFIG, tools: ToolBox | None = None) -> "GPTOssLLM":
        """Build the client from the "llm" section of a YAML file; environment variables win over the file."""
        with open(path) as file:
            settings = yaml.safe_load(file)["llm"]
        for key, variable in ENV_OVERRIDES.items():
            if os.environ.get(variable):
                settings[key] = os.environ[variable]
        return cls(**settings, tools=tools)

    def warm_up(self) -> None:
        """Warm up the LLM with a dummy request that carries the tool list, so the server has it cached.

        Nothing enters the history, and no tool is run.
        """
        for _ in self._request([{"role": "user", "content": "Say hello."}], tool_calls={}):
            pass

    def chat(self, user_text: str) -> Iterator[str]:
        """Answer one user turn with the whole conversation as context, yielding the spoken answer chunk by chunk.

        The model may call tools first; their calls and results are kept in the history together with the
        user message and the answer. If the caller stops early or the request fails, whatever part of the
        turn was produced is kept.
        """
        turn = [{"role": "user", "content": user_text}]
        self.last_tool_calls = []
        try:
            yield from self._respond(turn)
        finally:
            self.history.extend(turn)

    def reset(self) -> None:
        """Forget the conversation, e.g. to start a new dialogue."""
        self.history = []

    def stream(self, messages: list[dict]) -> Iterator[str]:
        """One plain request without tools: yield the answer text chunk by chunk. The reasoning is skipped.

        `messages` holds only the conversation (user and assistant turns); the system prompt is added here.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self._with_system_prompt(messages),
            max_tokens=self.max_tokens,
            reasoning_effort=self.reasoning_effort,
            stream=True,
        )
        for chunk in response:
            if not chunk.choices:
                continue
            text = chunk.choices[0].delta.content
            if text:
                yield text

    def _respond(self, turn: list[dict]) -> Iterator[str]:
        """Request, run the tools the model asks for, and request again until it answers without tools.

        Every message of the turn (assistant answers, tool calls, tool results) is appended to `turn`.
        """
        for _ in range(self.max_tool_rounds):
            tool_calls: dict[int, dict] = {}
            reply = ""
            try:
                for text in self._request([*self.history, *turn], tool_calls):
                    reply += text
                    yield text
            finally:
                message = {"role": "assistant", "content": reply}
                if tool_calls:
                    message["tool_calls"] = [tool_calls[index] for index in sorted(tool_calls)]
                turn.append(message)
            if not tool_calls:
                return
            for call in message["tool_calls"]:
                turn.append({"role": "tool", "tool_call_id": call["id"],
                             "content": self._run_tool(call["function"])})
        turn.append({"role": "assistant", "content": TOO_MANY_TOOL_ROUNDS})
        yield TOO_MANY_TOOL_ROUNDS

    def _request(self, messages: list[dict], tool_calls: dict[int, dict]) -> Iterator[str]:
        """One streamed request with the tool list. Yields answer text; collects tool calls into `tool_calls`.

        A tool call arrives in pieces: the first piece of each carries its id and name, the following ones
        carry more of its JSON arguments. `index` says which call a piece belongs to.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self._with_system_prompt(messages),
            max_tokens=self.max_tokens,
            reasoning_effort=self.reasoning_effort,
            tools=self.tools.schemas if self.tools else None,
            stream=True,
        )
        for chunk in response:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content
            for piece in delta.tool_calls or []:
                call = tool_calls.setdefault(piece.index, {
                    "id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                if piece.id:
                    call["id"] = piece.id
                if piece.function and piece.function.name:
                    call["function"]["name"] += piece.function.name
                if piece.function and piece.function.arguments:
                    call["function"]["arguments"] += piece.function.arguments

    def _run_tool(self, function: dict) -> str:
        started = time.perf_counter()
        if self.tools is None:
            result = json.dumps({"error": "No tools are available."})
        else:
            result = self.tools.call(function["name"], function["arguments"])
        self.last_tool_calls.append({
            "name": function["name"],
            "arguments": function["arguments"],
            "result": result,
            "milliseconds": round((time.perf_counter() - started) * 1000, 3),
        })
        return result

    def _with_system_prompt(self, messages: list[dict]) -> list[dict]:
        if self.system_prompt:
            return [{"role": "system", "content": self.system_prompt}, *messages]
        return messages
