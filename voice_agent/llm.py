import os
from collections.abc import Iterator
from pathlib import Path

import yaml
from openai import OpenAI

DEFAULT_CONFIG = Path(__file__).with_name("config.yaml")

# Environment variables that override a key of the "llm" section in the config file.
ENV_OVERRIDES = {"base_url": "LLM_BASE_URL", "model": "LLM_MODEL", "api_key": "LLM_API_KEY"}


class GPTOssLLM:
    """Client for a gpt-oss model served by an already-running vLLM (OpenAI-compatible) server."""

    def __init__(self, base_url: str, model: str, api_key: str = "EMPTY",
                 max_tokens: int = 512, reasoning_effort: str = "low",
                 system_prompt: str | None = None) -> None:
        self.model = model
        # gpt-oss thinks before it answers, and those thinking tokens count against max_tokens.
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.system_prompt = system_prompt
        # The conversation so far: user and assistant messages only (no system prompt, no reasoning).
        self.history: list[dict] = []
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    @classmethod
    def from_config(cls, path: Path = DEFAULT_CONFIG) -> "GPTOssLLM":
        """Build the client from the "llm" section of a YAML file; environment variables win over the file."""
        with open(path) as file:
            settings = yaml.safe_load(file)["llm"]
        for key, variable in ENV_OVERRIDES.items():
            if os.environ.get(variable):
                settings[key] = os.environ[variable]
        return cls(**settings)

    def warm_up(self) -> None:
        """Warm up the LLM by making a dummy request. Uses stream(), so it never enters the history."""
        list(self.stream([{"role": "user", "content": "Say hello."}]))

    def chat(self, user_text: str) -> Iterator[str]:
        """Answer one user turn with the whole conversation as context, yielding the answer chunk by chunk.

        Both sides of the turn are added to the history once the answer has finished streaming. If the
        caller stops early or the request fails, whatever part of the answer was produced is kept.
        """
        user_message = {"role": "user", "content": user_text}
        reply = ""
        try:
            for text in self.stream([*self.history, user_message]):
                reply += text
                yield text
        finally:
            self.history.append(user_message)
            self.history.append({"role": "assistant", "content": reply})

    def reset(self) -> None:
        """Forget the conversation, e.g. to start a new dialogue."""
        self.history = []

    def stream(self, messages: list[dict]) -> Iterator[str]:
        """Yield the answer text chunk by chunk. The model's reasoning is skipped.

        `messages` holds only the conversation (user and assistant turns); the system prompt is added here.
        """
        if self.system_prompt:
            messages = [{"role": "system", "content": self.system_prompt}, *messages]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
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
