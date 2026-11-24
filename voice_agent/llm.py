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
                 max_tokens: int = 512, reasoning_effort: str = "low") -> None:
        self.model = model
        # gpt-oss thinks before it answers, and those thinking tokens count against max_tokens.
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
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
        """Warm up the LLM by making a dummy request."""
        list(self.stream([{"role": "user", "content": "Say hello."}]))

    def stream(self, messages: list[dict]) -> Iterator[str]:
        """Yield the answer text chunk by chunk. The model's reasoning is skipped."""
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
