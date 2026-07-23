"""Provider adapters (README.md section 4).

Each adapter normalizes its provider's tool-calling wire format into a common
AssistantTurn, so harness/episode.py's loop is provider-agnostic. Tool schemas
are defined once in harness/tools.py and translated here per provider.
"""

import json
from dataclasses import dataclass, field


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict


@dataclass
class AssistantTurn:
    text: str
    tool_calls: list = field(default_factory=list)
    stop_reason: str = ""
    raw: object = None


class ModelAdapter:
    """Common interface every provider adapter implements."""

    def next_action(self, messages, tool_schemas, system):
        raise NotImplementedError

    def assistant_message(self, turn):
        """Turn an AssistantTurn back into the provider's message-history format."""
        raise NotImplementedError

    def tool_result_message(self, tool_call, result_text):
        """Build the provider's message-history entry for one tool result."""
        raise NotImplementedError


class AnthropicAdapter(ModelAdapter):
    def __init__(self, model="claude-sonnet-4-5", max_tokens=2048):
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens

    @staticmethod
    def _convert_tools(tool_schemas):
        return [
            {"name": t["name"], "description": t["description"], "input_schema": t["input_schema"]}
            for t in tool_schemas
        ]

    def next_action(self, messages, tool_schemas, system):
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=messages,
            tools=self._convert_tools(tool_schemas),
        )
        text_parts = [b.text for b in response.content if b.type == "text"]
        tool_calls = [
            ToolCall(id=b.id, name=b.name, input=b.input)
            for b in response.content
            if b.type == "tool_use"
        ]
        return AssistantTurn(
            text="\n".join(text_parts),
            tool_calls=tool_calls,
            stop_reason=response.stop_reason,
            raw=response,
        )

    def assistant_message(self, turn):
        return {"role": "assistant", "content": turn.raw.content}

    def tool_result_message(self, tool_call, result_text):
        return {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": tool_call.id, "content": result_text}],
        }


class OpenAIAdapter(ModelAdapter):
    def __init__(self, model="gpt-4.1", max_tokens=2048):
        import openai

        self.client = openai.OpenAI()
        self.model = model
        self.max_tokens = max_tokens

    @staticmethod
    def _convert_tools(tool_schemas):
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["input_schema"],
                },
            }
            for t in tool_schemas
        ]

    def next_action(self, messages, tool_schemas, system):
        full_messages = [{"role": "system", "content": system}] + messages
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=full_messages,
            tools=self._convert_tools(tool_schemas),
        )
        choice = response.choices[0]
        msg = choice.message
        tool_calls = [
            ToolCall(id=tc.id, name=tc.function.name, input=json.loads(tc.function.arguments))
            for tc in (msg.tool_calls or [])
        ]
        return AssistantTurn(
            text=msg.content or "",
            tool_calls=tool_calls,
            stop_reason=choice.finish_reason,
            raw=msg,
        )

    def assistant_message(self, turn):
        return turn.raw.model_dump(exclude_none=True)

    def tool_result_message(self, tool_call, result_text):
        return {"role": "tool", "tool_call_id": tool_call.id, "content": result_text}
