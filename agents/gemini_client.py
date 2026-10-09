"""Real Gemini-backed implementation of agents.loop.AgentClient, with one Langfuse
generation span per call. This is the only module that translates between loop.py's
plain-dict message history and google-genai's native Content/Part types -- the loop
itself never sees a google-genai type.
"""

import json
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types
from langfuse import get_client
from pydantic import BaseModel

import config
from agents.loop import ToolCallRequest, ToolTurn

load_dotenv()


def _to_contents(messages: list[dict]) -> list[types.Content]:
    contents = []
    for message in messages:
        role = message["role"]
        if role == "user":
            contents.append(types.Content(role="user", parts=[types.Part(text=message["content"])]))
        elif role == "model":
            parts = [
                types.Part(
                    function_call=types.FunctionCall(name=c["name"], args=c["arguments"]),
                    thought_signature=c.get("thought_signature"),
                )
                for c in message["tool_calls"]
            ]
            contents.append(types.Content(role="model", parts=parts))
        elif role == "tool":
            part = types.Part.from_function_response(
                name=message["name"], response={"result": message["content"]}
            )
            contents.append(types.Content(role="user", parts=[part]))
        else:
            raise ValueError(f"unknown message shape: {message!r}")
    return contents


class GeminiAgentClient:
    """One instance per agent (coverage/risk), holding that agent's system prompt."""

    def __init__(self, system_prompt: str):
        self._system_prompt = system_prompt
        self._genai = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self._langfuse = get_client()

    async def call_with_tools(
        self, agent_name: str, prompt_version: str, messages: list[dict], tools: list
    ) -> ToolTurn:
        gen_config = types.GenerateContentConfig(
            system_instruction=self._system_prompt,
            tools=tools,
            temperature=0,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        with self._langfuse.start_as_current_observation(
            name=f"{agent_name}.tool_turn",
            as_type="generation",
            model=config.GEMINI_MODEL_LIGHT,
            input=messages,
            metadata={"agent_name": agent_name, "prompt_version": prompt_version},
        ) as span:
            response = self._genai.models.generate_content(
                model=config.GEMINI_MODEL_LIGHT,
                contents=_to_contents(messages),
                config=gen_config,
            )
            # Iterate parts directly, not the response.function_calls shortcut: that
            # property drops each part's thought_signature, which the API requires
            # back verbatim on the next turn's replayed function_call part.
            parts = response.candidates[0].content.parts or []
            tool_calls = [
                ToolCallRequest(
                    name=part.function_call.name,
                    arguments=part.function_call.args or {},
                    thought_signature=part.thought_signature,
                )
                for part in parts
                if part.function_call is not None
            ]
            span.update(output=[c.model_dump() for c in tool_calls] if tool_calls else response.text)
        return ToolTurn(tool_calls)

    async def call_for_schema(
        self, agent_name: str, prompt_version: str, messages: list[dict], schema: type[BaseModel]
    ) -> dict:
        gen_config = types.GenerateContentConfig(
            system_instruction=self._system_prompt,
            temperature=0,
            response_mime_type="application/json",
            response_schema=schema,
        )
        with self._langfuse.start_as_current_observation(
            name=f"{agent_name}.final_answer",
            as_type="generation",
            model=config.GEMINI_MODEL_LIGHT,
            input=messages,
            metadata={"agent_name": agent_name, "prompt_version": prompt_version},
        ) as span:
            response = self._genai.models.generate_content(
                model=config.GEMINI_MODEL_LIGHT,
                contents=_to_contents(messages),
                config=gen_config,
            )
            raw = response.parsed.model_dump() if response.parsed else json.loads(response.text)
            span.update(output=raw)
        return raw
