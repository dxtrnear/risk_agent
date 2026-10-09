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
        elif role == "model" and "tool_calls" in message:
            parts = [
                types.Part(
                    function_call=types.FunctionCall(name=c["name"], args=c["arguments"]),
                    thought_signature=c.get("thought_signature"),
                )
                for c in message["tool_calls"]
            ]
            contents.append(types.Content(role="model", parts=parts))
        elif role == "model" and "text" in message:
            contents.append(types.Content(role="model", parts=[types.Part(text=message["text"])]))
        elif role == "tool":
            part = types.Part.from_function_response(
                name=message["name"], response={"result": message["content"]}
            )
            contents.append(types.Content(role="user", parts=[part]))
        else:
            raise ValueError(f"unknown message shape: {message!r}")
    return contents


class GeminiAgentClient:
    """One instance per agent (coverage/risk), holding that agent's system prompt.

    `parent_trace_id`/`parent_span_id` pin every generation span to an explicit
    parent (Langfuse's `trace_context`) rather than relying on ambient OTel
    context -- ambient nesting does not reliably survive the stdio_client/
    ClientSession anyio task-group boundaries a few frames up the call stack
    (confirmed broken in practice: every call was landing as its own trace).
    """

    def __init__(self, system_prompt: str, parent_trace_id: str, parent_span_id: str):
        self._system_prompt = system_prompt
        self._parent_trace_id = parent_trace_id
        self._parent_span_id = parent_span_id
        self._genai = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self._langfuse = get_client()
        self.input_tokens = 0
        self.output_tokens = 0

    def _trace_context(self) -> dict:
        return {"trace_id": self._parent_trace_id, "parent_span_id": self._parent_span_id}

    def _record_usage(self, response) -> None:
        usage = response.usage_metadata
        if usage is None:
            return
        self.input_tokens += usage.prompt_token_count or 0
        self.output_tokens += usage.candidates_token_count or 0

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
            trace_context=self._trace_context(),
        ) as span:
            response = self._genai.models.generate_content(
                model=config.GEMINI_MODEL_LIGHT,
                contents=_to_contents(messages),
                config=gen_config,
            )
            self._record_usage(response)
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
        text = response.text if not tool_calls else None
        return ToolTurn(tool_calls, text=text)

    async def call_for_schema(
        self, agent_name: str, prompt_version: str, messages: list[dict], schema: type[BaseModel]
    ) -> dict:
        # The model already reached its conclusion in the tool-turn text kept in
        # `messages` (role="model", "text"). This call must restate that
        # conclusion in the required schema, never re-decide from scratch --
        # re-deciding here would be a second, independent judgment silently
        # overriding the first one. The instruction must be a final USER turn,
        # not a system-prompt suffix: Gemini's API rejects any request whose
        # `contents` ends on a model turn ("Requests ending with a model turn
        # are not supported"), which a bare `messages` ending in {"role":
        # "model", "text": ...} does.
        restate_instruction = (
            "Ta conclusion a déjà été donnée ci-dessus. Mets-la simplement au "
            "format demandé, sans réévaluer le dossier."
        )
        gen_config = types.GenerateContentConfig(
            system_instruction=self._system_prompt,
            temperature=0,
            response_mime_type="application/json",
            response_schema=schema,
        )
        contents = _to_contents(messages)
        contents.append(types.Content(role="user", parts=[types.Part(text=restate_instruction)]))
        with self._langfuse.start_as_current_observation(
            name=f"{agent_name}.final_answer",
            as_type="generation",
            model=config.GEMINI_MODEL_LIGHT,
            input=messages,
            metadata={"agent_name": agent_name, "prompt_version": prompt_version},
            trace_context=self._trace_context(),
        ) as span:
            response = self._genai.models.generate_content(
                model=config.GEMINI_MODEL_LIGHT,
                contents=contents,
                config=gen_config,
            )
            self._record_usage(response)
            raw = response.parsed.model_dump() if response.parsed else json.loads(response.text)
            span.update(output=raw)
        return raw
