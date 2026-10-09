"""_to_contents message translation -- pure function, no LLM call."""

from agents.gemini_client import _to_contents


def test_user_turn_becomes_text_content():
    contents = _to_contents([{"role": "user", "content": "bonjour"}])
    assert len(contents) == 1
    assert contents[0].role == "user"
    assert contents[0].parts[0].text == "bonjour"


def test_model_turn_becomes_function_call_part():
    messages = [{"role": "model", "tool_calls": [{"name": "get_contract", "arguments": {}}]}]
    contents = _to_contents(messages)
    assert contents[0].role == "model"
    assert contents[0].parts[0].function_call.name == "get_contract"


def test_model_turn_replays_thought_signature():
    """Regression test: the Gemini API rejects a replayed function_call part that
    drops the thought_signature the model originally attached to it."""
    messages = [
        {
            "role": "model",
            "tool_calls": [{"name": "get_contract", "arguments": {}, "thought_signature": b"sig123"}],
        }
    ]
    contents = _to_contents(messages)
    assert contents[0].parts[0].thought_signature == b"sig123"


def test_model_turn_without_signature_defaults_to_none():
    messages = [{"role": "model", "tool_calls": [{"name": "get_contract", "arguments": {}}]}]
    contents = _to_contents(messages)
    assert contents[0].parts[0].thought_signature is None


def test_tool_turn_becomes_function_response_part():
    messages = [{"role": "tool", "name": "get_contract", "content": '{"formule": "tous_risques"}'}]
    contents = _to_contents(messages)
    assert contents[0].role == "user"
    function_response = contents[0].parts[0].function_response
    assert function_response.name == "get_contract"
    assert function_response.response == {"result": '{"formule": "tous_risques"}'}


def test_full_conversation_role_order_preserved():
    messages = [
        {"role": "user", "content": "bonjour"},
        {"role": "model", "tool_calls": [{"name": "get_contract", "arguments": {"x": 1}}]},
        {"role": "tool", "name": "get_contract", "content": "ok"},
    ]
    contents = _to_contents(messages)
    assert [c.role for c in contents] == ["user", "model", "user"]


def test_unknown_role_raises():
    import pytest

    with pytest.raises(ValueError, match="unknown message shape"):
        _to_contents([{"role": "system", "content": "nope"}])
