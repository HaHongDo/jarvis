from app.llm import LLMResponse, OllamaLLM


def test_chat_returns_nonempty_response():
    llm = OllamaLLM()
    messages = [{"role": "user", "content": "Say exactly: hello"}]

    response = llm.chat(messages)

    assert isinstance(response, LLMResponse)
    assert response.content
    assert not response.has_tool_calls()
