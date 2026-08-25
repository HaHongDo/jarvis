from app.llm import OllamaLLM


def test_chat_returns_nonempty_response():
    llm = OllamaLLM()
    messages = [{"role": "user", "content": "Say exactly: hello"}]

    response = llm.chat(messages)

    assert response
    assert isinstance(response, str)
