# Day 1 — Local LLM Foundation

## Goal

By the end of Day 1, the project should have a working local LLM that can be called from Python:

```text
Python application
       │
       ▼
Ollama
       │
       ▼
Local LLM
       │
       ▼
Text response
```

Do **not** implement STT, TTS, wake-word detection, web search, RAG, MCP, caching, or the React frontend today.

The only success criterion is:

> A Python program sends a prompt to a locally running model and receives a response.

Ollama exposes a local API at `http://localhost:11434/api` by default and provides official Python and JavaScript libraries. See the official documentation:
- https://docs.ollama.com/quickstart
- https://docs.ollama.com/api/introduction

---

## 1. Create the project

Suggested structure:

```text
jarvis/
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   └── llm/
│   │       ├── __init__.py
│   │       └── ollama.py
│   ├── tests/
│   │   └── test_ollama.py
│   ├── requirements.txt
│   └── README.md
└── .gitignore
```

Create a virtual environment inside `backend`:

```bash
cd backend
python -m venv .venv
```

Activate it.

### Windows PowerShell

```powershell
.\.venv\Scripts\Activate.ps1
```

### Linux/macOS

```bash
source .venv/bin/activate
```

Verify:

```bash
python --version
pip --version
```

Use Python 3.11 or 3.12 for this first version.

---

## 2. Install Ollama

Install Ollama for your operating system.

Official installation information:

https://docs.ollama.com/quickstart

Ollama currently supports macOS, Windows, and Linux.

After installation, verify that the CLI is available:

```bash
ollama --version
```

Then check that Ollama is running.

Its default local API is:

```text
http://localhost:11434/api
```

You can verify the service with:

```bash
curl http://localhost:11434/api/version
```

If you're on PowerShell and do not have `curl` behaving as expected, you can instead test the endpoint through Python or a browser.

---

## 3. Choose a first model

For Day 1, choose a model that your machine can comfortably run.

Do not optimize model quality yet.

The purpose of today's exercise is to prove the architecture, not benchmark models.

Example:

```bash
ollama pull gemma3
```

Then run it interactively:

```bash
ollama run gemma3
```

Ask:

```text
Explain what a goroutine is in Go.
```

Confirm that the model responds.

Exit the interactive session when finished.

### Important

The exact model is intentionally left configurable.

Your application should not become coupled to one model.

Use configuration such as:

```text
OLLAMA_MODEL=gemma3
```

rather than hardcoding the model name throughout the codebase.

---

## 4. Understand the first API call

Ollama's API supports chat requests such as:

```text
POST http://localhost:11434/api/chat
```

Conceptually:

```json
{
  "model": "gemma3",
  "messages": [
    {
      "role": "user",
      "content": "Explain what a goroutine is."
    }
  ]
}
```

The response contains the model's generated message.

The official quickstart demonstrates the `/api/chat` endpoint, while the API documentation covers the available endpoints and request formats.

Reference:

https://docs.ollama.com/api/introduction

---

## 5. Install the Python client

Add the official Ollama Python library:

```bash
pip install ollama
```

Then record the dependency:

```bash
pip freeze > requirements.txt
```

The official Ollama project provides `ollama-python` as its Python library.

---

## 6. Create an LLM abstraction

Create:

```text
backend/app/llm/ollama.py
```

Do not call Ollama directly from `main.py`.

Create a small abstraction around it.

The intended interface is approximately:

```python
class LLM:
    def chat(self, messages: list[dict]) -> str:
        ...
```

Then implement:

```python
class OllamaLLM(LLM):
    ...
```

The important design goal is:

```text
Application
     │
     ▼
   LLM interface
     │
     ▼
Ollama implementation
```

rather than:

```text
Application
     │
     ▼
Ollama-specific API calls everywhere
```

This will make it easy to replace Ollama later.

---

## 7. Implement the first chat call

Your Python code should:

1. Create an Ollama client.
2. Select the configured model.
3. Send a list of chat messages.
4. Extract the assistant's response.
5. Return it as a string.

Use a simple conversation:

```text
System:
You are Jarvis, a concise and helpful local virtual assistant.

User:
What is a goroutine?
```

Expected result:

```text
Jarvis:
A goroutine is a lightweight concurrent execution unit in Go...
```

The exact wording will depend on the model.

---

## 8. Create a minimal CLI

Create:

```text
backend/app/main.py
```

For Day 1, make it interactive:

```text
Jarvis is ready.

You: What is a goroutine?
Jarvis: A goroutine is...

You: Explain channels.
Jarvis: ...

You: exit
Goodbye.
```

The loop should conceptually be:

```text
read input
    ↓
LLM.chat()
    ↓
print response
    ↓
repeat
```

No audio yet.

---

## 9. Add conversation history

Do not send every question independently.

Maintain messages:

```python
messages = [
    {
        "role": "system",
        "content": "You are Jarvis, a helpful local virtual assistant."
    }
]
```

After each user message:

```text
messages.append(user message)
```

After each model response:

```text
messages.append(assistant message)
```

The next request therefore contains the previous conversation.

Example:

```text
User:
What is Go?

Assistant:
Go is a programming language...

User:
Who created it?

Assistant:
Go was created at Google by...
```

The model should be able to resolve `"it"` from the conversation history.

For Day 1, keep the entire history in memory.

Do not implement:

- Redis
- databases
- persistent memory
- summarization
- embeddings
- vector databases

---

## 10. Add a system prompt

Create a single system prompt:

```text
You are Jarvis, a local virtual assistant.

Be concise when answering simple questions.
Give detailed explanations when the user asks for them.
Do not claim to have access to the internet unless a web-search tool is explicitly provided.
```

Keep this prompt in configuration or a dedicated constant rather than scattering it through the code.

This will eventually become the foundation for the assistant's personality and tool instructions.

---

## 11. Add basic error handling

Your application should handle at least:

### Ollama unavailable

Display:

```text
Unable to connect to Ollama.
Make sure Ollama is running.
```

### Model unavailable

Display:

```text
The configured model is not available.
Run the appropriate `ollama pull` command.
```

### Empty input

Ignore it.

### User exits

Support:

```text
exit
quit
```

---

## 12. Add basic logging

For now, simple logging is enough.

Record:

```text
LLM request started
LLM request completed
LLM request failed
```

Also record elapsed time:

```text
LLM latency: 2.84s
```

You will eventually need latency measurements because voice assistants are extremely sensitive to response time.

---

## 13. Write one integration test

Create:

```text
backend/tests/test_ollama.py
```

The test should verify:

```text
Python
  ↓
Ollama
  ↓
model
  ↓
response
```

A simple test can ask:

```text
Say exactly: hello
```

and verify that a non-empty response was returned.

Do not make the test depend on exact generated wording.

Bad:

```python
assert response == "Hello!"
```

Better:

```python
assert response
assert isinstance(response, str)
```

The test is checking connectivity and integration, not model intelligence.

---

## 14. Verify the complete Day 1 flow

Run:

```bash
python -m app.main
```

Then test several prompts.

### Test 1 — Simple question

```text
You:
What is a goroutine?
```

### Test 2 — Follow-up

```text
You:
How does it relate to channels?
```

### Test 3 — Context

```text
You:
What programming language were we discussing?
```

The model should understand the conversation history.

### Test 4 — Exit

```text
You:
exit
```

---

# Day 1 Definition of Done

Do not move on to Day 2 until all of these work:

- [ ] Python virtual environment works.
- [ ] Ollama is installed.
- [ ] Ollama is running locally.
- [ ] A local model can be pulled.
- [ ] The model works through `ollama run`.
- [ ] Python can connect to Ollama.
- [ ] Python can send a chat request.
- [ ] Python receives a response.
- [ ] The model is accessed through an `LLM` abstraction.
- [ ] The model name is configurable.
- [ ] Conversation history works.
- [ ] A system prompt is present.
- [ ] Basic Ollama errors are handled.
- [ ] Basic latency logging exists.
- [ ] One integration test passes.
- [ ] The CLI can conduct a multi-turn conversation.

---

# What NOT to build today

It is important to resist expanding the scope.

Do **not** add:

```text
❌ Whisper
❌ Microphone capture
❌ OpenWakeWord
❌ Kokoro
❌ TTS
❌ FastAPI
❌ React
❌ WebSockets
❌ Web search
❌ SearXNG
❌ MCP
❌ RAG
❌ Vector DB
❌ Redis
❌ Prompt caching
❌ Semantic caching
❌ Long-term memory
❌ Agent framework
```

Today's architecture should remain:

```text
┌───────────────┐
│ Python CLI    │
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ LLM interface │
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ Ollama        │
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ Local model   │
└───────────────┘
```

Once this works, Day 2 can replace the CLI's text input with **microphone → speech-to-text**, while leaving the LLM layer unchanged.

---

# Useful official references

- Ollama documentation: https://docs.ollama.com/
- Ollama quickstart: https://docs.ollama.com/quickstart
- Ollama API introduction: https://docs.ollama.com/api/introduction
- Ollama GitHub: https://github.com/ollama/ollama
