# Day 6 — Tool Calling Foundation

## Goal

Day 6 introduces the foundation for giving Jarvis external capabilities.

Target loop:

```text
User
 ↓
STT
 ↓
LLM
 ↓
LLM decides it needs a tool
 ↓
Tool executes
 ↓
Tool result
 ↓
LLM
 ↓
Final response
 ↓
TTS
```

Do not build the complete web-search system or MCP integration yet. First prove that the tool-calling architecture works locally.

## 1. Keep the existing architecture

Keep the components from Days 1–5:

```text
AudioRecorder
SpeechToText
WakeWordDetector
LLM
TextToSpeech
ResponseStreamer
```

Add:

```text
tools/
├── __init__.py
├── registry.py
├── calculator.py
├── time.py
└── fake_search.py
```

## 2. Don't build MCP yet

For Day 6, use an internal tool registry:

```text
LLM
 ↓
Tool Registry
 ├── calculator
 ├── get_time
 └── search
```

Later this can become:

```text
LLM
 ↓
MCP Client
 ↓
MCP Server
 ↓
Tool
```

The goal is to establish the tool contract before introducing another protocol.

## 3. Create a Tool abstraction

Conceptually:

```python
class Tool:
    name: str
    description: str
    parameters: dict

    def execute(self, arguments):
        ...
```

The LLM should only need the tool's name, description, and parameters.

## 4. Create a Tool Registry

Create `tools/registry.py`.

Conceptually:

```python
class ToolRegistry:

    def register(self, tool):
        ...

    def get(self, name):
        ...

    def execute(self, name, arguments):
        ...
```

Register:

```python
registry.register(CalculatorTool())
registry.register(TimeTool())
```

The registry is the controlled gateway through which the model accesses capabilities.

## 5. Start with simple tools

### Calculator

```text
calculator(expression)
```

Example:

```text
calculator("123 * 456")
```

### Current time

```text
get_time(timezone)
```

Example:

```text
get_time("Asia/Ho_Chi_Minh")
```

These are deliberately simple. The objective is testing the architecture.

## 6. Give the LLM tool definitions

The LLM needs a structured description of each available tool.

Conceptually:

```json
{
  "type": "function",
  "function": {
    "name": "calculator",
    "description": "Performs mathematical calculations.",
    "parameters": {
      "type": "object",
      "properties": {
        "expression": {
          "type": "string"
        }
      },
      "required": ["expression"]
    }
  }
}
```

Keep this model/provider-specific representation inside the LLM adapter.

## 7. Add tools to the LLM abstraction

Existing interface:

```python
llm.chat(messages)
```

Change toward:

```python
llm.chat(
    messages=messages,
    tools=tools,
)
```

And for streaming:

```python
llm.stream_chat(
    messages=messages,
    tools=tools,
)
```

Keep Ollama-specific structures inside `llm/ollama.py`.

## 8. Detect tool calls

The model response can be either normal text or a tool call.

```python
response = llm.chat(...)

if response.has_tool_calls():
    ...
else:
    ...
```

Architecture:

```text
                  LLM
                   │
             ┌─────┴─────┐
             │           │
         text answer   tool call
             │           │
             ▼           ▼
           TTS        Execute tool
                         │
                         ▼
                    Tool result
                         │
                         ▼
                        LLM
                         │
                         ▼
                        TTS
```

## 9. Execute the tool

Example model request:

```json
{
  "name": "calculator",
  "arguments": {
    "expression": "123 * 456"
  }
}
```

Execute:

```python
result = registry.execute(
    "calculator",
    {
        "expression": "123 * 456"
    }
)
```

Result:

```text
56088
```

## 10. Send the tool result back to the LLM

The conversation should effectively become:

```text
User:
"What is 123 × 456?"

LLM:
"I need the calculator."

Application:
calculator("123 * 456")

Tool:
56088

Application → LLM:
"calculator returned 56088"

LLM:
"123 × 456 is 56,088."

TTS:
"123 times 456 is 56,088."
```

The LLM turns raw tool output into the final natural-language response.

## 11. Never send raw tool output directly to TTS

Use:

```text
Tool
 ↓
LLM
 ↓
Natural-language response
 ↓
TTS
```

not:

```text
Tool
 ↓
TTS
```

## 12. Support multiple tool calls

Eventually the model may request multiple tools:

```text
tool A
tool B
tool C
```

Support a collection of tool calls in the response. Sequential execution is sufficient for Day 6.

## 13. Add a maximum tool-call loop

Prevent accidental loops:

```python
MAX_TOOL_ROUNDS = 5
```

Conceptually:

```python
for _ in range(MAX_TOOL_ROUNDS):
    response = llm.chat(...)

    if not response.has_tool_calls():
        break

    execute_tools(...)
```

If the limit is reached, stop execution and return an error.

## 14. Add tool errors

Tools can fail.

Return structured information:

```json
{
  "success": false,
  "error": "Invalid expression"
}
```

Give the result back to the LLM so it can explain the failure naturally.

## 15. Validate tool arguments

Never blindly trust model-generated arguments.

Architecture:

```text
LLM
 ↓
Schema validation
 ↓
Tool
```

This becomes especially important when tools eventually perform privileged operations.

## 16. Treat tools as privileged capabilities

Do not give the LLM arbitrary access to your application.

Bad:

```text
LLM
 ↓
arbitrary Python execution
```

Good:

```text
LLM
 ↓
approved tool
 ↓
validated arguments
 ↓
controlled execution
```

Explicitly expose approved capabilities instead of arbitrary shell, filesystem, or Python execution.

## 17. Add tool logging

Log:

```text
timestamp
tool name
arguments
execution duration
success/failure
result size
```

Example:

```text
[10:31:04] TOOL calculator
arguments={"expression":"123*456"}
duration=3ms
success=true
```

Be careful about logging sensitive arguments or results.

## 18. Integrate with the Day 5 streaming pipeline

```text
Microphone
    ↓
Wake Word
    ↓
STT
    ↓
User Text
    ↓
Ollama
    │
    ├── normal response ──────────────┐
    │                                 │
    └── tool call → Tool Registry     │
                         │             │
                         ▼             │
                    Tool Result        │
                         │             │
                         ▼             │
                       Ollama ─────────┘
                         │
                         ▼
                   Response Stream
                         │
                         ▼
                     Text Queue
                         │
                         ▼
                       Kokoro
                         │
                         ▼
                      Speaker
```

When streaming tool calls, do not feed partial tool-call fragments into TTS. Accumulate the tool-call data, execute the tool once the call is complete, then continue the model/tool loop.

## 19. Build a fake search tool

Do not implement real web search yet.

Create:

```text
search(query)
```

that returns hardcoded results.

Example:

```python
def search(query):
    return [
        {
            "title": "Example Result",
            "url": "https://example.com",
            "snippet": "This is an example search result."
        }
    ]
```

Test:

> "Search for information about Redis."

Expected flow:

```text
LLM
 ↓
search("Redis")
 ↓
fake search result
 ↓
LLM
 ↓
answer
```

## 20. Why this prepares you for the real search system

Once the fake search tool works, replace:

```text
FakeSearchTool
```

with:

```text
SearchTool
 ↓
Your Search API
 ↓
SearXNG
 ↓
Normalized results
 ↓
LLM
```

Later:

```text
LLM
 ↓
MCP Client
 ↓
Search MCP Server
 ↓
SearXNG
```

The Day 6 work remains useful because the core capability contract stays the same.

# Day 6 Definition of Done

- [ ] `Tool` abstraction created.
- [ ] `ToolRegistry` created.
- [ ] Calculator tool works.
- [ ] Time tool works.
- [ ] Ollama receives tool definitions.
- [ ] LLM tool calls are detected.
- [ ] Tool arguments are validated.
- [ ] Tool execution errors are handled.
- [ ] Tool results are returned to the LLM.
- [ ] LLM produces the final natural-language response.
- [ ] Final response goes through the Day 5 streaming/TTS pipeline.
- [ ] Maximum tool-call rounds implemented.
- [ ] Tool execution is logged.
- [ ] Fake search tool implemented.
- [ ] User can ask the LLM to perform a fake search.
- [ ] No real web-search implementation yet.

# What NOT to build

```text
❌ Real web search
❌ SearXNG integration
❌ MCP
❌ RAG
❌ Vector database
❌ Long-term memory
❌ Autonomous agents
❌ Browser automation
❌ Shell execution
❌ Complex tool planning
❌ Parallel tool execution
```

The objective is:

```text
LLM → Tool → Result → LLM
```

# Final Day 6 Architecture

```text
                         ┌──────────────┐
                         │  Microphone  │
                         └──────┬───────┘
                                │
                                ▼
                         ┌──────────────┐
                         │ Wake Word    │
                         └──────┬───────┘
                                │
                                ▼
                         ┌──────────────┐
                         │     STT      │
                         └──────┬───────┘
                                │
                                ▼
                              Text
                                │
                                ▼
                         ┌──────────────┐
                         │    Ollama    │
                         └──────┬───────┘
                                │
                     ┌──────────┴──────────┐
                     │                     │
                 normal text           tool call
                     │                     │
                     │                     ▼
                     │              ┌──────────────┐
                     │              │ Tool Registry│
                     │              └──────┬───────┘
                     │                     │
                     │                     ▼
                     │                Tool Result
                     │                     │
                     │                     ▼
                     │                 Ollama
                     │                     │
                     └──────────┬──────────┘
                                │
                                ▼
                         Response Stream
                                │
                                ▼
                           Text Queue
                                │
                                ▼
                             Kokoro
                                │
                                ▼
                            Speaker
```

## Day 6 Milestone

Jarvis should be able to handle:

> "What is 123 times 456?"

```text
STT
 ↓
Ollama
 ↓
calculator("123 * 456")
 ↓
56088
 ↓
Ollama
 ↓
"123 times 456 is 56,088."
 ↓
Kokoro
 ↓
Speaker
```

And:

> "Search for information about Redis."

```text
STT
 ↓
Ollama
 ↓
search("Redis")
 ↓
Fake search result
 ↓
Ollama
 ↓
Natural-language answer
 ↓
Kokoro
 ↓
Speaker
```

The important milestone is establishing a clean:

```text
LLM → Tool → Result → LLM
```

loop.

Once this works, Day 7 can replace the fake search implementation with the actual **SearXNG/search API + result normalization pipeline**.

## References

- Ollama Tool Calling: https://docs.ollama.com/capabilities/tool-calling
- Ollama API: https://docs.ollama.com/api
