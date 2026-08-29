# Day 5 — Streaming Responses + Lower-Latency Voice Pipeline

## Goal

Make the Day 4 voice loop feel faster by streaming the LLM response into TTS instead of waiting for the complete response.

```text
Day 4:
STT → wait for complete LLM → complete TTS → Speaker

Day 5:
STT → LLM streaming → sentence chunk → TTS → Speaker
                       ↓
                    next chunk
```

Ollama supports streaming responses and documents streaming as useful for real-time generation and lower perceived latency. Streaming can be enabled in its SDK with `stream=True`.
Reference: https://docs.ollama.com/capabilities/streaming

---

## 1. Keep the existing architecture

Keep the components from Days 1–4:

```text
AudioRecorder
SpeechToText
WakeWordDetector
LLM
TextToSpeech
```

Add:

```text
ResponseStreamer
```

Suggested structure:

```text
backend/app/
├── audio/
├── wakeword/
├── stt/
├── llm/
│   └── ollama.py
├── tts/
│   └── kokoro.py
└── pipeline/
    └── response_streamer.py
```

---

## 2. Add streaming to the LLM abstraction

Add:

```python
def stream_chat(messages):
    ...
```

Example:

```python
from ollama import chat

stream = chat(
    model="your-model",
    messages=messages,
    stream=True,
)

for chunk in stream:
    text = chunk.message.content
```

Keep Ollama-specific code inside the LLM implementation so the rest of the assistant does not depend directly on Ollama.

---

## 3. Accumulate the complete response

Streaming should not replace conversation history.

```python
assistant_response = ""

for chunk in llm.stream_chat(messages):
    assistant_response += chunk
```

After generation:

```python
messages.append({
    "role": "assistant",
    "content": assistant_response,
})
```

You therefore have two paths:

```text
                    ┌→ Complete response → Conversation history
LLM streaming ──────┤
                    └→ Partial chunks → TTS
```

Ollama's documentation specifically recommends accumulating streamed content to preserve conversation history.

---

## 4. Do not send individual tokens to TTS

Avoid:

```text
"The" → TTS
" answer" → TTS
" is" → TTS
```

Instead accumulate:

```text
The
The answer
The answer is
The answer is relatively
The answer is relatively simple.
```

Then send a useful chunk to TTS.

For V1, sentence-level chunks are sufficient.

---

## 5. Implement a simple text chunker

Create:

```text
pipeline/response_streamer.py
```

Conceptually:

```python
buffer = ""

for token in llm.stream_chat(messages):
    buffer += token

    if should_flush(buffer):
        text_queue.put(buffer)
        buffer = ""

if buffer:
    text_queue.put(buffer)
```

Start with simple punctuation:

```python
def should_flush(text: str) -> bool:
    return text.endswith((".", "?", "!"))
```

Do not build sophisticated NLP sentence segmentation yet.

---

## 6. Add a minimum chunk size

A punctuation-only approach can produce tiny chunks such as:

```text
"Yes."
```

Add a configurable value:

```yaml
tts:
  min_chunk_characters: 20
```

Treat this as a tuning parameter.

You are balancing:

```text
smaller chunks
    ↓
lower time-to-first-audio
    ↓
more TTS overhead
```

against:

```text
larger chunks
    ↓
higher time-to-first-audio
    ↓
more efficient TTS
```

---

## 7. Introduce a text queue

The LLM must not wait for TTS.

```text
LLM producer
     ↓
Text Queue
     ↓
TTS consumer
```

Example architecture:

```text
                 ┌──────────────┐
                 │    Ollama    │
                 └──────┬───────┘
                        │
                  text chunks
                        │
                        ▼
                 ┌──────────────┐
                 │  Text Queue  │
                 └──────┬───────┘
                        │
                        ▼
                 ┌──────────────┐
                 │    Kokoro    │
                 └──────────────┘
```

---

## 8. Separate TTS generation from playback

Use two queues:

```text
LLM
 ↓
Text Queue
 ↓
TTS Worker
 ↓
Audio Queue
 ↓
Audio Player
```

This allows:

```text
Sentence A → Kokoro → Audio A → playing

while simultaneously:

Sentence B → Kokoro → Audio B
```

The audio player should remain responsible for playback ordering.

---

## 9. Target pipeline

```text
                    ┌──────────────┐
                    │    Ollama    │
                    └──────┬───────┘
                           │
                    streamed text
                           │
                           ▼
                    ┌──────────────┐
                    │ Text Buffer  │
                    └──────┬───────┘
                           │
                     sentence
                           │
                           ▼
                    ┌──────────────┐
                    │  Text Queue  │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │  TTS Worker  │
                    │    Kokoro    │
                    └──────┬───────┘
                           │
                      audio chunks
                           │
                           ▼
                    ┌──────────────┐
                    │ Audio Queue  │
                    └──────┬───────┘
                           │
                           ▼
                         Speaker
```

---

## 10. Guarantee sequential playback

If the LLM produces:

```text
Sentence A.
Sentence B.
Sentence C.
```

the speaker must produce:

```text
A → B → C
```

Never allow B to start before A finishes.

---

## 11. Add cancellation infrastructure

Eventually the user should be able to interrupt Jarvis.

Cancellation needs to affect:

```text
LLM generation
TTS generation
queued audio
current playback
```

For Day 5, implement only the infrastructure.

Conceptually:

```python
cancel_event.set()
```

Workers check:

```python
if cancel_event.is_set():
    return
```

Do not implement a voice "stop" command yet.

---

## 12. Update the state machine

Keep the logical states:

```text
LISTENING
    ↓
RECORDING
    ↓
TRANSCRIBING
    ↓
THINKING
    ↓
SPEAKING
    ↓
LISTENING
```

Internally, however, generation and speaking now overlap:

```text
Ollama
  ↓
Text Queue
  ↓
TTS
  ↓
Audio Queue
  ↓
Speaker
```

While sentence 1 is playing, Ollama can generate sentence 2.

---

## 13. Measure latency

Record:

```text
T0 = user stops speaking
T1 = STT finished
T2 = first LLM token
T3 = first TTS chunk ready
T4 = first audio playback
```

Calculate:

```text
STT latency          = T1 - T0
LLM first-token      = T2 - T1
TTS first-chunk      = T3 - T2
Time-to-first-audio  = T4 - T0
```

The most important metric is:

**Time-to-first-audio.**

Ollama also exposes generation statistics such as evaluation count and evaluation duration in the final streaming response, which can be used to calculate token generation speed.

---

## 14. Compare Day 4 and Day 5

### Day 4

```text
User stops speaking
        ↓
STT
        ↓
Complete LLM response
        ↓
Complete TTS
        ↓
Audio
```

### Day 5

```text
User stops speaking
        ↓
STT
        ↓
LLM starts streaming
        ↓
First sentence
        ↓
TTS
        ↓
Audio starts
        ↓
LLM continues generating
```

Benchmark both on your actual hardware.

Do not assume streaming is faster for every workload; measure time-to-first-audio.

---

## 15. Handle short responses

For:

```text
"Yes."
```

there is little benefit to complicated streaming.

Ollama's documentation notes that non-streaming can be simpler for short responses or structured output.

For V1, keep one streaming path and optimize this later.

---

## 16. Preserve TTS preprocessing

Do not send raw LLM output directly to Kokoro.

Use:

```text
LLM
 ↓
Markdown/code cleanup
 ↓
Number/URL normalization
 ↓
Text Queue
 ↓
Kokoro
```

Reuse the TTS preprocessing from Day 3.

---

## 17. Don't implement web search yet

The eventual architecture can become:

```text
LLM
 ↓
Tool call
 ↓
Search API
 ↓
Search result
 ↓
LLM
 ↓
TTS
```

Ollama's current chat API also supports tool calls, including streamed tool-call events.

But do not implement this on Day 5.

Keep the scope focused on:

```text
LLM → TTS latency
```

---

# Day 5 Definition of Done

- [ ] Ollama streaming enabled.
- [ ] `LLM.stream_chat()` abstraction exists.
- [ ] Complete assistant response is accumulated.
- [ ] Conversation history still works.
- [ ] LLM output is split into reasonable TTS chunks.
- [ ] Text queue implemented.
- [ ] TTS worker implemented.
- [ ] Audio queue implemented.
- [ ] Audio playback is sequential.
- [ ] Next TTS chunk can be generated while previous audio is playing.
- [ ] Basic cancellation mechanism exists.
- [ ] TTS preprocessing still works.
- [ ] Time-to-first-audio is measured.
- [ ] Day 4 wake-word flow still works.
- [ ] Streaming is perceptibly faster than waiting for the complete response.

---

# What NOT to build

```text
❌ Web search
❌ MCP
❌ RAG
❌ Prompt caching
❌ Long-term memory
❌ Custom wake-word training
❌ Speaker identification
❌ React UI
❌ Distributed architecture
❌ Cloud APIs
❌ Complex NLP sentence segmentation
❌ Token-level TTS
```

---

# Final Day 5 Architecture

```text
                       ┌──────────────┐
                       │  Microphone  │
                       └──────┬───────┘
                              │
                              ▼
                       ┌──────────────┐
                       │ Wake Word    │
                       │ openWakeWord │
                       └──────┬───────┘
                              │
                         Hey Jarvis
                              │
                              ▼
                       ┌──────────────┐
                       │     STT      │
                       │ faster-      │
                       │ whisper      │
                       └──────┬───────┘
                              │
                              ▼
                             Text
                              │
                              ▼
                       ┌──────────────┐
                       │    Ollama    │
                       │   Streaming  │
                       └──────┬───────┘
                              │
                       streamed text
                              │
                              ▼
                       ┌──────────────┐
                       │ Text Buffer  │
                       └──────┬───────┘
                              │
                        text chunks
                              │
                              ▼
                       ┌──────────────┐
                       │  Text Queue  │
                       └──────┬───────┘
                              │
                              ▼
                       ┌──────────────┐
                       │    Kokoro    │
                       │  TTS Worker  │
                       └──────┬───────┘
                              │
                        audio chunks
                              │
                              ▼
                       ┌──────────────┐
                       │ Audio Queue  │
                       └──────┬───────┘
                              │
                              ▼
                           Speaker
```

## Day 5 Milestone

Jarvis should now feel less like:

> "I will finish thinking, then eventually answer."

and more like:

> "I understood you and have already started answering."

The core improvement:

```text
DAY 4

STT → [wait for LLM] → [wait for TTS] → SPEAK


DAY 5

STT → LLM streaming → TTS chunk 1 → SPEAK
                       ↓
                    TTS chunk 2
                       ↓
                    TTS chunk 3
```

This is the first major latency/UX optimization rather than another major subsystem.

## References

- Ollama Streaming API: https://docs.ollama.com/api/streaming
- Ollama Streaming Capabilities: https://docs.ollama.com/capabilities/streaming
- Ollama API Reference: https://docs.ollama.com/api
