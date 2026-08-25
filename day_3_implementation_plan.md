# Day 3 — Text-to-Speech with Kokoro

## Goal

Take the Day 2 LLM response and turn it into spoken audio.

```text
Microphone
    ↓
faster-whisper
    ↓
Text
    ↓
Ollama
    ↓
Response text
    ↓
Kokoro
    ↓
Audio
    ↓
Speaker
```

By the end of Day 3, Jarvis should hear a spoken question, generate an LLM response, and speak the answer back.

Kokoro is an open-weight 82M-parameter TTS model. Its official Python examples use `KPipeline` and generate 24 kHz audio.

Official repository:
https://github.com/NVIDIA/kokoro

## 1. Install Kokoro

Inside the existing virtual environment:

```bash
pip install "kokoro>=0.9.4" soundfile
```

Kokoro also uses `espeak-ng` for phoneme generation/fallback.

On Windows, install eSpeak NG separately and verify:

```bash
espeak-ng --version
```

## 2. Test Kokoro independently

Create:

```text
backend/app/tts/
├── __init__.py
└── kokoro.py
```

Create:

```python
class TextToSpeech:
    def synthesize(self, text: str):
        ...
```

Then:

```python
class KokoroTTS(TextToSpeech):
    ...
```

Start with:

```text
"Hello, I am Jarvis."
```

Generate:

```text
hello.wav
```

Verify that you can actually hear the generated audio.

## 3. Play generated audio

Use the audio dependencies from Day 2:

```text
Text
  ↓
Kokoro
  ↓
Audio samples
  ↓
sounddevice
  ↓
Speaker
```

Do not connect it to the LLM until this standalone test works.

## 4. Choose a voice

Make the voice configurable:

```yaml
tts:
  language: en-us
  voice: af_heart
  speed: 1.0
```

Kokoro provides multiple voices and language/accent options. Start with one voice and don't implement dynamic voice switching yet.

## 5. Keep TTS behind an abstraction

The rest of the application should use something like:

```python
tts.speak(text)
```

or:

```python
audio = tts.synthesize(text)
```

It should not directly call Kokoro APIs.

Architecture:

```text
Assistant
    ↓
TextToSpeech
    ↓
KokoroTTS
    ↓
Kokoro
```

This keeps the assistant independent from the specific TTS engine.

## 6. Load Kokoro once

Initialize the TTS pipeline when the application starts:

```text
Application starts
       ↓
Load Kokoro
       ↓
Wait
       ↓
Generate speech
       ↓
Play
       ↓
Wait for next response
```

Do not reload the model for every response.

## 7. Test different response lengths

Test:

```text
Hello.
```

A normal response:

```text
Go is a statically typed programming language designed at Google.
```

And a longer response.

Evaluate:

- pronunciation
- pauses
- sentence boundaries
- generation speed
- audio quality
- memory usage

Don't optimize streaming yet.

## 8. Add basic TTS preprocessing

Create:

```text
LLM response
     ↓
TTS preprocessing
     ↓
Kokoro
```

Remove or transform things that should not be spoken literally, such as Markdown:

```text
**bold**
```

or code formatting:

```text
`fmt.Println()`
```

For example:

```text
"Use `fmt.Println()` to print a value."
```

can become:

```text
"Use fmt.Println() to print a value."
```

Keep preprocessing conservative.

## 9. Connect TTS to the Day 2 pipeline

Replace:

```text
Microphone
    ↓
Whisper
    ↓
Ollama
    ↓
print(response)
```

with:

```text
Microphone
    ↓
Whisper
    ↓
Ollama
    ↓
Kokoro
    ↓
Speaker
```

Keep printing the response while debugging so you can compare the LLM output with what was spoken.

## 10. Add TTS latency logging

Measure:

```text
LLM response length
TTS generation time
Audio duration
```

Example:

```text
[LLM] response: 284 chars
[TTS] generation: 0.72s
[TTS] audio duration: 4.13s
```

Eventually total perceived latency will be:

```text
Recording
    +
STT
    +
LLM
    +
TTS
    +
Playback
```

## 11. Do not implement streaming yet

For Day 3:

```text
Complete LLM response
        ↓
Complete TTS generation
        ↓
Play audio
```

Do not implement token-by-token LLM → TTS streaming yet. Make the basic pipeline reliable first.

## 12. Update the assistant state machine

Day 2:

```text
IDLE
  ↓
RECORDING
  ↓
TRANSCRIBING
  ↓
THINKING
  ↓
IDLE
```

Day 3:

```text
IDLE
  ↓
RECORDING
  ↓
TRANSCRIBING
  ↓
THINKING
  ↓
SPEAKING
  ↓
IDLE
```

This will become important when the wake-word/always-listening system is introduced.

## 13. Day 3 Definition of Done

- [ ] Kokoro installed locally.
- [ ] eSpeak NG works.
- [ ] Kokoro model loads successfully.
- [ ] Text → audio works.
- [ ] Generated audio plays through the speaker.
- [ ] At least one voice is selected.
- [ ] TTS is behind a `TextToSpeech` abstraction.
- [ ] Kokoro is initialized only once.
- [ ] TTS latency is logged.
- [ ] Basic text preprocessing exists.
- [ ] Day 2 STT → Ollama → Kokoro pipeline works.
- [ ] Jarvis can answer a spoken question aloud.
- [ ] Assistant state machine includes `SPEAKING`.

## What NOT to build

```text
❌ OpenWakeWord
❌ "Hey Jarvis"
❌ Streaming TTS
❌ Streaming Whisper
❌ React
❌ WebSockets
❌ Web search
❌ SearXNG
❌ MCP
❌ RAG
❌ Redis
❌ Prompt caching
❌ Long-term memory
❌ Speaker identification
```

Keep Day 3 focused on completing the core voice loop.

## Final Day 3 Architecture

```text
                   ┌──────────────┐
                   │ Microphone   │
                   └──────┬───────┘
                          │
                          ▼
                   ┌──────────────┐
                   │ AudioRecorder│
                   └──────┬───────┘
                          │
                          ▼
                   ┌──────────────┐
                   │ faster-      │
                   │ whisper      │
                   └──────┬───────┘
                          │
                          ▼
                         text
                          │
                          ▼
                   ┌──────────────┐
                   │    Ollama    │
                   └──────┬───────┘
                          │
                          ▼
                    response text
                          │
                          ▼
                   ┌──────────────┐
                   │   Kokoro     │
                   └──────┬───────┘
                          │
                          ▼
                       Speaker
```

At the end of Day 3:

```text
voice → STT → LLM → TTS → voice
```

Day 4 can focus specifically on the wake-word / always-listening layer.
