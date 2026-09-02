# Day 2 — Microphone + Speech-to-Text

## Goal

Replace the Day 1 text input with microphone input while keeping the Ollama/LLM layer unchanged.

```text
Microphone
    ↓
AudioRecorder
    ↓
16 kHz mono audio
    ↓
faster-whisper + Silero VAD
    ↓
Text
    ↓
Day 1 Ollama LLM
    ↓
Text response
```

The Day 2 success criterion is a simple push-to-talk voice interaction:

```text
Jarvis is ready.

Press Enter to speak...

[Recording...]

You:
"Explain how the Go scheduler works."

[Thinking...]

Jarvis:
"The Go scheduler..."
```

Do not implement wake-word detection, TTS, React, WebSockets, web search, MCP, RAG, or caching today.

---

## 1. Install audio dependencies

```bash
pip install sounddevice soundfile numpy
```

Keep `faster-whisper` from Day 1.

`sounddevice` handles microphone capture, `soundfile` can be used for WAV files, and NumPy provides the in-memory audio representation.

---

## 2. Test the microphone independently

Create:

```text
backend/app/audio/
├── __init__.py
└── microphone.py
```

Create an `AudioRecorder` abstraction that:

- enumerates available microphones
- selects the default/input device
- captures mono audio
- records at 16 kHz
- returns NumPy audio samples
- can save a recording as WAV

Start with a fixed recording duration, such as 5 seconds.

Test:

```text
Microphone → WAV → Playback
```

The recording should sound correct before introducing Whisper.

---

## 3. Build the STT abstraction

Create:

```text
backend/app/stt/
├── __init__.py
└── whisper.py
```

Use an interface similar to:

```python
class SpeechToText:
    def transcribe(self, audio) -> str:
        ...
```

Then implement it with faster-whisper:

```python
class FasterWhisperSTT(SpeechToText):
    ...
```

Keep Whisper-specific code inside this implementation so the rest of the assistant is not coupled to faster-whisper.

---

## 4. Load Whisper once

Load the Whisper model when the application starts rather than for every transcription.

Conceptually:

```text
Application starts
       ↓
Load Whisper
       ↓
Wait for speech
       ↓
Transcribe
       ↓
Wait for next speech
```

Do not repeatedly initialize the model.

faster-whisper exposes `WhisperModel` and supports CPU/GPU configurations and quantized inference.

Reference:

https://github.com/SYSTRAN/faster-whisper

---

## 5. Test WAV → text first

Before connecting the microphone:

```text
test_recording.wav
       ↓
faster-whisper
       ↓
"Hello, this is a microphone test."
```

This isolates transcription problems from microphone problems.

Test:

- short questions
- long questions
- pauses
- silence
- background noise
- normal conversational speech

Do not optimize the model yet.

---

## 6. Enable VAD

Use faster-whisper's built-in Silero VAD integration:

```python
segments, info = model.transcribe(
    audio,
    vad_filter=True,
)
```

faster-whisper integrates Silero VAD to filter portions of audio without speech. Its VAD parameters can be customized if necessary.

For example:

```python
segments, info = model.transcribe(
    audio,
    vad_filter=True,
    vad_parameters=dict(
        min_silence_duration_ms=500
    ),
)
```

Reference:

https://github.com/SYSTRAN/faster-whisper

Do not build your own VAD implementation yet.

---

## 7. Convert segments into text

faster-whisper returns transcription segments.

Combine them:

```python
text = " ".join(segment.text for segment in segments)
```

Then clean the result and return a plain string from the `SpeechToText` abstraction.

The rest of the application should not need to know about Whisper segments.

---

## 8. Add configuration

Keep audio/STT settings centralized:

```yaml
audio:
  sample_rate: 16000
  channels: 1
  max_recording_seconds: 15

stt:
  model: small
  device: cpu
  compute_type: int8
  language: en
  vad_filter: true
```

The exact model can be changed later without modifying application code.

---

## 9. Add recording limits

For Day 2, use push-to-talk with a maximum recording duration:

```yaml
audio:
  max_recording_seconds: 15
```

The flow is:

```text
Press Enter
    ↓
Record
    ↓
Maximum 15 seconds
    ↓
Stop
    ↓
Transcribe
```

Do not implement sophisticated end-of-speech detection yet.

---

## 10. Add latency logging

Measure:

```text
Recording duration
STT processing time
Transcription result
```

Example:

```text
[STT] recording: 4.21s
[STT] processing: 0.83s
[STT] text: "Explain goroutines"
```

This will eventually let you measure the complete voice-assistant latency.

---

## 11. Connect STT to the Day 1 LLM

Replace the Day 1:

```python
text = input("You: ")
```

with:

```text
Press Enter
    ↓
Microphone
    ↓
AudioRecorder
    ↓
SpeechToText
    ↓
text
    ↓
existing Ollama conversation
```

The LLM abstraction should not need to change.

The architecture should now be:

```text
Microphone
    ↓
AudioRecorder
    ↓
SpeechToText
    ↓
faster-whisper
    ↓
text
    ↓
LLM
    ↓
Ollama
```

This verifies that the Day 1 and Day 2 abstractions are working together.

---

## 12. Add prerecorded STT tests

Keep a small collection of WAV files:

```text
backend/tests/audio/
├── hello.wav
├── goroutine.wav
└── empty.wav
```

Use these to repeatedly test:

```text
WAV → STT → text
```

Do not require exact punctuation or exact wording in the tests. Verify important content instead.

For example:

```python
assert "goroutine" in text.lower()
```

is preferable to asserting an exact generated transcription.

---

## 13. Day 2 Definition of Done

- [ ] Microphone detected.
- [ ] Correct microphone selected.
- [ ] 16 kHz mono recording works.
- [ ] WAV recording sounds correct.
- [ ] faster-whisper loads successfully.
- [ ] WAV → text works.
- [ ] Microphone → text works.
- [ ] Silero VAD is enabled.
- [ ] STT latency is logged.
- [ ] STT is behind a `SpeechToText` abstraction.
- [ ] Day 1 Ollama receives the transcription.
- [ ] Multi-turn conversation works using voice.
- [ ] Prerecorded WAV tests exist.

---

## What NOT to build

```text
❌ OpenWakeWord
❌ "Hey Jarvis"
❌ Kokoro / TTS
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
❌ Custom VAD
```

Keep Day 2 deliberately small.

The next milestone is Day 3: **Text-to-Speech with Kokoro**, after which the core loop becomes:

```text
voice → text → LLM → voice
```
