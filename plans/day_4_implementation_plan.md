# Day 4 — Wake Word + Always-Listening

## Goal

Replace the Day 3 push-to-talk trigger with an always-listening wake-word detector.

```text
Microphone
    ↓
Always listening
    ↓
openWakeWord
    ↓
"Hey Jarvis"
    ↓
Record command
    ↓
faster-whisper
    ↓
Ollama
    ↓
Kokoro
    ↓
Speaker
```

For V1, use openWakeWord. It includes a pre-trained `hey jarvis` model, accepts 16 kHz 16-bit PCM audio, and recommends audio frames in multiples of 80 ms for good latency/efficiency.

Official documentation:
https://github.com/dscripka/openWakeWord

## 1. Install openWakeWord

```bash
pip install openwakeword
```

Download its pre-trained models:

```python
import openwakeword

openwakeword.utils.download_models()
```

## 2. Test "Hey Jarvis" independently

Create:

```text
backend/app/wakeword/
├── __init__.py
└── detector.py
```

Create:

```python
class WakeWordDetector:
    def process(self, audio_frame) -> bool:
        ...
```

Then:

```python
class OpenWakeWordDetector(WakeWordDetector):
    ...
```

Verify:

```text
Microphone
    ↓
audio frame
    ↓
openWakeWord
    ↓
detected = True
```

Saying "Hey Jarvis" should produce a detection.

## 3. Change microphone capture to continuous streaming

Day 3 used:

```text
Press Enter
    ↓
Record 15 seconds
```

Day 4 needs:

```text
Microphone
    ↓
continuous stream
    ↓
small audio frames
    ↓
wake-word detector
```

At 16 kHz, an 80 ms frame contains:

```text
16,000 × 0.08 = 1,280 samples
```

openWakeWord recommends 80 ms multiples for efficient, low-latency processing.

Your microphone layer should support something like:

```python
for frame in microphone.stream():
    detector.process(frame)
```

## 4. Separate microphone streaming from wake-word detection

Keep:

```text
Microphone
    ↓
AudioStream
    ↓
WakeWordDetector
```

as separate components.

Suggested structure:

```text
audio/
    microphone.py

wakeword/
    detector.py
```

This lets the same microphone system eventually support wake-word detection, VAD, recording, STT, and debugging.

## 5. Add the assistant state machine

```text
                ┌──────────────┐
                │   LISTENING  │
                └──────┬───────┘
                       │
                "Hey Jarvis"
                       │
                       ▼
                ┌──────────────┐
                │  RECORDING   │
                └──────┬───────┘
                       │
                       ▼
                ┌──────────────┐
                │ TRANSCRIBING │
                └──────┬───────┘
                       │
                       ▼
                ┌──────────────┐
                │   THINKING   │
                └──────┬───────┘
                       │
                       ▼
                ┌──────────────┐
                │   SPEAKING   │
                └──────┬───────┘
                       │
                       ▼
                ┌──────────────┐
                │   LISTENING  │
                └──────────────┘
```

`LISTENING` is now the default state.

## 6. Don't send wake-word audio directly to Whisper

If the user says:

```text
"Hey Jarvis, what's the weather?"
```

conceptually:

```text
"Hey Jarvis"
       ↓
Wake detector
       ↓
ACTIVATE
       ↓
capture command
       ↓
"What's the weather?"
       ↓
Whisper
```

For V1, don't try to perfectly remove the wake word from the same audio buffer. Detect the wake word and begin a new command-recording window.

## 7. Add a short activation delay

After detecting:

```text
Hey Jarvis
```

transition to:

```text
RECORDING
```

with a small configurable delay if necessary.

For example:

```text
Hey Jarvis
    ↓
~200 ms
    ↓
start command capture
```

Then the user's command is sent to Whisper.

## 8. Use VAD for command termination

Eventually:

```text
Wake word detected
        ↓
Start recording
        ↓
Speech detected
        ↓
Speech continues
        ↓
Silence
        ↓
Command finished
        ↓
Whisper
```

For Day 4, don't build sophisticated VAD from scratch.

Start with:

```yaml
audio:
  max_command_seconds: 15
  silence_timeout_ms: 1000
```

Use a maximum duration plus a silence timeout.

## 9. Tune false activations

Wake-word systems have two important failure modes:

### False rejection

You say:

```text
Hey Jarvis
```

but nothing happens.

### False acceptance

Background audio causes:

```text
Wake detected
```

when you did not intend to activate it.

openWakeWord returns a confidence score and recommends threshold tuning for the deployment environment. Its documentation also discusses techniques such as VAD gating and custom verifier models for reducing false activations.

Start with the default threshold and measure performance before changing it.

## 10. Create a wake-word evaluation test

Test repeated intentional activations:

```text
Hey Jarvis
Hey Jarvis
Hey Jarvis
```

Similar phrases:

```text
Hey Jason
Hey Travis
Okay Jarvis
Hi Jarvis
```

Background audio:

```text
YouTube
Music
TV
Conversation
Silence
```

Track:

```text
True detections
Missed detections
False activations
```

The important metrics are:

```text
false-positive rate
false-negative rate
```

## 11. Add activation debouncing

After:

```text
Hey Jarvis
```

is detected, prevent another wake detection from triggering the same command.

Use:

```text
LISTENING
    ↓
WAKE DETECTED
    ↓
RECORDING
```

and disable wake-word detection while the assistant is processing.

Only return to `LISTENING` after TTS finishes.

## 12. Ignore wake words while Jarvis is speaking

For V1:

```text
SPEAKING
    ↓
ignore microphone for wake activation
```

This prevents Jarvis's own voice from accidentally activating itself.

Use:

```text
LISTENING    → wake detection enabled
RECORDING    → disabled
TRANSCRIBING → disabled
THINKING     → disabled
SPEAKING     → disabled
```

Later, you can introduce acoustic echo cancellation or more advanced full-duplex handling.

## 13. Add the wake-word abstraction

The application should not directly depend on openWakeWord:

```python
class WakeWordDetector:
    def process(self, audio_frame) -> bool:
        ...
```

Implementation:

```python
class OpenWakeWordDetector(WakeWordDetector):
    ...
```

The assistant only needs to know:

```python
if wakeword.detected(frame):
    assistant.activate()
```

This follows the same abstraction pattern already used for:

```text
LLM
SpeechToText
TextToSpeech
AudioRecorder
```

## 14. Integrate the complete pipeline

```text
                  ┌──────────────┐
                  │  Microphone  │
                  └──────┬───────┘
                         │
                         ▼
                  ┌──────────────┐
                  │ Audio Stream │
                  └──────┬───────┘
                         │
                         ▼
                  ┌──────────────┐
                  │ openWakeWord │
                  └──────┬───────┘
                         │
                   Hey Jarvis
                         │
                         ▼
                  ┌──────────────┐
                  │   Recorder   │
                  └──────┬───────┘
                         │
                         ▼
                  ┌──────────────┐
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
                  └──────┬───────┘
                         │
                         ▼
                  Response Text
                         │
                         ▼
                  ┌──────────────┐
                  │   Kokoro     │
                  └──────┬───────┘
                         │
                         ▼
                      Speaker
```

## Day 4 Definition of Done

- [ ] `openwakeword` installed.
- [ ] Pre-trained `hey jarvis` model works.
- [ ] Microphone provides continuous audio frames.
- [ ] Audio is 16 kHz / mono / 16-bit PCM.
- [ ] Wake-word detection works without pressing Enter.
- [ ] Wake-word threshold has been tested/tuned.
- [ ] False-activation testing performed.
- [ ] Command recording starts after wake detection.
- [ ] Command recording has a maximum duration.
- [ ] Silence can terminate recording.
- [ ] Wake detection is disabled while processing.
- [ ] Wake detection is disabled while Jarvis speaks.
- [ ] Assistant returns to listening after TTS.
- [ ] `WakeWordDetector` is an abstraction.
- [ ] Full voice interaction works:

```text
"Hey Jarvis"
     ↓
"What is a goroutine?"
     ↓
Jarvis speaks answer
     ↓
back to listening
```

## What NOT to build

```text
❌ Custom wake-word training
❌ Speaker identification
❌ Personal voice recognition
❌ Streaming LLM → TTS
❌ Web search
❌ MCP
❌ RAG
❌ Long-term memory
❌ Redis
❌ Prompt caching
❌ React UI
❌ Distributed architecture
```

There is no need to train your own "Hey Jarvis" model yet because openWakeWord already includes a pre-trained model for that phrase.

Custom training can come later if the included model does not perform well enough in your environment.

## Day 4 Milestone

At the end of Day 4, you should be able to say:

> "Hey Jarvis."

Then:

> "Explain how Redis expiration works."

Jarvis should answer aloud without touching the keyboard and then return to its listening state.

The complete local voice-assistant loop is:

```text
always listening
      ↓
wake word
      ↓
speech
      ↓
STT
      ↓
LLM
      ↓
TTS
      ↓
speaker
      ↓
listening again
```

## Reference

openWakeWord:
https://github.com/dscripka/openWakeWord
