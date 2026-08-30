# Jarvis backend — Day 7

Always-listening voice CLI: wake word -> speech-to-text -> local LLM through Ollama (with
tool calling) -> text-to-speech.

```text
Microphone -> MicrophoneStream -> OpenWakeWordDetector -> "Hey Jarvis"
    -> AudioRecorder (command, VAD-terminated) -> FasterWhisperSTT
    -> LLM interface -> Ollama -> local model
        -> normal text -> ResponseStreamer -> KokoroTTS -> AudioPlayer -> Speaker
        -> tool call -> ToolRegistry -> tool result -> back to Ollama -> (repeat)
```

### Tool calling

The LLM can call tools registered in `app/tools/`:

| Tool         | Purpose                                             |
|--------------|------------------------------------------------------|
| `calculator` | Evaluates a math expression (safe AST eval, no `eval()`) |
| `get_time`   | Current date/time for an IANA timezone               |
| `search`     | Web search backed by a local SearXNG instance          |

`ResponseStreamer` runs the tool-calling loop itself: if the model asks for a tool instead
of answering, the tool is executed via `ToolRegistry` (which validates arguments against
the tool's JSON-schema `parameters` and logs every call), the result is fed back to the
model, and the loop repeats — up to `tools.max_rounds` in `config.yaml` — until the model
answers in plain text. Raw tool output is never sent to TTS directly; only the model's
natural-language response is spoken.

### Web search (`app/search/`)

`search` is backed by a local [SearXNG](https://docs.searxng.org/) instance's JSON API:

```text
search(query) -> SearchService
    -> SearchCache (normalized-query, TTL)     [cache miss ->]
    -> SearXNGClient (HTTP GET /search?format=json)
    -> SearchNormalizer (drop malformed/missing fields, dedup by normalized URL)
    -> top N results
    -> SearchContextBuilder (Markdown-ish text for the LLM, with source URLs)
```

Requires a local SearXNG instance with `search.formats: [json]` enabled in its
`settings.yml` (see the [Day 7 plan](../day_7_implementation_plan.md)). If SearXNG is
unreachable, times out, or returns no results, the tool returns a plain-language message
(e.g. "I couldn't reach the web search service right now.") instead of raising, so the
assistant can still respond.

## Setup

```bash
cd backend
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Install and start [Ollama](https://docs.ollama.com/quickstart), then pull a model:

```bash
ollama pull gemma3
```

Kokoro (TTS) uses eSpeak NG for phoneme generation/fallback. Install it separately and verify:

```bash
espeak-ng --version
```

openWakeWord needs its pre-trained models downloaded once:

```bash
python -c "import openwakeword; openwakeword.utils.download_models()"
```

## Configuration

Ollama connection is configured via environment variables:

| Variable       | Default                   | Purpose                     |
|----------------|----------------------------|------------------------------|
| `OLLAMA_HOST`  | `http://localhost:11434`   | Ollama server address        |
| `OLLAMA_MODEL` | `gemma3`                   | Model used for chat requests |

Audio and STT settings live in `config.yaml`:

```yaml
audio:
  sample_rate: 16000
  channels: 1
  max_recording_seconds: 15
  frame_ms: 80              # wake-word / VAD frame size
  silence_timeout_ms: 1000  # command recording stops after this much silence
  silence_threshold: 0.02   # RMS energy (0-1) above which a frame counts as speech

stt:
  model: small
  device: cpu
  compute_type: int8
  language: en
  vad_filter: true

tts:
  language: en-us
  voice: af_heart
  speed: 1.0

wakeword:
  model: hey_jarvis
  threshold: 0.5             # openWakeWord confidence threshold
  activation_delay_ms: 200   # pause between "Hey Jarvis" and command capture

tools:
  max_rounds: 5   # max LLM<->tool round-trips per user turn before giving up

search:
  searxng_url: http://localhost:8080
  timeout_seconds: 5
  fetch_results: 10   # raw results considered before dedup/normalization
  max_results: 5      # results kept after normalization, sent to the LLM
  cache_ttl_seconds: 900
  language: en
  safesearch: 1
```

`search.searxng_url` can also be overridden with the `SEARXNG_URL` environment variable.

## Run

```bash
python -m app.main
```

Say "Hey Jarvis", then speak your command. Jarvis records until you go quiet
(or `max_recording_seconds` elapses), transcribes, replies, and goes back to
listening automatically. Press Ctrl+C to quit.

The assistant cycles through: `LISTENING -> RECORDING -> TRANSCRIBING ->
THINKING -> SPEAKING -> LISTENING`. Wake-word detection is only active while
`LISTENING`, so Jarvis can't be triggered by its own voice or while it's
already handling a command.

To manually test wake-word detection accuracy (true detections, missed
detections, false activations against background noise/similar phrases):

```bash
python -m app.wakeword.evaluate
```

To try Kokoro on its own (writes `hello.wav` and plays it back):

```bash
python -m app.tts.kokoro
```

## Test

```bash
pytest
```

- `tests/test_ollama.py` is an integration test that requires Ollama to be running with the configured model pulled.
- `tests/test_tools.py` covers `CalculatorTool`, `TimeTool`, `SearchTool`, and `ToolRegistry` (argument validation, unknown-tool handling, schema shape) — no external services required.
- `tests/test_search.py` covers the SearXNG search pipeline (URL normalization/dedup, result normalization, context formatting, query cache, and `SearchService` orchestration) using a fake SearXNG client — no external services required.
- `tests/test_response_streamer.py` also covers the tool-calling loop: a stub LLM that requests a tool then answers, the max-tool-rounds cutoff, and tool-error handling.
- `tests/test_stt.py` runs prerecorded WAV files in `tests/audio/` through the STT pipeline. See `tests/audio/README.md` for which fixtures are checked in.
- `tests/test_tts.py` covers Markdown preprocessing and an integration check that Kokoro produces audio (requires Kokoro + eSpeak NG installed).
- `tests/test_wakeword.py` covers the `WakeWordDetector` abstraction with a stub, plus an integration check against `tests/audio/hey_jarvis.wav` (requires `openwakeword` installed and the fixture present; skipped otherwise).
