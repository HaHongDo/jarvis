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

| Tool               | Purpose                                             |
|--------------------|------------------------------------------------------|
| `calculator`       | Evaluates a math expression (safe AST eval, no `eval()`) |
| `get_time`         | Current date/time for an IANA timezone               |
| `search`           | Web search backed by a local SearXNG instance          |
| `fetch_page`       | Fetches and extracts a single webpage's content        |
| `research`         | Multi-source web research (search + fetch + dedup + chunk) |
| `search_knowledge` | Semantic search over the user's private local knowledge base |

`ResponseStreamer` runs the tool-calling loop itself: if the model asks for a tool instead
of answering, the tool is executed via `ToolRegistry` (which validates arguments against
the tool's JSON-schema `parameters` and logs every call), the result is fed back to the
model, and the loop repeats — up to `tools.max_rounds` in `config.yaml` — until the model
answers in plain text. Raw tool output is never sent to TTS directly; only the model's
natural-language response is spoken.

### Private knowledge base / RAG (`app/knowledge/`)

`search_knowledge` is a separate, local-only retrieval path over the user's own notes and
docs — never sent to the web:

```text
knowledge/<notes|docs|projects>/*.md, *.txt
    ↓ ingest (python -m app.ingest [directory])
load_documents -> chunk_document -> OllamaEmbedder -> VectorStore (SQLite)
    ↓ search_knowledge(query)
OllamaEmbedder.embed_query -> cosine similarity over stored chunks -> top-K matches
    -> KnowledgeContextBuilder (title/path/score-attributed context for the LLM)
```

Ingestion is incremental: each document's SHA-256 content hash is stored, so
unchanged files are skipped on re-ingestion, changed files are re-chunked/re-embedded,
and files removed from disk have their chunks deleted. Run ingestion after adding or
editing notes:

```bash
python -m app.ingest knowledge/
```

Embeddings use a local Ollama embedding model (`knowledge.embedding_model` in
`config.yaml`, default `nomic-embed-text` — run `ollama pull nomic-embed-text` once).
The vector store is a single SQLite file (`knowledge.db`); similarity search is a
brute-force cosine scan, which is fine for a personal-scale knowledge base.


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

### Speech normalization (`app/speech/`)

Technical vocabulary is what a small Whisper model gets wrong most often
("gore teen" for goroutine, "jay vee em" for JVM, "coopernetes" for Kubernetes).
Instead of switching to a bigger model, one vocabulary source feeds *both* sides
of the STT stage:

```text
                 Conversation
                      |
                      v
              VocabularyManager  -- active vocabulary (a few domains, not the whole DB)
                 |          |
 initial_prompt /           | VocabularyContext
 hotwords                   |
                 v          v
      Audio -> FasterWhisperSTT -> raw transcript -> SpeechNormalizer -> normalized text -> LLM
```

`VocabularyManager` decides what the conversation is currently about by keyword
scoring, explicit topic switches ("let's switch to Java"), and timestamp-based
decay, then exposes the matching slice of `app/speech/vocabulary.py` as an
`ActiveVocabulary`. That slice becomes a Whisper `initial_prompt` (plus
faster-whisper `hotwords`) *before* transcription, and correction context
*after* it.

The normalizer stays in place even when the hints work, because hints are
probabilistic. It is deliberately conservative — corrections are gated by
per-term confidence:

| Confidence   | Behavior                                                              |
|--------------|-----------------------------------------------------------------------|
| `>= 0.95`    | corrected automatically ("gore teen" -> goroutine)                     |
| `0.75–0.95`  | corrected only if the active vocabulary confirms the topic ("wait group" -> WaitGroup) |
| `< 0.75`     | left alone unless the optional LLM fallback confirms it ("g c" -> GC)  |

Every change records where it came from (`exact_match`, `context_match`,
`llm_fallback`), and the normalizer keeps per-run counters (`total`,
`corrected`, `unchanged`, `uncertain`, `llm_fallback`).

Two evaluation CLIs:

```bash
python -m app.speech.evaluate --by-category   # normalizer accuracy over tests/speech/terminology.json
python -m app.speech.evaluate_stt             # does feeding vocabulary into Whisper actually help?
```

`evaluate_stt` needs recordings: copy `tests/speech/stt_cases.example.json` to
`tests/speech/stt_cases.json`, record the sentences, and it will transcribe each
clip four ways (baseline, baseline + normalizer, hinted, hinted + normalizer) so
the two mechanisms can be compared. If the hinted columns don't beat the
baseline on your setup, turn the hints off with
`speech.vocabulary.stt_prompt_enabled` / `stt_hotwords_enabled`.

New vocabulary is added by hand after a mistake is observed repeatedly and
verified — there is no automatic learning from corrections.

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
- `tests/test_stt.py` runs prerecorded WAV files in `tests/audio/` through the STT pipeline. See `tests/audio/README.md` for which fixtures are checked in. It also covers vocabulary-hint wiring (`initial_prompt`/`hotwords`) against a fake Whisper model — no model load required.
- `tests/test_speech_normalizer.py` runs the `tests/speech/terminology.json` corpus (positive, negative, and context-dependent cases) through the normalizer, plus the confidence tiers, correction provenance, metrics, and LLM fallback.
- `tests/test_vocabulary_manager.py` covers topic detection, explicit topic switching, vocabulary decay, and the STT prompt/hotwords the active vocabulary produces.
- `tests/test_tts.py` covers Markdown preprocessing and an integration check that Kokoro produces audio (requires Kokoro + eSpeak NG installed).
- `tests/test_wakeword.py` covers the `WakeWordDetector` abstraction with a stub, plus an integration check against `tests/audio/hey_jarvis.wav` (requires `openwakeword` installed and the fixture present; skipped otherwise).
