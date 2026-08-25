# Jarvis backend — Day 3

Voice CLI: microphone -> speech-to-text -> local LLM through Ollama -> text-to-speech.

```text
Microphone -> AudioRecorder -> FasterWhisperSTT -> LLM interface -> Ollama -> local model -> KokoroTTS -> AudioPlayer -> Speaker
```

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
```

## Run

```bash
python -m app.main
```

Press Enter, speak, and wait for the transcription and spoken reply. Type `exit` (or `quit`) instead of pressing Enter to end the session.

To try Kokoro on its own (writes `hello.wav` and plays it back):

```bash
python -m app.tts.kokoro
```

## Test

```bash
pytest
```

- `tests/test_ollama.py` is an integration test that requires Ollama to be running with the configured model pulled.
- `tests/test_stt.py` runs prerecorded WAV files in `tests/audio/` through the STT pipeline. See `tests/audio/README.md` for which fixtures are checked in.
- `tests/test_tts.py` covers Markdown preprocessing and an integration check that Kokoro produces audio (requires Kokoro + eSpeak NG installed).
