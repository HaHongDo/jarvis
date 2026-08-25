# STT test fixtures

`empty.wav` (2s of silence) is checked in and used to verify that VAD/Whisper
return an empty transcription for non-speech audio.

`hello.wav` and `goroutine.wav` are not checked in because they require an
actual voice recording. To add them, record short WAV files (16 kHz mono) and
save them here:

- `hello.wav` — someone saying "Hello, this is a microphone test."
- `goroutine.wav` — someone asking a question containing the word "goroutine"

The corresponding tests in `test_stt.py` are skipped automatically when a
fixture file is missing.
