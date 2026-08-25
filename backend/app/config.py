import os
from pathlib import Path

import yaml

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma3")

SYSTEM_PROMPT = """You are Jarvis, a local virtual assistant.

Be concise when answering simple questions.
Give detailed explanations when the user asks for them.
Do not claim to have access to the internet unless a web-search tool is explicitly provided.
"""

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


def _load_yaml_config() -> dict:
    if not _CONFIG_PATH.exists():
        return {}
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f) or {}


_config = _load_yaml_config()
_audio_config = _config.get("audio", {})
_stt_config = _config.get("stt", {})
_tts_config = _config.get("tts", {})

AUDIO_SAMPLE_RATE = _audio_config.get("sample_rate", 16000)
AUDIO_CHANNELS = _audio_config.get("channels", 1)
MAX_RECORDING_SECONDS = _audio_config.get("max_recording_seconds", 15)

STT_MODEL = _stt_config.get("model", "small")
STT_DEVICE = _stt_config.get("device", "cpu")
STT_COMPUTE_TYPE = _stt_config.get("compute_type", "int8")
STT_LANGUAGE = _stt_config.get("language", "en")
STT_VAD_FILTER = _stt_config.get("vad_filter", True)

TTS_LANGUAGE = _tts_config.get("language", "en-us")
TTS_VOICE = _tts_config.get("voice", "af_heart")
TTS_SPEED = _tts_config.get("speed", 1.0)
