"""
config.py — Central configuration loader.
Reads API keys and audio constants from the .env file.
"""

import os
from dotenv import load_dotenv

_ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(dotenv_path=_ENV_PATH, override=True)

# ── API Keys ──────────────────────────────────────────────────────────────────
OPENAI_API_KEY      = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY   = os.getenv("ANTHROPIC_API_KEY", "")
ELEVENLABS_API_KEY  = os.getenv("ELEVENLABS_API_KEY", "")

# Default ElevenLabs voice used during onboarding (before the user clones theirs).
# "Adam" — friendly male voice, supports Arabic via eleven_multilingual_v2.
ELEVENLABS_DEFAULT_VOICE_ID = os.getenv(
    "ELEVENLABS_DEFAULT_VOICE_ID", "pNInz6obpgDQGcFmaJgB"
)

# ── Audio recording constants ─────────────────────────────────────────────────
SAMPLE_RATE    = 16_000  # Hz — Whisper prefers 16 kHz
CHANNELS       = 1       # mono
CHUNK_DURATION = 0.1     # seconds per analysis window (100 ms)

# ── Voice Activity Detection (VAD) thresholds ─────────────────────────────────
SILENCE_THRESHOLD   = 400   # RMS below this = silence
SILENCE_DURATION    = 1.5   # seconds of consecutive silence to stop recording
MAX_RECORD_SECONDS  = 30    # hard cap
SPEECH_START_CHUNKS = 2     # consecutive loud chunks needed to start recording
PRE_BUFFER_CHUNKS   = 5     # frames kept before speech is confirmed
