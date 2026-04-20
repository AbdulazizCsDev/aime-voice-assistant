"""
voice_manager.py — Manages cloned voices stored in voices.json.

voices.json structure:
{
  "voices": [
    {"name": "عزيز", "voice_id": "abc123", "created_at": "2026-04-19T10:00:00"}
  ],
  "active_voice_id": "abc123"
}
"""

import json
import os
from datetime import datetime

from elevenlabs.client import ElevenLabs

_DIR        = os.path.dirname(os.path.abspath(__file__))
_VOICES_PATH = os.path.join(_DIR, "voices.json")


# ── File I/O ───────────────────────────────────────────────────────────────────

def _load() -> dict:
    """Load voices.json. Returns empty structure if file is missing or corrupt."""
    try:
        with open(_VOICES_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"voices": [], "active_voice_id": None}


def _save(data: dict) -> None:
    with open(_VOICES_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ── Public API ─────────────────────────────────────────────────────────────────

def get_voices() -> list[dict]:
    """Return list of all saved voice entries."""
    return _load()["voices"]


def get_active_voice_id() -> str | None:
    """Return the currently active voice_id, or None if none is set."""
    data = _load()
    return data.get("active_voice_id")


def set_active_voice_id(voice_id: str) -> None:
    """Set the active voice by voice_id."""
    data = _load()
    data["active_voice_id"] = voice_id
    _save(data)


def get_voice_by_name(name: str) -> dict | None:
    """Find a voice entry whose name contains the given string (case-insensitive)."""
    for v in get_voices():
        if name.strip() in v["name"]:
            return v
    return None


def add_voice(name: str, voice_id: str) -> None:
    """Add a new voice entry and set it as active."""
    data = _load()
    data["voices"].append({
        "name": name,
        "voice_id": voice_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    })
    data["active_voice_id"] = voice_id
    _save(data)
    print(f"  ✅ Voice '{name}' saved (id: {voice_id})")


def clone_voice(name: str, wav_path: str, api_key: str) -> str:
    """
    Upload wav_path to ElevenLabs Instant Voice Clone (IVC).
    Returns the new voice_id.
    Raises RuntimeError on failure.
    """
    client = ElevenLabs(api_key=api_key)

    print("  ⬆️  Uploading audio to ElevenLabs for voice cloning...")
    with open(wav_path, "rb") as f:
        voice = client.voices.add(
            name=name,
            files=[f],
            description=f"Voice cloned by Aime assistant for {name}",
        )

    return voice.voice_id
