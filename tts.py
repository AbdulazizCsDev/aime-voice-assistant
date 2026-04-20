"""
tts.py — Text-to-speech via ElevenLabs API, played back with pygame.

speak(text, voice_id) accepts any ElevenLabs voice_id so the caller
can switch between the default onboarding voice and the user's cloned voice.
"""

import os
import tempfile

import pygame
from elevenlabs.client import ElevenLabs

from config import ELEVENLABS_API_KEY

_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

# Initialise pygame mixer once at import time
pygame.mixer.init()


def speak(text: str, voice_id: str) -> None:
    """
    Convert text to speech with ElevenLabs and play it through the speakers.
    Blocks until playback finishes.

    Args:
        text:     The text to synthesise.
        voice_id: ElevenLabs voice ID (premade or cloned).
    """
    if not text.strip():
        return

    # Request audio from ElevenLabs (returns a bytes generator)
    audio_stream = _client.text_to_speech.convert(
        text=text,
        voice_id=voice_id,
        model_id="eleven_multilingual_v2",   # supports Arabic + English
        output_format="mp3_44100_128",
    )

    # Collect all chunks into a single bytes object
    audio_bytes = b"".join(audio_stream)

    # Write to a temp MP3 file so pygame can load it
    tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    try:
        tmp.write(audio_bytes)
        tmp.close()

        pygame.mixer.music.load(tmp.name)
        pygame.mixer.music.play()

        # Block until playback finishes
        while pygame.mixer.music.get_busy():
            pygame.time.wait(50)
    finally:
        pygame.mixer.music.unload()
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)

