"""
stt.py — Always-on audio recording + transcription via OpenAI Whisper API.

Flow:
  1. record_until_silence() — wait for speech to begin, capture until silence
  2. transcribe(audio)       — send WAV bytes to Whisper API, return text

No wake word needed: the function waits silently until the user speaks.
"""

import os
import tempfile

import numpy as np
import sounddevice as sd
from scipy.io import wavfile
from openai import OpenAI

from config import (
    OPENAI_API_KEY,
    SAMPLE_RATE,
    CHANNELS,
    CHUNK_DURATION,
    SILENCE_THRESHOLD,
    SILENCE_DURATION,
    MAX_RECORD_SECONDS,
    SPEECH_START_CHUNKS,
    PRE_BUFFER_CHUNKS,
)

_client         = OpenAI(api_key=OPENAI_API_KEY)
_CHUNK_FRAMES   = int(SAMPLE_RATE * CHUNK_DURATION)
_SILENCE_CHUNKS = int(SILENCE_DURATION / CHUNK_DURATION)
_MAX_CHUNKS     = int(MAX_RECORD_SECONDS / CHUNK_DURATION)


def record_until_silence(silence_duration: float | None = None) -> np.ndarray:
    """
    Always-on recording: waits silently until speech is detected,
    then records until silence follows.

    Args:
        silence_duration: seconds of silence to stop recording.
                          Defaults to config.SILENCE_DURATION (1.5 s).

    Returns a 1-D int16 numpy array, or an empty array if nothing was captured.
    """
    _silence_chunks = int(
        (silence_duration if silence_duration is not None else SILENCE_DURATION)
        / CHUNK_DURATION
    )

    pre_buffer: list[np.ndarray] = []   # rolling buffer of frames before speech
    chunks:     list[np.ndarray] = []
    silent_count   = 0
    speech_started = False
    loud_count     = 0   # consecutive loud frames (used to confirm speech start)

    print("  🔇 منتظر...", end="", flush=True)

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="int16",
        blocksize=_CHUNK_FRAMES,
    ) as stream:
        for _ in range(_MAX_CHUNKS):
            frame, _ = stream.read(_CHUNK_FRAMES)
            rms = float(np.sqrt(np.mean(frame.astype(np.float32) ** 2)))

            if not speech_started:
                # Keep a short rolling buffer so we don't clip the start of speech
                pre_buffer.append(frame.copy())
                if len(pre_buffer) > PRE_BUFFER_CHUNKS:
                    pre_buffer.pop(0)

                if rms > SILENCE_THRESHOLD:
                    loud_count += 1
                    if loud_count >= SPEECH_START_CHUNKS:
                        # Confirmed speech — flush pre-buffer into chunks
                        speech_started = True
                        chunks.extend(pre_buffer)
                        pre_buffer.clear()
                        print(" 🔴 يسجّل...", flush=True)
                else:
                    loud_count = 0   # reset if silence resumes before confirmation
            else:
                chunks.append(frame.copy())
                if rms <= SILENCE_THRESHOLD:
                    silent_count += 1
                    if silent_count >= _silence_chunks:
                        break   # enough silence → end of utterance
                else:
                    silent_count = 0   # reset silence counter on any speech

    if not chunks:
        return np.array([], dtype=np.int16)

    return np.concatenate(chunks).flatten()


def transcribe(audio: np.ndarray) -> str:
    """
    Transcribe a PCM int16 audio array using the OpenAI Whisper API.
    Returns the transcribed text, or an empty string on failure.
    """
    if len(audio) == 0:
        return ""

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    try:
        wavfile.write(tmp.name, SAMPLE_RATE, audio)
        tmp.close()
        with open(tmp.name, "rb") as f:
            result = _client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
            )
        return result.text.strip()
    finally:
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)

        if os.path.exists(tmp.name):
            os.unlink(tmp.name)
