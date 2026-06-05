"""
api.py — Aime Web API (FastAPI)

Endpoints:
  POST /transcribe  — audio file (webm/wav/mp3) → { text }
  POST /chat        — { message, history }       → { reply }
  POST /speak       — { text }                   → audio/mpeg stream

Run with:
  uvicorn api:app --reload --host 0.0.0.0 --port 8000
"""

import io
import tempfile
import os

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from openai import OpenAI
from elevenlabs.client import ElevenLabs

import llm
from config import OPENAI_API_KEY, ELEVENLABS_API_KEY
import voice_manager

# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(title="Aime Voice Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve index.html at /
_DIR = os.path.dirname(os.path.abspath(__file__))
app.mount("/static", StaticFiles(directory=_DIR), name="static")

# ── Clients ───────────────────────────────────────────────────────────────────

_openai    = OpenAI(api_key=OPENAI_API_KEY)
_elevenlabs = ElevenLabs(api_key=ELEVENLABS_API_KEY)


# ── Pydantic models ───────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []   # [{"role": "user"|"assistant", "content": str}]

class SpeakRequest(BaseModel):
    text: str


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_voice_id() -> str:
    """Return the active cloned voice ID from voices.json."""
    voice_id = voice_manager.get_active_voice_id()
    if not voice_id:
        voices = voice_manager.get_voices()
        if voices:
            voice_id = voices[-1]["voice_id"]
    if not voice_id:
        raise HTTPException(status_code=500, detail="No cloned voice found in voices.json")
    return voice_id


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
async def index():
    """Serve the frontend."""
    return FileResponse(os.path.join(_DIR, "index.html"))


@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    """
    Receive an audio file (webm / wav / mp3) and return the Whisper transcription.
    Response: { "text": "..." }
    """
    suffix = os.path.splitext(audio.filename or "audio.webm")[1] or ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name

    try:
        with open(tmp_path, "rb") as f:
            result = _openai.audio.transcriptions.create(
                model="whisper-1",
                file=f,
            )
        return {"text": result.text.strip()}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/chat")
async def chat(req: ChatRequest):
    """
    Receive a user message + conversation history, return Claude's reply (with RAG).
    Response: { "reply": "..." }

    The caller is responsible for maintaining history and passing it on each turn.
    llm.py's internal history is synced from the request so multi-turn context works
    across stateless HTTP calls.
    """
    # Sync the module-level history with what the client sent
    llm._history.clear()
    llm._history.extend(req.history)

    try:
        reply = llm.get_response(req.message)
        return {"reply": reply}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/speak")
async def speak(req: SpeakRequest):
    """
    Convert text to speech using the active cloned ElevenLabs voice.
    Returns an audio/mpeg stream.
    """
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")

    voice_id = _get_voice_id()

    try:
        audio_stream = _elevenlabs.text_to_speech.convert(
            text=req.text,
            voice_id=voice_id,
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128",
        )
        audio_bytes = b"".join(audio_stream)
        return StreamingResponse(
            io.BytesIO(audio_bytes),
            media_type="audio/mpeg",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
