"""
main.py — Aime Voice Assistant main loop.

Startup flow:
  1. Check voices.json — if empty, run first-launch onboarding
     (conversational voice cloning: 4 Arabic questions → clone voice)
  2. If multiple voices saved, ask the user which voice to use
  3. Enter the main always-on conversation loop:
       a. Wait silently for speech
       b. Record until 1.5 s of silence
       c. Transcribe with OpenAI Whisper API
       d. Detect special commands ("غير الصوت", "ابدأ من جديد", …)
       e. Send to Claude Haiku → get response
       f. Speak response with the active ElevenLabs cloned voice
       g. Loop

Special voice commands (say them aloud):
  "غير الصوت" / "بدل الصوت"     — switch to another saved voice
  "أضف صوت جديد"                 — record a new voice via mini-onboarding
  "ابدأ من جديد" / "امسح السجل"  — clear conversation history
"""

import sys
import time

import stt
import llm
import tts
import voice_manager
import onboarding
from config import (
    OPENAI_API_KEY,
    ANTHROPIC_API_KEY,
    ELEVENLABS_API_KEY,
    ELEVENLABS_DEFAULT_VOICE_ID,
)

# ── Trigger phrases (lowercased for comparison) ───────────────────────────────
_CHANGE_VOICE   = {"غير الصوت", "بدل الصوت", "صوت ثاني", "change voice"}
_ADD_VOICE      = {"أضف صوت جديد", "صوت جديد", "add new voice"}
_RESET_HISTORY  = {"ابدأ من جديد", "امسح السجل", "reset", "clear history", "اريد تبدأ من اول"}


def _check_config() -> None:
    """Exit early with a helpful message if any required API key is missing."""
    missing = []
    if not OPENAI_API_KEY:     missing.append("OPENAI_API_KEY")
    if not ANTHROPIC_API_KEY:  missing.append("ANTHROPIC_API_KEY")
    if not ELEVENLABS_API_KEY: missing.append("ELEVENLABS_API_KEY")
    if missing:
        print("❌ المفاتيح التالية غير موجودة في .env:")
        for k in missing:
            print(f"   • {k}")
        sys.exit(1)


def _choose_voice_interactively(voices: list[dict]) -> str:
    """
    If multiple voices are saved, ask the user which one to use.
    Returns the chosen voice_id.
    """
    if len(voices) == 1:
        return voices[0]["voice_id"]

    # Build a list of names and ask aloud
    names = "، ".join(v["name"] for v in voices)
    question = f"عندي {len(voices)} أصوات محفوظة: {names}. بصوت مين تبيني أكلمك؟"
    print(f"\n  🗣️  {question}")
    tts.speak(question, ELEVENLABS_DEFAULT_VOICE_ID)

    # Record and transcribe the user's choice
    audio = stt.record_until_silence()
    if len(audio) == 0:
        # No answer → use the first (most recently added) voice
        return voices[-1]["voice_id"]

    choice_text = stt.transcribe(audio)
    print(f"  اخترت: {choice_text}")

    # Try to match the spoken name to a saved voice
    for v in voices:
        if v["name"] in choice_text:
            tts.speak(f"حسناً! بكلمك بصوت {v['name']}.", v["voice_id"])
            return v["voice_id"]

    # No match → use last added voice
    fallback = voices[-1]
    tts.speak(f"ما فهمت، بكلمك بصوت {fallback['name']}.", fallback["voice_id"])
    return fallback["voice_id"]


def _handle_add_voice(current_voice_id: str) -> str:
    """
    Mini-onboarding: run a new voice recording session and return the new voice_id.
    Falls back to current_voice_id if anything fails.
    """
    tts.speak("حسناً! بنضيف صوت جديد. سأسألك بعض الأسئلة لأسجل صوتك.", current_voice_id)
    try:
        _, new_voice_id = onboarding.run()
        return new_voice_id
    except Exception as exc:
        print(f"  ❌ فشل إضافة الصوت: {exc}")
        tts.speak("ما قدرت أضيف الصوت الجديد. سأكمل بالصوت الحالي.", current_voice_id)
        return current_voice_id


def main() -> None:
    print("=" * 52)
    print("   🤖 آيمي — المساعدة الصوتية")
    print("=" * 52)

    _check_config()

    # ── Step 1: First-launch onboarding ──────────────────────────────────────
    voices = voice_manager.get_voices()
    if not voices:
        print("\n  [أول تشغيل] جاري الأونبوردينغ...")
        user_name, active_voice_id = onboarding.run()
        llm.set_user_name(user_name)
    else:
        # ── Step 2: Multi-voice selection ─────────────────────────────────────
        active_voice_id = voice_manager.get_active_voice_id() or voices[-1]["voice_id"]
        active_voice_id = _choose_voice_interactively(voices)
        voice_manager.set_active_voice_id(active_voice_id)

        # Restore user name for the active voice
        for v in voices:
            if v["voice_id"] == active_voice_id:
                llm.set_user_name(v["name"])
                break

    print(f"\n  ✅ جاهز! تكلم بأي وقت (Ctrl+C للخروج)\n")
    turn = 0

    # ── Step 3: Main conversation loop ────────────────────────────────────────
    while True:
        try:
            # Wait for speech, then record until silence
            audio = stt.record_until_silence()

            if len(audio) < int(0.5 * 16_000):
                continue   # too short — probably noise

            # Transcribe
            print("  📝 جاري التفريغ النصي...")
            text = stt.transcribe(audio)
            if not text:
                continue

            text_lower = text.lower().strip()
            turn += 1
            print(f"\n── [جولة {turn}] أنت: {text}")

            # ── Special commands ──────────────────────────────────────────────
            if any(t in text_lower for t in _RESET_HISTORY):
                llm.clear_history()
                reply = "تم! بدأنا من جديد. كيف أقدر أساعدك؟"
                print(f"  آيمي: {reply}")
                tts.speak(reply, active_voice_id)
                continue

            if any(t in text_lower for t in _CHANGE_VOICE):
                all_voices = voice_manager.get_voices()
                active_voice_id = _choose_voice_interactively(all_voices)
                voice_manager.set_active_voice_id(active_voice_id)
                continue

            if any(t in text_lower for t in _ADD_VOICE):
                active_voice_id = _handle_add_voice(active_voice_id)
                voice_manager.set_active_voice_id(active_voice_id)
                continue

            # ── LLM → TTS ─────────────────────────────────────────────────────
            print("  🤔 أفكر...")
            reply = llm.get_response(text)
            print(f"  آيمي: {reply}")

            print("  🔊 أتكلم...")
            tts.speak(reply, active_voice_id)

        except KeyboardInterrupt:
            print("\n\n👋 إلى اللقاء!")
            break

        except Exception as exc:
            print(f"  ❌ خطأ: {exc}")
            print("  ⏳ إعادة المحاولة خلال ثانيتين...")
            time.sleep(2)


if __name__ == "__main__":
    main()

