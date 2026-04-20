"""
onboarding.py — First-launch onboarding: conversational voice cloning.

Flow:
  1. Greet the user and ask their name
  2. Present a long Arabic reading passage for better clone quality
  3. Concatenate all recordings into voice_sample.wav
  4. Upload to ElevenLabs Instant Voice Clone
  5. Say confirmation in the new cloned voice
  6. Save voice_id and user name to voices.json

Run automatically on first launch when voices.json is empty.
"""

import os

import numpy as np
from scipy.io import wavfile

import stt
import tts
import voice_manager
from config import ELEVENLABS_API_KEY, ELEVENLABS_DEFAULT_VOICE_ID, SAMPLE_RATE

_DIR               = os.path.dirname(os.path.abspath(__file__))
_VOICE_SAMPLE_PATH = os.path.join(_DIR, "voice_sample.wav")

# Opening question to capture the user's name
_Q1 = "أهلاً! أنا آيمي، مساعدتك الصوتية. سعيدة بلقاك! وش اسمك؟"

# Long reading passage — covers varied vowels, consonants, and intonation patterns
# for high-quality ElevenLabs voice cloning (~90 seconds of speech)
_READING_PASSAGE = """\
الفقرة الأولى:
التقنية الحديثة غيّرت حياتنا اليومية بشكل كبير، فأصبح بإمكاننا التواصل مع أشخاص في مختلف أنحاء العالم في غضون ثوانٍ معدودة.
وقد أسهمت هذه التطورات في تسهيل كثير من الأعمال والمهام التي كانت تستغرق وقتاً وجهداً كبيرَين في الماضي.
ومن أبرز هذه التغييرات: الذكاء الاصطناعي، الذي بات يُستخدم في تشخيص الأمراض، وقيادة السيارات، وتأليف الموسيقى.

الفقرة الثانية:
القراءة عادةٌ تُثري العقل وتُوسّع آفاق التفكير، فمن خلالها نستطيع أن نسافر إلى عوالم لم نرَها، ونفهم ثقافات لم نعشها.
يقول بعض العلماء إن الشخص الذي يقرأ كتاباً في الأسبوع يتعرّض لخبرات تعادل عشرات السنوات من الحياة الفعلية.
لذلك، احرص على تخصيص وقتٍ يومي للقراءة، حتى لو كان ذلك لنصف ساعة فقط.

الفقرة الثالثة:
الرياضة ليست مجرد نشاط بدني، بل هي أسلوب حياة متكامل يشمل الجانب النفسي والاجتماعي أيضاً.
فممارسة الرياضة بانتظام تُقوّي الجهاز المناعي، وتُحسّن المزاج، وتزيد من الإنتاجية في العمل.
وقد أثبتت الدراسات أن الأشخاص الذين يمارسون النشاط البدني ثلاث مرات أسبوعياً على الأقل يتمتعون بصحة أفضل وعمرٍ أطول.

الفقرة الرابعة:
الطموح هو المحرّك الأساسي للإنسان نحو النجاح، فبدونه تتوقف عجلة التطور وتتراجع الحضارات.
كل إنجازٍ عظيم في تاريخ البشرية بدأ بفكرة جريئة في ذهن شخصٍ لم يخشَ المستحيل.
لذا، لا تتردد في رسم أحلامك بخطوطٍ عريضة، والسعي نحوها بخطواتٍ ثابتة ومدروسة، يوماً بعد يوم.
"""


def _extract_name(transcription: str) -> str:
    """
    Extract the user's name from responses like 'أنا عزيز' or just 'عزيز'.
    Falls back to the first word of the transcription.
    """
    text = transcription.strip()
    for prefix in ["أنا ", "اسمي ", "اسم ", "يقولون لي ", "ناديني "]:
        if text.startswith(prefix):
            text = text[len(prefix):]
    words = text.split()
    return words[0] if words else transcription.strip()


def _ask_and_record(question: str, voice_id: str) -> np.ndarray:
    """
    Speak the question aloud, then record the user's answer.
    Returns the raw 16 kHz int16 audio array.
    """
    tts.speak(question, voice_id)
    print("  🎤 يرجى الإجابة الآن...")
    return stt.record_until_silence()


def run() -> tuple[str, str]:
    """
    Run the full onboarding conversation.

    Returns:
        (user_name, voice_id) — the extracted name and the ElevenLabs voice ID.
    """
    default_vid  = ELEVENLABS_DEFAULT_VOICE_ID
    all_audio:   list[np.ndarray] = []
    user_name    = "صديقي"   # fallback if name extraction fails

    print("\n" + "=" * 52)
    print("  🎤 مرحباً بك في آيمي!")
    print("=" * 52)

    # ── Step 1: Ask for name ──────────────────────────────────────────────────
    print("\n[1/2] عرّف نفسك")
    q1_audio = _ask_and_record(_Q1, default_vid)
    if len(q1_audio) > 0:
        all_audio.append(q1_audio)
        print("  📝 جاري التعرف على اسمك...")
        name_text = stt.transcribe(q1_audio)
        if name_text:
            user_name = _extract_name(name_text)
            print(f"  👤 اسمك: {user_name}")

    # ── Step 2: Long reading passage ─────────────────────────────────────────
    print(f"\n[2/2] النص للقراءة")
    intro = (
        f"ممتاز {user_name}! الحين بعطيك نصاً اقرأه بصوت واضح وطبيعي — "
        "هذا يساعدني أستنسخ صوتك بجودة عالية. "
        "اقرأ النص اللي رح يظهر على الشاشة، وخذ وقتك."
    )
    tts.speak(intro, default_vid)

    print("\n" + "─" * 52)
    print(_READING_PASSAGE)
    print("─" * 52)
    print("  🎤 ابدأ القراءة الآن — سأوقف التسجيل بعد 3 ثوانٍ من الصمت...")

    reading_audio = stt.record_until_silence(silence_duration=3.0)
    if len(reading_audio) > 0:
        all_audio.append(reading_audio)

    # ── Save concatenated audio as voice_sample.wav ────────────────────────
    tts.speak(
        f"ممتاز {user_name}! الحين بسحب صوتك وأنشئ نسخة منه. هذا بياخذ ثانية...",
        default_vid,
    )

    combined    = np.concatenate(all_audio) if all_audio else np.array([], dtype=np.int16)
    total_secs  = len(combined) / SAMPLE_RATE
    wavfile.write(_VOICE_SAMPLE_PATH, SAMPLE_RATE, combined)
    print(f"\n  💾 تم حفظ {total_secs:.1f} ثانية من صوتك في voice_sample.wav")

    if total_secs < 10:
        print("  ⚠️  الصوت قصير جداً — قد لا تكون جودة الاستنساخ عالية.")

    # ── Clone voice on ElevenLabs ─────────────────────────────────────────────
    try:
        voice_id = voice_manager.clone_voice(user_name, _VOICE_SAMPLE_PATH, ELEVENLABS_API_KEY)
        voice_manager.add_voice(user_name, voice_id)

        print(f"  🎉 تم استنساخ صوتك بنجاح!")

        # Confirm in the cloned voice
        tts.speak(
            f"الحين أنت تكلم نفسك يا {user_name}! صوتك صار صوتي. كيف يبدو لك؟",
            voice_id,
        )
        return user_name, voice_id

    except Exception as exc:
        print(f"  ❌ فشل استنساخ الصوت: {exc}")
        print("  ⚠️  سأستخدم الصوت الافتراضي مؤقتاً.")
        tts.speak(
            f"عذراً {user_name}، ما قدرت أستنسخ صوتك الحين. بكلمك بالصوت الافتراضي.",
            default_vid,
        )
        voice_manager.add_voice(user_name, default_vid)
        return user_name, default_vid

