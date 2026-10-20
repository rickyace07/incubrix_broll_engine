import re
import asyncio
import unicodedata
from gtts import gTTS
from app.config import settings

VOICE_LANG_MAP = {
    "en-IN-female": ("en", "co.in"),
    "en-US-female": ("en", "com"),
    "en-GB-female": ("en", "co.uk"),
    "hi-IN": ("hi", "co.in"),
    "hindi": ("hi", "co.in"),
    "ta-IN": ("ta", "co.in"),
    "tamil": ("ta", "co.in"),
}


async def synthesize_speech(text: str, voice_key: str, out_mp3: str) -> str:
    """
    Synthesizes speech using gTTS with automatic Unicode normalization and script detection
    for Tamil (ta), Hindi (hi), and English (en).
    """
    normalized_text = unicodedata.normalize("NFC", text)
    voice_lower = (voice_key or "").lower()

    # 1. Auto-detect Tamil or Hindi script directly from the text characters
    if re.search(r"[\u0B80-\u0BFF]", normalized_text):
        lang, tld = ("ta", "co.in")
    elif re.search(r"[\u0900-\u097F]", normalized_text):
        lang, tld = ("hi", "co.in")
    # 2. Or check the requested voice_key
    elif voice_lower.startswith("ta") or "tamil" in voice_lower:
        lang, tld = ("ta", "co.in")
    elif voice_lower.startswith("hi") or "hindi" in voice_lower:
        lang, tld = ("hi", "co.in")
    else:
        lang, tld = VOICE_LANG_MAP.get(voice_key, ("en", "co.in"))

    def _run():
        tts = gTTS(text=normalized_text, lang=lang, tld=tld, slow=False)
        tts.save(out_mp3)

    await asyncio.to_thread(_run)
    return f"gtts-real-speech-{lang}"