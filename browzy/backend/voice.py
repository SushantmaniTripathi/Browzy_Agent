"""Voice input: speech-to-text for Hindi, Marathi, English and code-mixed speech.

Keys live in ``browzy/.env`` (loaded on import, real environment variables win).
The first configured provider is used:

    GROQ_API_KEY      -> Groq Whisper large-v3   (free tier, best value)
Nothing is transcribed on a cloud by accident: if no key is present the endpoint
fails loudly with the exact line to add to ``browzy/.env``.
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import re
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


import httpx

from .language import detect_language

BASE_DIR = Path(__file__).resolve().parents[1]
ENV_PATHS = [BASE_DIR / ".env", BASE_DIR.parent / ".env", Path.cwd() / ".env"]

MAX_BYTES = 24 * 1024 * 1024  # Groq free tier caps uploads at 25 MB.

# Words the model should spell the way the user says them.  Whisper's prompt is
# capped at 224 tokens, so this stays short.
WHISPER_PROMPT = (
    "Browzy browser voice command. The speaker mixes Hindi, Marathi and English "
    "(Hinglish). Keep Hindi and Marathi words in Devanagari and English words in "
    "Latin script. Sites: YouTube, Google, Flipkart, Amazon, Myntra, Wikipedia, "
    "GitHub, Spotify. Commands: kholo, chalao, bharo, search karo, price compare."
)

SARVAM_KEYTERMS = [
    "Browzy", "YouTube", "Google", "Gmail", "GitHub", "Wikipedia", "Amazon",
    "Flipkart", "Myntra", "Meesho", "Spotify", "Reddit", "Instagram",
    "LinkedIn", "Netflix", "WhatsApp", "form bharo", "search karo",
    "price compare", "kholo", "chalao",
]

# Whole-transcript hallucinations Whisper emits for silence / noise.
_HALLUCINATIONS = {
    "thank you", "thanks for watching", "thanks for watching!", "thank you for watching", "thank you for watching!",
    "you", "bye", "bye.", "please subscribe", "subtitles by the amara.org community",
    "subtitles by", ".", "..", "...", "♪", "okay", "ok",
}


class VoiceError(Exception):
    """Raised with a message that is safe and useful to show in the panel."""


def _load_env() -> list[Path]:
    """Load browzy/.env without overriding non-empty variables that are already set."""
    loaded: list[Path] = []
    seen: set[Path] = set()
    for path in ENV_PATHS:
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if resolved in seen or not path.is_file():
            continue
        seen.add(resolved)
        try:
            from dotenv import dotenv_values  # type: ignore

            values = dotenv_values(path, encoding="utf-8-sig")
            for k, v in values.items():
                if v and (k not in os.environ or not os.environ[k].strip()):
                    os.environ[k] = str(v).strip()
            loaded.append(path)
            continue
        except Exception:
            pass
        # Minimal fallback parser (KEY=VALUE, # comments) if dotenv is missing.
        try:
            for line in path.read_text(encoding="utf-8-sig").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                if "#" in value and not (value.startswith('"') or value.startswith("'")):
                    value = value.split("#")[0]
                value = value.strip().strip('"').strip("'")
                if key and value and (key not in os.environ or not os.environ[key].strip()):
                    os.environ[key] = value
            loaded.append(path)
        except Exception:
            continue
    return loaded


ENV_LOADED = _load_env()


def _env(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


def _normalize_language(language: Optional[str]) -> Optional[str]:
    """'auto'/'unknown'/'' -> None (auto-detect).  Otherwise a BCP-47 tag."""
    if not language:
        return None
    value = language.strip()
    if value.lower() in ("auto", "unknown", "none", "detect"):
        return None
    aliases = {
        "hi": "hi-IN", "hindi": "hi-IN",
        "mr": "mr-IN", "marathi": "mr-IN",
        "en": "en-IN", "english": "en-IN",
    }
    return aliases.get(value.lower(), value)


def _iso639(tag: Optional[str]) -> Optional[str]:
    if not tag:
        return None
    return tag.split("-")[0].lower()


def _groq_key() -> str:
    """Check for Groq key, also accepting GROK_API_KEY as a common alias."""
    return _env("GROQ_API_KEY") or _env("GROK_API_KEY")


def _provider() -> Optional[str]:
    """Which provider to use, or None when voice is not configured."""
    explicit = _env("BROWZY_STT_PROVIDER").lower()
    if explicit:
        return "groq" if explicit == "grok" else explicit
    if _env("BROWZY_WHISPER_MODEL"):
        return "local"
    if _groq_key():
        return "groq"
    if _env("SARVAM_API_KEY"):
        return "sarvam"
    if _env("OPENAI_API_KEY"):
        return "openai"
    if _env("DEEPGRAM_API_KEY"):
        return "deepgram"
    return None


def _model_for(provider: str) -> str:
    override = _env("BROWZY_STT_MODEL")
    if override:
        return override
    return {
        "groq": "whisper-large-v3",
        "sarvam": "saaras:v4",
        "openai": "whisper-1",
        "deepgram": "nova-3",
        "local": _env("BROWZY_WHISPER_MODEL", "small") or "small",
    }.get(provider, "")


def _missing_message(provider: Optional[str] = None) -> str:
    env_path = ENV_PATHS[0]
    if provider and provider != "local":
        var = {
            "groq": "GROQ_API_KEY",
            "sarvam": "SARVAM_API_KEY",
            "openai": "OPENAI_API_KEY",
            "deepgram": "DEEPGRAM_API_KEY",
        }.get(provider, "BROWZY_STT_PROVIDER")
        return (
            f"Voice input is set to {provider}, but {var} is empty.\n\n"
            f"Add this line to {env_path} and restart Browzy:\n{var}=your-key"
        )
    return (
        "Voice input needs a speech-to-text key.\n\n"
        f"Add one line to {env_path} and restart Browzy:\n"
        "GROQ_API_KEY=your-groq-key      (free tier, Whisper large-v3 at https://console.groq.com/keys)\n"
        "or\n"
        "SARVAM_API_KEY=your-sarvam-key  (best for Hindi/Marathi code-mixing)\n\n"
        "Note: Voice transcription uses Groq (groq.com), not xAI Grok."
    )


def provider_status() -> Dict[str, Any]:
    global ENV_LOADED
    if _provider() is None:
        ENV_LOADED = _load_env()
    provider = _provider()
    status: Dict[str, Any] = {
        "ready": provider is not None,
        "provider": provider,
        "model": _model_for(provider) if provider else "",
        "language": _env("BROWZY_STT_LANGUAGE", "auto") or "auto",
        "env_path": str(ENV_PATHS[0]),
        "env_loaded": [str(path) for path in ENV_LOADED],
    }
    if provider is None:
        status["message"] = _missing_message()
    elif provider == "local":
        try:
            import faster_whisper  # noqa: F401

            status["message"] = (
                f"Offline transcription with faster-whisper "
                f"{getattr(faster_whisper, '__version__', '')} "
                f"({_model_for('local')})."
            ).replace("  ", " ")
        except Exception:
            status["ready"] = False
            status["message"] = (
                "BROWZY_STT_PROVIDER=local needs the faster-whisper package:\n"
                "pip install faster-whisper"
            )
    else:
        status["message"] = f"Transcribing with {provider} ({_model_for(provider)})."
    return status


def _clean(text: str) -> str:
    text = (text or "").replace("\u200b", "").replace("\ufeff", "")
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^[\-\u2013\u2014,;:]+", "", text).strip()
    if text.lower().strip(" .!?") in _HALLUCINATIONS:
        return ""
    if not re.search(r"[A-Za-z\u0900-\u097F0-9]", text):
        return ""
    return text


def _error_detail(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except Exception:
        return response.text[:300]
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            return str(error.get("message") or error.get("code") or payload)[:300]
        if error:
            return str(error)[:300]
        for key in ("message", "detail", "detail_message"):
            if payload.get(key):
                return str(payload[key])[:300]
    return str(payload)[:300]


def _friendly(provider: str, response: httpx.Response) -> VoiceError:
    detail = _error_detail(response)
    if response.status_code in (401, 403):
        return VoiceError(
            f"{provider} rejected the API key. Check the key in "
            f"{ENV_PATHS[0]} and restart Browzy.\n\n{detail}"
        )
    if response.status_code == 429:
        return VoiceError(
            f"{provider} rate limit reached (free tier). Wait a minute and try "
            f"again.\n\n{detail}"
        )
    return VoiceError(f"{provider} could not transcribe that audio ({response.status_code}).\n\n{detail}")


async def _groq(data: bytes, filename: str, content_type: str, language: Optional[str]) -> Tuple[str, Optional[str]]:
    key = _groq_key()
    if not key:
        raise VoiceError(_missing_message("groq"))
    if key.startswith("xai-"):
        raise VoiceError(
            "It looks like an xAI Grok API key ('xai-...') was entered.\n\n"
            "Browzy uses Groq (https://groq.com) for Whisper large-v3 speech-to-text. "
            "xAI Grok is a text language model and does not provide audio speech-to-text.\n\n"
            "To enable voice commands, get a free Groq key (starts with 'gsk_'):\n"
            "1. Visit https://console.groq.com/keys\n"
            "2. Generate a free API key\n"
            f"3. Add it to {ENV_PATHS[0]}:\n"
            "   GROQ_API_KEY=gsk_your_key_here\n\n"
            "Or use Sarvam AI (SARVAM_API_KEY=...) for Indic speech."
        )
    form: Dict[str, str] = {
        "model": _model_for("groq"),
        "response_format": "json",
        "temperature": "0",
        "prompt": WHISPER_PROMPT,
    }
    iso = _iso639(language)
    if iso:
        form["language"] = iso
    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0)) as client:
        response = await client.post(
            "https://api.groq.com/openai/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {key}"},
            data=form,
            files={"file": (filename, data, content_type)},
        )
    if response.status_code != 200:
        raise _friendly("Groq", response)
    payload = response.json()
    return str(payload.get("text") or ""), None


async def _sarvam(data: bytes, filename: str, content_type: str, language: Optional[str]) -> Tuple[str, Optional[str]]:
    key = _env("SARVAM_API_KEY")
    if not key:
        raise VoiceError(_missing_message("sarvam"))
    lang_code = language or _normalize_language(_env("BROWZY_STT_LANGUAGE", "auto")) or "unknown"

    # saaras:v4 understands code-mixed speech best; fall back if the account or
    # API version does not expose it yet.
    attempts = [
        {"model": "saaras:v4", "mode": "codemix", "keyterms": json.dumps(SARVAM_KEYTERMS)},
        {"model": "saaras:v3", "mode": "codemix"},
        {"model": "saarika:v2.5"},
    ]
    last: Optional[httpx.Response] = None
    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0)) as client:
        for extra in attempts:
            form = {"language_code": lang_code, **extra}
            response = await client.post(
                "https://api.sarvam.ai/speech-to-text",
                headers={"api-subscription-key": key},
                data=form,
                files={"file": (filename, data, content_type)},
            )
            if response.status_code == 200:
                payload = response.json()
                return str(payload.get("transcript") or ""), payload.get("language_code")
            last = response
            if response.status_code not in (400, 404, 422):
                break
    assert last is not None
    raise _friendly("Sarvam", last)


async def _openai(data: bytes, filename: str, content_type: str, language: Optional[str]) -> Tuple[str, Optional[str]]:
    key = _env("OPENAI_API_KEY")
    if not key:
        raise VoiceError(_missing_message("openai"))
    form: Dict[str, str] = {
        "model": _model_for("openai"),
        "response_format": "json",
        "temperature": "0",
        "prompt": WHISPER_PROMPT,
    }
    iso = _iso639(language)
    if iso:
        form["language"] = iso
    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0)) as client:
        response = await client.post(
            "https://api.openai.com/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {key}"},
            data=form,
            files={"file": (filename, data, content_type)},
        )
    if response.status_code != 200:
        raise _friendly("OpenAI", response)
    payload = response.json()
    return str(payload.get("text") or ""), None


async def _deepgram(data: bytes, filename: str, content_type: str, language: Optional[str]) -> Tuple[str, Optional[str]]:
    key = _env("DEEPGRAM_API_KEY")
    if not key:
        raise VoiceError(_missing_message("deepgram"))
    params = {
        "model": _model_for("deepgram"),
        "language": "multi",  # code-switching
        "smart_format": "true",
        "punctuate": "true",
    }
    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0)) as client:
        response = await client.post(
            "https://api.deepgram.com/v1/listen",
            headers={"Authorization": f"Token {key}", "Content-Type": content_type},
            params=params,
            content=data,
        )
    if response.status_code != 200:
        raise _friendly("Deepgram", response)
    payload = response.json()
    try:
        channel = payload["results"]["channels"][0]
        return str(channel["alternatives"][0]["transcript"] or ""), channel.get("detected_language")
    except (KeyError, IndexError, TypeError):
        return "", None


_WHISPER_LOCK = threading.Lock()
_WHISPER_MODELS: Dict[str, Any] = {}


def _whisper(model_name: str):
    with _WHISPER_LOCK:
        if model_name not in _WHISPER_MODELS:
            try:
                from faster_whisper import WhisperModel  # type: ignore
            except Exception as exc:  # pragma: no cover - optional dependency
                raise VoiceError(
                    "Offline transcription needs faster-whisper:\n"
                    "pip install faster-whisper"
                ) from exc
            _WHISPER_MODELS[model_name] = WhisperModel(
                model_name,
                device=_env("BROWZY_WHISPER_DEVICE", "cpu") or "cpu",
                compute_type=_env("BROWZY_WHISPER_COMPUTE", "int8") or "int8",
            )
        return _WHISPER_MODELS[model_name]


def _local_sync(data: bytes, language: Optional[str]) -> Tuple[str, Optional[str]]:
    model = _whisper(_model_for("local"))
    segments, info = model.transcribe(
        io.BytesIO(data),
        language=_iso639(language),
        vad_filter=True,
        beam_size=5,
        condition_on_previous_text=False,
        initial_prompt=WHISPER_PROMPT,
    )
    text = " ".join(segment.text.strip() for segment in segments).strip()
    return text, getattr(info, "language", None)


async def _local(data: bytes, filename: str, content_type: str, language: Optional[str]) -> Tuple[str, Optional[str]]:
    try:
        return await asyncio.to_thread(_local_sync, data, language)
    except VoiceError:
        raise
    except Exception as exc:
        raise VoiceError(
            "That recording could not be read. Please record it again."
        ) from exc


async def transcribe(
    data: bytes,
    filename: Optional[str] = None,
    content_type: Optional[str] = None,
    language: Optional[str] = None,
) -> Dict[str, Any]:
    """Transcribe recorded audio and return {text, language, provider, model}."""
    if not data:
        raise VoiceError("No audio was recorded. Hold the mic button and speak.")
    if len(data) > MAX_BYTES:
        raise VoiceError(
            "That recording is too long (max 24 MB). Keep voice commands under "
            "about two minutes."
        )

    global ENV_LOADED
    if _provider() is None:
        ENV_LOADED = _load_env()
    provider = _provider()
    if provider is None:
        raise VoiceError(_missing_message())

    handler = {
        "groq": _groq,
        "sarvam": _sarvam,
        "openai": _openai,
        "deepgram": _deepgram,
        "local": _local,
    }.get(provider)
    if handler is None:
        raise VoiceError(
            f"Unknown BROWZY_STT_PROVIDER={provider!r}. Use groq, sarvam, openai, "
            "deepgram or local."
        )

    hint = _normalize_language(language) or _normalize_language(
        _env("BROWZY_STT_LANGUAGE", "auto")
    )
    safe_name = re.sub(r"[^A-Za-z0-9._-]", "", filename or "") or "voice.webm"
    mime = (content_type or "audio/webm").split(";")[0].strip() or "audio/webm"

    try:
        text, detected = await handler(data, safe_name, mime, hint)
    except VoiceError:
        raise
    except httpx.TimeoutException as exc:
        raise VoiceError(f"{provider} timed out. Try a shorter command.") from exc
    except httpx.HTTPError as exc:
        raise VoiceError(f"Could not reach {provider}: {exc}") from exc

    cleaned = _clean(text)
    tag = _normalize_language(detected) or detect_language(cleaned)
    return {
        "text": cleaned,
        "language": tag,
        "provider": provider,
        "model": _model_for(provider),
    }
