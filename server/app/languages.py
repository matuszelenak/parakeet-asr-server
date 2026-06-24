"""Languages understood by the Nemotron streaming ASR model.

The model is prompt-conditioned on a target language given as a BCP-47 locale
(e.g. ``en-US``), or the special value ``auto`` to auto-detect the spoken
language.  The list below is a curated subset of the model's supported locales;
``auto`` is always available and is the default.
"""
from __future__ import annotations

# Special value: let the model detect the language itself.
AUTO = "auto"

# Default used when the caller omits a language.
DEFAULT_LANGUAGE = AUTO

# locale code -> human-readable name. Ordered for dropdown display.
LANGUAGE_NAMES: dict[str, str] = {
    "en-US": "English",
    "es-ES": "Spanish",
    "fr-FR": "French",
    "de-DE": "German",
    "it-IT": "Italian",
    "pt-PT": "Portuguese",
    "nl-NL": "Dutch",
    "pl-PL": "Polish",
    "sv-SE": "Swedish",
    "cs-CZ": "Czech",
    "sk-SK": "Slovak",
    "da-DK": "Danish",
    "nb-NO": "Norwegian",
    "fi-FI": "Finnish",
    "hu-HU": "Hungarian",
    "ro-RO": "Romanian",
    "hr-HR": "Croatian",
    "bg-BG": "Bulgarian",
    "et-EE": "Estonian",
    "uk-UA": "Ukrainian",
    "ru-RU": "Russian",
    "tr-TR": "Turkish",
    "ar": "Arabic",
    "hi-IN": "Hindi",
    "ja-JP": "Japanese",
    "ko-KR": "Korean",
    "vi-VN": "Vietnamese",
    "zh-CN": "Mandarin Chinese",
}

# Full ordered list exposed to clients: Auto-detect first, then the locales.
LANGUAGES: list[tuple[str, str]] = [(AUTO, "Auto-detect")] + list(
    LANGUAGE_NAMES.items()
)

_VALID = {AUTO, *LANGUAGE_NAMES}


def is_supported(code: str | None) -> bool:
    """Whether ``code`` is a language this model can be prompted with."""
    return code is None or code in _VALID


def resolve_language(code: str | None, default: str) -> str:
    """Return a valid language prompt value, falling back to ``default``."""
    if code and code in _VALID:
        return code
    return default
