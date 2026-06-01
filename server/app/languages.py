"""Supported languages for models that accept source/target language selection.

Some models (e.g. ``nvidia/canary-1b-v2``) accept ``source_lang`` and
``target_lang`` arguments, enabling both transcription and speech translation.
The language set below matches Canary v2's supported languages.
"""
from __future__ import annotations

from enum import Enum


class Language(str, Enum):
    bg = "bg"
    hr = "hr"
    cs = "cs"
    da = "da"
    nl = "nl"
    en = "en"
    et = "et"
    fi = "fi"
    fr = "fr"
    de = "de"
    el = "el"
    hu = "hu"
    it = "it"
    lv = "lv"
    lt = "lt"
    mt = "mt"
    pl = "pl"
    pt = "pt"
    ro = "ro"
    sk = "sk"
    sl = "sl"
    es = "es"
    sv = "sv"
    ru = "ru"
    uk = "uk"


LANGUAGE_NAMES: dict[Language, str] = {
    Language.bg: "Bulgarian",
    Language.hr: "Croatian",
    Language.cs: "Czech",
    Language.da: "Danish",
    Language.nl: "Dutch",
    Language.en: "English",
    Language.et: "Estonian",
    Language.fi: "Finnish",
    Language.fr: "French",
    Language.de: "German",
    Language.el: "Greek",
    Language.hu: "Hungarian",
    Language.it: "Italian",
    Language.lv: "Latvian",
    Language.lt: "Lithuanian",
    Language.mt: "Maltese",
    Language.pl: "Polish",
    Language.pt: "Portuguese",
    Language.ro: "Romanian",
    Language.sk: "Slovak",
    Language.sl: "Slovenian",
    Language.es: "Spanish",
    Language.sv: "Swedish",
    Language.ru: "Russian",
    Language.uk: "Ukrainian",
}

# Default used when the model supports languages but the caller omits them.
DEFAULT_LANGUAGE = Language.en


def model_supports_languages(model_name: str) -> bool:
    """Whether the configured model accepts source/target language arguments."""
    return "canary" in model_name.lower()
