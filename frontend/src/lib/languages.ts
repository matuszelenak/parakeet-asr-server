// Languages the Nemotron streaming model can be prompted with. Mirrors the
// server-side list (see server/app/languages.py); the live list is fetched
// from /health at runtime, this is the fallback used before that resolves.

export interface LanguageInfo {
  code: string
  name: string
}

// "auto" lets the model detect and tag the spoken language.
export const DEFAULT_LANGUAGE = 'auto'

export const LANGUAGES: LanguageInfo[] = [
  { code: 'auto', name: 'Auto-detect' },
  { code: 'en-US', name: 'English' },
  { code: 'es-ES', name: 'Spanish' },
  { code: 'fr-FR', name: 'French' },
  { code: 'de-DE', name: 'German' },
  { code: 'it-IT', name: 'Italian' },
  { code: 'pt-PT', name: 'Portuguese' },
  { code: 'nl-NL', name: 'Dutch' },
  { code: 'pl-PL', name: 'Polish' },
  { code: 'sv-SE', name: 'Swedish' },
  { code: 'cs-CZ', name: 'Czech' },
  { code: 'sk-SK', name: 'Slovak' },
  { code: 'da-DK', name: 'Danish' },
  { code: 'nb-NO', name: 'Norwegian' },
  { code: 'fi-FI', name: 'Finnish' },
  { code: 'hu-HU', name: 'Hungarian' },
  { code: 'ro-RO', name: 'Romanian' },
  { code: 'hr-HR', name: 'Croatian' },
  { code: 'bg-BG', name: 'Bulgarian' },
  { code: 'et-EE', name: 'Estonian' },
  { code: 'uk-UA', name: 'Ukrainian' },
  { code: 'ru-RU', name: 'Russian' },
  { code: 'tr-TR', name: 'Turkish' },
  { code: 'ar', name: 'Arabic' },
  { code: 'hi-IN', name: 'Hindi' },
  { code: 'ja-JP', name: 'Japanese' },
  { code: 'ko-KR', name: 'Korean' },
  { code: 'vi-VN', name: 'Vietnamese' },
  { code: 'zh-CN', name: 'Mandarin Chinese' },
]
