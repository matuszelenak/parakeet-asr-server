// Languages supported by models with source/target language selection
// (e.g. nvidia/canary-1b-v2). Mirrors the server-side Language enum.

export enum Language {
  bg = 'bg',
  hr = 'hr',
  cs = 'cs',
  da = 'da',
  nl = 'nl',
  en = 'en',
  et = 'et',
  fi = 'fi',
  fr = 'fr',
  de = 'de',
  el = 'el',
  hu = 'hu',
  it = 'it',
  lv = 'lv',
  lt = 'lt',
  mt = 'mt',
  pl = 'pl',
  pt = 'pt',
  ro = 'ro',
  sk = 'sk',
  sl = 'sl',
  es = 'es',
  sv = 'sv',
  ru = 'ru',
  uk = 'uk',
}

export const LANGUAGE_NAMES: Record<Language, string> = {
  [Language.bg]: 'Bulgarian',
  [Language.hr]: 'Croatian',
  [Language.cs]: 'Czech',
  [Language.da]: 'Danish',
  [Language.nl]: 'Dutch',
  [Language.en]: 'English',
  [Language.et]: 'Estonian',
  [Language.fi]: 'Finnish',
  [Language.fr]: 'French',
  [Language.de]: 'German',
  [Language.el]: 'Greek',
  [Language.hu]: 'Hungarian',
  [Language.it]: 'Italian',
  [Language.lv]: 'Latvian',
  [Language.lt]: 'Lithuanian',
  [Language.mt]: 'Maltese',
  [Language.pl]: 'Polish',
  [Language.pt]: 'Portuguese',
  [Language.ro]: 'Romanian',
  [Language.sk]: 'Slovak',
  [Language.sl]: 'Slovenian',
  [Language.es]: 'Spanish',
  [Language.sv]: 'Swedish',
  [Language.ru]: 'Russian',
  [Language.uk]: 'Ukrainian',
}

export interface LanguageInfo {
  code: string
  name: string
}

// Ordered list for populating dropdowns.
export const LANGUAGES: LanguageInfo[] = Object.values(Language).map((code) => ({
  code,
  name: LANGUAGE_NAMES[code],
}))

export const DEFAULT_LANGUAGE = Language.en
