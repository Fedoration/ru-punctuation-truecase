import unicodedata

ACCENT_MAPPING = {
    "́": "",
    "̀": "",
    "а́": "а",
    "а̀": "а",
    "е́": "е",
    "é": "е",
    "ѐ": "е",
    "и́": "и",
    "ѝ": "и",
    "о́": "о",
    "о̀": "о",
    "у́": "у",
    "у̀": "у",
    "ы́": "ы",
    "ы̀": "ы",
    "э́": "э",
    "э̀": "э",
    "ю́": "ю",
    "̀ю": "ю",
    "я́́": "я",
    "я̀": "я",
    "ё": "е",
}
ACCENT_MAPPING = {
    unicodedata.normalize("NFKC", i): j for i, j in ACCENT_MAPPING.items()
}
