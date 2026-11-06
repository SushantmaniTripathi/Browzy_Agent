"""Multilingual command understanding for Browzy.

Voice input arrives as Hindi, Marathi, English, or — most often — a mix of all
three ("youtube pe lofi chalao", "माझा फॉर्म भर", "flipkart pe iphone ki
kimat").  The deterministic browser parser is English, so this module rewrites
a spoken instruction into a canonical English command *while keeping the search
query in the language the user actually spoke*:

    "यूट्यूब पर लोफी बीट्स चलाओ"      -> "play लोफी बीट्स on youtube"
    "flipkart pe iphone ki kimat"     -> "compare prices for iphone on flipkart"
    "माझा फॉर्म भरायचा आहे"           -> "fill this form with my details"
    "गूगल वर हवामान शोधा"             -> "search हवामान on google"

English text is returned untouched (``normalize_command`` is a no-op unless the
sentence contains Devanagari or a Hinglish/Marathi command word), so the
existing English behaviour cannot regress.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
_LATIN = "A-Za-z"
_DEV = "\u0900-\u097F"


def _compile(words: List[str]) -> Optional[re.Pattern]:
    """Word-boundary regex for a list of Latin/Devanagari terms."""
    terms = sorted({w for w in words if w}, key=len, reverse=True)
    if not terms:
        return None
    body = "|".join(re.escape(term) for term in terms)
    return re.compile(
        rf"(?<![{_LATIN}{_DEV}])(?:{body})(?![{_LATIN}{_DEV}])", re.IGNORECASE
    )


def _sub(text: str, pattern: Optional[re.Pattern]) -> str:
    return pattern.sub(" ", text) if pattern is not None else text


# ---------------------------------------------------------------------------
# Platforms (Devanagari + the romanised spellings people actually say)
# ---------------------------------------------------------------------------
PLATFORM_WORDS: Tuple[Tuple[str, List[str]], ...] = (
    ("youtube", ["youtube", "yt", "यूट्यूब", "युट्यूब", "यूट्युब", "यू ट्यूब", "यूट्युब"]),
    ("google", ["google", "गूगल", "गुगल"]),
    ("gmail", ["gmail", "जीमेल", "जिमेल"]),
    ("github", ["github", "गिटहब", "गीथब"]),
    ("wikipedia", ["wikipedia", "विकिपीडिया", "विकीपीडिया"]),
    ("amazon", ["amazon", "अमेज़न", "अमेजन", "अमेझॉन", "अॅमेझॉन"]),
    ("flipkart", ["flipkart", "फ्लिपकार्ट", "फ्लिपकर्ट"]),
    ("myntra", ["myntra", "मायंत्रा", "मिंत्रा"]),
    ("meesho", ["meesho", "मीशो"]),
    ("spotify", ["spotify", "स्पॉटिफाई", "स्पॉटिफाय"]),
    ("reddit", ["reddit", "रेडिट"]),
    ("instagram", ["instagram", "इंस्टाग्राम", "इन्स्टाग्राम"]),
    ("linkedin", ["linkedin", "लिंक्डइन", "लिंक्डीन"]),
    ("netflix", ["netflix", "नेटफ्लिक्स"]),
    ("whatsapp", ["whatsapp", "व्हाट्सएप", "व्हाट्सऐप"]),
    ("facebook", ["facebook", "फेसबुक"]),
    ("twitter", ["twitter", "ट्विटर"]),
    ("ebay", ["ebay", "ईबे"]),
    ("aliexpress", ["aliexpress", "अलीएक्सप्रेस"]),
)
PLATFORM_PATTERNS = [(name, _compile(words)) for name, words in PLATFORM_WORDS]
PLATFORM_ANY = _compile([w for _, words in PLATFORM_WORDS for w in words])


# ---------------------------------------------------------------------------
# Intent vocabulary
# ---------------------------------------------------------------------------
FILL_FORM_WORDS = [
    "फॉर्म भर", "फार्म भर", "फॉर्म भरायचा", "फॉर्म भरायची", "फॉर्म भरणे",
    "फॉर्म भरून", "फॉर्म भरा", "फॉर्म भरो", "फॉर्म भर दो", "फॉर्म भर द्या",
    "form bhar", "form bharo", "form bhara", "form bhar do", "form bharun",
    "form bharaycha", "form bharaychi", "form bharne", "form fill",
    "autofill", "auto fill",
]
PRICE_WORDS = [
    "किंमत", "किमत", "कीमत", "किंमती", "दाम", "भाव", "कितने का", "कितने",
    "किती", "प्राइस", "रेट",
    "kimat", "keemat", "kinti", "daam", "bhav", "kitne", "kitna", "kitni",
    "kiti", "price", "prices", "rate", "how much", "cost",
]
COMPARE_WORDS = [
    "तुलना", "तुलना कर", "तुलना करा", "तुलना करो", "तुलनात्मक",
    "compare", "comparison", "tulna", "tulnatmak",
]
PLAY_WORDS = [
    "चलाओ", "चलवा", "चला", "चालू कर", "चालू", "बजाओ", "बजा", "वाजवा",
    "वाजव", "लाव", "प्ले कर", "प्ले",
    "chalao", "chala do", "chalu kar", "chalu", "bajao", "baja", "vajava",
    "vajav", "lava", "play", "stream", "start",
]
SEARCH_WORDS = [
    "खोजो", "खोज", "खोजें", "ढूंढो", "ढूँढो", "ढूंढ", "ढूँढ", "शोधा", "शोध",
    "शोधून", "सर्च कर", "सर्च", "दिखाओ", "दाखवा",
    "search", "khojo", "khoj", "dhoondo", "dhundo", "dhoond", "shodha",
    "shodh", "search kar", "find", "look up", "dikhao", "dakhva",
]
OPEN_WORDS = [
    "खोलो", "खोल", "खोल दो", "खोल द्या", "उघडा", "उघड", "उघडून",
    "जाओ", "जा", "kholo", "khol do", "khol", "kholiye", "ughada", "ughad",
    "open kar", "open", "launch", "go to", "jao", "ja",
]
FILL_FORM = _compile(FILL_FORM_WORDS)
PRICE_RE = _compile(PRICE_WORDS)
COMPARE_RE = _compile(COMPARE_WORDS)
PLAY_RE = _compile(PLAY_WORDS)
SEARCH_RE = _compile(SEARCH_WORDS)
OPEN_RE = _compile(OPEN_WORDS)
ACTION_ANY = _compile(
    FILL_FORM_WORDS + PRICE_WORDS + COMPARE_WORDS + PLAY_WORDS + SEARCH_WORDS + OPEN_WORDS
)

# Words that only carry politeness / grammar, never meaning for a search.
FILLER_WORDS = [
    "कृपया", "ज़रा", "जरा", "थोड़ा", "थोडं", "थोडा", "मला", "मुझे", "मुझको",
    "हमें", "माझ्यासाठी", "मेरे लिए", "मेरेलिए", "बरं", "आता", "जल्दी",
    "please", "kindly", "for me", "mujhe", "mujhko", "mala", "majhyasathi",
    "zara", "thoda", "jaldi", "quickly", "bara", "aata",
]
POSTPOSITION_WORDS = [
    "के बारे में", "के बारे", "बारे में", "बारे", "विषयी", "बद्दल",
    "पर", "पे", "वर", "वरून", "मध्ये", "मधून", "मधे", "पासून", "कडून",
    "कडे", "साठी", "करिता", "से", "में", "का", "की", "के", "चा", "ची",
    "चे", "च्या", "ला", "ने", "ना",
    "pe", "par", "var", "mein", "se", "ka", "ki", "ke", "cha", "chi",
    "che", "liye", "sathi", "madhe", "madun",
]
CONJUNCTION_WORDS = ["और", "आणि", "अनि", "तर", "और", "aur", "ani", "फिर", "मग"]
POLITE_WORDS = [
    "बताओ", "बता", "बताइए", "सांगा", "सांग", "सांगून", "करो", "करा", "कर",
    "करून", "दो", "द्या", "दे", "चाहिए", "हवे", "हवा", "पाहिजे",
    "batao", "bata", "bataiye", "sanga", "sang", "sangun", "karo", "kara",
    "kar do", "kar", "karun", "dya", "chahiye", "pahije",
]
AUXILIARY_WORDS = [
    "है", "हैं", "हो", "था", "थी", "थे", "आहे", "आहेत", "होते", "होता",
    "होती", "नाही", "नहीं",
    "hai", "hain", "ahe", "ahet", "nahi", "nahin",
]
MEDIA_WORDS = [
    "गाना", "गाने", "गाणं", "गाणी", "गीत", "संगीत", "व्हिडिओ", "व्हिडीओ",
    "वीडियो", "भजन", "मूवी", "मूव्ही", "ट्रेलर", "फिल्म",
    "song", "songs", "gaana", "gana", "gane", "music", "video", "videos",
    "track", "clip", "movie", "trailer", "film",
]
FILLER = _compile(FILLER_WORDS)
POSTPOSITIONS = _compile(POSTPOSITION_WORDS)
CONJUNCTIONS = _compile(CONJUNCTION_WORDS)
POLITE = _compile(POLITE_WORDS)
AUXILIARIES = _compile(AUXILIARY_WORDS)
MEDIA = _compile(MEDIA_WORDS)

# Attached Marathi/Hindi postpositions ("नेहरूंबद्दल" -> "नेहरू").
ATTACHED_SUFFIX_RE = re.compile(
    r"([\u0900-\u097F]{3,}?)(?:बद्दल|विषयी|साठी|करिता)(?![\u0900-\u097F])"
)

# Only these words make us run the multilingual rewrite.  A bare "ka"/"hai"
# would fire on ordinary English, so the list is deliberately verb-heavy.
_STRONG_MARKERS = _compile(
    [
        "kholo", "khol do", "kholiye", "chalao", "chala do", "bajao", "vajava",
        "vajav", "bharo", "bhar do", "bharun", "bharaycha", "dhoondo", "dhundo",
        "khojo", "shodha", "ughada", "ughad", "kimat", "keemat", "batao",
        "sanga", "sangun", "mujhe", "mujhko", "mala", "ahe", "ahet", "nahi",
        "nahin", "karo", "kar do", "kiti", "kitne", "tulna", "saransh",
        "samjhao", "samjavun", "dya", "pahije", "jao",
    ]
)

STOPWORD_QUERY = {
    "", "the", "a", "an", "it", "this", "that", "please", "video", "song",
    "गाना", "गाणं", "व्हिडिओ", "वीडियो", "music",
}


def needs_normalization(text: str) -> bool:
    """True when the sentence is not plain English."""
    if not text:
        return False
    if DEVANAGARI_RE.search(text):
        return True
    return bool(_STRONG_MARKERS and _STRONG_MARKERS.search(text))


def _platform(text: str) -> Optional[str]:
    for name, pattern in PLATFORM_PATTERNS:
        if pattern is not None and pattern.search(text):
            return name
    return None


def _action(text: str) -> Optional[str]:
    """Order matters: 'youtube kholo aur gaana chalao' is a play, not an open."""
    if FILL_FORM is not None and FILL_FORM.search(text):
        return "fill_form"
    if COMPARE_RE is not None and COMPARE_RE.search(text):
        return "compare_prices"
    if PRICE_RE is not None and PRICE_RE.search(text):
        return "compare_prices"
    if PLAY_RE is not None and PLAY_RE.search(text):
        return "play"
    if SEARCH_RE is not None and SEARCH_RE.search(text):
        return "search"
    if OPEN_RE is not None and OPEN_RE.search(text):
        return "open"
    return None


def _query(work: str, action: str) -> str:
    for pattern in (
        PLATFORM_ANY, ACTION_ANY, FILLER, POSTPOSITIONS, CONJUNCTIONS,
        POLITE, AUXILIARIES,
    ):
        work = _sub(work, pattern)
    if action in ("play", "open"):
        work = _sub(work, MEDIA)
    work = ATTACHED_SUFFIX_RE.sub(r"\1", work)
    work = re.sub(r"\s+", " ", work).strip()
    work = work.strip(" .,!?;:'\"“”‘’()[]-–—।|/\\")
    work = re.sub(r"\s+", " ", work).strip()
    if work.lower() in STOPWORD_QUERY:
        return ""
    return work


def normalize_command(text: str) -> str:
    """Rewrite a Hindi/Marathi/Hinglish instruction into canonical English.

    The query itself keeps its original script so YouTube/Google search for the
    right thing.  Returns ``text`` unchanged when nothing is recognised.
    """
    if not text or not needs_normalization(text):
        return text

    raw = text.strip()
    platform = _platform(raw)
    action = _action(raw)
    if action is None:
        return text

    query = _query(raw, action)

    if action == "fill_form":
        return "fill this form with my details"
    if action == "compare_prices":
        if not query:
            return text
        return f"compare prices for {query}" + (f" on {platform}" if platform else "")
    if action == "play":
        if query:
            return f"play {query}" + (f" on {platform}" if platform else "")
        if platform:
            return f"open {platform}"
        # "गाणं लाव" / "gaana chalao" — play something, YouTube is the default.
        return "play music on youtube"
    if action == "search":
        if query:
            return f"search {query}" + (f" on {platform}" if platform else "")
        if platform:
            return f"open {platform}"
        return text
    # open
    target = platform or query
    if target:
        return f"open {target}"
    return text


# ---------------------------------------------------------------------------
# Language detection (UI chip + STT hints)
# ---------------------------------------------------------------------------
MARATHI_MARKERS = (
    "आहे", "आहेत", "आणि", "मला", "माझा", "माझी", "माझे", "तुम्ही", "करा",
    "उघडा", "शोधा", "वाजवा", "पाहिजे", "काय", "कसे", "कुठे", "बद्दल", "विषयी",
    "सांगा", "द्या", "हवे", "नाही", "गाणं", "पान", "कृपया", "मधून", "साठी",
)
HINDI_MARKERS = (
    "है", "हैं", "और", "मुझे", "आप", "करो", "खोलो", "चलाओ", "चाहिए", "क्या",
    "कैसे", "कहाँ", "मेरा", "मेरी", "मेरे", "बारे", "बताओ", "दो", "नहीं", "था",
    "गाना", "पेज", "कृपया", "लिए", "वाला",
)
HINGLISH_MARKERS = (
    "kholo", "karo", "chalao", "bajao", "bharo", "dhoondo", "dhundo", "khojo",
    "kimat", "keemat", "kitne", "kiti", "batao", "sanga", "mujhe", "mala",
    "nahi", "hai", "ahe", "bhai", "yaar", "matlab", "chahiye", "mera", "meri",
    "mere", "bhar", "karna",
)


def detect_language(text: str) -> str:
    """Best-effort BCP-47-ish label: hi-IN, mr-IN, en-IN or a '+'-joined mix."""
    if not text:
        return "en-IN"

    devanagari = bool(DEVANAGARI_RE.search(text))
    latin = bool(re.search(r"[A-Za-z]", text))
    lowered = text.lower()

    tags: List[str] = []
    if devanagari:
        if "आहे" in text or "आहेत" in text:
            tags.append("mr-IN")
        else:
            mr_score = sum(1 for marker in MARATHI_MARKERS if marker in text)
            hi_score = sum(1 for marker in HINDI_MARKERS if marker in text)
            tags.append("mr-IN" if mr_score > hi_score else "hi-IN")

    hinglish = any(marker in lowered for marker in HINGLISH_MARKERS)
    if hinglish and not tags:
        tags.append("hi-IN")

    if latin and (not devanagari or hinglish or not tags):
        tags.append("en-IN")
    if not tags:
        tags.append("en-IN")

    seen: List[str] = []
    for tag in tags:
        if tag not in seen:
            seen.append(tag)
    return "+".join(seen)


LANGUAGE_LABELS = {
    "hi-IN": "Hindi",
    "mr-IN": "Marathi",
    "en-IN": "English",
}


def language_label(tag: str) -> str:
    """'hi-IN+mr-IN' -> 'Hindi + Marathi'."""
    if not tag:
        return ""
    parts = [LANGUAGE_LABELS.get(part, part) for part in tag.split("+")]
    return " + ".join(parts)
