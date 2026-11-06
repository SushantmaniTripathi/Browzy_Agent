"""Deterministic form understanding for Browzy.

The model is good at language, not at guessing which HTML input means "pincode".
So field matching happens in Python, in two tiers:

  1. canonical alias table + normalisation (case, spaces, underscores, hyphens),
  2. the local model, only for labels that tier 1 could not resolve.

Playwright then fills the fields through stable data attributes. Nothing is
filled into a non-empty field, so a form that already has user input is never
overwritten.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

APP_HOME = (
    Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "Browzy"
    if os.name == "nt"
    else Path.home() / ".browzy"
)
PROFILE_PATH = APP_HOME / "profile.json"

# Browzy ships with no personal data. The vault is empty until the user uploads
# a document; these keys only define what can be stored and how the panel groups
# it. `section` drives the "Your Data" tab, `label` is what the user sees.
PROFILE_SCHEMA: List[Dict[str, str]] = [
    {"key": "full_name", "label": "Full name", "section": "Personal"},
    {"key": "first_name", "label": "First name", "section": "Personal"},
    {"key": "middle_name", "label": "Middle name", "section": "Personal"},
    {"key": "last_name", "label": "Last name", "section": "Personal"},
    {"key": "dob", "label": "Date of birth", "section": "Personal"},
    {"key": "gender", "label": "Gender", "section": "Personal"},
    {"key": "nationality", "label": "Nationality", "section": "Personal"},
    {"key": "email", "label": "Email", "section": "Contact"},
    {"key": "phone", "label": "Phone", "section": "Contact"},
    {"key": "alt_phone", "label": "Alternate phone", "section": "Contact"},
    {"key": "street_address", "label": "Address", "section": "Address"},
    {"key": "address_line2", "label": "Address line 2", "section": "Address"},
    {"key": "city", "label": "City", "section": "Address"},
    {"key": "state", "label": "State", "section": "Address"},
    {"key": "pincode", "label": "PIN / ZIP code", "section": "Address"},
    {"key": "country", "label": "Country", "section": "Address"},
    {"key": "company", "label": "Company", "section": "Work"},
    {"key": "job_title", "label": "Job title", "section": "Work"},
    {"key": "website", "label": "Website", "section": "Work"},
    {"key": "username", "label": "Username", "section": "Account"},
    {"key": "pan", "label": "PAN / Tax ID", "section": "Documents"},
    {"key": "aadhaar", "label": "Aadhaar / National ID", "section": "Documents"},
    {"key": "passport", "label": "Passport number", "section": "Documents"},
    {"key": "license", "label": "Driving licence", "section": "Documents"},
]

SCHEMA_KEYS = [item["key"] for item in PROFILE_SCHEMA]
SCHEMA_LABELS = {item["key"]: item["label"] for item in PROFILE_SCHEMA}
SCHEMA_SECTIONS = {item["key"]: item["section"] for item in PROFILE_SCHEMA}

# The old build seeded a fake profile. If that exact data is still on disk it was
# never uploaded by the user, so it must not be used to fill anything.
_LEGACY_DEFAULT_PROFILE: Dict[str, str] = {
    "first_name": "Aarav",
    "last_name": "Sharma",
    "full_name": "Aarav Sharma",
    "email": "aarav.sharma@example.com",
    "phone": "+919876543210",
    "street_address": "123 Marine Drive",
    "address_line2": "",
    "city": "Mumbai",
    "state": "Maharashtra",
    "pincode": "400020",
    "country": "India",
    "company": "",
    "dob": "",
}

# Canonical key -> the many ways a site may spell it. Order inside the list does
# not matter; matching scores by alias length, so "email address" beats "address".
SYNONYM_MAP: Dict[str, List[str]] = {
    "first_name": ["first name", "firstname", "fname", "given name", "forename", "first"],
    "middle_name": ["middle name", "middlename", "middle", "middle initial", "additional name"],
    "last_name": ["last name", "lastname", "lname", "surname", "family name", "last"],
    "full_name": ["full name", "fullname", "name", "your name", "complete name", "full"],
    "email": ["email", "e mail", "email address", "mail id", "email id", "emailaddress"],
    "phone": [
        "phone", "telephone", "mobile", "mobile number", "contact number",
        "phone number", "cell", "cellphone", "whatsapp", "tel",
    ],
    "street_address": [
        "address", "street address", "address line 1", "address line1", "addressline1",
        "house no", "house number", "flat no", "building", "billing address",
        "shipping address", "street",
    ],
    "address_line2": ["address line 2", "address line2", "addressline2", "apartment", "suite", "landmark"],
    "city": ["city", "town", "district", "locality"],
    "state": ["state", "province", "region", "state province"],
    "pincode": ["pincode", "pin code", "zip", "zipcode", "zip code", "postal code", "postcode", "post code"],
    "country": ["country", "nation"],
    "company": ["company", "organization", "organisation", "employer", "firm"],
    "dob": ["date of birth", "dob", "birth date", "birthday", "birthdate"],
    "gender": ["gender", "sex"],
    "nationality": ["nationality", "citizenship"],
    "alt_phone": ["alternate phone", "alternate mobile", "secondary phone", "other phone"],
    "job_title": ["job title", "designation", "position", "role", "occupation", "profession"],
    "website": ["website", "url", "portfolio", "web site", "linkedin"],
    "username": ["username", "user name", "handle", "login id"],
    "pan": ["pan", "pan number", "tax id", "tin", "permanent account number"],
    "aadhaar": ["aadhaar", "aadhar", "national id", "id number", "government id"],
    "passport": ["passport", "passport number"],
    "license": ["driving licence", "driving license", "license number", "dl number"],
}

AUTOCOMPLETE_MAP = {
    "given-name": "first_name",
    "additional-name": "middle_name",
    "family-name": "last_name",
    "name": "full_name",
    "email": "email",
    "tel": "phone",
    "tel-national": "phone",
    "street-address": "street_address",
    "address-line1": "street_address",
    "address-line2": "address_line2",
    "address-level2": "city",
    "address-level1": "state",
    "postal-code": "pincode",
    "country-name": "country",
    "country": "country",
    "organization": "company",
    "bday": "dob",
}

# Inputs that must never receive personal data.
SKIP_TOKENS = (
    "search", "query", "coupon", "promo", "voucher", "gift", "referral", "discount",
    "otp", "captcha", "password", "passcode", "newsletter", "subscribe", "username",
    "login", "card number", "cvv", "expiry", "amount", "quantity", "qty",
    "verification", "country code",
    "notes", "comment", "message", "subject", "feedback", "review", "captcha",
    # Meeting / invite / join-code fields are never personal profile data. This
    # is what stopped "Enter a code or link" from being filled with a website URL.
    "meeting", "meet code", "meeting code", "meeting link", "invite",
    "invite code", "invite link", "access code", "join code", "join link",
    "code or link", "link or code", "enter a code", "enter code",
)

SKIP_TYPES = {
    "hidden", "password", "submit", "button", "reset", "file", "checkbox", "radio",
    "image", "range", "color", "search", "date", "time", "datetime-local", "month",
    "week", "number", "url",
}


def normalize_string(text: str) -> str:
    """Strip casing, punctuation, spaces and separators: 'Full Name' -> 'fullname'."""
    if not text:
        return ""
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", str(text or "").lower())


def _read_store() -> Dict[str, Any]:
    """Read the whole vault file: {"fields": {...}, "sources": [...]}."""
    try:
        if not PROFILE_PATH.exists():
            return {"fields": {}, "sources": []}
        data = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[Forms] Could not read profile: {exc}", flush=True)
        return {"fields": {}, "sources": []}

    if not isinstance(data, dict):
        return {"fields": {}, "sources": []}

    fields = data.get("fields") if isinstance(data.get("fields"), dict) else data
    clean: Dict[str, str] = {}
    for key, value in fields.items():
        if isinstance(value, (dict, list)):
            continue
        text = str(value or "").strip()
        if text:
            clean[str(key)] = text

    sources = data.get("sources") if isinstance(data.get("sources"), list) else []
    return {"fields": clean, "sources": [s for s in sources if isinstance(s, dict)]}


def _is_legacy_demo(fields: Dict[str, str]) -> bool:
    """True if these are exactly the values the old build seeded."""
    if not fields:
        return False
    for key, value in fields.items():
        if key not in _LEGACY_DEFAULT_PROFILE:
            return False
        if str(value).strip() != str(_LEGACY_DEFAULT_PROFILE[key]).strip():
            return False
    return True


def load_profile() -> Dict[str, str]:
    """Load the uploaded profile. Empty until the user provides a document."""
    store = _read_store()
    fields = store["fields"]

    if _is_legacy_demo(fields):
        print("[Forms] Dropping the old seeded demo profile", flush=True)
        save_profile({}, sources=[])
        return {}

    return fields


def load_sources() -> List[Dict[str, Any]]:
    return _read_store()["sources"]


def save_profile(
    profile: Dict[str, str], sources: Optional[List[Dict[str, Any]]] = None
) -> None:
    """Persist the vault. Only extracted values are stored, never the document."""
    if sources is None:
        sources = load_sources()
    payload = {
        "fields": {str(k): str(v) for k, v in profile.items() if str(v or "").strip()},
        "sources": sources,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    try:
        PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        PROFILE_PATH.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except Exception as exc:
        print(f"[Forms] Could not save profile: {exc}", flush=True)


def merge_upload(existing: Dict[str, str], incoming: Dict[str, str]) -> Dict[str, str]:
    """Merge a new document's fields into the vault.

    A new upload updates real changes, but must not downgrade a clean value with
    the same value read worse: OCR often drops spaces ("Product Designer" ->
    "ProductDesigner"). When the two only differ in spacing/punctuation, keep the
    more readable (longer) one.

    When a document carries a fuller name (more words), its first/middle/last
    parts replace the old ones so the vault never mixes two documents' names.
    """
    merged = dict(existing)
    incoming = dict(incoming)
    old_full = str(merged.get("full_name") or "").strip()
    new_full = str(incoming.get("full_name") or "").strip()
    if new_full and old_full and len(new_full.split()) < len(old_full.split()):
        # The new document knows less about the name (e.g. a resume without the
        # middle name). Keep the fuller name block instead of mixing the two.
        for key in ("full_name", "first_name", "middle_name", "last_name"):
            incoming.pop(key, None)
    elif new_full and normalize_string(new_full) != normalize_string(old_full):
        for key in ("first_name", "middle_name", "last_name"):
            merged.pop(key, None)
    for key, value in incoming.items():
        text = str(value or "").strip()
        if not text:
            continue
        current = str(merged.get(key) or "").strip()
        if current and normalize_string(current) == normalize_string(text) and len(current) >= len(text):
            continue
        merged[key] = text
    return merged


def has_data() -> bool:
    return bool(load_profile())


# Labels worth remembering when a user types into a form Browzy has never seen.
# Kept deliberately narrow: free-text boxes ("message", "comments", "search")
# must never end up in the vault.
LEARN_HINTS = (
    "name", "fullname", "firstname", "lastname", "surname", "middlename",
    "email", "mail", "phone", "mobile", "contact", "whatsapp", "telephone",
    "address", "city", "town", "state", "province", "pin", "zip", "postal",
    "country", "dob", "birth", "age", "gender", "sex", "nationality",
    "roll", "seat", "enrol", "enroll", "registration", "regno", "student",
    "college", "school", "university", "course", "branch", "semester", "stream",
    "father", "mother", "guardian", "parent", "aadhaar", "aadhar", "pan",
    "passport", "licence", "license", "company", "organisation", "organization",
    "designation", "department", "job", "occupation", "website", "linkedin",
)

_LEARN_SEEN: Dict[str, Dict[str, str]] = {}

# Words that mean the box is not personal data, even if it contains "name".
LEARN_EXCLUDE = (
    "file", "folder", "list", "project", "search", "password", "passcode", "card",
    "cvv", "otp", "coupon", "promo", "captcha", "username", "login", "amount",
    "quantity", "message", "comment", "note", "subject", "title", "url", "link",
    "domain", "company name", "business name", "brand", "event", "team",
    "meeting", "invite", "code or link", "link or code", "join code",
)


def _learn_identity(field: Dict[str, Any]) -> str:
    parts = [str(field.get(key) or "") for key in ("label", "name", "id", "placeholder")]
    return "|".join(part.strip().lower() for part in parts)


def _slug_key(label: str) -> str:
    words = re.findall(r"[a-z0-9]+", str(label or "").lower())
    words = [word for word in words if word not in ("the", "your", "a", "an", "of")]
    if not words:
        return ""
    key = "_".join(words)[:40].strip("_")
    if not key or not key[0].isalpha():
        return ""
    return key


def _learn_key(field: Dict[str, Any], available: List[str]) -> str:
    if is_blocked(field):
        return ""
    key = match_field_key(field, available)
    if key:
        return key
    label = str(field.get("label") or field.get("placeholder") or "").strip()
    if not label or len(label) > 48:
        return ""
    normalized = normalize_string(label)
    if not normalized or not any(hint in normalized for hint in LEARN_HINTS):
        return ""
    if any(word in normalized for word in LEARN_EXCLUDE):
        return ""
    if field.get("tag") == "select":
        return ""
    return _slug_key(label)


async def learn_from_page(
    page,
    profile: Optional[Dict[str, str]] = None,
    stable_only: bool = True,
) -> Dict[str, Any]:
    """Remember values a user has typed into the current form.

    With ``stable_only`` the value must be identical to the previous inspection
    of the same page, so a half-typed word is not saved while the panel polls.
    """
    profile = profile or load_profile()
    try:
        fields = await collect_fields(page)
    except Exception as exc:
        print(f"[Forms] could not learn from page: {exc}", flush=True)
        return {"ok": False, "learned": 0, "keys": []}

    url = getattr(page, "url", "") or ""
    previous = _LEARN_SEEN.get(url, {})
    current: Dict[str, str] = {}
    learned: Dict[str, str] = {}
    available = list(dict.fromkeys(list(profile.keys()) + SCHEMA_KEYS))

    for field in fields:
        value = str(field.get("value") or "").strip()
        identity = _learn_identity(field)
        if not value or not identity or len(value) > 120:
            continue
        if field.get("tag") == "textarea" and not match_field_key(field, available):
            continue
        if is_blocked(field):
            continue
        current[identity] = value
        if stable_only and previous.get(identity) != value:
            continue
        key = _learn_key(field, available)
        if not key:
            continue
        if str(profile.get(key) or "").strip() == value:
            continue
        learned[key] = value

    _LEARN_SEEN.clear()
    if current:
        _LEARN_SEEN[url] = current

    if not learned:
        return {"ok": True, "learned": 0, "keys": []}

    merged = merge_upload(profile, learned)
    host = url.split("//")[-1].split("/")[0] or "this page"
    source_name = f"form: {host}"
    sources = [s for s in load_sources() if s.get("file") != source_name]
    sources.append(
        {
            "file": source_name,
            "kind": "form",
            "fields": len(learned),
            "chars": 0,
            "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
    )
    save_profile(merged, sources=sources[-8:])
    print(f"[Forms] learned {len(learned)} field(s) from {host}: {', '.join(learned)}", flush=True)
    return {"ok": True, "learned": len(learned), "keys": list(learned.keys())}


def candidate_strings(meta: Dict[str, Any]) -> List[str]:
    """Every string that might describe an input, from most to least reliable."""
    raw = [
        meta.get("id") or "",
        meta.get("name") or "",
        meta.get("placeholder") or "",
        meta.get("ariaLabel") or "",
        meta.get("label") or "",
    ]
    autocomplete = (meta.get("autocomplete") or "").lower()
    if autocomplete:
        raw.append(autocomplete.replace("-", " "))
    return [value for value in raw if str(value).strip()]


def _aliases_for(key: str) -> List[str]:
    aliases = SYNONYM_MAP.get(key)
    if aliases:
        return aliases
    return [key.replace("_", " ")]


# Very short, generic aliases. They are only accepted when the whole candidate
# equals them, so "country name" or "company name" is not read as a person's name.
WEAK_ALIASES = {"name", "full", "first", "middle", "last"}


def _plausible_model_label(label: str) -> bool:
    """Reject labels that are clearly not personal profile fields.

    Tier-2 LLM mapping is only allowed for labels that look like a person's
    data. Search boxes, Meet join codes, OTPs, coupons and links stay unfilled
    even if the model suggests a profile key for them.
    """
    tokens = _tokens(label)
    if not tokens or len(tokens) > 8:
        return False
    joined = " ".join(tokens)
    if any(word in joined for word in (
        "meeting", "invite", "join", "otp", "coupon", "promo", "captcha",
        "search", "query", "password", "passcode", "login", "username",
    )):
        return False
    if ("code" in tokens or "link" in tokens) and not any(
        hint in joined for hint in ("postal", "pin", "zip", "area", "dial", "country")
    ):
        return False
    normalized = normalize_string(label)
    if any(normalize_string(key) == normalized for key in SCHEMA_KEYS):
        return True
    return any(hint in normalized for hint in LEARN_HINTS)


def is_blocked(meta: Dict[str, Any]) -> bool:
    """True for fields that must never be filled (search boxes, coupons, ...)."""
    joined = " ".join(str(c).lower() for c in candidate_strings(meta))
    return any(token in joined for token in SKIP_TOKENS)


def match_field_key(meta: Dict[str, Any], available_keys: List[str]) -> Optional[str]:
    """Return the profile key that best describes this input, or None.

    Matching is score-based rather than first-hit, so "email address" resolves to
    ``email`` instead of ``street_address``. Exact matched aliases win, and a tie
    between two different keys is treated as ambiguous and skipped.
    """
    autocomplete = (meta.get("autocomplete") or "").lower().split()
    for token in autocomplete:
        mapped = AUTOCOMPLETE_MAP.get(token)
        if mapped and mapped in available_keys:
            return mapped

    candidates = candidate_strings(meta)
    normalized = [normalize_string(c) for c in candidates]

    if is_blocked(meta):
        return None

    scores: Dict[str, int] = {}
    for key in available_keys:
        best = 0
        for alias in _aliases_for(key):
            a = normalize_string(alias)
            if not a:
                continue
            for cn in normalized:
                if not cn:
                    continue
                if cn == a:
                    score = 1000 + len(a)
                elif a in cn and a not in WEAK_ALIASES:
                    score = 500 + len(a)
                elif cn in a and len(cn) >= 4:
                    score = 300 + len(cn)
                else:
                    continue
                best = max(best, score)
        if best:
            scores[key] = best

    if not scores:
        return None

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    top_key, top_score = ranked[0]
    if len(ranked) > 1 and ranked[1][1] == top_score:
        return None  # genuinely ambiguous: let the model decide
    return top_key if top_score >= 300 else None


def pick_option(options: List[str], value: str) -> Optional[str]:
    """Choose the <select> option matching the profile value."""
    if not options:
        return None
    target = normalize_string(value)
    if not target:
        return None
    for option in options:
        if normalize_string(option) == target:
            return option
    for option in options:
        normalized = normalize_string(option)
        if target and (target in normalized or normalized in target):
            return option
    return None


COLLECT_FIELDS_JS = r"""
() => {
    const visible = (el) => {
        const style = window.getComputedStyle(el);
        if (style.visibility === 'hidden' || style.display === 'none') return false;
        const rect = el.getBoundingClientRect();
        return rect.width > 0 && rect.height > 0;
    };
    const cleanLabel = (text) => (text || '')
        .replace(/\s+/g, ' ')
        .replace(/\s*[*✱]\s*$/, '')
        .replace(/\s*\((optional|required)\)\s*$/i, '')
        .replace(/^\s*(required|optional)\b\s*/i, '')
        .trim();
    const shortText = (node) => {
        if (!node || !node.innerText) return '';
        const text = cleanLabel(node.innerText);
        return text.length > 0 && text.length <= 120 ? text : '';
    };
    const labelFor = (el) => {
        if (el.id) {
            const explicit = document.querySelector(`label[for="${CSS.escape(el.id)}"]`);
            const text = shortText(explicit);
            if (text) return text;
        }

        const aria = cleanLabel(el.getAttribute('aria-label'));
        if (aria) return aria;

        const labelledBy = el.getAttribute('aria-labelledby');
        if (labelledBy) {
            const text = cleanLabel(
                labelledBy.split(/\s+/)
                    .map((id) => {
                        const node = document.getElementById(id);
                        return node ? node.innerText : '';
                    })
                    .join(' ')
            );
            if (text) return text;
        }

        const wrapping = el.closest('label');
        const wrappedText = shortText(wrapping);
        if (wrappedText) return wrappedText;

        // Google Forms, Microsoft Forms, Typeform and most custom layouts do not
        // use <label>: the question text sits in a heading a few levels above the
        // input. Climb to the question block, but stop before reaching a container
        // that holds several unrelated inputs.
        let node = el.parentElement;
        for (let depth = 0; node && depth < 7; depth += 1, node = node.parentElement) {
            const tag = node.tagName ? node.tagName.toLowerCase() : '';
            if (tag === 'body' || tag === 'form' || tag === 'html') break;
            const inputs = node.querySelectorAll('input, select, textarea').length;
            if (inputs > 3) break;
            const heading = node.querySelector(
                '[role="heading"], legend, h1, h2, h3, h4, h5, ' +
                '.freebirdFormviewerComponentsQuestionBaseTitle, ' +
                '[class*="question-title"], [class*="QuestionTitle"], ' +
                '[class*="questionTitle"], [class*="field-label"], [class*="fieldLabel"], ' +
                '[class*="input-label"], [class*="inputLabel"], [class*="control-label"]'
            );
            const text = shortText(heading);
            if (text) return text;
        }

        const placeholder = cleanLabel(el.getAttribute('placeholder'));
        if (placeholder && !/^(enter|type|your answer|search)/i.test(placeholder)) {
            return placeholder;
        }
        return '';
    };
    const out = [];
    let idx = 0;
    for (const el of document.querySelectorAll('input, select, textarea')) {
        const tag = el.tagName.toLowerCase();
        const type = (el.getAttribute('type') || (tag === 'select' ? 'select' : 'text')).toLowerCase();
        if (el.disabled || el.readOnly) continue;
        if (!visible(el)) continue;
        if (tag === 'input' && ['hidden','password','submit','button','reset','file','checkbox','radio','image','range','color','search'].includes(type)) continue;
        el.setAttribute('data-browzy-field', String(idx));
        out.push({
            idx: idx++,
            tag: tag,
            type: type,
            id: el.id || '',
            name: el.getAttribute('name') || '',
            placeholder: el.getAttribute('placeholder') || '',
            ariaLabel: el.getAttribute('aria-label') || '',
            autocomplete: el.getAttribute('autocomplete') || '',
            label: (labelFor(el) || '').replace(/\s+/g, ' ').trim().slice(0, 120),
            value: ('value' in el ? el.value : '') || '',
            options: tag === 'select'
                ? Array.from(el.options).map((o) => ({ text: o.text.trim(), value: o.value })).slice(0, 60)
                : [],
        });
    }
    return out;
}
"""


async def collect_fields(page) -> List[Dict[str, Any]]:
    try:
        return await page.evaluate(COLLECT_FIELDS_JS) or []
    except Exception as exc:
        print(f"[Forms] Could not inspect fields: {exc}", flush=True)
        return []


async def _apply_value(page, field: Dict[str, Any], value: str) -> bool:
    selector = f'[data-browzy-field="{field["idx"]}"]'
    locator = page.locator(selector).first
    try:
        if field.get("tag") == "select":
            options = [opt.get("text", "") for opt in field.get("options") or []]
            choice = pick_option(options, value)
            if not choice:
                return False
            await locator.select_option(label=choice, timeout=4000)
            return True

        await locator.scroll_into_view_if_needed(timeout=2500)
        await locator.fill(str(value), timeout=4000)

        # Chrome's own address autofill can append a character when a field is
        # blurred. Read the value back and force an exact match if needed.
        actual = await locator.input_value()
        if actual != str(value):
            await locator.evaluate(
                """(el, v) => {
                    el.value = v;
                    el.dispatchEvent(new Event('input', { bubbles: true }));
                    el.dispatchEvent(new Event('change', { bubbles: true }));
                }""",
                str(value),
            )
            actual = await locator.input_value()
            if actual != str(value):
                print(f"[Forms] value mismatch on {field.get('label')}: {actual!r}", flush=True)
                return False
        return True
    except Exception as exc:
        print(f"[Forms] fill failed for {field.get('label') or field.get('name')}: {exc}", flush=True)
        return False


async def autofill(
    page,
    profile: Optional[Dict[str, str]] = None,
    on_fill: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None,
    llm_map: Optional[Callable[[List[str], List[str]], Awaitable[Dict[str, str]]]] = None,
) -> Dict[str, Any]:
    """Match the page's visible fields against the profile and fill them."""
    profile = profile or load_profile()
    available = [k for k, v in profile.items() if str(v or "").strip()]
    fields = await collect_fields(page)

    filled: List[Dict[str, Any]] = []
    unmatched: List[Dict[str, Any]] = []
    already: List[str] = []
    blocked: List[str] = []

    for field in fields:
        if str(field.get("value") or "").strip():
            already.append(field.get("label") or field.get("name") or "")
            continue

        # Search boxes, coupon inputs, OTPs ... never touch these, and never
        # offer them to the model either.
        if is_blocked(field):
            blocked.append(
                field.get("label") or field.get("placeholder") or field.get("name") or ""
            )
            continue

        key = match_field_key(field, available)
        if not key:
            unmatched.append(field)
            continue

        value = str(profile.get(key) or "").strip()
        if not value or not await _apply_value(page, field, value):
            unmatched.append(field)
            continue

        record = {
            "key": key,
            "label": field.get("label") or field.get("name") or field.get("placeholder") or key,
            "value": value,
        }
        filled.append(record)
        if on_fill:
            await on_fill(record)

    # Tier 2: only ask the local model when the alias table barely understood the
    # page. On a normal checkout tier 1 fills everything in about a second, and a
    # 20-40s model call on CPU would be pure latency with nothing to show for it.
    if unmatched and llm_map and len(filled) < 2:
        labels = []
        for field in unmatched:
            if is_blocked(field):
                continue
            label = (field.get("label") or field.get("placeholder") or field.get("name") or "").strip()
            if label and label not in labels and _plausible_model_label(label):
                labels.append(label)
        if labels:
            mapping = await llm_map(labels, available)
            for field in list(unmatched):
                label = (field.get("label") or field.get("placeholder") or field.get("name") or "").strip()
                if not _plausible_model_label(label):
                    continue
                key = (mapping or {}).get(label)
                if not key or key not in available:
                    continue
                value = str(profile.get(key) or "").strip()
                if not value or not await _apply_value(page, field, value):
                    continue
                record = {"key": key, "label": label, "value": value}
                filled.append(record)
                unmatched.remove(field)
                if on_fill:
                    await on_fill(record)

    # Whatever is on the page after filling (typed by Browzy or by the user) is
    # remembered, so the next form of the same kind fills itself.
    learned = await learn_from_page(page, load_profile(), stable_only=False)

    return {
        "ok": True,
        "total": len(fields),
        "filled": len(filled),
        "fields": filled,
        "skipped": already,
        "blocked": [label for label in blocked if label],
        "unmatched": [
            (f.get("label") or f.get("placeholder") or f.get("name") or f"field {f.get('idx')}")
            for f in unmatched
        ],
        "learned": int(learned.get("learned") or 0),
        "url": page.url,
    }
