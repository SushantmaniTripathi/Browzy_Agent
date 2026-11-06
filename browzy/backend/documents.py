"""Read a document and turn it into profile fields.

Browzy never keeps the uploaded file. It extracts the text locally (PDF text
layer, DOCX paragraphs, or OCR for scans), pulls out the fields the user cares
about, stores those values, and deletes the upload.

Extraction is deterministic first (labels and well-known formats such as email,
phone, PIN, PAN, Aadhaar), then the local model fills gaps for unusual layouts.
The model never overrides a value that a regex already found.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from . import forms

MAX_BYTES = 20 * 1024 * 1024
MAX_OCR_PAGES = 8
MAX_MODEL_CHARS = 7000

TEXT_SUFFIXES = {".txt", ".md", ".csv", ".json"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
SUPPORTED = {".pdf", ".docx", ".doc"} | TEXT_SUFFIXES | IMAGE_SUFFIXES

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(?<![\d])(?:\+?\d{1,3}[\s.\-]?)?(?:\(?\d{3,5}\)?[\s.\-]?)?\d{3,5}[\s.\-]?\d{4}(?![\d])")
# Unlabelled numbers are only trusted when they look like a real Indian mobile.
MOBILE_RE = re.compile(r"(?<!\d)(?:\+?91[\s.\-]?)?[6-9]\d{4}[\s.\-]?\d{5}(?!\d)")
PIN_RE = re.compile(r"\b(?:pin|zip|postal)\s*(?:code)?\s*[:\-]?\s*(\d{5,6})\b", re.IGNORECASE)
DOB_RE = re.compile(
    r"\b(?:dob|date\s+of\s+birth|born)\s*[:\-]?\s*"
    r"(\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|\d{4}[./-]\d{1,2}[./-]\d{1,2}|"
    r"\d{1,2}\s+[A-Za-z]{3,9}\s+\d{2,4}|[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{2,4})",
    re.IGNORECASE,
)
PAN_RE = re.compile(r"\b([A-Z]{5}\d{4}[A-Z])\b")
AADHAAR_RE = re.compile(r"\b(\d{4}\s?\d{4}\s?\d{4})\b")
PASSPORT_RE = re.compile(r"\b([A-Z]\d{7})\b")
URL_RE = re.compile(r"(?:https?://|www\.)[^\s,;]+", re.IGNORECASE)
LINKEDIN_RE = re.compile(r"(?:https?://)?(?:www\.)?linkedin\.com/[^\s,;]+", re.IGNORECASE)
# Resumes often print just "linkedin/handle" or "github/handle".
LINKEDIN_SHORT_RE = re.compile(r"\blinkedin\s*/\s*([A-Za-z0-9._-]{3,40})", re.IGNORECASE)
GITHUB_SHORT_RE = re.compile(r"\bgithub\s*/\s*([A-Za-z0-9._-]{3,40})", re.IGNORECASE)

# Words that show up in headings, labels and table columns, never in a name.
# Without this "SEAT NO." and "HIGHER SECONDARY EDUCATION" become the full name.
NAME_STOPWORDS = {
    "seat", "roll", "reg", "registration", "index", "sr", "serial", "no", "number",
    "date", "exam", "examination", "board", "school", "college", "institute",
    "university", "education", "higher", "secondary", "senior", "marks", "subject",
    "father", "mother", "parent", "guardian", "signature", "photo", "centre",
    "center", "district", "division", "code", "session", "year", "month", "admit",
    "card", "hall", "ticket", "certificate", "statement", "result", "total",
    "grade", "percentage", "blood", "group", "address", "phone", "mobile", "email",
    "pin", "zip", "state", "city", "name", "full", "first", "last", "middle",
    "surname", "of", "the", "and", "for", "to", "from",
}
# Place and company names may legitimately contain connectives ("Jammu and
# Kashmir", "Head of Sales"), so they are not treated as document labels.
PLACE_STOPWORDS = NAME_STOPWORDS - {"and", "of", "the", "for", "to", "from"}
ADDRESS_HINT_RE = re.compile(
    r"\b(?:street|st\.|road|rd\.|avenue|ave\.|lane|ln\.|drive|sector|block|flat|"
    r"apartment|apt\.|house|plot|nagar|colony|layout|floor|building|bldg)\b",
    re.IGNORECASE,
)

# Document label -> canonical key, in addition to the form aliases.
DOC_LABELS: Dict[str, str] = {
    "name": "full_name",
    "full name": "full_name",
    "candidate name": "full_name",
    "applicant name": "full_name",
    "father's name": "",
    "mother's name": "",
    "date of birth": "dob",
    "dob": "dob",
    "birth date": "dob",
    "gender": "gender",
    "sex": "gender",
    "nationality": "nationality",
    "citizenship": "nationality",
    "email": "email",
    "e-mail": "email",
    "email address": "email",
    "mail": "email",
    "phone": "phone",
    "mobile": "phone",
    "mobile number": "phone",
    "phone number": "phone",
    "contact number": "phone",
    "contact": "phone",
    "alternate phone": "alt_phone",
    "address": "street_address",
    "current address": "street_address",
    "permanent address": "street_address",
    "residential address": "street_address",
    "address line 1": "street_address",
    "address line 2": "address_line2",
    "city": "city",
    "town": "city",
    "state": "state",
    "province": "state",
    "pincode": "pincode",
    "pin code": "pincode",
    "pin": "pincode",
    "zip": "pincode",
    "zip code": "pincode",
    "postal code": "pincode",
    "country": "country",
    "company": "company",
    "organisation": "company",
    "organization": "company",
    "employer": "company",
    "designation": "job_title",
    "job title": "job_title",
    "position": "job_title",
    "role": "job_title",
    "occupation": "job_title",
    "website": "website",
    "portfolio": "website",
    "linkedin": "website",
    "pan": "pan",
    "pan number": "pan",
    "pan no": "pan",
    "aadhaar": "aadhaar",
    "aadhar": "aadhaar",
    "aadhaar number": "aadhaar",
    "passport": "passport",
    "passport number": "passport",
    "passport no": "passport",
    "driving licence": "license",
    "driving license": "license",
    "licence number": "license",
    "license number": "license",
}


def _ocr_image(path: Path) -> str:
    """OCR with RapidOCR (bundled ONNX models, works offline)."""
    try:
        from rapidocr_onnxruntime import RapidOCR
    except Exception as exc:
        raise RuntimeError(
            "OCR is unavailable (rapidocr_onnxruntime is not installed)."
        ) from exc

    engine = RapidOCR()
    result, _ = engine(str(path))
    if not result:
        return ""
    lines = []
    for row in result:
        if len(row) >= 2 and str(row[1]).strip():
            lines.append(str(row[1]).strip())
    return "\n".join(lines)


def _pdf_text(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except Exception:
        return ""

    try:
        reader = PdfReader(str(path))
        pages = []
        for page in reader.pages[:30]:
            try:
                pages.append(page.extract_text() or "")
            except Exception:
                pages.append("")
        return "\n".join(pages).strip()
    except Exception as exc:
        print(f"[Docs] PDF text layer failed: {exc}", flush=True)
        return ""


def _pdf_ocr(path: Path) -> str:
    """Render the first pages of a scanned PDF and OCR them."""
    try:
        import pypdfium2 as pdfium
    except Exception as exc:
        raise RuntimeError("Cannot render scanned PDF pages (pypdfium2 missing).") from exc

    doc = pdfium.PdfDocument(str(path))
    chunks: List[str] = []
    try:
        for index in range(min(len(doc), MAX_OCR_PAGES)):
            page = doc[index]
            bitmap = page.render(scale=2.2)
            image = bitmap.to_pil()
            with tempfile.TemporaryDirectory() as tmp:
                target = Path(tmp) / f"page-{index}.png"
                image.save(target)
                chunks.append(_ocr_image(target))
    finally:
        try:
            doc.close()
        except Exception:
            pass
    return "\n".join(chunk for chunk in chunks if chunk).strip()


def _docx_text(path: Path) -> str:
    try:
        from docx import Document
    except Exception as exc:
        raise RuntimeError("DOCX support is unavailable (python-docx missing).") from exc

    document = Document(str(path))
    chunks = [p.text for p in document.paragraphs if p.text and p.text.strip()]
    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if cells:
                chunks.append(" | ".join(cells))
    return "\n".join(chunks).strip()


def extract_text(path: Path) -> Dict[str, Any]:
    """Return {ok, text, kind, note} for a supported document."""
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED:
        return {
            "ok": False,
            "text": "",
            "kind": suffix.lstrip("."),
            "error": f"Unsupported file type '{suffix}'. Use PDF, DOCX, TXT or an image scan.",
        }

    if path.stat().st_size > MAX_BYTES:
        return {
            "ok": False, "text": "", "kind": suffix.lstrip("."),
            "error": "That file is larger than 20 MB.",
        }

    try:
        if suffix == ".pdf":
            text = _pdf_text(path)
            kind = "pdf"
            if len(text) < 40:
                text = _pdf_ocr(path)
                kind = "scanned pdf"
        elif suffix == ".docx":
            text = _docx_text(path)
            kind = "docx"
        elif suffix == ".doc":
            return {
                "ok": False, "text": "", "kind": "doc",
                "error": "Old .doc files are not readable. Save it as .docx or PDF and upload again.",
            }
        elif suffix in IMAGE_SUFFIXES:
            text = _ocr_image(path)
            kind = "scan"
        else:
            text = path.read_text(encoding="utf-8", errors="ignore")
            kind = "text"
    except Exception as exc:
        return {
            "ok": False, "text": "", "kind": suffix.lstrip("."),
            "error": f"Could not read that document: {exc}",
        }

    text = re.sub(r"[ \t]+", " ", text or "")
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) < 15:
        return {
            "ok": False, "text": text, "kind": kind,
            "error": "I could not read any text from that document. Try a clearer scan or a text-based PDF.",
        }
    return {"ok": True, "text": text, "kind": kind, "error": ""}


def _clean_value(value: str) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    text = text.strip(" |,;:")
    return text[:160]


# Labels that contain a field word but describe somebody else or an institution.
IGNORE_LABEL_WORDS = (
    "father", "mother", "guardian", "parent", "spouse", "school", "college",
    "institute", "centre", "center",
)

# Resume section headings often contain field words ("POSITION of responsibility",
# "PROFESSIONAL summary"). They are headings, not labels for the next line.
SECTION_WORDS = {
    "achievements", "achievement", "awards", "award", "experience", "education",
    "projects", "project", "skills", "skill", "summary", "certifications",
    "certification", "interests", "languages", "references", "publications",
    "hobbies", "declaration", "profile", "objective", "training", "workshops",
    "activities", "positions", "responsibility", "responsibilities",
    "accomplishments", "coursework", "volunteer", "leadership",
}


def _key_for_label(label: str) -> Optional[str]:
    """Map a document label to a profile key, tolerating bilingual headers.

    Marksheets often print ``उमेदवाराचे ... / CANDIDATE'S FULL NAME`` on one line,
    so the English half is checked too, and a known label may appear inside a
    longer phrase ("CANDIDATE'S FULL NAME (SURNAME FIRST)").
    """
    candidates = [label]
    if "/" in label:
        candidates.append(label.rsplit("/", 1)[-1])
    for candidate in candidates:
        normalized = forms.normalize_string(candidate)
        if not normalized:
            continue
        if any(word in normalized for word in IGNORE_LABEL_WORDS):
            continue
        if any(word in SECTION_WORDS for word in re.findall(r"[a-z]+", candidate.lower())):
            continue

        for known, key in DOC_LABELS.items():
            if forms.normalize_string(known) == normalized:
                return key or None
        for key in forms.SCHEMA_KEYS:
            for alias in forms.SYNONYM_MAP.get(key, []):
                if forms.normalize_string(alias) == normalized:
                    return key

        # Fuzzy: whole words only, and never on a label that carries numbers or
        # looks like a sentence. Substring matching used to turn
        # "Regional Finalist (Top 3)" into a state because it contains "region".
        words = re.findall(r"[a-z]+", candidate.lower())
        if not words or any(ch.isdigit() for ch in candidate):
            continue
        best_key, best_len = None, 0
        for key in forms.SCHEMA_KEYS:
            aliases = list(forms.SYNONYM_MAP.get(key, [])) + [key]
            for alias in aliases:
                probe_words = re.findall(r"[a-z]+", alias.lower())
                if not probe_words or len("".join(probe_words)) < 6:
                    continue
                span = len(probe_words)
                found = any(
                    words[i:i + span] == probe_words
                    for i in range(0, len(words) - span + 1)
                )
                if found and span > best_len:
                    best_key, best_len = key, span
        if best_key:
            return best_key
    return None


def _valid_phone(raw: str) -> bool:
    digits = re.sub(r"\D", "", str(raw or ""))
    if not 10 <= len(digits) <= 13:
        return False
    # Numbers such as "002275 1012" are seat/roll numbers, not phones.
    return not digits.startswith("00")


PLACE_CONNECTORS = {"and", "of", "the", "&", "in"}


def _looks_like_place(value: str) -> bool:
    """A city/state is a short proper noun, never a sentence or a headline."""
    text = str(value or "").strip()
    if not text or len(text) > 48 or any(ch.isdigit() for ch in text):
        return False
    words = text.replace("&", " and ").split()
    if not 1 <= len(words) <= 4:
        return False
    for word in words:
        token = word.strip(".,'")
        if not token:
            return False
        if token.lower() in PLACE_CONNECTORS:
            continue
        if not token[0].isupper():
            return False
        if not re.fullmatch(r"[A-Za-z.'\-]+", token):
            return False
    return True


def _plausible_value(key: str, value: str) -> bool:
    """Reject values that are really another label, a heading or a serial number."""
    value = str(value or "").strip()
    if not value:
        return False
    if key in ("full_name", "first_name", "middle_name", "last_name"):
        if _looks_like_name(value):
            return True
        # A labelled single-word name (mononym) is still a real name.
        single = value.strip()
        return bool(re.fullmatch(r"[A-Za-z][A-Za-z.'\-]{1,40}", single)) and single.lower() not in NAME_STOPWORDS
    if key == "email":
        return bool(EMAIL_RE.search(value))
    if key in ("phone", "alt_phone"):
        return _valid_phone(value)
    if key == "pincode":
        return bool(re.fullmatch(r"\d{5,6}", value))
    if key == "dob":
        return bool(re.search(r"\d", value))
    if key in ("city", "state", "country"):
        if not _looks_like_place(value):
            return False
        tokens = [t for t in re.split(r"[^a-z]+", value.lower()) if t]
        return bool(tokens) and not any(token in PLACE_STOPWORDS for token in tokens)
    if key in ("company", "job_title"):
        tokens = [t for t in re.split(r"[^a-z]+", value.lower()) if t]
        if not tokens:
            return False
        return not any(token in PLACE_STOPWORDS for token in tokens)
    return len(value) >= 2


def _label_value_pairs(text: str) -> Dict[str, str]:
    """Read 'Label: value', 'Label - value' and 'Label   value' lines."""
    found: Dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or len(line) > 200:
            continue
        match = re.match(r"^([A-Za-z][A-Za-z0-9 .'/&()\-]{1,40}?)\s*[:\-–]\s*(.+)$", line)
        if match:
            label, value = match.group(1).strip(), match.group(2).strip()
        else:
            # Cards often align the label and value with a wide gap or a tab.
            match = re.match(r"^([A-Za-z][A-Za-z0-9 .'/&()]{1,40}?)\s{2,}(.+)$", line)
            if not match:
                continue
            label, value = match.group(1).strip(), match.group(2).strip()
        if len(value) < 2:
            continue
        key = _key_for_label(label)
        if key and key not in found and _plausible_value(key, value):
            found[key] = _clean_value(value)
    return found


def _label_next_line_pairs(text: str) -> Dict[str, str]:
    """Read labels whose value sits on the following line (common in cards)."""
    lines = [line.strip() for line in text.splitlines()]
    found: Dict[str, str] = {}
    for index, line in enumerate(lines[:-1]):
        if not line or any(ch.isdigit() for ch in line):
            continue
        label_text = line.rsplit("/", 1)[-1].strip() if "/" in line else line
        if not label_text or len(label_text) > 60:
            continue
        key = _key_for_label(label_text.rstrip(".:"))
        if not key or key in found:
            continue
        nxt = lines[index + 1].strip()
        if not nxt or len(nxt) > 120:
            continue
        if _key_for_label(nxt.rstrip(".:")):
            continue
        if _plausible_value(key, nxt):
            found[key] = _clean_value(nxt)
    return found


def _first(patterns: List[re.Pattern], text: str) -> str:
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            value = match.group(1) if match.groups() else match.group(0)
            if value:
                return _clean_value(value)
    return ""


def _phone(text: str) -> str:
    for match in MOBILE_RE.finditer(text):
        raw = match.group(0)
        if _valid_phone(raw):
            return _clean_value(raw)
    return ""


def _looks_like_name(line: str) -> bool:
    line = line.strip().strip(".,;:")
    if not line or len(line) > 60:
        return False
    if any(ch.isdigit() for ch in line) or "@" in line or ":" in line:
        return False
    words = line.split()
    if not 2 <= len(words) <= 4:
        return False
    tokens = [word.strip(".,'\"-") for word in words]
    if not all(re.fullmatch(r"[A-Za-z][A-Za-z.'\-]*", token) for token in tokens):
        return False
    if any(token.lower() in NAME_STOPWORDS for token in tokens):
        return False
    if not all(token[0].isupper() for token in tokens):
        return False
    # Require at least two words of two or more letters (rejects "K. Rao" noise
    # and initials-only lines).
    return sum(1 for token in tokens if len(token) >= 2) >= 2


def _split_name(full_name: str) -> Dict[str, str]:
    """Split a western-order name: first, optional middle, last."""
    parts = [part for part in full_name.split() if part]
    if not parts:
        return {}
    if len(parts) == 1:
        return {"first_name": parts[0]}
    if len(parts) == 2:
        return {"first_name": parts[0], "last_name": parts[1]}
    return {
        "first_name": parts[0],
        "middle_name": " ".join(parts[1:-1]),
        "last_name": parts[-1],
    }


def _name_from_text(text: str) -> str:
    """Look for a person's name, skipping headings and relatives' names."""
    lines = text.splitlines()
    skip_next = False
    for raw_line in lines[:25]:
        line = raw_line.strip()
        lowered = line.lower()
        if skip_next:
            skip_next = False
            if _looks_like_name(line):
                continue
        if re.search(r"\b(father|mother|parent|guardian|s/o|d/o|w/o)\b", lowered):
            skip_next = True
            continue
        if _looks_like_name(line):
            return _clean_value(line)
    return ""


ROLE_SUFFIXES = (
    "engineer", "developer", "designer", "analyst", "intern", "manager",
    "consultant", "scientist", "architect", "specialist", "associate",
    "administrator", "executive", "director", "programmer", "researcher",
)


def _role_from_text(text: str) -> str:
    """Find a job title line such as "AI/ML Engineer Intern" near the top."""
    for raw_line in text.splitlines()[:60]:
        line = raw_line.strip().strip("|,;")
        if not line or len(line) > 48 or any(ch.isdigit() for ch in line):
            continue
        words = line.replace("/", " ").replace("&", " ").split()
        if not 2 <= len(words) <= 5:
            continue
        if any(not word[0].isupper() and word.lower() not in ("of", "and") for word in words):
            continue
        if words[-1].strip(".,").lower() not in ROLE_SUFFIXES:
            continue
        return _clean_value(line)
    return ""


def extract_fields(text: str) -> Dict[str, str]:
    """Deterministic extraction: labels first, formats and heuristics second."""
    fields: Dict[str, str] = {}

    for key, value in _label_value_pairs(text).items():
        fields[key] = value
    for key, value in _label_next_line_pairs(text).items():
        fields.setdefault(key, value)

    email = _first([EMAIL_RE], text)
    if email:
        fields.setdefault("email", email.lower())

    phone = _phone(text)
    if phone:
        fields.setdefault("phone", phone)

    dob = _first([DOB_RE], text)
    if dob:
        fields.setdefault("dob", dob)

    pan = _first([PAN_RE], text)
    if pan:
        fields.setdefault("pan", pan)

    # Only read an Aadhaar/passport number when the document actually names it,
    # otherwise any 12-digit or A1234567-looking string would become an ID.
    if not fields.get("aadhaar") and re.search(r"\b(?:aadhaar|aadhar|uidai)\b", text, re.IGNORECASE):
        aadhaar = _first([AADHAAR_RE], text)
        if aadhaar:
            fields["aadhaar"] = aadhaar

    if not fields.get("passport") and re.search(r"\bpassport\b", text, re.IGNORECASE):
        passport = _first([PASSPORT_RE], text)
        if passport:
            fields["passport"] = passport

    linkedin = LINKEDIN_RE.search(text)
    if linkedin:
        fields.setdefault("website", linkedin.group(0).rstrip("/"))
    else:
        short = LINKEDIN_SHORT_RE.search(text)
        if short:
            fields.setdefault("website", f"https://www.linkedin.com/in/{short.group(1).strip('.')}")
        else:
            github = GITHUB_SHORT_RE.search(text)
            if github:
                fields.setdefault("website", f"https://github.com/{github.group(1).strip('.')}")
            else:
                url = URL_RE.search(text)
                if url:
                    fields.setdefault("website", url.group(0).rstrip(").,"))

    pincode = _first([PIN_RE], text)
    if pincode:
        fields.setdefault("pincode", pincode)

    if not fields.get("full_name"):
        name = _name_from_text(text)
        if name:
            fields["full_name"] = name

    if fields.get("full_name"):
        if not fields.get("first_name") or not fields.get("last_name"):
            parts = fields["full_name"].split()
            if len(parts) >= 2 and re.search(r"surname\s+first", text, re.IGNORECASE):
                # Indian marksheets print "SURNAME FIRST": Tripathi Sushantmani Rajeshmani
                # means last=Tripathi, first=Sushantmani, middle=Rajeshmani.
                fields.setdefault("last_name", parts[0])
                fields.setdefault("first_name", parts[1])
                if len(parts) >= 3:
                    fields.setdefault("middle_name", " ".join(parts[2:]))
            else:
                for key, value in _split_name(fields["full_name"]).items():
                    fields.setdefault(key, value)

    if not fields.get("job_title"):
        role = _role_from_text(text)
        if role:
            fields["job_title"] = role

    if not fields.get("street_address"):
        for line in text.splitlines():
            if ADDRESS_HINT_RE.search(line) and re.search(r"\d", line) and 12 < len(line) < 160:
                fields["street_address"] = _clean_value(line)
                break

    # City/state usually live at the end of the address: "..., Bengaluru, Karnataka 560025".
    if fields.get("street_address"):
        parts = [p.strip() for p in fields["street_address"].split(",") if p.strip()]
        if len(parts) >= 3:
            state = re.sub(r"\b\d{5,6}\b", "", parts[-1]).strip(" -")
            fields.setdefault("city", _clean_value(parts[-2]))
            if state:
                fields.setdefault("state", _clean_value(state))
        if not fields.get("pincode"):
            tail = re.search(r"\b(\d{5,6})\b", parts[-1] if parts else "")
            if tail:
                fields["pincode"] = tail.group(1)

    if not fields.get("country") and re.search(r"\bindia\b", text, re.IGNORECASE):
        fields["country"] = "India"

    return {key: value for key, value in fields.items() if value}


def build_model_prompt(text: str) -> List[Dict[str, str]]:
    keys = ", ".join(forms.SCHEMA_KEYS)
    system = (
        "You extract personal details from a document into JSON. Use only facts "
        "present in the document. Reply with a single JSON object; omit keys you "
        "cannot find. Never invent values."
    )
    user = (
        f"Allowed keys: {keys}\n"
        f"Document text:\n\"\"\"\n{text[:MAX_MODEL_CHARS]}\n\"\"\"\n"
        "Return JSON only."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


async def extract_fields_with_model(
    text: str, llm: Callable[[List[Dict[str, str]]], Any]
) -> Dict[str, str]:
    """Ask the local model for any fields the deterministic pass missed."""
    try:
        raw = await asyncio.wait_for(llm(build_model_prompt(text)), timeout=120)
    except Exception as exc:
        print(f"[Docs] model extraction failed: {exc}", flush=True)
        return {}

    content = ""
    if isinstance(raw, dict):
        content = str(raw.get("message", {}).get("content") or raw.get("content") or "")
    else:
        content = str(raw or "")
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if not match:
        return {}

    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}

    clean: Dict[str, str] = {}
    haystack = re.sub(r"[^a-z0-9]", "", text.lower())
    for key, value in data.items():
        key = str(key).strip()
        if key not in forms.SCHEMA_KEYS or isinstance(value, (dict, list)):
            continue
        text_value = _clean_value(str(value))
        if not text_value or text_value.lower() in ("null", "none", "n/a", "unknown"):
            continue
        # Only accept values that are actually printed in the document and that
        # pass the same sanity checks as the deterministic pass. A local model
        # will happily turn a seat number into a phone or a board name into a city.
        probe = re.sub(r"[^a-z0-9]", "", text_value.lower())
        if len(probe) < 2 or probe not in haystack:
            continue
        if not _plausible_value(key, text_value):
            continue
        clean[key] = text_value
    return clean


def _fill_name_parts(fields: Dict[str, str]) -> None:
    """Complete first/middle/last from the full name without reordering it.

    Order is unknown at this point (marksheets print "surname first"), so parts
    already present are removed from the full name and whatever is left becomes
    the missing middle/last part.
    """
    full = str(fields.get("full_name") or "").strip()
    parts = [part for part in full.split() if part]
    if not parts:
        return
    first, middle, last = fields.get("first_name"), fields.get("middle_name"), fields.get("last_name")
    if not first and not middle and not last:
        for key, value in _split_name(full).items():
            fields[key] = value
        return
    remaining = list(parts)
    for known in (first, last):
        if known:
            for token in str(known).split():
                if token in remaining:
                    remaining.remove(token)
    if middle or not remaining:
        return
    if last:
        fields["middle_name"] = " ".join(remaining)
    else:
        fields["last_name"] = remaining[-1]
        if len(remaining) > 1:
            fields["middle_name"] = " ".join(remaining[:-1])


def merge_fields(base: Dict[str, str], extra: Dict[str, str]) -> Dict[str, str]:
    """Deterministic values win; the model only fills gaps."""
    merged = dict(base)
    for key, value in extra.items():
        if not merged.get(key):
            merged[key] = value
    if merged.get("full_name"):
        _fill_name_parts(merged)
    return merged


def save_upload(data: bytes, filename: str) -> Path:
    """Write the upload to a temp file so it can be parsed, then delete it."""
    suffix = Path(filename or "upload").suffix.lower()
    handle, name = tempfile.mkstemp(prefix="browzy-upload-", suffix=suffix)
    with os.fdopen(handle, "wb") as fh:
        fh.write(data)
    return Path(name)


def cleanup(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except Exception:
        pass
