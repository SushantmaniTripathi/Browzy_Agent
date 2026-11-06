"""Price comparison that never trusts the model with arithmetic.

The browser extracts raw cards (title, price text, link); Python parses the
numbers, removes duplicates and sorts ascending. The model is not involved, so
"1,29,999" can never be ranked as a bigger number than "89,999".
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional
from urllib.parse import quote_plus

PRICE_RE = re.compile(
    r"(?:₹|(?<![a-z])Rs\.?|(?<![a-z])INR|(?<![a-z])\$|(?<![a-z])USD|€|EUR|£|GBP|¥|JPY)"
    r"\s*\d[\d.,]*"
    r"|\d[\d.,]*\s*(?:₹|€|£|\$)",
    re.IGNORECASE,
)
NUMBER_RE = re.compile(r"\d[\d.,]*")

SEARCH_URLS = {
    "amazon": "https://www.amazon.com/s?k={q}",
    "amazon.in": "https://www.amazon.in/s?k={q}",
    "amazon.co.uk": "https://www.amazon.co.uk/s?k={q}",
    "flipkart": "https://www.flipkart.com/search?q={q}",
    "ebay": "https://www.ebay.com/sch/i.html?_nkw={q}",
    "walmart": "https://www.walmart.com/search?q={q}",
    "bestbuy": "https://www.bestbuy.com/site/searchpage.jsp?st={q}",
    "myntra": "https://www.myntra.com/{q}",
    "meesho": "https://www.meesho.com/search?q={q}",
    "aliexpress": "https://www.aliexpress.com/wholesale?SearchText={q}",
    "croma": "https://www.croma.com/searchB?q={q}",
    "reliance-digital": "https://www.reliancedigital.in/search?q={q}",
    "vijay-sales": "https://www.vijaysales.com/search/{q}",
    "apple": "https://www.apple.com/in/search/{q}",
    "samsung": "https://www.samsung.com/in/search/?searchvalue={q}",
    "oneplus": "https://www.oneplus.in/search?query={q}",
    "mi": "https://www.mi.com/in/search/{q}",
    "realme": "https://www.realme.com/in/search?keyword={q}",
}

DISPLAY = {
    "amazon": "Amazon",
    "amazon.in": "Amazon India",
    "amazon.co.uk": "Amazon UK",
    "flipkart": "Flipkart",
    "ebay": "eBay",
    "walmart": "Walmart",
    "bestbuy": "Best Buy",
    "myntra": "Myntra",
    "meesho": "Meesho",
    "aliexpress": "AliExpress",
    "croma": "Croma",
    "reliance-digital": "Reliance Digital",
    "vijay-sales": "Vijay Sales",
    "apple": "Apple India",
    "samsung": "Samsung India",
    "oneplus": "OnePlus India",
    "mi": "Mi India",
    "realme": "realme India",
}

INDIA_SOURCES = (
    "amazon.in", "flipkart", "croma", "reliance-digital", "vijay-sales",
    "ebay", "apple", "samsung", "oneplus", "mi", "realme",
)


def _to_number(token: str) -> Optional[float]:
    """Convert '1,29,999', '1.299,00' or '$1,299.00' into a float."""
    text = str(token or "").strip().rstrip(".,")
    if not text:
        return None

    has_dot = "." in text
    has_comma = "," in text

    if has_dot and has_comma:
        decimal = "." if text.rfind(".") > text.rfind(",") else ","
        thousands = "," if decimal == "." else "."
        text = text.replace(thousands, "").replace(decimal, ".")
    elif has_comma:
        parts = text.split(",")
        if len(parts) > 2:
            text = text.replace(",", "")            # 1,29,999
        elif len(parts[1]) == 3 and len(parts[0]) <= 3:
            text = text.replace(",", "")            # 1,299
        else:
            text = text.replace(",", ".")           # 1299,00
    elif has_dot:
        parts = text.split(".")
        if len(parts) > 2:
            text = text.replace(".", "")
        elif len(parts[1]) == 3 and len(parts[0]) <= 3:
            text = text.replace(".", "")            # 1.299 (EU thousands)
        # otherwise a plain decimal such as 1299.99

    try:
        value = float(text)
    except ValueError:
        return None
    return value if value > 0 else None


def parse_price(raw: Any) -> Optional[float]:
    """Extract the price from a scraped string.

    Cards often contain several numbers ("₹1,59,999 ₹99,999", "11% off",
    "₹1,499/month"). Discounts, percentages and EMI figures are filtered out and
    the lowest remaining price is used - that is the one the buyer pays.
    """
    if raw is None:
        return None
    text = str(raw)

    values: List[float] = []
    for match in PRICE_RE.finditer(text):
        # Look only at the text before the next price; a card reads
        # "₹69,999 ₹79,999 12% off", so the "%" belongs to the second number.
        before = text[max(0, match.start() - 24): match.start()].lower()
        if re.search(r"(emi|month|/mo|per month)\b", before):
            continue
        tail = text[match.end(): match.end() + 20]
        tail = re.split(r"[₹$€£¥]|Rs\.?|INR|USD|EUR|GBP", tail, maxsplit=1)[0].lower()
        if re.search(r"(off\b|%|month|/mo|emi|save)", tail):
            continue
        digits = NUMBER_RE.search(match.group(0))
        if not digits:
            continue
        value = _to_number(digits.group(0))
        if value is not None:
            values.append(value)

    if not values and not PRICE_RE.search(text):
        # No currency at all: only accept a plausible plain price. This keeps
        # rating counts ("19 product ratings") out of the results.
        for token in NUMBER_RE.findall(text):
            value = _to_number(token)
            if value is not None and value >= 100:
                values.append(value)

    return min(values) if values else None


def _clean_title(title: str) -> str:
    text = re.sub(r"\s+", " ", str(title or "")).strip()
    # "Sponsored Ad - ..." needs the prefix removed twice.
    for _ in range(3):
        cleaned = re.sub(
            r"^(sponsored|ad|advertisement)\b[:\s-]*", "", text, flags=re.IGNORECASE
        )
        if cleaned == text:
            break
        text = cleaned
    text = re.sub(r"\s+opens in a new window or tab\s*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+image\s+\d+\s+of\s+\d+\s*$", "", text, flags=re.IGNORECASE)
    return text[:180]


# Listings that are cheap because they are not the product the user asked for.
ACCESSORY_WORDS = (
    "case", "cover", "silicone", "screen protector", "tempered glass", "cable",
    "charger", "adapter", "strap", "pouch", "sleeve", "skin", "bumper", "holder",
    "stand", "mount", "refurbished", "renewed", "spare",
)

# Cheap because they are broken, used, or a part instead of the whole device.
CONDITION_WORDS = (
    "parts only", "for parts", "not working", "faulty", "damaged", "cracked",
    "compatible with", "replacement", "pre-owned", "preowned", "restored", "used",
    "open box", "display only", "read description", "read desc",
)


def _title_tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", str(text or "").lower())


GENERIC_QUERY_WORDS = {
    "phone", "phones", "mobile", "mobiles", "smartphone", "smartphones",
    "product", "products", "item", "items", "device", "devices",
}
MODEL_VARIANTS = {
    "pro", "max", "plus", "ultra", "mini", "se", "fe", "edge",
    "fold", "flip", "play", "turbo", "prime", "active",
}


def matches_query(title: str, query: str) -> bool:
    """Reject unrelated listings (a "Samsung" card in an iPhone search).

    Numbers must appear as standalone model numbers: "15" matches "iPhone 15"
    but not the "15.40 cm" display size inside an iPhone 17e listing.
    """
    tokens = [t for t in _title_tokens(query) if len(t) > 1]
    if not tokens:
        return True

    lowered = str(title or "").lower()
    words = [t for t in tokens if not t.isdigit()]
    specific_words = [word for word in words if word not in GENERIC_QUERY_WORDS]
    numbers = [t for t in tokens if t.isdigit()]

    def contains_word(word: str) -> bool:
        return word in lowered or (word.endswith("s") and word[:-1] in lowered)

    # Generic terms such as "phones" are useful category filters, but named
    # products/brands must all match; "iPhone" must never return a realme card.
    if specific_words and not all(contains_word(word) for word in specific_words):
        return False
    for num in numbers:
        if not re.search(rf"(?<![a-z0-9.]){re.escape(num)}(?![\d.])", lowered):
            return False
    has_model_identifier = bool(numbers) or any(
        re.search(r"[a-z]\d", word) for word in specific_words
    )
    if has_model_identifier:
        requested_variants = {word for word in specific_words if word in MODEL_VARIANTS}
        title_tokens = set(_title_tokens(title))
        title_variants = title_tokens & MODEL_VARIANTS
        # A numbered model request is exact: asking for "iPhone 15" must not
        # silently return 15 Pro/Plus/Pro Max, and "S24 Ultra" must stay Ultra.
        if title_variants != requested_variants:
            return False
    return True


def is_listing_usable(title: str, query: str) -> bool:
    """True unless the listing is a part or a broken/used unit the user didn't ask for."""
    lowered = str(title or "").lower()
    query_lower = (query or "").lower()

    if any(word in query_lower for word in ACCESSORY_WORDS):
        blocked = [w for w in CONDITION_WORDS if w not in query_lower]
    else:
        blocked = list(ACCESSORY_WORDS) + [
            w for w in CONDITION_WORDS if w not in query_lower
        ]
    return not any(word in lowered for word in blocked)


def extract_and_sort_products(
    cards: List[Dict[str, Any]], limit: int = 5, query: str = "",
    filters: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Normalise scraped cards, drop duplicates, sort numerically, return the cheapest."""
    items: List[Dict[str, Any]] = []
    seen_titles = set()
    seen_urls = set()
    filters = filters or {}

    for card in cards or []:
        title = _clean_title(card.get("title") or "")
        if len(title) < 4:
            continue
        if not matches_query(title, query):
            continue
        if not is_listing_usable(title, query):
            continue
        price = parse_price(card.get("raw_price"))
        if price is None:
            continue
        if filters.get("max_price") is not None and price > float(filters["max_price"]):
            continue
        if filters.get("min_price") is not None and price < float(filters["min_price"]):
            continue
        searchable = f"{title} {card.get('text') or ''}".lower()
        if filters.get("ram_gb") is not None and not re.search(
            rf"\b{int(filters['ram_gb'])}\s*gb\s*(?:ram|memory)?\b", searchable
        ):
            continue
        if filters.get("storage_gb") is not None and not re.search(
            rf"\b{int(filters['storage_gb'])}\s*gb\s*(?:storage|rom)?\b", searchable
        ):
            continue

        url = (card.get("url") or "").strip()
        if not re.match(r"^https?://", url, re.IGNORECASE):
            continue
        title_key = re.sub(r"[^a-z0-9]", "", title.lower())[:70]
        if not title_key:
            continue
        if url and url in seen_urls:
            continue
        if not url and title_key in seen_titles:
            continue
        seen_titles.add(title_key)
        if url:
            seen_urls.add(url)

        items.append(
            {
                "title": title,
                "price": price,
                "formatted_price": re.sub(r"\s+", " ", str(card.get("raw_price") or "")).strip(),
                "url": url,
                "source": card.get("source") or "",
            }
        )

    items.sort(key=lambda item: item["price"])
    return items[: max(1, limit)]


SCRAPE_CARDS_JS = r"""
() => {
    const PRICE = /(?:₹|(?<![a-z])Rs\.?|(?<![a-z])INR|(?<![a-z])\$|(?<![a-z])USD|€|EUR|£|GBP|¥|JPY)\s*\d[\d.,]*|\d[\d.,]*\s*(?:₹|€|£|\$)/i;
    const clean = (s) => (s || '').replace(/\s+/g, ' ').trim();
    const RATING_ONLY = /^(?:\d(?:\.\d)?\s*out of 5 stars\.?\s*)?(?:\(?\d[\d,.]*\)?\s*product ratings?\s*[-–]?\s*)+$/i;

    const priceIn = (el) => {
        const nodes = el.querySelectorAll(
            '[class*="price" i], [data-testid*="price" i], .a-price .a-offscreen, .s-item__price'
        );
        for (const node of nodes) {
            const value = clean(node.innerText || node.textContent || '');
            const found = value.match(PRICE);
            if (found) return clean(found[0]);
        }
        return '';
    };

    const containers = new Set();
    const selectors = [
        'div[data-component-type="s-search-result"]',
        'li.s-item',
        'div[data-id]',
        'div[class*="product" i]',
        'div[class*="card" i]',
        'li',
        'article'
    ];

    for (const selector of selectors) {
        let nodes;
        try { nodes = document.querySelectorAll(selector); } catch (e) { continue; }
        for (const el of nodes) {
            if (containers.size > 600) break;
            const text = el.innerText || '';
            if (text.length < 24 || text.length > 1400) continue;
            if (!PRICE.test(text) && !priceIn(el)) continue;
            if (!el.querySelector('a[href]')) continue;
            containers.add(el);
        }
    }

    const out = [];
    for (const el of containers) {
        const text = el.innerText || '';

        const link = el.querySelector('a[href]');
        const href = link ? link.href : '';

        let title = '';
        const titleEl = el.querySelector(
            'h1, h2, h3, h4, [class*="title" i], [class*="name" i], img[alt]'
        );
        if (titleEl) title = titleEl.getAttribute('alt') || titleEl.innerText || '';
        if (!title && link) title = link.getAttribute('title') || link.innerText || '';
        title = clean(title);
        if (title.length < 5) title = clean(text.split('\n')[0]);
        // eBay prefixes titles with "5.0 out of 5 stars. 19 product ratings -".
        title = clean(title.replace(/^\d(?:\.\d)?\s*out of 5 stars\.?\s*(?:\(?\d[\d,.]*\)?\s*product ratings?\s*[-–]?\s*)?/i, ''));
        if (title.length < 5 || RATING_ONLY.test(title)) continue;

        let rawPrice = priceIn(el);
        if (!rawPrice) {
            const match = text.match(PRICE);
            rawPrice = match ? match[0] : '';
        }
        if (!rawPrice) continue;

        out.push({
            title: title.slice(0, 180),
            raw_price: clean(rawPrice),
            url: href,
            text: clean(text).slice(0, 400)
        });
    }
    return out.slice(0, 150);
}
"""


def search_url(platform: str, query: str) -> str:
    site = (platform or "").strip().lower()
    if not site:
        # An unnamed marketplace means "compare across stores", not Amazon.
        return "https://www.google.com/search?tbm=shop&q=" + quote_plus(query)
    template = SEARCH_URLS.get(site)
    if template:
        return template.format(q=quote_plus(query))
    if re.fullmatch(r"[a-z0-9.-]+\.[a-z]{2,}", site):
        return f"https://{site}/search?q={quote_plus(query)}"
    return "https://www.google.com/search?tbm=shop&q=" + quote_plus(
        f"{site} {query}"
    )


def display_name(platform: str) -> str:
    site = (platform or "").strip().lower()
    if not site:
        return "Google Shopping"
    return DISPLAY.get(site, site)


async def scrape_products(page) -> List[Dict[str, Any]]:
    """Scroll a results page a little (lazy loading) and read the product cards."""
    for _ in range(3):
        try:
            await page.mouse.wheel(0, 1500)
        except Exception:
            pass
        await page.wait_for_timeout(900)
    try:
        return await page.evaluate(SCRAPE_CARDS_JS) or []
    except Exception as exc:
        print(f"[Prices] scrape failed: {exc}", flush=True)
        return []


def format_results(
    items: List[Dict[str, Any]], query: str, site: str, scanned: int, url: str
) -> str:
    label = display_name(site)
    results_url = url
    if not items:
        return (
            f"I opened {label} for \u201c{query}\u201d but couldn't read any prices "
            f"(the site may have shown a captcha or a different layout).\n\n{url}"
        )

    lines = [
        f"Cheapest {len(items)} for \u201c{query}\u201d on {label} "
        f"(sorted by price, {scanned} listings scanned):"
    ]
    for index, item in enumerate(items, start=1):
        price = item.get("formatted_price") or f"{item['price']:.2f}"
        url = str(item.get("url") or "").strip()
        source = f" ({item['source']})" if item.get("source") else ""
        if re.match(r"^https?://", url, re.IGNORECASE):
            lines.append(f"{index}. [{item['title']}{source}]({url}) \u2014 {price}")
        else:
            lines.append(f"{index}. {item['title']}{source} \u2014 {price}")
    lines.append("")
    lines.append("Prices come straight from the page; tax and shipping may differ.")
    lines.append(results_url)
    return "\n".join(lines)
