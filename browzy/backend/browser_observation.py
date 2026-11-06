"""Compact, structured observations of the current Playwright page."""

from __future__ import annotations

import hashlib
from typing import Any, Dict

_OBSERVE_JS = r"""
() => {
  const sel = 'a[href], button, input:not([type=hidden]), textarea, select, [role=button], [role=link], [role=menuitem], [role=tab], [contenteditable="true"]';
  const norm = (s) => (s || '').replace(/\s+/g, ' ').trim();
  const cut = (s, n) => norm(s).slice(0, n);
  const items = [];
  const seen = new Set();
  for (const el of document.querySelectorAll(sel)) {
    const rect = el.getBoundingClientRect();
    const style = getComputedStyle(el);
    if (rect.width < 2 || rect.height < 2) continue;
    if (style.visibility === 'hidden' || style.display === 'none' || style.opacity === '0') continue;
    const tag = el.tagName.toLowerCase();
    const type = (el.getAttribute('type') || '').toLowerCase();
    const name = cut(
      el.getAttribute('aria-label') ||
      el.getAttribute('placeholder') ||
      (['submit', 'button'].includes(type) ? el.value : '') ||
      el.innerText || el.textContent ||
      el.getAttribute('name') || el.id,
      80
    );
    if (!name) continue;
    const role = el.getAttribute('role') || (tag === 'a' ? 'link' : tag === 'button' ? 'button' : tag);
    const key = role + '|' + name;
    if (seen.has(key)) continue;
    seen.add(key);
    items.push({role, name, tag, type, id: el.id || ''});
  }
  const text = cut(document.body ? document.body.innerText : '', 4000);
  const video = document.querySelector('video');
  return {
    url: location.href,
    title: document.title,
    elements: items,
    text: text,
    media: video ? {
      present: true,
      paused: video.paused,
      current_time: video.currentTime,
      ended: video.ended
    } : null,
    signature: location.href + '|' + document.title + '|' +
      items.map(i => i.role + ':' + i.name).join('|') + '|' + text.slice(0, 600)
  };
}
"""

_FINGERPRINT_JS = r"""
() => {
  const els = document.querySelectorAll('a[href], button, input:not([type=hidden]), textarea, select, [role=button], [role=link], [role=menuitem], [role=tab]');
  const names = [];
  for (const el of Array.from(els).slice(0, 150)) {
    const rect = el.getBoundingClientRect();
    if (rect.width < 2 || rect.height < 2) continue;
    const label = (el.innerText || el.value || el.getAttribute('aria-label') || el.getAttribute('placeholder') || '').replace(/\s+/g, ' ').trim().slice(0, 40);
    names.push((el.getAttribute('role') || el.tagName) + ':' + label);
  }
  const text = document.body ? document.body.innerText.slice(0, 800) : '';
  return location.href + '|' + document.title + '|' + names.join('|') + '|' + text;
}
"""


async def observe_page(
    page, max_chars: int = 2500, max_elements: int = 50
) -> Dict[str, Any]:
    """Return a bounded snapshot of visible controls, text, and media state."""
    try:
        data = await page.evaluate(_OBSERVE_JS)
    except Exception as exc:
        return {"ok": False, "error": str(exc)[:200]}

    all_elements = data.get("elements") or []
    limit = max(0, min(int(max_elements or 0), 80))
    text_limit = max(200, min(int(max_chars or 0), 6000))
    signature = str(data.get("signature") or "")
    return {
        "ok": True,
        "url": data.get("url") or page.url,
        "title": data.get("title") or "",
        "elements": all_elements[:limit],
        "count": len(all_elements),
        "media": data.get("media"),
        "text": str(data.get("text") or "")[:text_limit],
        "fingerprint": hashlib.sha1(signature.encode("utf-8")).hexdigest()[:16],
    }


async def page_fingerprint(page) -> str:
    """Return a cheap page-state hash for stale-batch detection."""
    try:
        raw = await page.evaluate(_FINGERPRINT_JS)
    except Exception:
        return ""
    return hashlib.sha1(str(raw).encode("utf-8")).hexdigest()[:16]


def observation_payload(observation: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "url": observation.get("url"),
        "title": observation.get("title"),
        "fingerprint": observation.get("fingerprint"),
        "elements": observation.get("elements"),
        "media": observation.get("media"),
        "text": observation.get("text"),
    }
