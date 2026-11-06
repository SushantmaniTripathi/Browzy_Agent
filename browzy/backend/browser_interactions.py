"""Semantic Playwright interactions shared by deterministic browser routes."""

from __future__ import annotations

import re
from typing import Any, Dict


async def click_visible_control(page, target: str) -> Dict[str, Any]:
    """Click a visible control by its accessible or visible user-facing name."""
    cleaned = " ".join((target or "").split()).strip(" .,!?:;\"'“”‘’")
    if not cleaned:
        return {"ok": False, "error": "Tell me which control to click."}
    if len(cleaned) > 80:
        return {"ok": False, "error": f"\"{cleaned[:40]}…\" is not a usable control name."}

    patterns = [re.compile(rf"^{re.escape(cleaned)}$", re.IGNORECASE)]
    if len(cleaned) >= 3:
        patterns.append(re.compile(re.escape(cleaned), re.IGNORECASE))

    candidates = []
    for role in ("button", "link", "menuitem", "tab", "option"):
        for pattern in patterns:
            candidates.append(page.get_by_role(role, name=pattern))
    for pattern in patterns:
        candidates.append(page.get_by_text(pattern))
        candidates.append(page.get_by_label(pattern))

    for candidate in candidates:
        try:
            count = min(await candidate.count(), 8)
        except Exception:
            continue
        for index in range(count):
            locator = candidate.nth(index)
            try:
                if not await locator.is_visible() or not await locator.is_enabled():
                    continue
                await locator.scroll_into_view_if_needed()
                await locator.click(timeout=5000)
                return {
                    "ok": True,
                    "text": cleaned,
                    "url": page.url,
                    "title": await page.title(),
                    "page_state": (await page.evaluate(
                        "document.body ? document.body.innerText : ''"
                    ))[:1200],
                }
            except Exception:
                continue

    return {
        "ok": False,
        "error": f"I couldn't find a visible \u201c{cleaned}\u201d control on this page.",
    }
