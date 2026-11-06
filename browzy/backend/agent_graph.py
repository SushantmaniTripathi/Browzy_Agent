"""LangGraph orchestration for Browzy.

The graph deliberately separates *thinking* from *doing*:

    think ──► open ──► search ──► select ──► play ──► verify
      │                         ▲                     │
      └──► agent (fallback)     └──── retry ◄─────────┘

* ``think`` turns an open-ended instruction into a tiny JSON plan
  ``{platform, query, action}``.  A deterministic parser answers first
  (instant, no model); the local model is only consulted for instructions the
  parser does not recognise.  That is the "command parser" step, and it is the
  only place the model is used for planning.
* The action nodes never see the raw instruction, only the clean plan.  They
  drive the visible browser: open the platform, type the *query* into the real
  search box, pick the result that actually matches, click it, verify playback.

Keeping these apart is what makes a small 3–4B model behave predictably: it
only has to emit three fields, and every follow-up step is deterministic.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, START, StateGraph
from .task_runtime import MAX_SPECIALIZED_RETRIES

PLAY_ACTIONS = {"play", "play_first_video", "play_video"}
DISPLAY_NAMES = {
    "youtube": "YouTube",
    "yt": "YouTube",
    "google": "Google",
    "google_flights": "Google Flights",
    "skyscanner": "Skyscanner",
    "google_hotels": "Google Travel",
    "google_travel": "Google Travel",
    "duckduckgo": "DuckDuckGo",
    "ddg": "DuckDuckGo",
    "github": "GitHub",
    "wikipedia": "Wikipedia",
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
    "spotify": "Spotify",
    "reddit": "Reddit",
    "gmail": "Gmail",
    "croma": "Croma",
    "reliance-digital": "Reliance Digital",
    "vijay-sales": "Vijay Sales",
    "apple": "Apple India",
    "samsung": "Samsung India",
    "oneplus": "OnePlus India",
    "mi": "Mi India",
    "realme": "realme India",
}


class BrowzyState(TypedDict, total=False):
    """Shared state passed between graph nodes."""

    instruction: str
    session_id: str
    model: str
    thinking_model: bool
    ws: Any

    plan: Dict[str, Any]
    plan_source: str          # "rules" | "model"
    route: str                # "open" | "search" | "play" | "agent"
    home_url: str
    results_url: str

    candidates: List[Dict[str, Any]]
    tried: List[int]
    index: int
    hits: int
    attempts: int

    items: List[Dict[str, Any]]
    filled: int
    scanned: int

    title: str
    url: str
    error: str
    message: str


def _display(platform: str) -> str:
    if not platform:
        return "the web"
    return DISPLAY_NAMES.get(platform.lower(), platform)


def _plan_label(plan: Dict[str, Any]) -> str:
    """Human-readable trace of the plan, shown in the panel as a ⚡ chip."""
    platform = plan.get("platform") or ""
    query = plan.get("query") or ""
    action = plan.get("action") or "open"
    site = _display(platform)

    if action == "new_meeting":
        return "Plan: open Google Meet \u2192 New \u2192 Start an instant meeting"
    if action == "click":
        target = plan.get("target") or "the visible control"
        return f"Plan: find \u201c{target}\u201d on this page and click it"
    if action == "fill_form":
        return "Plan: form filling is manual (Your Data \u2192 Fill this form)"
    if action == "travel":
        origin = plan.get("origin") or ""
        destination = plan.get("destination") or ""
        travel_date = plan.get("travel_date") or ""
        if origin and destination:
            route = f"{origin} → {destination}"
            when = f" on {travel_date}" if travel_date else ""
            return_date = plan.get("return_date") or ""
            if return_date:
                when += f"; return {return_date}"
            sort = "; sort by cheapest" if plan.get("sort") == "cheapest" else ""
            return f"Plan: search {site} flights {route}{when}{sort} → inspect booking options"
        if query:
            return f"Plan: open {site} for \u201c{query}\u201d \u2192 read live options"
        return f"Plan: open {site} \u2192 read live options"
    if action in ("compare_prices", "compare"):
        return f"Plan: search {site} for \u201c{query}\u201d \u2192 extract prices \u2192 sort in Python"
    if action in PLAY_ACTIONS:
        return f"Plan: open {site} \u2192 search \u201c{query}\u201d \u2192 play the best match"
    if action == "search":
        if query:
            return f"Plan: search \u201c{query}\u201d on {site}"
        return f"Plan: open {site}"
    return f"Plan: open {site}"


def build_browser_graph(deps: Any):
    """Compile the Browzy graph.

    ``deps`` is a namespace of callables injected by the backend so this module
    stays free of imports from ``backend.py`` (which runs as ``__main__``).
    """

    async def think(state: BrowzyState) -> Dict[str, Any]:
        """THINKING: instruction -> deterministic route or {platform, query, action}."""
        text = (state.get("instruction") or "").strip()

        # 1) Deterministic parser first. Explicit UI commands ("click New",
        # "start a new meeting") must never be reinterpreted by a model.
        plan: Optional[Dict[str, Any]] = deps.parse(text)
        source = "rules"

        if plan:
            raw_action = (plan.get("action") or "open").lower()
            if raw_action in ("click", "new_meeting"):
                plan["action"] = raw_action
                await deps.emit(
                    state["ws"], state["session_id"], "plan", {"plan": plan},
                    _plan_label(plan),
                )
                return {
                    "plan": plan,
                    "plan_source": source,
                    "route": raw_action,
                    "attempts": 0,
                    "tried": [],
                }
            if raw_action in ("fill_form", "fillform", "autofill", "fill"):
                # Personal data is only filled from the Your Data panel button.
                await deps.emit(
                    state["ws"], state["session_id"], "plan", {"mode": "guidance"},
                    _plan_label({"action": "fill_form"}),
                )
                return {
                    "plan": {},
                    "plan_source": source,
                    "route": "fill_guidance",
                    "attempts": 0,
                    "tried": [],
                }
            if raw_action == "travel":
                plan["action"] = "travel"
                await deps.emit(
                    state["ws"], state["session_id"], "plan", {"plan": plan},
                    _plan_label(plan),
                )
                return {
                    "plan": plan,
                    "plan_source": source,
                    "route": "travel",
                    "attempts": 0,
                    "tried": [],
                }

        # 2) Multi-step website tasks must be decided by the browser agent. A
        # deterministic search plan cannot safely infer which compose/reply
        # controls are present on a live site.
        if deps.needs_agent(text):
            await deps.emit(
                state["ws"], state["session_id"], "plan", {"mode": "agent"},
                "Plan: inspect the page and complete the requested task",
            )
            return {"plan": {}, "route": "agent", "plan_source": "agent"}

        # 3) Model planner: only for open-ended navigation/search/play phrasing.
        if not plan:
            plan = await deps.plan_with_model(state.get("model"), text)
            source = "model"

        if not plan:
            return {"plan": {}, "route": "agent", "plan_source": "none"}

        action = (plan.get("action") or "open").lower()
        if action in PLAY_ACTIONS:
            action = "play"
        elif action in ("fill_form", "fillform", "autofill", "fill"):
            return {
                "plan": {},
                "plan_source": source,
                "route": "fill_guidance",
                "attempts": 0,
                "tried": [],
            }
        elif action in ("compare_prices", "price_compare", "compare", "cheapest"):
            action = "compare"
        elif action not in ("open", "search"):
            # A model must never invent an action outside this allow-list.
            return {"plan": {}, "route": "agent", "plan_source": source}

        # Only YouTube has a playback strategy; everywhere else "play X" means
        # "search for X" and stop at the results.
        if action == "play" and (plan.get("platform") or "").lower() not in ("youtube", "yt"):
            action = "search"
        store = (plan.get("platform") or "").strip().lower()
        if action == "compare" and (
            not store
            or store in ("google", "google shopping", "google_shopping", "googleshopping")
        ):
            # No marketplace was named. Let the reasoning layer research live
            # stores (Google Shopping or ask) instead of assuming Amazon.
            plan["action"] = "compare_prices"
            query = (plan.get("query") or "the product").strip()
            await deps.emit(
                state["ws"], state["session_id"], "plan", {"mode": "agent"},
                f"Plan: compare live prices for \u201c{query}\u201d without assuming a store",
            )
            return {
                "plan": plan,
                "plan_source": source,
                "route": "agent",
                "attempts": 0,
                "tried": [],
            }
        plan["action"] = action

        await deps.emit(
            state["ws"], state["session_id"], "plan", {"plan": plan},
            _plan_label(plan),
        )
        return {
            "plan": plan,
            "plan_source": source,
            "route": action,
            "attempts": 0,
            "tried": [],
        }

    async def open_site(state: BrowzyState) -> Dict[str, Any]:
        """DOING: navigate to the platform home."""
        page = deps.page()
        if page is None:
            raise RuntimeError("The browser window is not open.")

        plan = state.get("plan") or {}
        platform = (plan.get("platform") or "").lower()
        query = plan.get("query") or ""
        action = state.get("route") or "open"

        home = plan.get("url") or await deps.home(platform, query)
        await deps.focus(page)
        await deps.emit(
            state["ws"], state["session_id"], "open_url", {"url": home},
            f"Opening {_display(platform)}",
        )
        await deps.goto(page, home)
        await deps.dismiss(page)

        if action == "open":
            title = await page.title()
            return {
                "message": f"Opened {title or _display(platform)}.\n\n{page.url}",
                "title": title,
                "url": page.url,
            }
        return {"home_url": home}

    async def travel(state: BrowzyState) -> Dict[str, Any]:
        """DOING: open the right travel surface, then reason on live results."""
        page = deps.page()
        if page is None:
            raise RuntimeError("The browser window is not open.")

        plan = state.get("plan") or {}
        platform = (plan.get("platform") or "").lower()
        query = (plan.get("query") or "").strip()
        url = plan.get("url") or await deps.home(platform, query)

        await deps.focus(page)
        await deps.emit(
            state["ws"], state["session_id"], "open_url", {"url": url},
            f"Opening {_display(platform)}",
        )
        await deps.goto(page, url)
        await deps.dismiss(page)

        # Stop cleanly when a travel site presents a CAPTCHA or automation
        # challenge. Passing a blocked interstitial to the local model wastes
        # several slow CPU turns and can produce a false "searched" response.
        try:
            try:
                await page.locator("body").wait_for(state="visible", timeout=5000)
                await page.wait_for_load_state("networkidle", timeout=7000)
            except Exception:
                # Some travel pages keep background requests open; inspect the
                # current DOM after the bounded readiness wait either way.
                pass
            try:
                await page.wait_for_function(
                    "() => /captcha|verify you are human|automated requests/i.test(" 
                    "+ ((document.body && document.body.innerText) || '') + ' ' + location.href)",
                    timeout=6000,
                )
            except Exception:
                pass
            current_url = (page.url or "").lower()
            body_text = await page.evaluate(
                "document.body ? document.body.innerText : ''"
            )
            body_lower = (body_text or "").lower()
            challenge_markers = (
                "captcha-v2",
                "captcha",
                "verify you are human",
                "automated requests",
                '"reason":"blocked"',
            )
            if any(
                marker in current_url or marker in body_lower
                for marker in challenge_markers
            ):
                return {
                    "message": (
                        f"{_display(platform)} blocked automated access with a "
                        "verification challenge. I left the page open and did "
                        "not try to bypass it. Complete the challenge in the "
                        "browser, then ask me to continue.\n\n"
                        f"{page.url}"
                    ),
                    "url": page.url,
                }
        except Exception as exc:
            print(f"[Travel] challenge check unavailable: {exc}", flush=True)

        try:
            answer = await deps.agent(
                state["ws"], state["session_id"], state.get("model"),
                bool(state.get("thinking_model")),
            )
        except Exception as exc:
            print(f"[Travel] live reasoning failed: {exc}", flush=True)
            answer = ""

        if not (answer or "").strip():
            title = await page.title()
            answer = (
                f"Opened {title or _display(platform)}, but couldn't verify the "
                f"flight options for \u201c{query}\u201d.\n\n{page.url}"
            )
        return {"message": answer, "url": page.url}

    async def search(state: BrowzyState) -> Dict[str, Any]:
        """DOING: type the clean query into the site's own search box."""
        page = deps.page()
        if page is None:
            raise RuntimeError("The browser window is not open.")

        plan = state.get("plan") or {}
        platform = (plan.get("platform") or "").lower()
        query = plan.get("query") or ""
        action = state.get("route") or "search"

        if not query:
            return {"results_url": page.url}

        if platform == "youtube":
            await deps.emit(
                state["ws"], state["session_id"], "type_text",
                {"text": query, "selector": "input#search"},
                f"Searching YouTube for \u201c{query}\u201d",
            )
            await deps.search(page, query)
            result: Dict[str, Any] = {"results_url": page.url}
            if action != "play":
                result["message"] = (
                    f"Here are YouTube results for \u201c{query}\u201d.\n\n{page.url}"
                )
            return result

        if platform in ("", "google", "duckduckgo", "ddg"):
            url = (
                "https://www.google.com/search?q=" + deps.quote(query)
                if platform == "google"
                else "https://duckduckgo.com/?q=" + deps.quote(query)
            )
            await deps.emit(
                state["ws"], state["session_id"], "search", {"query": query},
                f"Searching for \u201c{query}\u201d",
            )
            await deps.goto(page, url)
            await deps.dismiss(page)
            return {"results_url": page.url, "message": f"Here are results for \u201c{query}\u201d.\n\n{page.url}"}

        # Any other site: use its own search field when it has one.
        await deps.emit(
            state["ws"], state["session_id"], "search", {"query": query},
            f"Searching {_display(platform)} for \u201c{query}\u201d",
        )
        try:
            field = page.locator(
                "input[type='search'], input[name='q'], input[type='text']"
            ).first
            await field.fill(query, timeout=5000)
            await page.keyboard.press("Enter")
            try:
                await page.wait_for_load_state("domcontentloaded", timeout=8000)
            except Exception:
                # Single page applications may update without a navigation.
                pass
        except Exception:
            url = "https://duckduckgo.com/?q=" + deps.quote(f"site:{platform} {query}")
            await deps.goto(page, url)
        return {
            "results_url": page.url,
            "message": f"Searched {_display(platform)} for \u201c{query}\u201d.\n\n{page.url}",
        }

    async def select(state: BrowzyState) -> Dict[str, Any]:
        """DOING: score the results and choose the best untried match."""
        page = deps.page()
        plan = state.get("plan") or {}
        query = plan.get("query") or ""
        tried = list(state.get("tried") or [])

        rows = await deps.rank(page, query)
        usable = [row for row in rows if row.get("index") not in tried]
        if not usable:
            usable = rows

        if not usable:
            return {"error": "No playable video was found in the results."}

        # Highest token overlap wins; ties fall back to page order.
        usable.sort(key=lambda row: (-int(row.get("hits") or 0), int(row.get("index") or 0)))
        best = usable[0]
        return {
            "index": int(best.get("index") or 0),
            "hits": int(best.get("hits") or 0),
            "candidates": rows,
        }

    async def play(state: BrowzyState) -> Dict[str, Any]:
        """DOING: click the chosen result."""
        page = deps.page()
        index = int(state.get("index") or 0)
        attempts = int(state.get("attempts") or 0)

        await deps.emit(
            state["ws"], state["session_id"], "click",
            {"text": f"result #{index + 1}"},
            "Playing the first video" if not attempts else "Trying the next matching result",
        )
        started = await deps.click(page, index)
        if not started:
            return {"attempts": attempts + 1, "error": "The result would not open."}
        return {"attempts": attempts + 1}

    async def verify(state: BrowzyState) -> Dict[str, Any]:
        """DOING: confirm the player is really running, then answer."""
        page = deps.page()
        info = await deps.player(page)
        plan = state.get("plan") or {}
        query = plan.get("query") or ""
        title = (info.get("title") or "").replace(" - YouTube", "").strip()

        if info.get("unavailable"):
            return {"error": "That video is unavailable.", "url": page.url}

        if info.get("playing"):
            if state.get("hits"):
                detail = f"Playing \u201c{title}\u201d."
            else:
                detail = (
                    f"No result matched \u201c{query}\u201d exactly, so I played the "
                    f"top result: \u201c{title}\u201d."
                )
            return {
                "message": f"{detail}\n\n{info.get('url') or page.url}",
                "url": info.get("url") or page.url,
            }

        if int(state.get("attempts") or 0) >= MAX_SPECIALIZED_RETRIES:
            return {
                "error": (
                    "Playback did not start after "
                    f"{MAX_SPECIALIZED_RETRIES} attempts."
                ),
                "url": page.url,
            }
        return {"error": "Playback did not start."}

    async def retry(state: BrowzyState) -> Dict[str, Any]:
        """Recover: return to the results page and try the next candidate."""
        page = deps.page()
        tried = list(state.get("tried") or []) + [int(state.get("index") or 0)]
        results_url = state.get("results_url")
        if results_url:
            await deps.goto(page, results_url)
        else:
            try:
                await page.go_back()
            except Exception:
                pass
        return {"tried": tried, "error": None}

    async def click_control(state: BrowzyState) -> Dict[str, Any]:
        """DOING: click the named visible control on the current page."""
        page = deps.page()
        if page is None:
            raise RuntimeError("The browser window is not open.")

        target = ((state.get("plan") or {}).get("target") or "").strip()
        if not target:
            return {"error": "Tell me which button or link to click."}

        await deps.focus(page)
        await deps.emit(
            state["ws"], state["session_id"], "click", {"text": target},
            f"Clicking \u201c{target}\u201d",
        )
        result = await deps.click_control(page, target)
        if not result.get("ok"):
            return {
                "error": result.get("error")
                or f"I couldn't find a visible \u201c{target}\u201d control on this page."
            }

        clicked = result.get("text") or target
        title = await page.title()
        return {
            "message": f"Clicked \u201c{clicked}\u201d.\n\n{page.url}",
            "title": title,
            "url": page.url,
        }

    async def new_meeting(state: BrowzyState) -> Dict[str, Any]:
        """DOING: open Google Meet and start an instant meeting."""
        page = deps.page()
        if page is None:
            raise RuntimeError("The browser window is not open.")

        await deps.focus(page)
        result = await deps.start_meeting(
            page, state["ws"], state["session_id"]
        )
        if not result.get("ok"):
            return {
                "error": result.get("error")
                or "I couldn't start a new Google Meet. Please try again."
            }

        return {
            "message": result.get("message") or "Started a new Google Meet.",
            "url": page.url,
            "title": await page.title(),
        }

    async def fill_guidance(state: BrowzyState) -> Dict[str, Any]:
        """Form filling is manual: guide the user, never autofill from chat."""
        return {
            "message": (
                "Form filling only happens when you press \u201cFill this form\u201d "
                "in the Your Data tab. Open Your Data, review your saved details, "
                "then press the Fill this form button while on the page you want "
                "filled. I won't type personal data from chat."
            )
        }

    async def compare_prices(state: BrowzyState) -> Dict[str, Any]:
        """DOING: scrape listings, then sort them numerically in Python."""
        page = deps.page()
        if page is None:
            raise RuntimeError("The browser window is not open.")

        plan = state.get("plan") or {}
        query = (plan.get("query") or "").strip()
        platform = (plan.get("platform") or "").strip().lower()
        if not query:
            return {"error": "Tell me which product you want to compare."}
        if not platform:
            return {
                "error": (
                    "Which store should I compare prices on? Name a marketplace "
                    "such as Amazon, Flipkart, eBay or Walmart."
                )
            }

        await deps.focus(page)
        result = await deps.compare(
            page, platform, query, state["ws"], state["session_id"],
            plan.get("filters") or {},
        )
        return {
            "items": result.get("items") or [],
            "scanned": int(result.get("scanned") or 0),
            "message": result.get("message") or "",
            "url": result.get("url") or page.url,
        }

    async def agent(state: BrowzyState) -> Dict[str, Any]:
        """Fallback lane: open-ended task for the model's tool loop."""
        answer = await deps.agent(
            state["ws"], state["session_id"], state.get("model"),
            bool(state.get("thinking_model")),
        )
        if not (answer or "").strip():
            raise RuntimeError("The model did not return a final answer.")
        return {"message": answer}

    def after_think(state: BrowzyState) -> str:
        route = state.get("route") or "open"
        if route in ("click", "new_meeting", "fill_guidance", "travel", "compare", "agent"):
            return route
        return "open"

    def after_open(state: BrowzyState) -> str:
        route = state.get("route") or "open"
        return "done" if route == "open" else "search"

    def after_search(state: BrowzyState) -> str:
        return "select" if state.get("route") == "play" else "done"

    def after_verify(state: BrowzyState) -> str:
        if state.get("message") or state.get("error"):
            return "done"
        return "retry"

    builder = StateGraph(BrowzyState)
    builder.add_node("think", think)
    builder.add_node("open", open_site)
    builder.add_node("search", search)
    builder.add_node("select", select)
    builder.add_node("play", play)
    builder.add_node("verify", verify)
    builder.add_node("retry", retry)
    builder.add_node("click", click_control)
    builder.add_node("new_meeting", new_meeting)
    builder.add_node("fill_guidance", fill_guidance)
    builder.add_node("travel", travel)
    builder.add_node("compare", compare_prices)
    builder.add_node("agent", agent)

    builder.add_edge(START, "think")
    builder.add_conditional_edges(
        "think",
        after_think,
        {
            "open": "open",
            "click": "click",
            "new_meeting": "new_meeting",
            "fill_guidance": "fill_guidance",
            "travel": "travel",
            "compare": "compare",
            "agent": "agent",
        },
    )
    builder.add_conditional_edges("open", after_open, {"search": "search", "done": END})
    builder.add_conditional_edges("search", after_search, {"select": "select", "done": END})
    builder.add_edge("select", "play")
    builder.add_edge("play", "verify")
    builder.add_conditional_edges("verify", after_verify, {"retry": "retry", "done": END})
    builder.add_edge("retry", "select")
    builder.add_edge("click", END)
    builder.add_edge("new_meeting", END)
    builder.add_edge("fill_guidance", END)
    builder.add_edge("travel", END)
    builder.add_edge("compare", END)
    builder.add_edge("agent", END)

    return builder.compile()


async def run_browser_graph(app: Any, state: Dict[str, Any]) -> Dict[str, Any]:
    """Invoke the compiled graph and return its final state."""
    return await app.ainvoke(state, config={"recursion_limit": 30})
