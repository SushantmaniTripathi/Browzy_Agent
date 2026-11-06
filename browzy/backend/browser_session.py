"""Ownership and lifecycle for Browzy's shared Chromium session."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional, Set

from fastapi import WebSocket


class ConnectionManager:
    """Tracks panel clients and the shared Playwright browser session."""

    def __init__(self) -> None:
        self.active_connections: Set[WebSocket] = set()
        self.browser = None
        self.context = None
        self.page = None
        self.playwright = None
        self.launch_lock = asyncio.Lock()
        self.browser_state = "stopped"
        self.last_error: Optional[str] = None
        self.models: List[Dict[str, Any]] = []
        self.ollama_online: Optional[bool] = None
        self.model_signature: tuple = ()
        self.model_poll_task: Optional[asyncio.Task] = None
        self.prefer_maximized: bool = True
        self.last_page_at: float = 0.0
        self.normal_since: Optional[float] = None
        self.window_watch_task: Optional[asyncio.Task] = None

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections.add(websocket)
        print(f"Client connected. Total connections: {len(self.active_connections)}", flush=True)

    def disconnect(self, websocket: WebSocket) -> None:
        self.active_connections.discard(websocket)
        print(f"Client disconnected. Total connections: {len(self.active_connections)}", flush=True)

    async def broadcast(self, message: dict) -> None:
        if not self.active_connections:
            return
        data = json.dumps(message)
        disconnected = set()
        for conn in self.active_connections:
            try:
                await conn.send_text(data)
            except Exception:
                disconnected.add(conn)
        for conn in disconnected:
            self.active_connections.discard(conn)

    async def send_personal(self, websocket: WebSocket, message: dict) -> None:
        try:
            await websocket.send_text(json.dumps(message))
        except Exception:
            self.disconnect(websocket)

    def is_browser_alive(self) -> bool:
        if self.context is None:
            return False
        try:
            _ = self.context.pages
            return True
        except Exception:
            return False
