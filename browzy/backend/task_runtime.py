"""Shared lifecycle for tasks that operate Browzy's single browser session."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional, TypeVar

T = TypeVar("T")
StateCallback = Callable[[str, dict[str, Any]], Awaitable[None]]

MAX_AGENT_TURNS = 10
MAX_AGENT_ACTIONS = 20
MAX_ACTION_FAILURES = 2
MAX_SPECIALIZED_RETRIES = 2


@dataclass
class BrowserTask:
    """Small, serializable view of the browser task currently in progress."""

    task_id: str
    instruction: str
    state: str = "queued"
    current_step: int = 0
    result: Optional[str] = None
    error: Optional[str] = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "instruction": self.instruction,
            "state": self.state,
            "current_step": self.current_step,
            "result": self.result,
            "error": self.error,
        }


class BrowserTaskRuntime:
    """Serialize access to the shared Playwright page and track task outcomes."""

    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        self.active: Optional[BrowserTask] = None

    async def run(
        self,
        task_id: str,
        instruction: str,
        operation: Callable[[], Awaitable[T]],
        on_state: Optional[StateCallback] = None,
    ) -> T:
        task = BrowserTask(task_id=task_id, instruction=instruction)
        await self._notify(task, on_state)

        async with self.lock:
            self.active = task
            task.state = "running"
            task.current_step = 1
            await self._notify(task, on_state)
            try:
                result = await operation()
                task.state = "done"
                task.result = str(result)[:2000]
                return result
            except asyncio.CancelledError:
                task.state = "stopped"
                raise
            except Exception as exc:
                task.state = "error"
                task.error = str(exc)[:1000]
                raise
            finally:
                await self._notify(task, on_state)
                self.active = None

    @staticmethod
    async def _notify(task: BrowserTask, callback: Optional[StateCallback]) -> None:
        if callback is None:
            return
        try:
            await callback(task.state, task.as_dict())
        except Exception:
            # Progress reporting must never mask a browser-task result.
            pass
