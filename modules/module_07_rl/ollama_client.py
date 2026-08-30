"""Shared, bounded Ollama client for synthetic-data generation."""

from __future__ import annotations

import asyncio
import random
from collections import Counter
from typing import Any

import httpx


class BoundedOllamaClient:
    """Reuse HTTP connections and cap in-flight requests per model.

    The client deliberately separates candidate/interviewer traffic from judge
    traffic. This makes concurrency explicit and prevents an episode-level
    semaphore from hiding unbounded API fan-out in future callers.
    """

    def __init__(
        self,
        host: str,
        model_limits: dict[str, int],
        timeout: float = 300.0,
        max_retries: int = 2,
        single_model_residency: bool = True,
    ):
        if not model_limits or any(limit <= 0 for limit in model_limits.values()):
            raise ValueError("Every Ollama model concurrency limit must be positive")
        self.host = host.rstrip("/")
        self.timeout = float(timeout)
        self.max_retries = int(max_retries)
        self.single_model_residency = bool(single_model_residency)
        self._limits = dict(model_limits)
        self._semaphores = {
            model: asyncio.Semaphore(limit) for model, limit in model_limits.items()
        }
        self._phase_condition = asyncio.Condition()
        self._active_model: str | None = None
        self._active_requests = 0
        self._waiting = Counter()
        max_connections = max(sum(model_limits.values()) * 2, 4)
        self._client = httpx.AsyncClient(
            limits=httpx.Limits(
                max_connections=max_connections,
                max_keepalive_connections=max_connections,
            ),
            timeout=self.timeout,
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _enter_model_phase(self, model: str) -> None:
        if not self.single_model_residency:
            return
        async with self._phase_condition:
            self._waiting[model] += 1
            try:
                await self._phase_condition.wait_for(
                    lambda: self._active_model in (None, model)
                )
                if self._active_model is None:
                    self._active_model = model
                self._active_requests += 1
            finally:
                self._waiting[model] -= 1

    async def _leave_model_phase(self, model: str) -> None:
        if not self.single_model_residency:
            return
        async with self._phase_condition:
            self._active_requests -= 1
            if self._active_requests == 0:
                # Continue the current model while it has queued work; otherwise
                # yield residency to whichever model has been waiting.
                if self._waiting[model] == 0:
                    self._active_model = None
                self._phase_condition.notify_all()

    async def generate(self, payload: dict[str, Any]) -> dict[str, Any]:
        model = str(payload.get("model", ""))
        if model not in self._semaphores:
            raise ValueError(f"No concurrency limit configured for Ollama model {model!r}")
        semaphore = self._semaphores[model]
        await self._enter_model_phase(model)
        try:
            async with semaphore:
                for attempt in range(self.max_retries + 1):
                    try:
                        response = await self._client.post(
                            f"{self.host}/api/generate",
                            json=payload,
                        )
                        response.raise_for_status()
                        return response.json()
                    except (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError):
                        if attempt >= self.max_retries:
                            raise
                        delay = min(0.5 * (2**attempt), 4.0)
                        await asyncio.sleep(delay + random.random() * delay * 0.2)
        finally:
            await self._leave_model_phase(model)
        raise RuntimeError("Unreachable Ollama retry state")
