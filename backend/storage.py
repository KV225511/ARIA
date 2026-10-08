from __future__ import annotations

import asyncio
from pathlib import Path
import re
import uuid

from backend.settings import AppSettings


class LocalPrivateStorage:
    def __init__(self, settings: AppSettings):
        self.root = settings.storage_root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def object_key(owner_id: uuid.UUID, document_id: uuid.UUID) -> str:
        return f"{owner_id}/{document_id}.pdf"

    def resolve(self, object_key: str) -> Path:
        if not re.fullmatch(r"[0-9a-f-]{36}/[0-9a-f-]{36}\.pdf", object_key):
            raise ValueError("Invalid private object key")
        path = (self.root / object_key).resolve()
        if self.root not in path.parents:
            raise ValueError("Private object escaped storage root")
        return path

    async def put(self, object_key: str, content: bytes) -> None:
        path = self.resolve(object_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f".tmp-{uuid.uuid4()}")
        await asyncio.to_thread(temporary.write_bytes, content)
        await asyncio.to_thread(temporary.replace, path)

    async def read(self, object_key: str) -> bytes:
        return await asyncio.to_thread(self.resolve(object_key).read_bytes)

    async def delete(self, object_key: str) -> None:
        path = self.resolve(object_key)
        try:
            await asyncio.to_thread(path.unlink)
        except FileNotFoundError:
            pass

    def path(self, object_key: str) -> Path:
        return self.resolve(object_key)
