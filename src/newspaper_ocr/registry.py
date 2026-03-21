"""Simple registry for backend discovery."""
from __future__ import annotations


class Registry:
    def __init__(self, name: str):
        self.name = name
        self._entries: dict[str, type] = {}

    def register(self, key: str, cls: type) -> None:
        self._entries[key] = cls

    def get(self, key: str) -> type:
        if key not in self._entries:
            raise KeyError(f"Unknown {self.name}: '{key}'. Available: {self.list()}")
        return self._entries[key]

    def list(self) -> list[str]:
        return sorted(self._entries.keys())
