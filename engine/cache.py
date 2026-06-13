"""
查询缓存 — 相同问题直接返回缓存，避免重复计算

缓存到 data/query_cache.json，文档变动时自动清空。
"""

import hashlib
import json
import time
from pathlib import Path

import config

MAX_ENTRIES = 500


class QueryCache:
    """查询缓存 (单例)"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._path = config.DATA_DIR / "query_cache.json"
        self._data: dict[str, dict] = self._load()

    def _key(self, question: str) -> str:
        return hashlib.md5(question.strip().encode()).hexdigest()

    def get(self, question: str) -> dict | None:
        """命中返回 {answer, hits, ...}，未命中返回 None"""
        k = self._key(question)
        return self._data.get(k)

    def set(self, question: str, answer: str, hits: list[dict]):
        """写入缓存"""
        k = self._key(question)
        self._data[k] = {
            "question": question,
            "answer": answer,
            "hits": [{"source": h.get("source", ""), "page": h.get("page", 1),
                      "content": h.get("content", "")[:200]} for h in hits],
            "ts": time.time(),
        }
        # LRU: 超限删最旧
        while len(self._data) > MAX_ENTRIES:
            oldest = min(self._data, key=lambda k: self._data[k]["ts"])
            del self._data[oldest]
        self._save()

    def clear(self):
        self._data = {}
        if self._path.exists():
            self._path.unlink()

    def _load(self) -> dict:
        if self._path.exists():
            try:
                return json.loads(self._path.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def _save(self):
        self._path.write_text(
            json.dumps(self._data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def __len__(self):
        return len(self._data)


cache = QueryCache()
